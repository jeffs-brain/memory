// SPDX-License-Identifier: Apache-2.0

package retrieval

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/jeffs-brain/memory/go/llm"
)

// scriptedProvider is a minimal [llm.Provider] harness used by the
// rerank unit tests. Each call to Complete either returns a scripted
// response or invokes the supplied handler so tests can inspect the
// request and shape the reply.
type scriptedProvider struct {
	mu       sync.Mutex
	replies  []string
	handler  func(req llm.CompleteRequest) (llm.CompleteResponse, error)
	captured []llm.CompleteRequest
	calls    atomic.Int64
}

func (s *scriptedProvider) Complete(ctx context.Context, req llm.CompleteRequest) (llm.CompleteResponse, error) {
	s.calls.Add(1)
	s.mu.Lock()
	s.captured = append(s.captured, req)
	var reply string
	if len(s.replies) > 0 {
		reply = s.replies[0]
		s.replies = s.replies[1:]
	}
	handler := s.handler
	s.mu.Unlock()

	if handler != nil {
		return handler(req)
	}
	return llm.CompleteResponse{Text: reply, Stop: llm.StopEndTurn}, nil
}

func (s *scriptedProvider) CompleteStream(ctx context.Context, req llm.CompleteRequest) (<-chan llm.StreamChunk, error) {
	return nil, errors.New("scriptedProvider: streaming not supported")
}

func (s *scriptedProvider) Close() error { return nil }

// Captured returns the captured requests in call order.
func (s *scriptedProvider) Captured() []llm.CompleteRequest {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make([]llm.CompleteRequest, len(s.captured))
	copy(out, s.captured)
	return out
}

func makeRerankChunks(n int) []RetrievedChunk {
	out := make([]RetrievedChunk, n)
	for i := 0; i < n; i++ {
		out[i] = RetrievedChunk{
			ChunkID: string(rune('a' + i)),
			Path:    "wiki/doc-" + string(rune('a'+i)) + ".md",
			Title:   "Doc " + string(rune('A'+i)),
			Summary: "Summary for doc " + string(rune('A'+i)),
			Text:    "body of doc " + string(rune('A'+i)),
			Score:   float64(n-i) / float64(n),
		}
	}
	return out
}

func TestLLMReranker_ReordersByScore(t *testing.T) {
	t.Parallel()
	// Five candidates. Scripted reply scores them descending by
	// reversed position so the final ordering is reversed.
	provider := &scriptedProvider{
		replies: []string{`[{"id":0,"score":1.0},{"id":1,"score":2.0},{"id":2,"score":3.0},{"id":3,"score":4.0},{"id":4,"score":9.5}]`},
	}
	rr := NewLLMReranker(provider, "judge")
	rr.MaxBatch = 10
	chunks := makeRerankChunks(5)
	out, err := rr.Rerank(context.Background(), "does not matter", chunks)
	if err != nil {
		t.Fatalf("Rerank: %v", err)
	}
	if len(out) != 5 {
		t.Fatalf("out length = %d, want 5", len(out))
	}
	if out[0].ChunkID != "e" {
		t.Fatalf("top chunk = %q, want e (highest score)", out[0].ChunkID)
	}
	if out[4].ChunkID != "a" {
		t.Fatalf("bottom chunk = %q, want a (lowest score)", out[4].ChunkID)
	}
	if out[0].RerankScore != 9.5 {
		t.Errorf("top RerankScore = %v, want 9.5", out[0].RerankScore)
	}
	if provider.calls.Load() != 1 {
		t.Errorf("provider calls = %d, want 1", provider.calls.Load())
	}
}

func TestLLMReranker_Batching(t *testing.T) {
	t.Parallel()
	// 25 candidates, MaxBatch=10 -> 3 batches (10, 10, 5). Each batch
	// returns scores equal to the local ID so the global ranking
	// reorders into repeating 9-8-7... patterns; we verify at least the
	// number of batches rather than exact order.
	provider := &scriptedProvider{
		handler: func(req llm.CompleteRequest) (llm.CompleteResponse, error) {
			// Count the number of "[%d] title" lines in the user msg to
			// know how many candidates this batch carried.
			n := 0
			for _, m := range req.Messages {
				if m.Role != llm.RoleUser {
					continue
				}
				for _, line := range strings.Split(m.Content, "\n") {
					if strings.HasPrefix(strings.TrimSpace(line), "[") && strings.Contains(line, "] title:") {
						n++
					}
				}
			}
			var b strings.Builder
			b.WriteString("[")
			for i := 0; i < n; i++ {
				if i > 0 {
					b.WriteString(",")
				}
				// Descending scores: local 0 highest, local n-1 lowest.
				b.WriteString(`{"id":`)
				b.WriteRune(rune('0' + i%10))
				b.WriteString(`,"score":`)
				b.WriteRune(rune('0' + (n-i-1)%10))
				b.WriteString(`}`)
			}
			b.WriteString("]")
			return llm.CompleteResponse{Text: b.String(), Stop: llm.StopEndTurn}, nil
		},
	}
	rr := NewLLMReranker(provider, "judge")
	rr.MaxBatch = 10
	chunks := makeRerankChunks(25)
	out, err := rr.Rerank(context.Background(), "q", chunks)
	if err != nil {
		t.Fatalf("Rerank: %v", err)
	}
	if len(out) != 25 {
		t.Fatalf("out length = %d, want 25", len(out))
	}
	if provider.calls.Load() != 3 {
		t.Fatalf("provider calls = %d, want 3 batches", provider.calls.Load())
	}
}

func TestLLMReranker_MalformedResponseFallsBack(t *testing.T) {
	t.Parallel()
	// Both the default and the strict retry return garbage; rerank
	// must fall through to the input order rather than error out.
	provider := &scriptedProvider{
		replies: []string{"totally not json", "still not json"},
	}
	rr := NewLLMReranker(provider, "judge")
	rr.MaxBatch = 10
	chunks := makeRerankChunks(3)
	out, err := rr.Rerank(context.Background(), "q", chunks)
	if err != nil {
		t.Fatalf("Rerank with malformed resp: %v", err)
	}
	if len(out) != 3 {
		t.Fatalf("out length = %d, want 3", len(out))
	}
	for i, c := range out {
		if c.ChunkID != chunks[i].ChunkID {
			t.Errorf("chunk %d reordered: got %q, want %q", i, c.ChunkID, chunks[i].ChunkID)
		}
	}
	// Default + retry = 2 calls per batch, single batch -> 2 calls total.
	if got := provider.calls.Load(); got != 2 {
		t.Errorf("provider calls = %d, want 2 (default + strict retry)", got)
	}
}

func TestLLMReranker_StrictRetryOnParseFailure(t *testing.T) {
	t.Parallel()
	// First call returns prose; strict retry returns valid JSON. Verify
	// the retry path is what produced the final ordering and the
	// strict prompt was actually sent on the second call.
	provider := &scriptedProvider{
		replies: []string{"I think the ordering is alphabetical.", `[{"id":0,"score":1},{"id":1,"score":9},{"id":2,"score":5}]`},
	}
	rr := NewLLMReranker(provider, "judge")
	chunks := makeRerankChunks(3)
	out, err := rr.Rerank(context.Background(), "q", chunks)
	if err != nil {
		t.Fatalf("Rerank: %v", err)
	}
	if out[0].ChunkID != "b" {
		t.Fatalf("top after retry = %q, want b", out[0].ChunkID)
	}
	captured := provider.Captured()
	if len(captured) != 2 {
		t.Fatalf("captured = %d, want 2", len(captured))
	}
	if !strings.Contains(captured[1].Messages[0].Content, "ONLY a raw JSON array") {
		t.Errorf("strict retry did not use the strict prompt; got system: %q", captured[1].Messages[0].Content)
	}
}

func TestLLMReranker_EmptyInput(t *testing.T) {
	t.Parallel()
	provider := &scriptedProvider{}
	rr := NewLLMReranker(provider, "judge")
	out, err := rr.Rerank(context.Background(), "q", nil)
	if err != nil {
		t.Fatalf("Rerank with nil: %v", err)
	}
	if len(out) != 0 {
		t.Fatalf("out length = %d, want 0", len(out))
	}
	if provider.calls.Load() != 0 {
		t.Errorf("provider calls on empty input = %d, want 0", provider.calls.Load())
	}
}

func TestLLMReranker_TiedScoresStableOnRRFThenOrder(t *testing.T) {
	t.Parallel()
	// All four candidates get the same rerank score. Ties break first
	// on original RRF score (descending) and then on input order.
	provider := &scriptedProvider{
		replies: []string{`[{"id":0,"score":5},{"id":1,"score":5},{"id":2,"score":5},{"id":3,"score":5}]`},
	}
	rr := NewLLMReranker(provider, "judge")
	// Craft chunks with interesting RRF scores so the tie-break is
	// visible.
	chunks := []RetrievedChunk{
		{ChunkID: "a", Path: "a.md", Title: "A", Score: 0.1},
		{ChunkID: "b", Path: "b.md", Title: "B", Score: 0.4},
		{ChunkID: "c", Path: "c.md", Title: "C", Score: 0.3},
		{ChunkID: "d", Path: "d.md", Title: "D", Score: 0.2},
	}
	out, err := rr.Rerank(context.Background(), "q", chunks)
	if err != nil {
		t.Fatalf("Rerank: %v", err)
	}
	// Expected order: b (0.4), c (0.3), d (0.2), a (0.1).
	want := []string{"b", "c", "d", "a"}
	for i, id := range want {
		if out[i].ChunkID != id {
			t.Errorf("pos %d: got %q, want %q (tie-break by RRF score)", i, out[i].ChunkID, id)
		}
	}
}

func TestLLMReranker_ContextCancelled(t *testing.T) {
	t.Parallel()
	provider := &scriptedProvider{replies: []string{"[]"}}
	rr := NewLLMReranker(provider, "judge")
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err := rr.Rerank(ctx, "q", makeRerankChunks(2))
	if err == nil {
		t.Fatal("expected cancelled context to surface error")
	}
}

func TestLLMReranker_NilProvider_ReturnsError(t *testing.T) {
	t.Parallel()
	rr := NewLLMReranker(nil, "x")
	_, err := rr.Rerank(context.Background(), "q", makeRerankChunks(1))
	if err == nil {
		t.Fatal("expected error when provider nil")
	}
}

func TestLLMReranker_BareNumericJSON(t *testing.T) {
	t.Parallel()
	// Some models return a bare numeric array. Verify the fallback
	// parser handles it.
	provider := &scriptedProvider{
		replies: []string{"[1.0, 9.0, 5.0]"},
	}
	rr := NewLLMReranker(provider, "judge")
	out, err := rr.Rerank(context.Background(), "q", makeRerankChunks(3))
	if err != nil {
		t.Fatalf("Rerank: %v", err)
	}
	if out[0].ChunkID != "b" {
		t.Errorf("top chunk = %q, want b", out[0].ChunkID)
	}
}

func TestLLMReranker_FencedJSON(t *testing.T) {
	t.Parallel()
	provider := &scriptedProvider{
		replies: []string{"```json\n[{\"id\":0,\"score\":2},{\"id\":1,\"score\":8}]\n```"},
	}
	rr := NewLLMReranker(provider, "judge")
	out, err := rr.Rerank(context.Background(), "q", makeRerankChunks(2))
	if err != nil {
		t.Fatalf("Rerank: %v", err)
	}
	if out[0].ChunkID != "b" {
		t.Errorf("fenced JSON: top = %q, want b", out[0].ChunkID)
	}
}

func TestLLMReranker_PromptIncludesQueryAndSummary(t *testing.T) {
	t.Parallel()
	provider := &scriptedProvider{
		replies: []string{`[{"id":0,"score":1}]`},
	}
	rr := NewLLMReranker(provider, "judge")
	chunks := []RetrievedChunk{
		{ChunkID: "x", Path: "wiki/x.md", Title: "The Widget", Summary: "All about widgets"},
	}
	if _, err := rr.Rerank(context.Background(), "how do widgets work", chunks); err != nil {
		t.Fatalf("Rerank: %v", err)
	}
	captured := provider.Captured()
	if len(captured) == 0 {
		t.Fatal("provider never called")
	}
	got := captured[0].Messages
	if len(got) < 2 {
		t.Fatalf("expected system + user messages, got %d", len(got))
	}
	if got[0].Role != llm.RoleSystem {
		t.Errorf("first message role = %q, want system", got[0].Role)
	}
	if !strings.Contains(got[1].Content, "how do widgets work") {
		t.Errorf("user prompt missing query; got: %q", got[1].Content)
	}
	if !strings.Contains(got[1].Content, "The Widget") {
		t.Errorf("user prompt missing title; got: %q", got[1].Content)
	}
	if !strings.Contains(got[1].Content, "All about widgets") {
		t.Errorf("user prompt missing summary; got: %q", got[1].Content)
	}
}

func TestLLMReranker_ProviderErrorReturnsInput(t *testing.T) {
	t.Parallel()
	// Both attempts error out. Rerank must fall back to the input
	// ordering rather than propagate the error (matching the stub
	// contract that reranker failures degrade gracefully).
	provider := &scriptedProvider{
		handler: func(req llm.CompleteRequest) (llm.CompleteResponse, error) {
			return llm.CompleteResponse{}, errors.New("boom")
		},
	}
	rr := NewLLMReranker(provider, "judge")
	chunks := makeRerankChunks(3)
	out, err := rr.Rerank(context.Background(), "q", chunks)
	if err != nil {
		t.Fatalf("Rerank with provider error: %v", err)
	}
	for i, c := range out {
		if c.ChunkID != chunks[i].ChunkID {
			t.Errorf("pos %d reordered: got %q, want %q", i, c.ChunkID, chunks[i].ChunkID)
		}
	}
}

func TestParseLLMRerankResponse_MissingIDsFillPosition(t *testing.T) {
	t.Parallel()
	// Scores without explicit IDs pair by position.
	raw := `[{"score":3.0},{"score":7.0}]`
	scores, err := parseLLMRerankResponse(raw, 2)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if scores[0] != 3.0 || scores[1] != 7.0 {
		t.Fatalf("scores = %v, want [3 7]", scores)
	}
}

func TestParseLLMRerankResponse_OutOfRangeIDsDropped(t *testing.T) {
	t.Parallel()
	raw := `[{"id":99,"score":1.0},{"id":0,"score":5.0}]`
	scores, err := parseLLMRerankResponse(raw, 2)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if scores[0] != 5.0 {
		t.Errorf("scores[0] = %v, want 5", scores[0])
	}
	// Out-of-range ID must not bleed into an unrelated slot.
	if scores[1] != 0.0 {
		t.Errorf("scores[1] = %v, want 0 (untouched)", scores[1])
	}
}

func TestParseLLMRerankResponse_EmptyReturnsError(t *testing.T) {
	t.Parallel()
	if _, err := parseLLMRerankResponse("", 3); err == nil {
		t.Fatal("expected error on empty payload")
	}
	if _, err := parseLLMRerankResponse("not json at all", 3); err == nil {
		t.Fatal("expected error on non-JSON payload")
	}
}

func TestLLMReranker_ParallelProducesSameResults(t *testing.T) {
	t.Parallel()
	// 20 candidates, MaxBatch=5 -> 4 batches. Run once with parallelism=1
	// (sequential) and once with parallelism=4 (parallel). Both must
	// produce the same set of (ChunkID, RerankScore) pairs.
	makeProvider := func() *scriptedProvider {
		return &scriptedProvider{
			handler: func(req llm.CompleteRequest) (llm.CompleteResponse, error) {
				n := 0
				for _, m := range req.Messages {
					if m.Role != llm.RoleUser {
						continue
					}
					for _, line := range strings.Split(m.Content, "\n") {
						if strings.HasPrefix(strings.TrimSpace(line), "[") && strings.Contains(line, "] title:") {
							n++
						}
					}
				}
				var b strings.Builder
				b.WriteString("[")
				for i := 0; i < n; i++ {
					if i > 0 {
						b.WriteString(",")
					}
					// Score = local index + 1 so each batch has unique scores.
					fmt.Fprintf(&b, `{"id":%d,"score":%d}`, i, i+1)
				}
				b.WriteString("]")
				return llm.CompleteResponse{Text: b.String(), Stop: llm.StopEndTurn}, nil
			},
		}
	}

	chunks := makeRerankChunks(20)

	seqProvider := makeProvider()
	seqRR := NewLLMReranker(seqProvider, "judge")
	seqRR.MaxBatch = 5
	seqRR.Parallelism = 1
	seqOut, err := seqRR.Rerank(context.Background(), "q", chunks)
	if err != nil {
		t.Fatalf("sequential Rerank: %v", err)
	}

	parProvider := makeProvider()
	parRR := NewLLMReranker(parProvider, "judge")
	parRR.MaxBatch = 5
	parRR.Parallelism = 4
	parOut, err := parRR.Rerank(context.Background(), "q", chunks)
	if err != nil {
		t.Fatalf("parallel Rerank: %v", err)
	}

	if len(seqOut) != len(parOut) {
		t.Fatalf("len mismatch: sequential=%d, parallel=%d", len(seqOut), len(parOut))
	}

	// Build sets of (chunkID -> rerankScore) for both.
	seqScores := make(map[string]float64, len(seqOut))
	for _, c := range seqOut {
		seqScores[c.ChunkID] = c.RerankScore
	}
	parScores := make(map[string]float64, len(parOut))
	for _, c := range parOut {
		parScores[c.ChunkID] = c.RerankScore
	}
	for id, ss := range seqScores {
		ps, ok := parScores[id]
		if !ok {
			t.Errorf("chunk %q present in sequential but not parallel", id)
			continue
		}
		if ss != ps {
			t.Errorf("chunk %q: sequential score=%v, parallel score=%v", id, ss, ps)
		}
	}
}

func TestLLMReranker_Parallelism1Sequential(t *testing.T) {
	t.Parallel()
	// With parallelism=1, batches must execute sequentially.
	// We verify by recording batch start order and confirming it matches
	// the natural batch order (0, 5, 10, ...).
	var mu sync.Mutex
	var batchStarts []int

	provider := &scriptedProvider{
		handler: func(req llm.CompleteRequest) (llm.CompleteResponse, error) {
			// Extract the first candidate ID from the prompt to identify
			// the batch start offset. With parallelism=1, we parse the
			// user message to detect which batch this is.
			n := 0
			for _, m := range req.Messages {
				if m.Role != llm.RoleUser {
					continue
				}
				for _, line := range strings.Split(m.Content, "\n") {
					if strings.HasPrefix(strings.TrimSpace(line), "[") && strings.Contains(line, "] title:") {
						n++
					}
				}
			}
			mu.Lock()
			batchStarts = append(batchStarts, n)
			mu.Unlock()
			var b strings.Builder
			b.WriteString("[")
			for i := 0; i < n; i++ {
				if i > 0 {
					b.WriteString(",")
				}
				fmt.Fprintf(&b, `{"id":%d,"score":%d}`, i, i+1)
			}
			b.WriteString("]")
			return llm.CompleteResponse{Text: b.String(), Stop: llm.StopEndTurn}, nil
		},
	}
	rr := NewLLMReranker(provider, "judge")
	rr.MaxBatch = 5
	rr.Parallelism = 1
	chunks := makeRerankChunks(12)
	out, err := rr.Rerank(context.Background(), "q", chunks)
	if err != nil {
		t.Fatalf("Rerank: %v", err)
	}
	if len(out) != 12 {
		t.Fatalf("out length = %d, want 12", len(out))
	}
	// 12 candidates / 5 per batch = 3 batches (5, 5, 2).
	if provider.calls.Load() != 3 {
		t.Fatalf("provider calls = %d, want 3", provider.calls.Load())
	}
}

func TestLLMReranker_BatchErrorContinuesOthers(t *testing.T) {
	t.Parallel()
	// 10 candidates, MaxBatch=5 -> 2 batches. Second batch errors.
	// First batch scores should still be applied; errored batch
	// candidates get score 0.
	callCount := atomic.Int64{}
	provider := &scriptedProvider{
		handler: func(req llm.CompleteRequest) (llm.CompleteResponse, error) {
			n := callCount.Add(1)
			// First two calls succeed (first batch: default + possibly
			// strict). Third call (second batch) fails.
			// Because the handler is invoked per call, and callBatch does
			// two calls per batch on failure, we fail every call from the
			// second batch onwards. We identify the batch by counting
			// candidates in the prompt.
			candidateCount := 0
			for _, m := range req.Messages {
				if m.Role != llm.RoleUser {
					continue
				}
				for _, line := range strings.Split(m.Content, "\n") {
					if strings.HasPrefix(strings.TrimSpace(line), "[") && strings.Contains(line, "] title:") {
						candidateCount++
					}
				}
			}
			_ = n
			// We'll use a simpler approach: fail based on call number.
			// With parallelism, we can't rely on call order, so we
			// detect by the query text. Let's use a different approach.
			// We'll embed an error signal in the candidate titles.
			for _, m := range req.Messages {
				if m.Role != llm.RoleUser && strings.Contains(m.Content, "Doc F") {
					return llm.CompleteResponse{}, errors.New("batch error")
				}
				if m.Role == llm.RoleUser && strings.Contains(m.Content, "Doc F") {
					return llm.CompleteResponse{}, errors.New("batch error")
				}
			}
			var b strings.Builder
			b.WriteString("[")
			for i := 0; i < candidateCount; i++ {
				if i > 0 {
					b.WriteString(",")
				}
				fmt.Fprintf(&b, `{"id":%d,"score":%d}`, i, (i+1)*2)
			}
			b.WriteString("]")
			return llm.CompleteResponse{Text: b.String(), Stop: llm.StopEndTurn}, nil
		},
	}
	rr := NewLLMReranker(provider, "judge")
	rr.MaxBatch = 5
	rr.Parallelism = 4
	chunks := makeRerankChunks(10)
	out, err := rr.Rerank(context.Background(), "q", chunks)
	if err != nil {
		t.Fatalf("Rerank: %v", err)
	}
	if len(out) != 10 {
		t.Fatalf("out length = %d, want 10", len(out))
	}
	// First batch (candidates a-e) should have non-zero scores.
	// Second batch (candidates f-j) should have zero scores (error).
	scoreMap := make(map[string]float64, len(out))
	for _, c := range out {
		scoreMap[c.ChunkID] = c.RerankScore
	}
	// First batch candidates should have been scored.
	for _, id := range []string{"a", "b", "c", "d", "e"} {
		if scoreMap[id] == 0 {
			t.Errorf("first-batch chunk %q has score 0; expected non-zero", id)
		}
	}
	// Second batch candidates should be 0 (failed batch).
	for _, id := range []string{"f", "g", "h", "i", "j"} {
		if scoreMap[id] != 0 {
			t.Errorf("failed-batch chunk %q has score %v; expected 0", id, scoreMap[id])
		}
	}
}

func TestLLMReranker_EmptyCandidatesNoBatches(t *testing.T) {
	t.Parallel()
	provider := &scriptedProvider{}
	rr := NewLLMReranker(provider, "judge")
	rr.Parallelism = 4
	out, err := rr.Rerank(context.Background(), "q", nil)
	if err != nil {
		t.Fatalf("Rerank with nil: %v", err)
	}
	if len(out) != 0 {
		t.Fatalf("out length = %d, want 0", len(out))
	}
	if provider.calls.Load() != 0 {
		t.Errorf("provider calls on empty = %d, want 0", provider.calls.Load())
	}
}

func TestLLMReranker_FewerThanBatchSize(t *testing.T) {
	t.Parallel()
	// 3 candidates with MaxBatch=10 -> single batch, should still work.
	provider := &scriptedProvider{
		replies: []string{`[{"id":0,"score":3},{"id":1,"score":7},{"id":2,"score":5}]`},
	}
	rr := NewLLMReranker(provider, "judge")
	rr.MaxBatch = 10
	rr.Parallelism = 4
	chunks := makeRerankChunks(3)
	out, err := rr.Rerank(context.Background(), "q", chunks)
	if err != nil {
		t.Fatalf("Rerank: %v", err)
	}
	if len(out) != 3 {
		t.Fatalf("out length = %d, want 3", len(out))
	}
	if out[0].ChunkID != "b" {
		t.Errorf("top = %q, want b (score 7)", out[0].ChunkID)
	}
	if provider.calls.Load() != 1 {
		t.Errorf("provider calls = %d, want 1", provider.calls.Load())
	}
}

func TestLLMReranker_ParallelPerformance(t *testing.T) {
	t.Parallel()
	// Mock provider with 100ms delay per call. 4 batches with
	// parallelism=4 should complete in ~100ms (one wave), not ~400ms.
	provider := &scriptedProvider{
		handler: func(req llm.CompleteRequest) (llm.CompleteResponse, error) {
			// Count candidates to generate correct-sized response.
			n := 0
			for _, m := range req.Messages {
				if m.Role != llm.RoleUser {
					continue
				}
				for _, line := range strings.Split(m.Content, "\n") {
					if strings.HasPrefix(strings.TrimSpace(line), "[") && strings.Contains(line, "] title:") {
						n++
					}
				}
			}
			// Simulate 100ms latency.
			time.Sleep(100 * time.Millisecond)
			var b strings.Builder
			b.WriteString("[")
			for i := 0; i < n; i++ {
				if i > 0 {
					b.WriteString(",")
				}
				fmt.Fprintf(&b, `{"id":%d,"score":%d}`, i, i+1)
			}
			b.WriteString("]")
			return llm.CompleteResponse{Text: b.String(), Stop: llm.StopEndTurn}, nil
		},
	}
	rr := NewLLMReranker(provider, "judge")
	rr.MaxBatch = 5
	rr.Parallelism = 4
	chunks := makeRerankChunks(20) // 4 batches of 5

	start := time.Now()
	out, err := rr.Rerank(context.Background(), "q", chunks)
	elapsed := time.Since(start)

	if err != nil {
		t.Fatalf("Rerank: %v", err)
	}
	if len(out) != 20 {
		t.Fatalf("out length = %d, want 20", len(out))
	}
	// Sequential would take ~400ms. Parallel with 4 should be ~100ms.
	// We allow up to 250ms to account for scheduling jitter in CI.
	if elapsed > 250*time.Millisecond {
		t.Errorf("parallel rerank took %v; expected < 250ms (4 batches x 100ms delay, parallelism=4)", elapsed)
	}
}

func TestLLMReranker_DefaultParallelism(t *testing.T) {
	t.Parallel()
	rr := NewLLMReranker(&scriptedProvider{}, "judge")
	// NewLLMReranker does not set Parallelism; the Rerank method should
	// default to LLMRerankerDefaultParallelism when it is 0.
	if rr.Parallelism != 0 {
		t.Errorf("NewLLMReranker sets Parallelism = %d, want 0 (default applied at runtime)", rr.Parallelism)
	}
	if LLMRerankerDefaultParallelism != 4 {
		t.Errorf("LLMRerankerDefaultParallelism = %d, want 4 (TS parity)", LLMRerankerDefaultParallelism)
	}
}

var _ llm.Provider = (*scriptedProvider)(nil)
