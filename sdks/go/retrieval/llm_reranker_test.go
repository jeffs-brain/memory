// SPDX-License-Identifier: Apache-2.0

package retrieval

import (
	"context"
	"errors"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

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

var _ llm.Provider = (*scriptedProvider)(nil)
