// SPDX-License-Identifier: Apache-2.0

package retrieval

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"sort"
	"strings"

	"github.com/jeffs-brain/memory/go/llm"
)

// LLMRerankerDefaultMaxBatch is the default number of candidates shipped
// to the provider in a single rerank call when LLMReranker.MaxBatch is
// zero or negative. Five is small enough for a local Gemma-class model
// to chew through in well under a second; larger batches amortise the
// request overhead at the cost of response latency.
const LLMRerankerDefaultMaxBatch = 5

// llmRerankMaxTokens is the max_tokens budget given to the provider for
// the rerank response. A batch of five candidates at ~40 bytes per
// entry is roughly 200 bytes; 2048 tokens is comfortably above that
// even with whitespace and fences in the response.
const llmRerankMaxTokens = 2048

// llmRerankSnippetLimit caps the per-candidate body excerpt length
// emitted into the prompt so the assembled request stays inside
// typical provider input budgets even on a full-batch call. 1200 chars
// gives the reranker enough room to see dates, amounts, and short
// comparative clauses that often sit beyond the summary line.
const llmRerankSnippetLimit = 1200

// llmRerankSystemPrompt is the default instruction block shipped to
// the provider. Ported verbatim from
// jeff/apps/jeff/internal/knowledge/search_hybrid.go (rerankSystemPrompt
// at lines 724-728 in the 2026-04-15 revision) to preserve downstream
// parity with the jeff cross-encoder. British English is mandatory in
// both the prompt and the model's responses.
const llmRerankSystemPrompt = `You are scoring wiki articles against a user's question.
Return ONLY a JSON array of objects matching the input order:
[{"id": 0, "score": 8.5}, {"id": 1, "score": 2.0}, ...]
Score 0 means irrelevant. Score 10 means perfectly answers the
question. Use British English.`

// llmRerankSystemPromptStrict is the retry fallback used after the
// default prompt returns unparseable output. Ported verbatim from
// jeff/apps/jeff/internal/knowledge/search_hybrid.go
// (rerankSystemPromptStrict at lines 730-736 in the 2026-04-15
// revision). The stricter wording reliably coaxes a raw JSON array out
// of a model that started with prose on the first turn.
const llmRerankSystemPromptStrict = `You are scoring wiki articles against a user's question.
Return ONLY a raw JSON array and NOTHING ELSE. No prose, no markdown,
no backticks, no commentary. The array must have one object per input
article in the same order:
[{"id": 0, "score": 8.5}, {"id": 1, "score": 2.0}, ...]
Each score is a number between 0 (irrelevant) and 10 (perfect match).
Use British English.`

// LLMReranker scores each candidate with an LLM judge and re-orders
// the slice by the returned score. Failures degrade gracefully: any
// parse error, provider error, or context cancellation returns the
// input slice unchanged and records the reason for the caller.
//
// The prompt is ported verbatim from jeff's cross-encoder reference so
// rerank quality matches the upstream LongMemEval baseline. Batching is
// driven by MaxBatch so a large RerankTopN splits across multiple
// provider calls rather than blowing past the input window.
type LLMReranker struct {
	// Provider drives the completion calls. Required.
	Provider llm.Provider
	// Model is forwarded to the provider as CompleteRequest.Model. When
	// empty the provider's configured default is used.
	Model string
	// MaxBatch caps how many candidates go into one provider call. When
	// zero or negative, LLMRerankerDefaultMaxBatch applies.
	MaxBatch int
	// Logger receives warnings when a batch fails or produces malformed
	// output. Defaults to slog.Default when nil.
	Logger *slog.Logger
}

// NewLLMReranker constructs an [LLMReranker]. The provider is held by
// reference; the caller retains lifecycle ownership. An empty model is
// allowed because several providers (Anthropic, Ollama) read the model
// from their own configuration when CompleteRequest.Model is blank.
func NewLLMReranker(provider llm.Provider, model string) *LLMReranker {
	return &LLMReranker{
		Provider: provider,
		Model:    model,
		MaxBatch: LLMRerankerDefaultMaxBatch,
	}
}

// Rerank splits candidates into MaxBatch-sized chunks, scores each,
// then re-orders the full slice by score desc, RRF score, input
// position. A batch that fails or returns unparseable JSON contributes
// zero scores so affected candidates sink rather than winning silently.
func (r *LLMReranker) Rerank(ctx context.Context, query string, candidates []RetrievedChunk) ([]RetrievedChunk, error) {
	if r == nil {
		return candidates, nil
	}
	if r.Provider == nil {
		return nil, errors.New("retrieval: LLMReranker.Provider is nil")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if len(candidates) == 0 {
		return candidates, nil
	}
	release, err := acquireRerankSlot(ctx)
	if err != nil {
		return nil, err
	}
	defer release()

	batch := r.MaxBatch
	if batch <= 0 {
		batch = LLMRerankerDefaultMaxBatch
	}

	payloads := make([]llmRerankCandidate, len(candidates))
	for i, c := range candidates {
		payloads[i] = llmRerankCandidate{
			ID:      i,
			Path:    c.Path,
			Title:   c.Title,
			Summary: c.Summary,
			Snippet: composeLLMRerankSnippet(c),
		}
	}

	scores := make([]float64, len(candidates))
	scored := make([]bool, len(candidates))
	failures := 0
	for start := 0; start < len(candidates); start += batch {
		end := start + batch
		if end > len(candidates) {
			end = len(candidates)
		}
		slice := payloads[start:end]

		// Re-ID the slice so the prompt IDs are always 0..len(slice)-1.
		local := make([]llmRerankCandidate, len(slice))
		for li, c := range slice {
			local[li] = c
			local[li].ID = li
		}

		batchScores, err := r.callBatch(ctx, query, local)
		if err != nil {
			failures++
			r.warn("retrieval: LLMReranker batch failed", "query", query, "start", start, "err", err)
			continue
		}
		for li, s := range batchScores {
			gi := start + li
			if gi >= len(candidates) {
				break
			}
			scores[gi] = s
			scored[gi] = true
		}
	}

	// Every batch failed: hand back the input order so callers see a
	// stable fallback rather than an empty slice.
	anyScored := false
	for _, ok := range scored {
		if ok {
			anyScored = true
			break
		}
	}
	if !anyScored {
		out := make([]RetrievedChunk, len(candidates))
		copy(out, candidates)
		return out, nil
	}

	type scoredPair struct {
		chunk RetrievedChunk
		score float64
		order int
	}
	pairs := make([]scoredPair, len(candidates))
	for i, c := range candidates {
		pairs[i] = scoredPair{chunk: c, score: scores[i], order: i}
	}
	sort.SliceStable(pairs, func(i, j int) bool {
		if pairs[i].score != pairs[j].score {
			return pairs[i].score > pairs[j].score
		}
		if pairs[i].chunk.Score != pairs[j].chunk.Score {
			return pairs[i].chunk.Score > pairs[j].chunk.Score
		}
		return pairs[i].order < pairs[j].order
	})

	out := make([]RetrievedChunk, len(pairs))
	for i, p := range pairs {
		chunk := p.chunk
		chunk.RerankScore = p.score
		out[i] = chunk
	}
	if failures > 0 {
		r.warn("retrieval: LLMReranker partial rerank", "query", query, "failed_batches", failures)
	}
	return out, nil
}

// Name implements the namedReranker interface used by the retrieval
// trace, so consumers can attribute the rerank pass to the configured
// model.
func (r *LLMReranker) Name() string {
	if r == nil || r.Model == "" {
		return "llm-reranker"
	}
	return "llm:" + r.Model
}

func (r *LLMReranker) IsAvailable(_ context.Context) bool {
	return r != nil && r.Provider != nil
}

// callBatch tries the default system prompt once, then retries with
// the strict variant on parse failure so hallucinated prose collapses
// to a raw JSON array on the second turn.
func (r *LLMReranker) callBatch(ctx context.Context, query string, batch []llmRerankCandidate) ([]float64, error) {
	if len(batch) == 0 {
		return nil, nil
	}
	user := renderLLMRerankUserPrompt(query, batch)

	messages := []llm.Message{
		{Role: llm.RoleSystem, Content: llmRerankSystemPrompt},
		{Role: llm.RoleUser, Content: user},
	}
	req := llm.CompleteRequest{
		Model:       r.Model,
		Messages:    messages,
		MaxTokens:   llmRerankMaxTokens,
		Temperature: 0.1,
	}
	resp, err := r.Provider.Complete(ctx, req)
	if err == nil {
		if scores, parseErr := parseLLMRerankResponse(resp.Text, len(batch)); parseErr == nil {
			return scores, nil
		}
	}

	messages[0].Content = llmRerankSystemPromptStrict
	req.Messages = messages
	resp, err = r.Provider.Complete(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("llm reranker provider: %w", err)
	}
	scores, parseErr := parseLLMRerankResponse(resp.Text, len(batch))
	if parseErr != nil {
		return nil, fmt.Errorf("llm reranker parse: %w", parseErr)
	}
	return scores, nil
}

func (r *LLMReranker) warn(msg string, attrs ...any) {
	logger := r.Logger
	if logger == nil {
		logger = slog.Default()
	}
	logger.Warn(msg, attrs...)
}

// llmRerankCandidate carries the 0-indexed batch ID so responses can
// be zipped back to candidates without matching on path strings.
type llmRerankCandidate struct {
	ID      int
	Path    string
	Title   string
	Summary string
	Snippet string
}

// composeLLMRerankSnippet returns a normalised body excerpt for the
// rerank prompt so numeric values and dates survive even when title and
// summary are generic.
func composeLLMRerankSnippet(r RetrievedChunk) string {
	body := strings.Join(strings.Fields(r.Text), " ")
	if body == "" {
		return ""
	}
	if len(body) <= llmRerankSnippetLimit {
		return body
	}
	return body[:llmRerankSnippetLimit] + "..."
}

// renderLLMRerankUserPrompt mirrors jeff's renderRerankUserPrompt so
// rerank quality stays on parity with the upstream bench.
func renderLLMRerankUserPrompt(query string, candidates []llmRerankCandidate) string {
	var b strings.Builder
	b.WriteString("## Question\n")
	b.WriteString(strings.TrimSpace(query))
	b.WriteString("\n\n## Articles\n")
	for _, c := range candidates {
		title := strings.TrimSpace(c.Title)
		if title == "" {
			title = "(untitled)"
		}
		fmt.Fprintf(&b, "[%d] title: %s\n", c.ID, title)
		fmt.Fprintf(&b, "    path: %s\n", c.Path)
		summary := strings.TrimSpace(c.Summary)
		if summary == "" {
			summary = "(no summary available)"
		}
		fmt.Fprintf(&b, "    summary: %s\n\n", summary)
		snippet := strings.TrimSpace(c.Snippet)
		if snippet == "" {
			snippet = "(no body excerpt available)"
		}
		fmt.Fprintf(&b, "    content: %s\n\n", snippet)
	}
	return b.String()
}

// parseLLMRerankResponse coerces the provider's response text into an
// ordered slice of scores. Tolerates leading/trailing whitespace,
// markdown fences, trailing commentary after the array, and both
// object-form (`[{"id":0,"score":8.5}, ...]`) and bare-numeric
// (`[8.5, 2.0, ...]`) entries.
//
// expected is the number of candidates sent in the batch. The returned
// slice is always exactly that length: missing entries default to 0 and
// extra entries are dropped silently.
func parseLLMRerankResponse(raw string, expected int) ([]float64, error) {
	payload := extractJSONArray(raw)
	if payload == "" {
		return nil, fmt.Errorf("no JSON array found in response")
	}

	var objForm []struct {
		ID    *int     `json:"id"`
		Score *float64 `json:"score"`
	}
	if err := json.Unmarshal([]byte(payload), &objForm); err == nil && len(objForm) > 0 {
		scores := make([]float64, expected)
		for i, entry := range objForm {
			if entry.Score == nil {
				continue
			}
			idx := i
			if entry.ID != nil {
				idx = *entry.ID
			}
			if idx < 0 || idx >= expected {
				continue
			}
			scores[idx] = *entry.Score
		}
		return scores, nil
	}

	var numForm []float64
	if err := json.Unmarshal([]byte(payload), &numForm); err == nil {
		scores := make([]float64, expected)
		for i, v := range numForm {
			if i >= expected {
				break
			}
			scores[i] = v
		}
		return scores, nil
	}

	return nil, fmt.Errorf("response is neither object nor numeric JSON array: %q", truncateForError(payload))
}

// extractJSONArray pulls the outermost [...] substring out of the raw
// response, stripping any fenced-code wrappers.
func extractJSONArray(raw string) string {
	s := strings.TrimSpace(raw)
	if strings.HasPrefix(s, "```") {
		s = strings.TrimPrefix(s, "```")
		if nl := strings.IndexByte(s, '\n'); nl >= 0 {
			s = s[nl+1:]
		}
		if end := strings.LastIndex(s, "```"); end >= 0 {
			s = s[:end]
		}
		s = strings.TrimSpace(s)
	}
	start := strings.IndexByte(s, '[')
	end := strings.LastIndexByte(s, ']')
	if start < 0 || end < 0 || end <= start {
		return ""
	}
	return s[start : end+1]
}

func truncateForError(s string) string {
	if len(s) <= 120 {
		return s
	}
	return s[:120] + "..."
}

var _ Reranker = (*LLMReranker)(nil)
