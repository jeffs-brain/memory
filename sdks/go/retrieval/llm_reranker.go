// SPDX-License-Identifier: Apache-2.0

package retrieval

import (
	"context"
	"errors"

	"github.com/jeffs-brain/memory/go/llm"
)

// LLMReranker is a [Reranker] backed by an [llm.Provider]. The
// reference implementation prompts the model to rank candidates by
// relevance and parses the response back into a permutation. The
// production-quality prompt and parser are deliberately deferred:
// landing the wiring without the prompt churn lets downstream callers
// integrate the rerank type today and swap the implementation in once
// the prompt template is tuned.
//
// TODO(rerank): replace the pass-through Rerank with the LLM
// cross-encoder prompt described in spec/RERANK.md once the prompt
// template stabilises. Until then this acts as an identity reranker:
// it does not reorder, does not assign RerankScore, and traces still
// flag Reranked=true so consumers can confirm the wiring is intact.
type LLMReranker struct {
	provider llm.Provider
	model    string
}

// NewLLMReranker constructs an [LLMReranker]. The provider is held by
// reference; the caller retains lifecycle ownership.
func NewLLMReranker(provider llm.Provider, model string) (*LLMReranker, error) {
	if provider == nil {
		return nil, errors.New("retrieval: NewLLMReranker requires a non-nil provider")
	}
	if model == "" {
		return nil, errors.New("retrieval: NewLLMReranker requires a non-empty model name")
	}
	return &LLMReranker{provider: provider, model: model}, nil
}

// Rerank implements [Reranker]. Currently a stub: returns the input
// chunks in their original order. See the package-level TODO(rerank).
func (r *LLMReranker) Rerank(ctx context.Context, query string, chunks []RetrievedChunk) ([]RetrievedChunk, error) {
	if r == nil {
		return chunks, nil
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	// TODO(rerank): build the cross-encoder prompt from
	// (query, ComposeRerankText(chunk)) pairs, call r.provider.Complete,
	// parse the response into a permutation, and reassemble chunks in
	// the new order with RerankScore populated.
	out := make([]RetrievedChunk, len(chunks))
	copy(out, chunks)
	return out, nil
}

// Name implements the namedReranker interface used by the retrieval
// trace, so consumers can attribute the rerank pass to the configured
// model.
func (r *LLMReranker) Name() string {
	if r == nil {
		return "llm-reranker"
	}
	return "llm:" + r.model
}

// compile-time interface check.
var _ Reranker = (*LLMReranker)(nil)
