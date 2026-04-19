// SPDX-License-Identifier: Apache-2.0

package retrieval

import (
	"context"
	"errors"
)

type availabilityReranker interface {
	IsAvailable(ctx context.Context) bool
}

func rerankerAvailable(ctx context.Context, reranker Reranker) bool {
	if reranker == nil {
		return false
	}
	if probe, ok := reranker.(availabilityReranker); ok {
		return probe.IsAvailable(ctx)
	}
	return true
}

// AutoReranker prefers a primary reranker when it reports healthy and
// falls back to a secondary reranker when the primary is unavailable or
// the live call fails.
type AutoReranker struct {
	Primary  Reranker
	Fallback Reranker
	Label    string
}

func NewAutoReranker(primary, fallback Reranker) *AutoReranker {
	return &AutoReranker{
		Primary:  primary,
		Fallback: fallback,
		Label:    "auto-rerank",
	}
}

func (r *AutoReranker) Name() string {
	if r == nil || r.Label == "" {
		return "auto-rerank"
	}
	return r.Label
}

func (r *AutoReranker) IsAvailable(ctx context.Context) bool {
	if r == nil {
		return false
	}
	if rerankerAvailable(ctx, r.Primary) {
		return true
	}
	return rerankerAvailable(ctx, r.Fallback)
}

func (r *AutoReranker) Rerank(ctx context.Context, query string, chunks []RetrievedChunk) ([]RetrievedChunk, error) {
	if r == nil {
		return chunks, nil
	}
	if rerankerAvailable(ctx, r.Primary) {
		out, err := r.Primary.Rerank(ctx, query, chunks)
		if err == nil {
			return out, nil
		}
		if r.Fallback == nil {
			return nil, err
		}
	}
	if r.Fallback != nil {
		return r.Fallback.Rerank(ctx, query, chunks)
	}
	return nil, errors.New("auto-rerank: no available reranker")
}
