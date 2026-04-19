// SPDX-License-Identifier: Apache-2.0

package retrieval

import (
	"context"
	"errors"
	"testing"
)

type availabilityStubReranker struct {
	name      string
	available bool
	err       error
	calls     int
}

func (r *availabilityStubReranker) Rerank(_ context.Context, _ string, chunks []RetrievedChunk) ([]RetrievedChunk, error) {
	r.calls++
	if r.err != nil {
		return nil, r.err
	}
	out := make([]RetrievedChunk, len(chunks))
	copy(out, chunks)
	if len(out) > 0 {
		out[0].RerankScore = float64(r.calls)
	}
	return out, nil
}

func (r *availabilityStubReranker) Name() string { return r.name }

func (r *availabilityStubReranker) IsAvailable(_ context.Context) bool { return r.available }

func TestAutoReranker_FallsBackWhenPrimaryUnavailable(t *testing.T) {
	t.Parallel()

	primary := &availabilityStubReranker{name: "primary", available: false}
	fallback := &availabilityStubReranker{name: "fallback", available: true}
	rr := NewAutoReranker(primary, fallback)

	out, err := rr.Rerank(context.Background(), "q", []RetrievedChunk{{ChunkID: "a"}})
	if err != nil {
		t.Fatalf("Rerank: %v", err)
	}
	if primary.calls != 0 {
		t.Fatalf("primary calls = %d, want 0", primary.calls)
	}
	if fallback.calls != 1 {
		t.Fatalf("fallback calls = %d, want 1", fallback.calls)
	}
	if out[0].RerankScore != 1 {
		t.Fatalf("fallback rerank score = %v, want 1", out[0].RerankScore)
	}
}

func TestAutoReranker_FallsBackWhenPrimaryErrors(t *testing.T) {
	t.Parallel()

	primary := &availabilityStubReranker{name: "primary", available: true, err: errors.New("boom")}
	fallback := &availabilityStubReranker{name: "fallback", available: true}
	rr := NewAutoReranker(primary, fallback)

	_, err := rr.Rerank(context.Background(), "q", []RetrievedChunk{{ChunkID: "a"}})
	if err != nil {
		t.Fatalf("Rerank: %v", err)
	}
	if primary.calls != 1 {
		t.Fatalf("primary calls = %d, want 1", primary.calls)
	}
	if fallback.calls != 1 {
		t.Fatalf("fallback calls = %d, want 1", fallback.calls)
	}
}
