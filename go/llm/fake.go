// SPDX-License-Identifier: Apache-2.0

package llm

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"math"
	"strings"
	"sync"
	"sync/atomic"
)

// NewFake returns a [Provider] that hands out responses from the provided
// slice in round-robin order. It is deterministic and requires no network.
// This is the provider downstream packages use in unit tests.
func NewFake(responses []string) Provider {
	copyOf := make([]string, len(responses))
	copy(copyOf, responses)
	return &fakeProvider{responses: copyOf}
}

type fakeProvider struct {
	responses []string
	idx       atomic.Uint64
	closed    atomic.Bool
}

func (f *fakeProvider) next() string {
	if len(f.responses) == 0 {
		return ""
	}
	i := f.idx.Add(1) - 1
	return f.responses[int(i%uint64(len(f.responses)))]
}

func (f *fakeProvider) Complete(ctx context.Context, req CompleteRequest) (CompleteResponse, error) {
	if f.closed.Load() {
		return CompleteResponse{}, context.Canceled
	}
	if err := ctx.Err(); err != nil {
		return CompleteResponse{}, err
	}
	if len(req.Messages) == 0 {
		return CompleteResponse{}, ErrEmpty
	}
	text := f.next()
	return CompleteResponse{
		Text:      text,
		Stop:      StopEndTurn,
		TokensIn:  len(concatUserText(req.Messages)),
		TokensOut: len(text),
	}, nil
}

func (f *fakeProvider) CompleteStream(ctx context.Context, req CompleteRequest) (<-chan StreamChunk, error) {
	if f.closed.Load() {
		return nil, context.Canceled
	}
	if len(req.Messages) == 0 {
		return nil, ErrEmpty
	}
	text := f.next()
	ch := make(chan StreamChunk, len(text)+1)
	go func() {
		defer close(ch)
		for _, r := range text {
			select {
			case <-ctx.Done():
				return
			case ch <- StreamChunk{DeltaText: string(r)}:
			}
		}
		select {
		case <-ctx.Done():
		case ch <- StreamChunk{Stop: StopEndTurn}:
		}
	}()
	return ch, nil
}

func (f *fakeProvider) Close() error {
	f.closed.Store(true)
	return nil
}

func concatUserText(msgs []Message) string {
	var b strings.Builder
	for _, m := range msgs {
		b.WriteString(m.Content)
	}
	return b.String()
}

// NewFakeEmbedder returns an [Embedder] that emits deterministic pseudo-
// random vectors seeded by a SHA-256 hash of each input string. This lets
// tests verify embedding-driven logic without touching a real model.
func NewFakeEmbedder(dims int) Embedder {
	if dims <= 0 {
		dims = 16
	}
	return &fakeEmbedder{dims: dims}
}

type fakeEmbedder struct {
	dims int
	mu   sync.Mutex
}

func (f *fakeEmbedder) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	out := make([][]float32, len(texts))
	for i, t := range texts {
		out[i] = seedVector(t, f.dims)
	}
	return out, nil
}

func (f *fakeEmbedder) Dimensions() int { return f.dims }

func (f *fakeEmbedder) Close() error { return nil }

// seedVector produces a unit-length deterministic vector for t.
func seedVector(t string, dims int) []float32 {
	vec := make([]float32, dims)
	if dims == 0 {
		return vec
	}
	seed := sha256.Sum256([]byte(t))
	var sumSq float64
	for i := 0; i < dims; i++ {
		offset := (i * 4) % len(seed)
		end := offset + 4
		if end > len(seed) {
			// Re-hash for bigger dims.
			nextSeed := sha256.Sum256(append(seed[:], byte(i)))
			seed = nextSeed
			offset = 0
			end = 4
		}
		raw := binary.BigEndian.Uint32(seed[offset:end])
		val := float64(raw)/float64(math.MaxUint32)*2 - 1
		vec[i] = float32(val)
		sumSq += val * val
	}
	if sumSq == 0 {
		return vec
	}
	norm := float32(math.Sqrt(sumSq))
	for i := range vec {
		vec[i] /= norm
	}
	return vec
}
