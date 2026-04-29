// SPDX-License-Identifier: Apache-2.0

package lme

import (
	"context"
	"fmt"

	"github.com/jeffs-brain/memory/go/llm"
)

// scriptedProvider is a minimal [llm.Provider] for testing the judge,
// reader and runner. It hands out pre-canned responses (and optional
// errors) in call order. Concurrent callers are serialised through the
// lock-free increment of callIdx, so the tests must not assume per-call
// ordering when they dispatch goroutines.
type scriptedProvider struct {
	responses []llm.CompleteResponse
	errors    []error
	callIdx   int
	maxCtx    int
	lastReqs  []llm.CompleteRequest
}

func (f *scriptedProvider) Complete(_ context.Context, req llm.CompleteRequest) (llm.CompleteResponse, error) {
	snap := req
	snap.Messages = append([]llm.Message(nil), req.Messages...)
	f.lastReqs = append(f.lastReqs, snap)
	if f.callIdx >= len(f.responses) && f.callIdx >= len(f.errors) {
		return llm.CompleteResponse{}, fmt.Errorf("scriptedProvider: no more canned responses")
	}
	idx := f.callIdx
	f.callIdx++

	var err error
	if idx < len(f.errors) {
		err = f.errors[idx]
	}
	if err != nil {
		return llm.CompleteResponse{}, err
	}

	if idx < len(f.responses) {
		return f.responses[idx], nil
	}
	return llm.CompleteResponse{}, fmt.Errorf("scriptedProvider: index %d out of range", idx)
}

func (f *scriptedProvider) CompleteStream(_ context.Context, _ llm.CompleteRequest) (<-chan llm.StreamChunk, error) {
	return nil, fmt.Errorf("scriptedProvider: streaming not implemented")
}

func (f *scriptedProvider) Close() error { return nil }

func (f *scriptedProvider) MaxContextTokens() int {
	if f.maxCtx == 0 {
		return 32_768
	}
	return f.maxCtx
}

// maxContextOnlyProvider is a [scriptedProvider] variant that only
// advertises a context window (no scripted responses) so the
// budget-inference tests can exercise the interface assertion path.
type maxContextOnlyProvider struct {
	maxCtx int
}

func (p *maxContextOnlyProvider) Complete(_ context.Context, _ llm.CompleteRequest) (llm.CompleteResponse, error) {
	return llm.CompleteResponse{}, fmt.Errorf("maxContextOnlyProvider: complete not implemented")
}

func (p *maxContextOnlyProvider) CompleteStream(_ context.Context, _ llm.CompleteRequest) (<-chan llm.StreamChunk, error) {
	return nil, fmt.Errorf("maxContextOnlyProvider: stream not implemented")
}

func (p *maxContextOnlyProvider) Close() error          { return nil }
func (p *maxContextOnlyProvider) MaxContextTokens() int { return p.maxCtx }
