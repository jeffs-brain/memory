// SPDX-License-Identifier: Apache-2.0

package retrieval

import (
	"context"
	"os"
	"strconv"
	"strings"
	"sync"
)

const rerankMaxConcurrent = 4

var rerankGates sync.Map

func acquireRerankSlot(ctx context.Context) (func(), error) {
	limit := rerankConcurrencyCap()
	gate := rerankGateFor(limit)
	select {
	case gate <- struct{}{}:
		return func() { <-gate }, nil
	case <-ctx.Done():
		return nil, ctx.Err()
	}
}

func rerankConcurrencyCap() int {
	raw := strings.TrimSpace(os.Getenv("JB_RERANK_CONCURRENCY"))
	if raw == "" {
		return rerankMaxConcurrent
	}
	parsed, err := strconv.Atoi(raw)
	if err != nil || parsed <= 0 {
		return rerankMaxConcurrent
	}
	return parsed
}

func rerankGateFor(limit int) chan struct{} {
	if limit <= 0 {
		limit = rerankMaxConcurrent
	}
	if gate, ok := rerankGates.Load(limit); ok {
		return gate.(chan struct{})
	}
	created := make(chan struct{}, limit)
	actual, _ := rerankGates.LoadOrStore(limit, created)
	return actual.(chan struct{})
}
