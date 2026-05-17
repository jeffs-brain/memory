// SPDX-License-Identifier: Apache-2.0

package queue

import (
	"context"
	"sync/atomic"
)

// defaultMaxQueueDepth is the backpressure threshold per tenant.
// Modelled on RabbitMQ/SQS patterns where 1000 pending items per
// tenant signals the system should stop accepting new work.
const defaultMaxQueueDepth int64 = 1000

// BackpressureChecker evaluates whether the queue has exceeded its
// capacity threshold. When backpressured, producers should stop
// enqueuing new jobs until depth drops below the limit.
type BackpressureChecker struct {
	adapter      Adapter
	maxDepth     int64
	backpressured atomic.Bool
}

// NewBackpressureChecker creates a checker with the given depth limit.
// A maxDepth of zero or negative falls back to defaultMaxQueueDepth.
func NewBackpressureChecker(adapter Adapter, maxDepth int64) *BackpressureChecker {
	safeMax := maxDepth
	if safeMax <= 0 {
		safeMax = defaultMaxQueueDepth
	}
	return &BackpressureChecker{
		adapter:  adapter,
		maxDepth: safeMax,
	}
}

// Check queries the adapter for current queue depth and updates the
// backpressure state. The brainID parameter scopes the depth check;
// pass an empty string for global depth.
func (bc *BackpressureChecker) Check(ctx context.Context, brainID string) (bool, error) {
	depth, err := bc.adapter.Depth(ctx, brainID)
	if err != nil {
		return bc.backpressured.Load(), err
	}
	pressured := depth >= bc.maxDepth
	bc.backpressured.Store(pressured)
	return pressured, nil
}

// IsBackpressured returns the last known backpressure state without
// querying the adapter. This is safe to call from hot paths where a
// stale value is acceptable.
func (bc *BackpressureChecker) IsBackpressured() bool {
	return bc.backpressured.Load()
}

// MaxDepth returns the configured threshold.
func (bc *BackpressureChecker) MaxDepth() int64 {
	return bc.maxDepth
}
