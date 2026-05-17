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
	adapter       Adapter
	maxDepth      int64
	backpressured atomic.Bool
	lastDepth     atomic.Int64
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

// Check queries the adapter for current pending job count using
// CountByStatus and updates the backpressure state. The brainID
// parameter scopes the count; pass an empty string for global depth.
func (bc *BackpressureChecker) Check(ctx context.Context, brainID string) (bool, error) {
	counts, err := bc.adapter.CountByStatus(ctx, brainID)
	if err != nil {
		return bc.backpressured.Load(), err
	}
	depth := int64(counts[StatusPending])
	bc.lastDepth.Store(depth)
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

// LastDepth returns the last observed queue depth from the most recent
// Check call. Returns 0 before the first check.
func (bc *BackpressureChecker) LastDepth() int64 {
	return bc.lastDepth.Load()
}
