// SPDX-License-Identifier: Apache-2.0

package ratelimit

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"sync"
	"sync/atomic"
	"time"

	"golang.org/x/time/rate"
)

// maxRetryAfter caps the maximum retry-after duration to 5 minutes,
// preventing a malicious or buggy server from pausing indefinitely.
const maxRetryAfter = 5 * time.Minute

// tokenBucket implements [Limiter] using golang.org/x/time/rate for the
// token bucket and an optional semaphore for concurrency control.
//
// NOTE: This is a single-process implementation. For multi-worker
// deployments requiring shared state, a Redis-backed bucket is planned
// as a follow-up (see LLE-XXXX).
type tokenBucket struct {
	mu       sync.RWMutex
	limiter  *rate.Limiter
	max      int
	refill   float64
	tenantID string
	logger   *slog.Logger

	// concurrency semaphore; nil when maxConcurrency <= 0.
	sem chan struct{}

	// stopCh is closed by Close() to cancel any in-flight retry-after
	// goroutines spawned by UpdateFromHeaders.
	stopCh chan struct{}

	// metrics
	waiting   atomic.Int64
	throttled atomic.Int64
}

// NewBucket creates a new in-memory token bucket limiter. Panics if
// MaxTokens or RefillRatePerSec are not positive.
func NewBucket(opts BucketOptions) Limiter {
	if opts.MaxTokens <= 0 {
		panic(fmt.Sprintf("ratelimit: MaxTokens must be positive, got %d", opts.MaxTokens))
	}
	if opts.RefillRatePerSec <= 0 {
		panic(fmt.Sprintf("ratelimit: RefillRatePerSec must be positive, got %f", opts.RefillRatePerSec))
	}

	logger := opts.Logger
	if logger == nil {
		logger = slog.New(slog.DiscardHandler)
	}

	b := &tokenBucket{
		limiter:  rate.NewLimiter(rate.Limit(opts.RefillRatePerSec), opts.MaxTokens),
		max:      opts.MaxTokens,
		refill:   opts.RefillRatePerSec,
		tenantID: opts.TenantID,
		logger:   logger,
		stopCh:   make(chan struct{}),
	}

	if opts.MaxConcurrency > 0 {
		b.sem = make(chan struct{}, opts.MaxConcurrency)
	}

	return b
}

// validateCost returns an error if cost is not a positive integer.
func validateCost(cost int) error {
	if cost < 1 {
		return errors.New("ratelimit: cost must be a positive integer (>= 1)")
	}
	return nil
}

// Acquire blocks until cost tokens are available, then optionally
// acquires a concurrency slot. Respects context cancellation.
func (b *tokenBucket) Acquire(ctx context.Context, cost int) (Token, error) {
	if err := validateCost(cost); err != nil {
		return Token{}, err
	}

	b.waiting.Add(1)
	defer b.waiting.Add(-1)

	b.mu.RLock()
	lim := b.limiter
	b.mu.RUnlock()

	if err := lim.WaitN(ctx, cost); err != nil {
		b.throttled.Add(1)
		return Token{}, err
	}

	if b.sem != nil {
		select {
		case b.sem <- struct{}{}:
		case <-ctx.Done():
			b.throttled.Add(1)
			return Token{}, ctx.Err()
		}
	}

	released := atomic.Bool{}
	return Token{
		Release: func() {
			if !released.CompareAndSwap(false, true) {
				return
			}
			if b.sem != nil {
				<-b.sem
			}
		},
	}, nil
}

// TryAcquire attempts a non-blocking token acquisition.
func (b *tokenBucket) TryAcquire(cost int) (Token, bool) {
	if err := validateCost(cost); err != nil {
		return Token{}, false
	}

	b.mu.RLock()
	lim := b.limiter
	b.mu.RUnlock()

	if !lim.AllowN(time.Now(), cost) {
		return Token{}, false
	}

	if b.sem != nil {
		select {
		case b.sem <- struct{}{}:
		default:
			return Token{}, false
		}
	}

	released := atomic.Bool{}
	return Token{
		Release: func() {
			if !released.CompareAndSwap(false, true) {
				return
			}
			if b.sem != nil {
				<-b.sem
			}
		},
	}, true
}

// SetRefillRate overrides the bucket's refill rate. Thread-safe.
func (b *tokenBucket) SetRefillRate(r float64) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.refill = r
	b.limiter.SetLimit(rate.Limit(r))
}

// UpdateFromHeaders adjusts the limiter parameters based on provider
// response headers. When remaining tokens drop below burst/4, the
// refill rate is halved. When a Retry-After header is present, the
// limiter is paused for the specified duration (capped at 5 minutes).
func (b *tokenBucket) UpdateFromHeaders(h Headers) {
	b.mu.Lock()
	defer b.mu.Unlock()

	// Retry-After: temporarily set rate to zero, schedule restoration.
	if h.RetryAfter > 0 {
		capped := h.RetryAfter
		if capped > maxRetryAfter {
			capped = maxRetryAfter
		}
		b.logger.Info("rate limiter pausing due to retry-after",
			"tenant", b.tenantID,
			"retryAfter", capped,
		)
		b.throttled.Add(1)
		savedRate := b.refill
		b.refill = 0
		b.limiter.SetLimit(0)
		stopCh := b.stopCh
		go func() {
			timer := time.NewTimer(capped)
			defer timer.Stop()
			select {
			case <-timer.C:
				b.mu.Lock()
				defer b.mu.Unlock()
				b.refill = savedRate
				b.limiter.SetLimit(rate.Limit(savedRate))
				b.logger.Info("rate limiter resumed after retry-after",
					"tenant", b.tenantID,
					"rate", savedRate,
				)
			case <-stopCh:
				// Bucket is closing; abandon the scheduled restore.
			}
		}()
		return
	}

	// Back off when remaining < burst/4.
	if h.Remaining > 0 && h.Limit > 0 {
		threshold := b.max / 4
		if threshold < 1 {
			threshold = 1
		}
		if h.Remaining < threshold {
			newRate := b.refill / 2
			if newRate < 0.1 {
				newRate = 0.1
			}
			b.logger.Info("rate limiter backing off",
				"tenant", b.tenantID,
				"remaining", h.Remaining,
				"threshold", threshold,
				"newRate", newRate,
			)
			b.throttled.Add(1)
			b.refill = newRate
			b.limiter.SetLimit(rate.Limit(newRate))
		}
	}

	// Update burst from limit header when it differs.
	if h.Limit > 0 && h.Limit != b.max {
		b.max = h.Limit
		b.limiter.SetBurst(h.Limit)
	}
}

// Metrics returns a point-in-time snapshot of the limiter state.
func (b *tokenBucket) Metrics() Metrics {
	b.mu.RLock()
	defer b.mu.RUnlock()

	return Metrics{
		AvailableTokens:  b.limiter.Tokens(),
		MaxTokens:        b.max,
		RefillRatePerSec: b.refill,
		WaitingRequests:  b.waiting.Load(),
		ThrottledTotal:   b.throttled.Load(),
	}
}

// Close cancels any in-flight retry-after goroutines and releases
// resources held by the bucket.
func (b *tokenBucket) Close() error {
	b.mu.Lock()
	defer b.mu.Unlock()
	select {
	case <-b.stopCh:
		// Already closed.
	default:
		close(b.stopCh)
	}
	return nil
}
