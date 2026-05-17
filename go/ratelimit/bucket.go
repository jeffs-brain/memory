// SPDX-License-Identifier: Apache-2.0

package ratelimit

import (
	"context"
	"log/slog"
	"sync"
	"sync/atomic"
	"time"

	"golang.org/x/time/rate"
)

// tokenBucket implements [Limiter] using golang.org/x/time/rate for the
// token bucket and an optional semaphore for concurrency control.
type tokenBucket struct {
	mu       sync.RWMutex
	limiter  *rate.Limiter
	max      int
	refill   float64
	tenantID string
	logger   *slog.Logger

	// concurrency semaphore; nil when maxConcurrency <= 0.
	sem chan struct{}

	// metrics
	waiting   atomic.Int64
	throttled atomic.Int64
}

// NewBucket creates a new in-memory token bucket limiter.
func NewBucket(opts BucketOptions) Limiter {
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
	}

	if opts.MaxConcurrency > 0 {
		b.sem = make(chan struct{}, opts.MaxConcurrency)
	}

	return b
}

// Acquire blocks until cost tokens are available, then optionally
// acquires a concurrency slot. Respects context cancellation.
func (b *tokenBucket) Acquire(ctx context.Context, cost int) (Token, error) {
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

// UpdateFromHeaders adjusts the limiter parameters based on provider
// response headers. When remaining tokens drop below burst/4, the
// refill rate is halved. When a Retry-After header is present, the
// limiter is paused for the specified duration.
func (b *tokenBucket) UpdateFromHeaders(h Headers) {
	b.mu.Lock()
	defer b.mu.Unlock()

	// Retry-After: temporarily set rate to zero, schedule restoration.
	if h.RetryAfter > 0 {
		b.logger.Info("rate limiter pausing due to retry-after",
			"tenant", b.tenantID,
			"retryAfter", h.RetryAfter,
		)
		b.throttled.Add(1)
		savedRate := b.refill
		b.refill = 0
		b.limiter.SetLimit(0)
		go func() {
			time.Sleep(h.RetryAfter)
			b.mu.Lock()
			defer b.mu.Unlock()
			b.refill = savedRate
			b.limiter.SetLimit(rate.Limit(savedRate))
			b.logger.Info("rate limiter resumed after retry-after",
				"tenant", b.tenantID,
				"rate", savedRate,
			)
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

// Close is a no-op for in-memory buckets.
func (b *tokenBucket) Close() error {
	return nil
}
