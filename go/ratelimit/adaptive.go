// SPDX-License-Identifier: Apache-2.0

package ratelimit

import (
	"context"
	"log/slog"
	"math"
	"sync"
)

// defaultMinRefillRate is the floor refill rate (tokens/sec).
const defaultMinRefillRate = 1.0

// defaultMaxRefillRate is the ceiling refill rate (tokens/sec).
const defaultMaxRefillRate = 100.0

// defaultRecoveryFactor controls the ramp-up multiplier after a
// throttle period. After each successful batch of requests, the rate
// is multiplied by this factor until maxRefillRate is reached.
const defaultRecoveryFactor = 1.5

// adaptiveLimiter wraps a base [Limiter] and adjusts its parameters
// in response to provider rate-limit headers. It enforces floor and
// ceiling bounds on the refill rate.
type adaptiveLimiter struct {
	mu             sync.Mutex
	bucket         Limiter
	minRefillRate  float64
	maxRefillRate  float64
	recoveryFactor float64
	currentRate    float64
	logger         *slog.Logger
}

// NewAdaptive wraps the given bucket in an adaptive layer that adjusts
// the refill rate based on provider response headers.
func NewAdaptive(opts AdaptiveOptions) Limiter {
	minRate := opts.MinRefillRate
	if minRate <= 0 {
		minRate = defaultMinRefillRate
	}
	maxRate := opts.MaxRefillRate
	if maxRate <= 0 {
		maxRate = defaultMaxRefillRate
	}
	recovery := opts.RecoveryFactor
	if recovery <= 0 {
		recovery = defaultRecoveryFactor
	}
	logger := opts.Logger
	if logger == nil {
		logger = slog.New(slog.DiscardHandler)
	}

	m := opts.Bucket.Metrics()

	return &adaptiveLimiter{
		bucket:         opts.Bucket,
		minRefillRate:  minRate,
		maxRefillRate:  maxRate,
		recoveryFactor: recovery,
		currentRate:    m.RefillRatePerSec,
		logger:         logger,
	}
}

func (a *adaptiveLimiter) Acquire(ctx context.Context, cost int) (Token, error) {
	return a.bucket.Acquire(ctx, cost)
}

func (a *adaptiveLimiter) TryAcquire(cost int) (Token, bool) {
	return a.bucket.TryAcquire(cost)
}

// UpdateFromHeaders adjusts the refill rate based on headers, clamping
// to [minRefillRate, maxRefillRate]. When remaining tokens are above
// the back-off threshold, the rate recovers by recoveryFactor.
func (a *adaptiveLimiter) UpdateFromHeaders(h Headers) {
	a.mu.Lock()
	defer a.mu.Unlock()

	// Delegate retry-after handling to the underlying bucket.
	if h.RetryAfter > 0 {
		a.bucket.UpdateFromHeaders(h)
		return
	}

	m := a.bucket.Metrics()
	maxTokens := m.MaxTokens
	if maxTokens < 1 {
		maxTokens = 1
	}

	threshold := maxTokens / 4
	if threshold < 1 {
		threshold = 1
	}

	switch {
	case h.Remaining > 0 && h.Remaining < threshold:
		// Throttle: reduce rate.
		ratio := float64(h.Remaining) / float64(maxTokens)
		newRate := a.currentRate * ratio
		newRate = math.Max(newRate, a.minRefillRate)
		a.logger.Info("adaptive limiter throttling",
			"remaining", h.Remaining,
			"threshold", threshold,
			"oldRate", a.currentRate,
			"newRate", newRate,
		)
		a.currentRate = newRate

	case h.Remaining >= threshold:
		// Recover: ramp up towards max.
		newRate := a.currentRate * a.recoveryFactor
		newRate = math.Min(newRate, a.maxRefillRate)
		a.currentRate = newRate
	}

	// Propagate the adjusted headers to the bucket so it updates its
	// internal state (burst, etc.).
	a.bucket.UpdateFromHeaders(Headers{
		Remaining: h.Remaining,
		Limit:     h.Limit,
		ResetAt:   h.ResetAt,
	})
}

func (a *adaptiveLimiter) Metrics() Metrics {
	m := a.bucket.Metrics()
	a.mu.Lock()
	m.RefillRatePerSec = a.currentRate
	a.mu.Unlock()
	return m
}

func (a *adaptiveLimiter) Close() error {
	return a.bucket.Close()
}
