// SPDX-License-Identifier: Apache-2.0

package connector

import (
	"context"
	"math"
	"math/rand"
	"net/http"
	"strconv"
	"sync"
	"time"
)

// RateLimiterConfig configures a token-bucket rate limiter.
type RateLimiterConfig struct {
	// MaxTokens is the bucket capacity.
	MaxTokens int
	// RefillRate is the number of tokens added per second.
	RefillRate float64
	// RefillInterval is the interval between refill ticks. Defaults to 1s.
	RefillInterval time.Duration
	// MaxRetries is the maximum number of retry attempts on rate limit.
	// Defaults to 5.
	MaxRetries int
	// BaseBackoff is the base delay for exponential backoff. Defaults to 1s.
	BaseBackoff time.Duration
	// MaxBackoff caps the backoff duration. Defaults to 60s.
	MaxBackoff time.Duration
}

func (c RateLimiterConfig) withDefaults() RateLimiterConfig {
	out := c
	if out.RefillInterval <= 0 {
		out.RefillInterval = time.Second
	}
	if out.MaxRetries <= 0 {
		out.MaxRetries = 5
	}
	if out.BaseBackoff <= 0 {
		out.BaseBackoff = time.Second
	}
	if out.MaxBackoff <= 0 {
		out.MaxBackoff = 60 * time.Second
	}
	return out
}

// RateLimiter implements a token-bucket rate limiter with exponential
// backoff and adaptive header-based adjustment. It is safe for
// concurrent use.
type RateLimiter struct {
	mu                 sync.Mutex
	config             RateLimiterConfig
	originalRefillRate float64
	tokens             float64
	lastTick           time.Time
	stopCh             chan struct{}
	stopped            bool
}

// NewRateLimiter creates a new token-bucket rate limiter. The bucket
// starts full.
func NewRateLimiter(config RateLimiterConfig) *RateLimiter {
	cfg := config.withDefaults()
	return &RateLimiter{
		config:             cfg,
		originalRefillRate: cfg.RefillRate,
		tokens:             float64(cfg.MaxTokens),
		lastTick:           time.Now(),
		stopCh:             make(chan struct{}),
	}
}

// refill adds tokens based on elapsed time since last refill.
func (rl *RateLimiter) refill() {
	now := time.Now()
	elapsed := now.Sub(rl.lastTick).Seconds()
	rl.tokens = math.Min(
		float64(rl.config.MaxTokens),
		rl.tokens+elapsed*rl.config.RefillRate,
	)
	rl.lastTick = now
}

// Acquire blocks until count tokens are available, then consumes them.
// Returns an error if the context is cancelled.
func (rl *RateLimiter) Acquire(ctx context.Context, count int) error {
	if count <= 0 {
		count = 1
	}
	needed := float64(count)

	for {
		rl.mu.Lock()
		rl.refill()
		if rl.tokens >= needed {
			rl.tokens -= needed
			rl.mu.Unlock()
			return nil
		}
		// Calculate wait time until enough tokens are available.
		deficit := needed - rl.tokens
		waitDuration := time.Duration(deficit / rl.config.RefillRate * float64(time.Second))
		rl.mu.Unlock()

		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-rl.stopCh:
			return context.Canceled
		case <-time.After(waitDuration):
			// Loop back and try again after refill.
		}
	}
}

// TryAcquire attempts a non-blocking token acquisition. Returns true if
// tokens were consumed, false if insufficient tokens are available.
func (rl *RateLimiter) TryAcquire(count int) bool {
	if count <= 0 {
		count = 1
	}
	needed := float64(count)

	rl.mu.Lock()
	defer rl.mu.Unlock()
	rl.refill()

	if rl.tokens >= needed {
		rl.tokens -= needed
		return true
	}
	return false
}

// Reset refills the bucket to maximum capacity. Useful after receiving
// rate limit headers that indicate the limit has reset.
func (rl *RateLimiter) Reset() {
	rl.mu.Lock()
	defer rl.mu.Unlock()
	rl.tokens = float64(rl.config.MaxTokens)
	rl.config.RefillRate = rl.originalRefillRate
	rl.lastTick = time.Now()
}

// AdjustFromHeaders reads standard rate limit headers and adjusts the
// bucket fill level accordingly. Supported headers:
//   - x-ratelimit-remaining: set tokens to remaining value
//   - x-ratelimit-limit: used to detect proactive throttling threshold
//   - retry-after: pause for the specified number of seconds
//
// When remaining < 10% of limit, the refill rate is halved to avoid
// hitting the limit. The rate is restored on Reset or the next
// AdjustFromHeaders call with healthy remaining.
func (rl *RateLimiter) AdjustFromHeaders(headers http.Header) {
	rl.mu.Lock()
	defer rl.mu.Unlock()

	remaining := headers.Get("X-Ratelimit-Remaining")
	limit := headers.Get("X-Ratelimit-Limit")

	if remaining != "" {
		if rem, err := strconv.ParseFloat(remaining, 64); err == nil {
			rl.tokens = math.Min(rem, float64(rl.config.MaxTokens))
		}
	}

	if remaining != "" && limit != "" {
		rem, remErr := strconv.ParseFloat(remaining, 64)
		lim, limErr := strconv.ParseFloat(limit, 64)
		if remErr == nil && limErr == nil && lim > 0 {
			ratio := rem / lim
			if ratio < 0.1 {
				// Proactively throttle: halve the original refill rate.
				rl.config.RefillRate = rl.originalRefillRate / 2
			} else {
				// Restore original refill rate when remaining is healthy.
				rl.config.RefillRate = rl.originalRefillRate
			}
		}
	}
}

// Backoff sleeps for an exponentially increasing duration based on the
// attempt number. The delay is: min(baseDelay * 2^attempt + jitter, maxDelay).
// Jitter is random between 0 and 500ms.
func (rl *RateLimiter) Backoff(ctx context.Context, attempt int) error {
	base := rl.config.BaseBackoff
	multiplier := math.Pow(2, float64(attempt))
	delay := time.Duration(float64(base) * multiplier)

	// Add jitter: 0-500ms.
	jitter := time.Duration(rand.Int63n(int64(500 * time.Millisecond)))
	delay += jitter

	maxBackoff := rl.config.MaxBackoff
	if delay > maxBackoff {
		delay = maxBackoff
	}

	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-time.After(delay):
		return nil
	}
}

// RetryAfter parses the Retry-After header and sleeps for the specified
// duration. Returns immediately if the header is empty or unparseable.
func (rl *RateLimiter) RetryAfter(ctx context.Context, headers http.Header) error {
	retryAfter := headers.Get("Retry-After")
	if retryAfter == "" {
		return nil
	}

	seconds, err := strconv.ParseFloat(retryAfter, 64)
	if err != nil {
		return nil
	}

	delay := time.Duration(seconds * float64(time.Second))
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-time.After(delay):
		return nil
	}
}

// Tokens returns the current number of available tokens.
func (rl *RateLimiter) Tokens() float64 {
	rl.mu.Lock()
	defer rl.mu.Unlock()
	rl.refill()
	return rl.tokens
}

// Config returns the current rate limiter configuration.
func (rl *RateLimiter) Config() RateLimiterConfig {
	rl.mu.Lock()
	defer rl.mu.Unlock()
	return rl.config
}

// Close stops the rate limiter and releases resources.
func (rl *RateLimiter) Close() {
	rl.mu.Lock()
	defer rl.mu.Unlock()
	if !rl.stopped {
		close(rl.stopCh)
		rl.stopped = true
	}
}
