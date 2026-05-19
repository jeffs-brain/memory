// SPDX-License-Identifier: Apache-2.0

package connector_test

import (
	"context"
	"net/http"
	"testing"
	"time"

	"github.com/jeffs-brain/memory/go/connector"
)

func TestRateLimiter_AcquireWithinBudget(t *testing.T) {
	rl := connector.NewRateLimiter(connector.RateLimiterConfig{
		MaxTokens:  10,
		RefillRate: 1,
	})
	defer rl.Close()

	ctx := context.Background()
	if err := rl.Acquire(ctx, 1); err != nil {
		t.Fatalf("Acquire: %v", err)
	}
	// Should have consumed 1 token.
	remaining := rl.Tokens()
	if remaining < 8 || remaining > 10 {
		t.Errorf("remaining tokens = %f, want ~9", remaining)
	}
}

func TestRateLimiter_AcquireExhaustsBudget(t *testing.T) {
	rl := connector.NewRateLimiter(connector.RateLimiterConfig{
		MaxTokens:  2,
		RefillRate: 100, // Fast refill so test completes quickly.
	})
	defer rl.Close()

	ctx := context.Background()

	// Consume all tokens.
	if err := rl.Acquire(ctx, 2); err != nil {
		t.Fatalf("Acquire 2: %v", err)
	}

	// Next acquire should wait for refill (fast in this test).
	start := time.Now()
	if err := rl.Acquire(ctx, 1); err != nil {
		t.Fatalf("Acquire after exhaust: %v", err)
	}
	elapsed := time.Since(start)
	// With 100 tokens/sec, refill of 1 token should take ~10ms.
	if elapsed > 500*time.Millisecond {
		t.Errorf("waited too long: %v", elapsed)
	}
}

func TestRateLimiter_TryAcquireAvailable(t *testing.T) {
	rl := connector.NewRateLimiter(connector.RateLimiterConfig{
		MaxTokens:  5,
		RefillRate: 1,
	})
	defer rl.Close()

	if !rl.TryAcquire(1) {
		t.Error("TryAcquire should succeed when tokens available")
	}
}

func TestRateLimiter_TryAcquireEmpty(t *testing.T) {
	rl := connector.NewRateLimiter(connector.RateLimiterConfig{
		MaxTokens:  1,
		RefillRate: 0.001, // Very slow refill.
	})
	defer rl.Close()

	// Drain the bucket.
	rl.TryAcquire(1)

	if rl.TryAcquire(1) {
		t.Error("TryAcquire should fail when bucket empty")
	}
}

func TestRateLimiter_Reset(t *testing.T) {
	rl := connector.NewRateLimiter(connector.RateLimiterConfig{
		MaxTokens:  10,
		RefillRate: 0.001,
	})
	defer rl.Close()

	// Drain.
	rl.TryAcquire(10)
	if rl.TryAcquire(1) {
		t.Fatal("should be empty after draining")
	}

	// Reset.
	rl.Reset()
	if !rl.TryAcquire(1) {
		t.Error("TryAcquire should succeed after Reset")
	}
}

func TestRateLimiter_ContextCancellation(t *testing.T) {
	rl := connector.NewRateLimiter(connector.RateLimiterConfig{
		MaxTokens:  1,
		RefillRate: 0.001,
	})
	defer rl.Close()

	// Drain.
	rl.TryAcquire(1)

	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()

	err := rl.Acquire(ctx, 1)
	if err == nil {
		t.Fatal("expected context error")
	}
}

func TestRateLimiter_AdjustFromHeaders(t *testing.T) {
	rl := connector.NewRateLimiter(connector.RateLimiterConfig{
		MaxTokens:  100,
		RefillRate: 10,
	})
	defer rl.Close()

	headers := http.Header{}
	headers.Set("X-Ratelimit-Remaining", "5")
	headers.Set("X-Ratelimit-Limit", "100")

	rl.AdjustFromHeaders(headers)

	tokens := rl.Tokens()
	if tokens > 6 {
		t.Errorf("tokens = %f, want <= 5 after header adjustment", tokens)
	}
}

func TestRateLimiter_Backoff_Timing(t *testing.T) {
	rl := connector.NewRateLimiter(connector.RateLimiterConfig{
		MaxTokens:   10,
		RefillRate:  1,
		BaseBackoff: 10 * time.Millisecond,
		MaxBackoff:  200 * time.Millisecond,
	})
	defer rl.Close()

	ctx := context.Background()

	start := time.Now()
	if err := rl.Backoff(ctx, 0); err != nil {
		t.Fatalf("Backoff attempt 0: %v", err)
	}
	elapsed := time.Since(start)
	// Attempt 0: ~10ms base + up to 500ms jitter.
	if elapsed > 600*time.Millisecond {
		t.Errorf("backoff attempt 0 too slow: %v", elapsed)
	}
}

func TestRateLimiter_Backoff_MaxCap(t *testing.T) {
	rl := connector.NewRateLimiter(connector.RateLimiterConfig{
		MaxTokens:   10,
		RefillRate:  1,
		BaseBackoff: 10 * time.Millisecond,
		MaxBackoff:  50 * time.Millisecond,
	})
	defer rl.Close()

	ctx := context.Background()

	start := time.Now()
	if err := rl.Backoff(ctx, 10); err != nil {
		t.Fatalf("Backoff attempt 10: %v", err)
	}
	elapsed := time.Since(start)
	// Should be capped at maxBackoff (50ms) regardless of attempt number.
	if elapsed > 200*time.Millisecond {
		t.Errorf("backoff exceeded max cap: %v", elapsed)
	}
}

func TestRateLimiter_RetryAfter(t *testing.T) {
	rl := connector.NewRateLimiter(connector.RateLimiterConfig{
		MaxTokens:  10,
		RefillRate: 1,
	})
	defer rl.Close()

	headers := http.Header{}
	headers.Set("Retry-After", "0.01") // 10ms

	ctx := context.Background()
	start := time.Now()
	if err := rl.RetryAfter(ctx, headers); err != nil {
		t.Fatalf("RetryAfter: %v", err)
	}
	elapsed := time.Since(start)
	if elapsed < 5*time.Millisecond {
		t.Errorf("RetryAfter returned too fast: %v", elapsed)
	}
}

func TestRateLimiter_RetryAfter_EmptyHeader(t *testing.T) {
	rl := connector.NewRateLimiter(connector.RateLimiterConfig{
		MaxTokens:  10,
		RefillRate: 1,
	})
	defer rl.Close()

	headers := http.Header{}
	ctx := context.Background()
	if err := rl.RetryAfter(ctx, headers); err != nil {
		t.Fatalf("RetryAfter empty: %v", err)
	}
}

func TestRateLimiter_TokenRefillOverTime(t *testing.T) {
	rl := connector.NewRateLimiter(connector.RateLimiterConfig{
		MaxTokens:  10,
		RefillRate: 1000, // 1000 tokens/sec for fast test.
	})
	defer rl.Close()

	// Drain.
	rl.TryAcquire(10)

	// Wait for refill.
	time.Sleep(20 * time.Millisecond) // Should refill ~20 tokens.

	if !rl.TryAcquire(1) {
		t.Error("token should have refilled after waiting")
	}
}

func TestRateLimiter_AdjustFromHeaders_RepeatedThrottle(t *testing.T) {
	rl := connector.NewRateLimiter(connector.RateLimiterConfig{
		MaxTokens:  100,
		RefillRate: 10,
	})
	defer rl.Close()

	lowHeaders := http.Header{}
	lowHeaders.Set("X-Ratelimit-Remaining", "5")
	lowHeaders.Set("X-Ratelimit-Limit", "100")

	// Repeatedly call AdjustFromHeaders with low remaining.
	// Before the fix, this would halve the rate each time,
	// converging to zero.
	for i := 0; i < 10; i++ {
		rl.AdjustFromHeaders(lowHeaders)
	}

	// The refill rate should be halved once (5), not 10/1024.
	cfg := rl.Config()
	if cfg.RefillRate != 5 {
		t.Errorf("RefillRate = %f after repeated throttle, want 5", cfg.RefillRate)
	}
}

func TestRateLimiter_AdjustFromHeaders_RestoreRate(t *testing.T) {
	rl := connector.NewRateLimiter(connector.RateLimiterConfig{
		MaxTokens:  100,
		RefillRate: 10,
	})
	defer rl.Close()

	// Throttle.
	lowHeaders := http.Header{}
	lowHeaders.Set("X-Ratelimit-Remaining", "5")
	lowHeaders.Set("X-Ratelimit-Limit", "100")
	rl.AdjustFromHeaders(lowHeaders)

	cfg := rl.Config()
	if cfg.RefillRate != 5 {
		t.Errorf("RefillRate = %f after throttle, want 5", cfg.RefillRate)
	}

	// Restore with healthy remaining.
	healthyHeaders := http.Header{}
	healthyHeaders.Set("X-Ratelimit-Remaining", "80")
	healthyHeaders.Set("X-Ratelimit-Limit", "100")
	rl.AdjustFromHeaders(healthyHeaders)

	cfg = rl.Config()
	if cfg.RefillRate != 10 {
		t.Errorf("RefillRate = %f after restore, want 10", cfg.RefillRate)
	}
}

func TestRateLimiter_ResetRestoresRefillRate(t *testing.T) {
	rl := connector.NewRateLimiter(connector.RateLimiterConfig{
		MaxTokens:  100,
		RefillRate: 10,
	})
	defer rl.Close()

	// Throttle.
	lowHeaders := http.Header{}
	lowHeaders.Set("X-Ratelimit-Remaining", "5")
	lowHeaders.Set("X-Ratelimit-Limit", "100")
	rl.AdjustFromHeaders(lowHeaders)

	cfg := rl.Config()
	if cfg.RefillRate != 5 {
		t.Errorf("RefillRate = %f after throttle, want 5", cfg.RefillRate)
	}

	// Reset should restore the original refill rate.
	rl.Reset()

	cfg = rl.Config()
	if cfg.RefillRate != 10 {
		t.Errorf("RefillRate = %f after Reset, want 10 (original)", cfg.RefillRate)
	}
}
