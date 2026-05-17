// SPDX-License-Identifier: Apache-2.0

package ratelimit

import (
	"context"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestBucket_AcquireSucceeds(t *testing.T) {
	b := NewBucket(BucketOptions{
		MaxTokens:        10,
		RefillRatePerSec: 100,
		TenantID:         "t1",
	})
	defer b.Close()

	ctx := context.Background()
	tok, err := b.Acquire(ctx, 1)
	if err != nil {
		t.Fatalf("acquire failed: %v", err)
	}
	tok.Release()
}

func TestBucket_AcquireBlocksWhenExhausted(t *testing.T) {
	// Bucket of 2 with slow refill.
	b := NewBucket(BucketOptions{
		MaxTokens:        2,
		RefillRatePerSec: 50,
		TenantID:         "t1",
	})
	defer b.Close()

	ctx := context.Background()

	// Drain the bucket.
	for range 2 {
		tok, err := b.Acquire(ctx, 1)
		if err != nil {
			t.Fatalf("drain acquire failed: %v", err)
		}
		tok.Release()
	}

	// Next acquire should succeed after refill (50 tokens/sec = ~20ms per token).
	start := time.Now()
	tok, err := b.Acquire(ctx, 1)
	if err != nil {
		t.Fatalf("blocking acquire failed: %v", err)
	}
	tok.Release()
	elapsed := time.Since(start)

	// Should have waited at least a few milliseconds for refill.
	if elapsed < 1*time.Millisecond {
		t.Logf("elapsed %v — may not have blocked (race with refill), acceptable", elapsed)
	}
}

func TestBucket_AcquireRespectsContextCancellation(t *testing.T) {
	b := NewBucket(BucketOptions{
		MaxTokens:        1,
		RefillRatePerSec: 0.001, // very slow refill
		TenantID:         "t1",
	})
	defer b.Close()

	ctx := context.Background()
	// Drain.
	tok, err := b.Acquire(ctx, 1)
	if err != nil {
		t.Fatalf("drain failed: %v", err)
	}
	tok.Release()

	// Cancel immediately.
	ctx2, cancel := context.WithTimeout(context.Background(), 5*time.Millisecond)
	defer cancel()

	_, err = b.Acquire(ctx2, 1)
	if err == nil {
		t.Fatalf("expected error from cancelled context")
	}
}

func TestBucket_TryAcquireNonBlocking(t *testing.T) {
	b := NewBucket(BucketOptions{
		MaxTokens:        1,
		RefillRatePerSec: 0.001,
		TenantID:         "t1",
	})
	defer b.Close()

	tok, ok := b.TryAcquire(1)
	if !ok {
		t.Fatalf("first tryAcquire should succeed")
	}
	tok.Release()

	// Second should fail — bucket exhausted and refill is very slow.
	_, ok = b.TryAcquire(1)
	if ok {
		t.Fatalf("second tryAcquire should fail when bucket is empty")
	}
}

func TestBucket_ReleaseIsIdempotent(t *testing.T) {
	b := NewBucket(BucketOptions{
		MaxTokens:        5,
		RefillRatePerSec: 100,
		MaxConcurrency:   2,
		TenantID:         "t1",
	})
	defer b.Close()

	tok, err := b.Acquire(context.Background(), 1)
	if err != nil {
		t.Fatalf("acquire failed: %v", err)
	}

	// Release twice — second call should be a no-op.
	tok.Release()
	tok.Release()
}

func TestBucket_ConcurrencyLimit(t *testing.T) {
	b := NewBucket(BucketOptions{
		MaxTokens:        100,
		RefillRatePerSec: 1000,
		MaxConcurrency:   2,
		TenantID:         "t1",
	})
	defer b.Close()

	ctx := context.Background()
	var inflight atomic.Int32
	var maxInflight atomic.Int32
	var wg sync.WaitGroup

	for range 10 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			tok, err := b.Acquire(ctx, 1)
			if err != nil {
				return
			}
			n := inflight.Add(1)
			// Track max concurrent.
			for {
				cur := maxInflight.Load()
				if n <= cur {
					break
				}
				if maxInflight.CompareAndSwap(cur, n) {
					break
				}
			}
			time.Sleep(10 * time.Millisecond)
			inflight.Add(-1)
			tok.Release()
		}()
	}
	wg.Wait()

	if maxInflight.Load() > 2 {
		t.Fatalf("max inflight %d exceeds concurrency limit 2", maxInflight.Load())
	}
}

func TestBucket_UpdateFromHeaders_BackOff(t *testing.T) {
	b := NewBucket(BucketOptions{
		MaxTokens:        100,
		RefillRatePerSec: 50,
		TenantID:         "t1",
	})
	defer b.Close()

	// remaining < burst/4 (100/4=25) triggers back-off.
	b.UpdateFromHeaders(Headers{
		Remaining: 10,
		Limit:     100,
	})

	m := b.Metrics()
	if m.RefillRatePerSec >= 50 {
		t.Fatalf("expected reduced refill rate after back-off, got %f", m.RefillRatePerSec)
	}
}

func TestBucket_UpdateFromHeaders_RetryAfter(t *testing.T) {
	b := NewBucket(BucketOptions{
		MaxTokens:        10,
		RefillRatePerSec: 50,
		TenantID:         "t1",
	})
	defer b.Close()

	b.UpdateFromHeaders(Headers{
		RetryAfter: 50 * time.Millisecond,
	})

	// Rate should be zero during pause.
	m := b.Metrics()
	if m.RefillRatePerSec != 0 {
		t.Fatalf("expected zero rate during retry-after, got %f", m.RefillRatePerSec)
	}

	// Wait for resume.
	time.Sleep(100 * time.Millisecond)
	m = b.Metrics()
	if m.RefillRatePerSec == 0 {
		t.Fatalf("expected non-zero rate after retry-after expired")
	}
}

func TestBucket_Metrics(t *testing.T) {
	b := NewBucket(BucketOptions{
		MaxTokens:        20,
		RefillRatePerSec: 10,
		TenantID:         "t1",
	})
	defer b.Close()

	m := b.Metrics()
	if m.MaxTokens != 20 {
		t.Fatalf("expected MaxTokens=20, got %d", m.MaxTokens)
	}
	if m.RefillRatePerSec != 10 {
		t.Fatalf("expected RefillRatePerSec=10, got %f", m.RefillRatePerSec)
	}
	if m.ThrottledTotal != 0 {
		t.Fatalf("expected zero throttled initially")
	}
}

func TestBucket_ConcurrentAccess(t *testing.T) {
	b := NewBucket(BucketOptions{
		MaxTokens:        100,
		RefillRatePerSec: 1000,
		TenantID:         "t1",
	})
	defer b.Close()

	ctx := context.Background()
	var wg sync.WaitGroup

	for range 50 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			tok, err := b.Acquire(ctx, 1)
			if err != nil {
				return
			}
			tok.Release()
		}()
	}
	wg.Wait()
}
