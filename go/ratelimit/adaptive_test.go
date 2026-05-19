// SPDX-License-Identifier: Apache-2.0

package ratelimit

import (
	"context"
	"testing"
	"time"
)

func TestAdaptive_ThrottlesOnLowRemaining(t *testing.T) {
	bucket := NewBucket(BucketOptions{
		MaxTokens:        100,
		RefillRatePerSec: 50,
		TenantID:         "t1",
	})
	a := NewAdaptive(AdaptiveOptions{
		Bucket:         bucket,
		MinRefillRate:  1,
		MaxRefillRate:  100,
		RecoveryFactor: 1.5,
	})
	defer a.Close()

	// remaining=5 < threshold (100/4=25) -> throttle.
	a.UpdateFromHeaders(Headers{
		Remaining: 5,
		Limit:     100,
	})

	m := a.Metrics()
	if m.RefillRatePerSec >= 50 {
		t.Fatalf("expected reduced rate after throttle, got %f", m.RefillRatePerSec)
	}
}

func TestAdaptive_RecoversWhenRemainingAboveThreshold(t *testing.T) {
	bucket := NewBucket(BucketOptions{
		MaxTokens:        100,
		RefillRatePerSec: 10,
		TenantID:         "t1",
	})
	a := NewAdaptive(AdaptiveOptions{
		Bucket:         bucket,
		MinRefillRate:  1,
		MaxRefillRate:  100,
		RecoveryFactor: 2.0,
	})
	defer a.Close()

	initialRate := a.Metrics().RefillRatePerSec

	// Remaining above threshold -> recovery.
	a.UpdateFromHeaders(Headers{
		Remaining: 80,
		Limit:     100,
	})

	m := a.Metrics()
	if m.RefillRatePerSec <= initialRate {
		t.Fatalf("expected increased rate after recovery, got %f (was %f)", m.RefillRatePerSec, initialRate)
	}
}

func TestAdaptive_RateNeverDropsBelowMin(t *testing.T) {
	bucket := NewBucket(BucketOptions{
		MaxTokens:        100,
		RefillRatePerSec: 2,
		TenantID:         "t1",
	})
	a := NewAdaptive(AdaptiveOptions{
		Bucket:         bucket,
		MinRefillRate:  5,
		MaxRefillRate:  100,
		RecoveryFactor: 1.5,
	})
	defer a.Close()

	// Repeatedly throttle to drive rate down.
	for range 20 {
		a.UpdateFromHeaders(Headers{
			Remaining: 1,
			Limit:     100,
		})
	}

	m := a.Metrics()
	if m.RefillRatePerSec < 5 {
		t.Fatalf("rate %f dropped below min 5", m.RefillRatePerSec)
	}
}

func TestAdaptive_RateNeverExceedsMax(t *testing.T) {
	bucket := NewBucket(BucketOptions{
		MaxTokens:        100,
		RefillRatePerSec: 80,
		TenantID:         "t1",
	})
	a := NewAdaptive(AdaptiveOptions{
		Bucket:         bucket,
		MinRefillRate:  1,
		MaxRefillRate:  100,
		RecoveryFactor: 2.0,
	})
	defer a.Close()

	// Repeatedly recover to drive rate up.
	for range 20 {
		a.UpdateFromHeaders(Headers{
			Remaining: 90,
			Limit:     100,
		})
	}

	m := a.Metrics()
	if m.RefillRatePerSec > 100 {
		t.Fatalf("rate %f exceeded max 100", m.RefillRatePerSec)
	}
}

func TestAdaptive_DelegatesRetryAfterToBucket(t *testing.T) {
	bucket := NewBucket(BucketOptions{
		MaxTokens:        10,
		RefillRatePerSec: 50,
		TenantID:         "t1",
	})
	a := NewAdaptive(AdaptiveOptions{
		Bucket:        bucket,
		MinRefillRate: 1,
		MaxRefillRate: 100,
	})
	defer a.Close()

	a.UpdateFromHeaders(Headers{
		RetryAfter: 50 * time.Millisecond,
	})

	// The bucket should be paused.
	bm := bucket.Metrics()
	if bm.RefillRatePerSec != 0 {
		t.Fatalf("expected bucket rate 0 during retry-after, got %f", bm.RefillRatePerSec)
	}
}

func TestAdaptive_AcquireWorks(t *testing.T) {
	bucket := NewBucket(BucketOptions{
		MaxTokens:        10,
		RefillRatePerSec: 100,
		TenantID:         "t1",
	})
	a := NewAdaptive(AdaptiveOptions{
		Bucket: bucket,
	})
	defer a.Close()

	tok, err := a.Acquire(context.Background(), 1)
	if err != nil {
		t.Fatalf("acquire failed: %v", err)
	}
	tok.Release()
}

func TestAdaptive_TryAcquireWorks(t *testing.T) {
	bucket := NewBucket(BucketOptions{
		MaxTokens:        10,
		RefillRatePerSec: 100,
		TenantID:         "t1",
	})
	a := NewAdaptive(AdaptiveOptions{
		Bucket: bucket,
	})
	defer a.Close()

	tok, ok := a.TryAcquire(1)
	if !ok {
		t.Fatalf("tryAcquire should succeed with tokens available")
	}
	tok.Release()
}

func TestAdaptive_DefaultOptions(t *testing.T) {
	bucket := NewBucket(BucketOptions{
		MaxTokens:        10,
		RefillRatePerSec: 10,
		TenantID:         "t1",
	})
	// All options at zero -> defaults should be applied.
	a := NewAdaptive(AdaptiveOptions{
		Bucket: bucket,
	})
	defer a.Close()

	// Verify it works without panicking.
	tok, err := a.Acquire(context.Background(), 1)
	if err != nil {
		t.Fatalf("acquire failed: %v", err)
	}
	tok.Release()
}

func TestAdaptive_ThrottleAppliesRateToBucket(t *testing.T) {
	bucket := NewBucket(BucketOptions{
		MaxTokens:        100,
		RefillRatePerSec: 50,
		TenantID:         "t1",
	})
	a := NewAdaptive(AdaptiveOptions{
		Bucket:         bucket,
		MinRefillRate:  1,
		MaxRefillRate:  100,
		RecoveryFactor: 1.5,
	})
	defer a.Close()

	// Throttle via adaptive.
	a.UpdateFromHeaders(Headers{
		Remaining: 5,
		Limit:     100,
	})

	// The adaptive rate should be applied to the underlying bucket.
	adaptiveRate := a.Metrics().RefillRatePerSec
	bucketRate := bucket.Metrics().RefillRatePerSec
	if adaptiveRate != bucketRate {
		t.Fatalf("adaptive rate %f should equal bucket rate %f", adaptiveRate, bucketRate)
	}
}

func TestAdaptive_NoDoubleThrottle(t *testing.T) {
	bucket := NewBucket(BucketOptions{
		MaxTokens:        100,
		RefillRatePerSec: 50,
		TenantID:         "t1",
	})
	a := NewAdaptive(AdaptiveOptions{
		Bucket:         bucket,
		MinRefillRate:  1,
		MaxRefillRate:  100,
		RecoveryFactor: 1.5,
	})
	defer a.Close()

	// The adaptive layer throttles based on remaining < threshold.
	a.UpdateFromHeaders(Headers{
		Remaining: 5,
		Limit:     100,
	})

	adaptiveRate := a.Metrics().RefillRatePerSec

	// The bucket should have the same rate as adaptive (no independent
	// back-off was applied on top).
	bucketRate := bucket.Metrics().RefillRatePerSec
	if adaptiveRate != bucketRate {
		t.Fatalf("double-throttle detected: adaptive=%f, bucket=%f", adaptiveRate, bucketRate)
	}
}

func TestAdaptive_SetRefillRate(t *testing.T) {
	bucket := NewBucket(BucketOptions{
		MaxTokens:        10,
		RefillRatePerSec: 50,
		TenantID:         "t1",
	})
	a := NewAdaptive(AdaptiveOptions{
		Bucket: bucket,
	})
	defer a.Close()

	a.SetRefillRate(25)
	m := a.Metrics()
	if m.RefillRatePerSec != 25 {
		t.Fatalf("expected 25, got %f", m.RefillRatePerSec)
	}
	bm := bucket.Metrics()
	if bm.RefillRatePerSec != 25 {
		t.Fatalf("expected bucket rate 25, got %f", bm.RefillRatePerSec)
	}
}
