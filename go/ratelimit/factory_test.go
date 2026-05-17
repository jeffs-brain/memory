// SPDX-License-Identifier: Apache-2.0

package ratelimit

import (
	"context"
	"testing"
)

func TestFactory_ForTenantReturnsIsolatedLimiters(t *testing.T) {
	f := NewFactory(FactoryOptions{
		DefaultMaxTokens:  10,
		DefaultRefillRate: 100,
	})
	defer f.Close()

	l1 := f.ForTenant("tenant-a")
	l2 := f.ForTenant("tenant-b")

	if l1 == l2 {
		t.Fatalf("expected different limiter instances for different tenants")
	}
}

func TestFactory_ForTenantReturnsSameInstance(t *testing.T) {
	f := NewFactory(FactoryOptions{
		DefaultMaxTokens:  10,
		DefaultRefillRate: 100,
	})
	defer f.Close()

	l1 := f.ForTenant("tenant-a")
	l2 := f.ForTenant("tenant-a")

	if l1 != l2 {
		t.Fatalf("expected same limiter instance for same tenant")
	}
}

func TestFactory_WithAdaptive(t *testing.T) {
	f := NewFactory(FactoryOptions{
		DefaultMaxTokens:  10,
		DefaultRefillRate: 50,
		AdaptiveEnabled:   true,
		MinRefillRate:     1,
		MaxRefillRate:     100,
		RecoveryFactor:    1.5,
	})
	defer f.Close()

	lim := f.ForTenant("tenant-a")
	tok, err := lim.Acquire(context.Background(), 1)
	if err != nil {
		t.Fatalf("acquire failed: %v", err)
	}
	tok.Release()

	// Verify adaptive behaviour: update headers and check metrics change.
	lim.UpdateFromHeaders(Headers{
		Remaining: 1,
		Limit:     10,
	})

	m := lim.Metrics()
	if m.RefillRatePerSec >= 50 {
		t.Fatalf("expected reduced rate after adaptive throttle, got %f", m.RefillRatePerSec)
	}
}

func TestFactory_CloseReleasesAll(t *testing.T) {
	f := NewFactory(FactoryOptions{
		DefaultMaxTokens:  10,
		DefaultRefillRate: 100,
	})

	_ = f.ForTenant("a")
	_ = f.ForTenant("b")
	_ = f.ForTenant("c")

	if err := f.Close(); err != nil {
		t.Fatalf("close failed: %v", err)
	}
}

func TestFactory_IndependentTenantBuckets(t *testing.T) {
	f := NewFactory(FactoryOptions{
		DefaultMaxTokens:  2,
		DefaultRefillRate: 0.001, // very slow
	})
	defer f.Close()

	l1 := f.ForTenant("t1")
	l2 := f.ForTenant("t2")

	// Drain t1.
	for range 2 {
		tok, err := l1.Acquire(context.Background(), 1)
		if err != nil {
			t.Fatalf("t1 acquire failed: %v", err)
		}
		tok.Release()
	}

	// t2 should still have tokens.
	tok, ok := l2.TryAcquire(1)
	if !ok {
		t.Fatalf("t2 should still have tokens after t1 is drained")
	}
	tok.Release()
}

func TestFactory_ForTenantAfterClosePanics(t *testing.T) {
	f := NewFactory(FactoryOptions{
		DefaultMaxTokens:  10,
		DefaultRefillRate: 100,
	})
	_ = f.Close()

	defer func() {
		if r := recover(); r == nil {
			t.Fatalf("expected panic when calling ForTenant after Close")
		}
	}()
	f.ForTenant("should-panic")
}

func TestFactory_PanicsOnInvalidOptions(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatalf("expected panic for zero DefaultMaxTokens")
		}
	}()
	NewFactory(FactoryOptions{
		DefaultMaxTokens:  0,
		DefaultRefillRate: 10,
	})
}
