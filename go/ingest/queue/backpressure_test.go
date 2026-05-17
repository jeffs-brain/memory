// SPDX-License-Identifier: Apache-2.0

package queue

import (
	"context"
	"testing"
)

func TestBackpressureChecker_DefaultThreshold(t *testing.T) {
	t.Parallel()
	adapter := newFakeAdapter()
	checker := NewBackpressureChecker(adapter, 0)
	if checker.MaxDepth() != defaultMaxQueueDepth {
		t.Fatalf("expected default threshold %d, got %d", defaultMaxQueueDepth, checker.MaxDepth())
	}
}

func TestBackpressureChecker_CustomThreshold(t *testing.T) {
	t.Parallel()
	adapter := newFakeAdapter()
	checker := NewBackpressureChecker(adapter, 500)
	if checker.MaxDepth() != 500 {
		t.Fatalf("expected threshold 500, got %d", checker.MaxDepth())
	}
}

func TestBackpressureChecker_CheckUpdatesState(t *testing.T) {
	t.Parallel()
	adapter := newFakeAdapter()
	adapter.pendingDepth = 100
	checker := NewBackpressureChecker(adapter, 50)

	ctx := context.Background()
	pressured, err := checker.Check(ctx, "")
	if err != nil {
		t.Fatalf("Check returned error: %v", err)
	}
	if !pressured {
		t.Fatal("expected backpressured when depth 100 >= threshold 50")
	}
	if !checker.IsBackpressured() {
		t.Fatal("IsBackpressured should reflect last check")
	}

	adapter.mu.Lock()
	adapter.pendingDepth = 30
	adapter.mu.Unlock()

	pressured, err = checker.Check(ctx, "")
	if err != nil {
		t.Fatalf("Check returned error: %v", err)
	}
	if pressured {
		t.Fatal("expected not backpressured when depth 30 < threshold 50")
	}
}

func TestBackpressureChecker_LastDepth(t *testing.T) {
	t.Parallel()
	adapter := newFakeAdapter()
	adapter.pendingDepth = 42
	checker := NewBackpressureChecker(adapter, 1000)

	// Before first check, LastDepth is 0.
	if checker.LastDepth() != 0 {
		t.Fatalf("expected LastDepth=0 before first check, got %d", checker.LastDepth())
	}

	ctx := context.Background()
	_, err := checker.Check(ctx, "")
	if err != nil {
		t.Fatalf("Check returned error: %v", err)
	}

	if checker.LastDepth() != 42 {
		t.Fatalf("expected LastDepth=42 after check, got %d", checker.LastDepth())
	}
}
