// SPDX-License-Identifier: Apache-2.0

package ratelimit

import (
	"errors"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestCircuitBreaker_StartsInClosedState(t *testing.T) {
	cb := NewCircuitBreaker(CircuitBreakerOptions{
		FailureThreshold:    3,
		ResetTimeout:        time.Second,
		HalfOpenMaxAttempts: 2,
		Name:                "test",
	})

	if got := cb.State(); got != CircuitClosed {
		t.Errorf("initial state = %v, want CircuitClosed", got)
	}
}

func TestCircuitBreaker_AllowsRequestsWhenClosed(t *testing.T) {
	cb := NewCircuitBreaker(CircuitBreakerOptions{
		FailureThreshold:    3,
		ResetTimeout:        time.Second,
		HalfOpenMaxAttempts: 2,
		Name:                "test",
	})

	err := cb.Allow()
	if err != nil {
		t.Fatalf("Allow() returned error in closed state: %v", err)
	}
	cb.RecordSuccess()
}

func TestCircuitBreaker_OpensAfterFailureThreshold(t *testing.T) {
	cb := NewCircuitBreaker(CircuitBreakerOptions{
		FailureThreshold:    3,
		ResetTimeout:        time.Minute,
		HalfOpenMaxAttempts: 1,
		Name:                "test",
	})

	// Record failures up to threshold.
	for i := 0; i < 3; i++ {
		if err := cb.Allow(); err != nil {
			t.Fatalf("Allow() failed before threshold: %v", err)
		}
		cb.RecordFailure()
	}

	// Circuit should now be open.
	err := cb.Allow()
	if !errors.Is(err, ErrCircuitOpen) {
		t.Errorf("Allow() = %v, want ErrCircuitOpen", err)
	}
}

func TestCircuitBreaker_RejectsWhenOpen(t *testing.T) {
	cb := NewCircuitBreaker(CircuitBreakerOptions{
		FailureThreshold:    1,
		ResetTimeout:        time.Minute,
		HalfOpenMaxAttempts: 1,
		Name:                "test",
	})

	_ = cb.Allow()
	cb.RecordFailure()

	err := cb.Allow()
	if !errors.Is(err, ErrCircuitOpen) {
		t.Errorf("Allow() = %v, want ErrCircuitOpen", err)
	}

	metrics := cb.Metrics()
	if metrics.TotalRejected != 1 {
		t.Errorf("TotalRejected = %d, want 1", metrics.TotalRejected)
	}
}

func TestCircuitBreaker_TransitionsToHalfOpenAfterTimeout(t *testing.T) {
	cb := NewCircuitBreaker(CircuitBreakerOptions{
		FailureThreshold:    1,
		ResetTimeout:        10 * time.Millisecond,
		HalfOpenMaxAttempts: 1,
		Name:                "test",
	})

	_ = cb.Allow()
	cb.RecordFailure()

	// Wait for reset timeout.
	time.Sleep(20 * time.Millisecond)

	// State should report half-open (or the Allow call transitions it).
	if got := cb.State(); got != CircuitHalfOpen {
		t.Errorf("state after timeout = %v, want CircuitHalfOpen", got)
	}

	// Allow should succeed (first probe request).
	err := cb.Allow()
	if err != nil {
		t.Fatalf("Allow() in half-open state: %v", err)
	}
}

func TestCircuitBreaker_ClosesAfterHalfOpenSuccesses(t *testing.T) {
	cb := NewCircuitBreaker(CircuitBreakerOptions{
		FailureThreshold:    1,
		ResetTimeout:        10 * time.Millisecond,
		HalfOpenMaxAttempts: 2,
		Name:                "test",
	})

	// Trip the circuit.
	_ = cb.Allow()
	cb.RecordFailure()

	time.Sleep(20 * time.Millisecond)

	// Two successful probes in half-open should close the circuit.
	for i := 0; i < 2; i++ {
		if err := cb.Allow(); err != nil {
			t.Fatalf("probe %d: Allow() = %v", i, err)
		}
		cb.RecordSuccess()
	}

	if got := cb.State(); got != CircuitClosed {
		t.Errorf("state after recovery = %v, want CircuitClosed", got)
	}
}

func TestCircuitBreaker_ReopensOnHalfOpenFailure(t *testing.T) {
	cb := NewCircuitBreaker(CircuitBreakerOptions{
		FailureThreshold:    1,
		ResetTimeout:        10 * time.Millisecond,
		HalfOpenMaxAttempts: 3,
		Name:                "test",
	})

	// Trip the circuit.
	_ = cb.Allow()
	cb.RecordFailure()

	time.Sleep(20 * time.Millisecond)

	// First probe succeeds, second fails.
	_ = cb.Allow()
	cb.RecordSuccess()

	_ = cb.Allow()
	cb.RecordFailure()

	// Should be back to open.
	err := cb.Allow()
	if !errors.Is(err, ErrCircuitOpen) {
		t.Errorf("Allow() after half-open failure = %v, want ErrCircuitOpen", err)
	}
}

func TestCircuitBreaker_HalfOpenLimitsProbes(t *testing.T) {
	cb := NewCircuitBreaker(CircuitBreakerOptions{
		FailureThreshold:    1,
		ResetTimeout:        10 * time.Millisecond,
		HalfOpenMaxAttempts: 2,
		Name:                "test",
	})

	_ = cb.Allow()
	cb.RecordFailure()

	time.Sleep(20 * time.Millisecond)

	// Allow the max number of probes (but do not record results yet).
	_ = cb.Allow()
	_ = cb.Allow()

	// Third probe should be rejected.
	err := cb.Allow()
	if !errors.Is(err, ErrCircuitHalfOpenLimit) {
		t.Errorf("Allow() beyond probe limit = %v, want ErrCircuitHalfOpenLimit", err)
	}
}

func TestCircuitBreaker_SuccessResetsConsecutiveFailures(t *testing.T) {
	cb := NewCircuitBreaker(CircuitBreakerOptions{
		FailureThreshold:    3,
		ResetTimeout:        time.Minute,
		HalfOpenMaxAttempts: 1,
		Name:                "test",
	})

	// Two failures, then a success should reset the counter.
	_ = cb.Allow()
	cb.RecordFailure()
	_ = cb.Allow()
	cb.RecordFailure()
	_ = cb.Allow()
	cb.RecordSuccess()

	// Two more failures should not trip (counter was reset).
	_ = cb.Allow()
	cb.RecordFailure()
	_ = cb.Allow()
	cb.RecordFailure()

	if got := cb.State(); got != CircuitClosed {
		t.Errorf("state = %v, want CircuitClosed (counter was reset)", got)
	}
}

func TestCircuitBreaker_Execute(t *testing.T) {
	cb := NewCircuitBreaker(CircuitBreakerOptions{
		FailureThreshold:    2,
		ResetTimeout:        time.Minute,
		HalfOpenMaxAttempts: 1,
		Name:                "test",
	})

	testErr := errors.New("downstream error")

	// Successful execution.
	err := cb.Execute(func() error { return nil })
	if err != nil {
		t.Fatalf("Execute(success) = %v", err)
	}

	// Failed executions that trip the circuit.
	_ = cb.Execute(func() error { return testErr })
	_ = cb.Execute(func() error { return testErr })

	// Should be rejected now.
	err = cb.Execute(func() error { return nil })
	if !errors.Is(err, ErrCircuitOpen) {
		t.Errorf("Execute() when open = %v, want ErrCircuitOpen", err)
	}
}

func TestCircuitBreaker_Reset(t *testing.T) {
	cb := NewCircuitBreaker(CircuitBreakerOptions{
		FailureThreshold:    1,
		ResetTimeout:        time.Minute,
		HalfOpenMaxAttempts: 1,
		Name:                "test",
	})

	_ = cb.Allow()
	cb.RecordFailure()

	cb.Reset()

	if got := cb.State(); got != CircuitClosed {
		t.Errorf("state after Reset = %v, want CircuitClosed", got)
	}

	err := cb.Allow()
	if err != nil {
		t.Fatalf("Allow() after Reset: %v", err)
	}
	cb.RecordSuccess()
}

func TestCircuitBreaker_Metrics(t *testing.T) {
	cb := NewCircuitBreaker(CircuitBreakerOptions{
		FailureThreshold:    2,
		ResetTimeout:        time.Minute,
		HalfOpenMaxAttempts: 1,
		Name:                "test",
	})

	_ = cb.Allow()
	cb.RecordSuccess()
	_ = cb.Allow()
	cb.RecordFailure()
	_ = cb.Allow()
	cb.RecordFailure()

	// Circuit should be open; one rejection.
	_ = cb.Allow()

	m := cb.Metrics()
	if m.TotalSuccesses != 1 {
		t.Errorf("TotalSuccesses = %d, want 1", m.TotalSuccesses)
	}
	if m.TotalFailures != 2 {
		t.Errorf("TotalFailures = %d, want 2", m.TotalFailures)
	}
	if m.TotalRejected != 1 {
		t.Errorf("TotalRejected = %d, want 1", m.TotalRejected)
	}
	if m.ConsecutiveFailures != 2 {
		t.Errorf("ConsecutiveFailures = %d, want 2", m.ConsecutiveFailures)
	}
	if m.State != CircuitOpen {
		t.Errorf("State = %v, want CircuitOpen", m.State)
	}
}

func TestCircuitBreaker_OnStateChange(t *testing.T) {
	var transitions []string
	var mu sync.Mutex
	done := make(chan struct{}, 4)

	cb := NewCircuitBreaker(CircuitBreakerOptions{
		FailureThreshold:    1,
		ResetTimeout:        10 * time.Millisecond,
		HalfOpenMaxAttempts: 1,
		Name:                "test-cb",
		OnStateChange: func(name string, from, to CircuitState) {
			mu.Lock()
			transitions = append(transitions, from.String()+"->"+to.String())
			mu.Unlock()
			done <- struct{}{}
		},
	})

	// Trip: closed -> open.
	_ = cb.Allow()
	cb.RecordFailure()
	<-done

	time.Sleep(20 * time.Millisecond)

	// Probe: open -> half_open, then success: half_open -> closed.
	_ = cb.Allow()
	<-done
	cb.RecordSuccess()
	<-done

	mu.Lock()
	defer mu.Unlock()

	expected := []string{"closed->open", "open->half_open", "half_open->closed"}
	if len(transitions) != len(expected) {
		t.Fatalf("transitions = %v, want %v", transitions, expected)
	}
	for i, want := range expected {
		if transitions[i] != want {
			t.Errorf("transitions[%d] = %q, want %q", i, transitions[i], want)
		}
	}
}

func TestCircuitBreaker_ConcurrentSafety(t *testing.T) {
	cb := NewCircuitBreaker(CircuitBreakerOptions{
		FailureThreshold:    100,
		ResetTimeout:        time.Minute,
		HalfOpenMaxAttempts: 10,
		Name:                "concurrent",
	})

	var wg sync.WaitGroup
	var successCount atomic.Int64

	for i := 0; i < 100; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < 100; j++ {
				if err := cb.Allow(); err == nil {
					successCount.Add(1)
					cb.RecordSuccess()
				}
			}
		}()
	}

	wg.Wait()

	if successCount.Load() != 10000 {
		t.Errorf("expected all 10000 requests to succeed, got %d", successCount.Load())
	}
}

func TestCircuitBreaker_StateString(t *testing.T) {
	tests := map[CircuitState]string{
		CircuitClosed:   "closed",
		CircuitOpen:     "open",
		CircuitHalfOpen: "half_open",
		CircuitState(99): "unknown(99)",
	}
	for state, want := range tests {
		if got := state.String(); got != want {
			t.Errorf("CircuitState(%d).String() = %q, want %q", int(state), got, want)
		}
	}
}

func TestCircuitBreaker_PanicsOnInvalidOptions(t *testing.T) {
	expectPanic := func(name string, opts CircuitBreakerOptions) {
		t.Helper()
		defer func() {
			if r := recover(); r == nil {
				t.Errorf("%s: expected panic, got none", name)
			}
		}()
		NewCircuitBreaker(opts)
	}

	expectPanic("zero threshold", CircuitBreakerOptions{
		FailureThreshold:    0,
		ResetTimeout:        time.Second,
		HalfOpenMaxAttempts: 1,
	})
	expectPanic("zero timeout", CircuitBreakerOptions{
		FailureThreshold:    1,
		ResetTimeout:        0,
		HalfOpenMaxAttempts: 1,
	})
	expectPanic("zero half-open", CircuitBreakerOptions{
		FailureThreshold:    1,
		ResetTimeout:        time.Second,
		HalfOpenMaxAttempts: 0,
	})
}
