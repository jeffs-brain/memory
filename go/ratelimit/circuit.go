// SPDX-License-Identifier: Apache-2.0

package ratelimit

import (
	"errors"
	"fmt"
	"log/slog"
	"sync"
	"time"
)

// CircuitState represents the current state of a circuit breaker.
type CircuitState int

const (
	// CircuitClosed is the normal operating state. Requests flow
	// through and failures are tracked.
	CircuitClosed CircuitState = iota

	// CircuitOpen indicates the circuit has tripped. All requests are
	// rejected immediately without execution.
	CircuitOpen

	// CircuitHalfOpen is the recovery probe state. A limited number
	// of requests are allowed through to test whether the downstream
	// dependency has recovered.
	CircuitHalfOpen
)

// circuitStateNames maps states to human-readable labels for logging
// and metrics.
var circuitStateNames = map[CircuitState]string{
	CircuitClosed:  "closed",
	CircuitOpen:    "open",
	CircuitHalfOpen: "half_open",
}

// String returns the human-readable name of the circuit state.
func (s CircuitState) String() string {
	name, ok := circuitStateNames[s]
	if !ok {
		return fmt.Sprintf("unknown(%d)", int(s))
	}
	return name
}

// Sentinel errors for circuit breaker operations.
var (
	// ErrCircuitOpen is returned when a call is rejected because the
	// circuit is in the open state.
	ErrCircuitOpen = errors.New("ratelimit: circuit breaker is open")

	// ErrCircuitHalfOpenLimit is returned when the maximum number of
	// half-open probe requests has been reached.
	ErrCircuitHalfOpenLimit = errors.New("ratelimit: circuit breaker half-open probe limit reached")
)

// CircuitBreakerOptions configures a circuit breaker instance.
type CircuitBreakerOptions struct {
	// FailureThreshold is the number of consecutive failures that
	// trips the circuit from closed to open. Must be >= 1.
	FailureThreshold int

	// ResetTimeout is the duration the circuit remains open before
	// transitioning to half-open. Must be > 0.
	ResetTimeout time.Duration

	// HalfOpenMaxAttempts is the maximum number of probe requests
	// allowed in the half-open state before the circuit decides to
	// close (all succeed) or re-open (any fail). Must be >= 1.
	HalfOpenMaxAttempts int

	// Name identifies the circuit for logging and metrics.
	Name string

	// Logger receives diagnostic messages. Nil means discard.
	Logger *slog.Logger

	// OnStateChange is an optional callback invoked when the circuit
	// transitions between states. It is called while the mutex is NOT
	// held, so it is safe to interact with other circuits.
	OnStateChange func(name string, from, to CircuitState)
}

// CircuitBreakerMetrics is a point-in-time snapshot of circuit breaker
// state for monitoring.
type CircuitBreakerMetrics struct {
	State              CircuitState
	ConsecutiveFailures int
	TotalSuccesses     int64
	TotalFailures      int64
	TotalRejected      int64
	LastFailureAt      time.Time
	LastSuccessAt      time.Time
}

// CircuitBreaker implements the Netflix Hystrix circuit breaker
// pattern. It tracks consecutive failures and transitions through
// three states: closed -> open -> half-open -> closed.
//
// Thread-safe: all methods are safe for concurrent use.
type CircuitBreaker struct {
	mu   sync.Mutex
	opts CircuitBreakerOptions

	state               CircuitState
	consecutiveFailures int
	halfOpenSuccesses   int
	halfOpenAttempts    int
	openedAt            time.Time

	totalSuccesses int64
	totalFailures  int64
	totalRejected  int64
	lastFailureAt  time.Time
	lastSuccessAt  time.Time
}

// NewCircuitBreaker creates a circuit breaker with the given options.
// Panics if FailureThreshold < 1, ResetTimeout <= 0, or
// HalfOpenMaxAttempts < 1.
func NewCircuitBreaker(opts CircuitBreakerOptions) *CircuitBreaker {
	if opts.FailureThreshold < 1 {
		panic(fmt.Sprintf("ratelimit: FailureThreshold must be >= 1, got %d", opts.FailureThreshold))
	}
	if opts.ResetTimeout <= 0 {
		panic(fmt.Sprintf("ratelimit: ResetTimeout must be > 0, got %v", opts.ResetTimeout))
	}
	if opts.HalfOpenMaxAttempts < 1 {
		panic(fmt.Sprintf("ratelimit: HalfOpenMaxAttempts must be >= 1, got %d", opts.HalfOpenMaxAttempts))
	}
	if opts.Logger == nil {
		opts.Logger = slog.New(slog.DiscardHandler)
	}

	return &CircuitBreaker{
		opts:  opts,
		state: CircuitClosed,
	}
}

// Allow checks whether a request should be permitted through the
// circuit. Returns nil if the request is allowed, or an error if it
// is rejected. Callers must invoke RecordSuccess or RecordFailure
// after the guarded operation completes (only when Allow returned
// nil).
func (cb *CircuitBreaker) Allow() error {
	cb.mu.Lock()
	defer cb.mu.Unlock()

	switch cb.state {
	case CircuitClosed:
		return nil

	case CircuitOpen:
		if time.Since(cb.openedAt) >= cb.opts.ResetTimeout {
			cb.transitionTo(CircuitHalfOpen)
			cb.halfOpenSuccesses = 0
			cb.halfOpenAttempts = 0
			// Allow this request as the first half-open probe.
			cb.halfOpenAttempts++
			return nil
		}
		cb.totalRejected++
		return ErrCircuitOpen

	case CircuitHalfOpen:
		if cb.halfOpenAttempts >= cb.opts.HalfOpenMaxAttempts {
			cb.totalRejected++
			return ErrCircuitHalfOpenLimit
		}
		cb.halfOpenAttempts++
		return nil

	default:
		return ErrCircuitOpen
	}
}

// RecordSuccess records a successful operation. In half-open state,
// if all probe attempts succeed, the circuit transitions back to
// closed.
func (cb *CircuitBreaker) RecordSuccess() {
	cb.mu.Lock()

	cb.totalSuccesses++
	cb.lastSuccessAt = time.Now()
	cb.consecutiveFailures = 0

	switch cb.state {
	case CircuitHalfOpen:
		cb.halfOpenSuccesses++
		if cb.halfOpenSuccesses >= cb.opts.HalfOpenMaxAttempts {
			cb.transitionTo(CircuitClosed)
			cb.mu.Unlock()
			return
		}
	}

	cb.mu.Unlock()
}

// RecordFailure records a failed operation. In closed state, if
// consecutive failures reach the threshold, the circuit opens. In
// half-open state, any failure immediately re-opens the circuit.
func (cb *CircuitBreaker) RecordFailure() {
	cb.mu.Lock()

	cb.totalFailures++
	cb.consecutiveFailures++
	cb.lastFailureAt = time.Now()

	switch cb.state {
	case CircuitClosed:
		if cb.consecutiveFailures >= cb.opts.FailureThreshold {
			cb.openedAt = time.Now()
			cb.transitionTo(CircuitOpen)
		}
	case CircuitHalfOpen:
		cb.openedAt = time.Now()
		cb.transitionTo(CircuitOpen)
	}

	cb.mu.Unlock()
}

// State returns the current circuit state.
func (cb *CircuitBreaker) State() CircuitState {
	cb.mu.Lock()
	defer cb.mu.Unlock()

	// Check for automatic transition from open to half-open.
	if cb.state == CircuitOpen && time.Since(cb.openedAt) >= cb.opts.ResetTimeout {
		return CircuitHalfOpen
	}
	return cb.state
}

// Metrics returns a point-in-time snapshot of the circuit breaker
// state for monitoring dashboards.
func (cb *CircuitBreaker) Metrics() CircuitBreakerMetrics {
	cb.mu.Lock()
	defer cb.mu.Unlock()

	return CircuitBreakerMetrics{
		State:               cb.state,
		ConsecutiveFailures: cb.consecutiveFailures,
		TotalSuccesses:      cb.totalSuccesses,
		TotalFailures:       cb.totalFailures,
		TotalRejected:       cb.totalRejected,
		LastFailureAt:       cb.lastFailureAt,
		LastSuccessAt:       cb.lastSuccessAt,
	}
}

// Reset forces the circuit back to the closed state. Intended for
// administrative or testing use.
func (cb *CircuitBreaker) Reset() {
	cb.mu.Lock()
	cb.consecutiveFailures = 0
	cb.halfOpenSuccesses = 0
	cb.halfOpenAttempts = 0
	cb.transitionTo(CircuitClosed)
	cb.mu.Unlock()
}

// transitionTo changes the circuit state and fires the OnStateChange
// callback. Must be called with cb.mu held. The callback is invoked
// after releasing the mutex to prevent deadlocks.
func (cb *CircuitBreaker) transitionTo(to CircuitState) {
	from := cb.state
	if from == to {
		return
	}
	cb.state = to
	cb.opts.Logger.Info("circuit breaker state change",
		"name", cb.opts.Name,
		"from", from.String(),
		"to", to.String(),
	)
	if cb.opts.OnStateChange != nil {
		// Schedule the callback after the caller releases the lock.
		// We use a goroutine to avoid holding the lock during the
		// callback, which could interact with other circuits.
		name := cb.opts.Name
		go cb.opts.OnStateChange(name, from, to)
	}
}

// Execute runs the given function if the circuit allows it, recording
// the outcome. Returns ErrCircuitOpen or ErrCircuitHalfOpenLimit if
// the call is rejected. Otherwise returns the function's error.
func (cb *CircuitBreaker) Execute(fn func() error) error {
	if err := cb.Allow(); err != nil {
		return err
	}
	if err := fn(); err != nil {
		cb.RecordFailure()
		return err
	}
	cb.RecordSuccess()
	return nil
}
