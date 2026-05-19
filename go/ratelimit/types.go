// SPDX-License-Identifier: Apache-2.0

package ratelimit

import (
	"context"
	"log/slog"
	"time"
)

// Token represents a successfully acquired rate limit token. Callers
// must invoke Release when the guarded operation completes so that the
// concurrency semaphore slot is freed.
type Token struct {
	Release func()
}

// Limiter is the primary rate limiting interface. Implementations must
// be safe for concurrent use from multiple goroutines.
type Limiter interface {
	// Acquire blocks until cost tokens are available or the context is
	// cancelled. The returned Token must be released after use.
	Acquire(ctx context.Context, cost int) (Token, error)

	// TryAcquire attempts a non-blocking acquisition of cost tokens.
	// Returns the token and true on success, or a zero Token and false
	// when insufficient tokens are available.
	TryAcquire(cost int) (Token, bool)

	// UpdateFromHeaders adjusts the bucket parameters based on
	// rate-limit headers received from a provider response.
	UpdateFromHeaders(headers Headers)

	// SetRefillRate overrides the refill rate (tokens/sec). Used by
	// the adaptive layer to inject computed rates into the bucket.
	SetRefillRate(rate float64)

	// Metrics returns a point-in-time snapshot of the limiter state.
	Metrics() Metrics

	// Close releases any resources held by the limiter.
	Close() error
}

// Headers contains parsed rate limit headers from a provider response.
type Headers struct {
	Remaining  int
	Limit      int
	ResetAt    time.Time
	RetryAfter time.Duration
}

// Metrics is a point-in-time snapshot of limiter state for monitoring.
type Metrics struct {
	AvailableTokens  float64
	MaxTokens        int
	RefillRatePerSec float64
	WaitingRequests  int64
	ThrottledTotal   int64
}

// BucketOptions configures a token bucket limiter.
type BucketOptions struct {
	// MaxTokens is the bucket capacity (burst size).
	MaxTokens int
	// RefillRatePerSec is the number of tokens replenished per second.
	RefillRatePerSec float64
	// MaxConcurrency limits the number of in-flight operations. Zero
	// means unlimited concurrency.
	MaxConcurrency int
	// TenantID identifies the tenant this bucket belongs to.
	TenantID string
	// Logger receives diagnostic messages. Nil means discard.
	Logger *slog.Logger
}

// AdaptiveOptions configures the adaptive wrapper around a base Limiter.
type AdaptiveOptions struct {
	// Bucket is the underlying rate limiter to adapt.
	Bucket Limiter
	// MinRefillRate is the floor refill rate (tokens/sec). Default 1.
	MinRefillRate float64
	// MaxRefillRate is the ceiling refill rate (tokens/sec). Default 100.
	MaxRefillRate float64
	// RecoveryFactor controls how fast the rate ramps back up after a
	// throttle period. Default 1.5.
	RecoveryFactor float64
	// Logger receives diagnostic messages. Nil means discard.
	Logger *slog.Logger
}

// Factory creates per-tenant Limiter instances. Thread-safe.
type Factory interface {
	// ForTenant returns the limiter for the given tenant, creating one
	// if it does not yet exist.
	ForTenant(tenantID string) Limiter
	// Close releases all tenant limiters.
	Close() error
}

// FactoryOptions configures the limiter factory.
type FactoryOptions struct {
	// DefaultMaxTokens is the bucket capacity for new tenants.
	DefaultMaxTokens int
	// DefaultRefillRate is the refill rate for new tenants.
	DefaultRefillRate float64
	// DefaultMaxConcurrency limits in-flight operations per tenant.
	DefaultMaxConcurrency int
	// AdaptiveEnabled wraps each bucket in an adaptive layer.
	AdaptiveEnabled bool
	// MinRefillRate for the adaptive layer.
	MinRefillRate float64
	// MaxRefillRate for the adaptive layer.
	MaxRefillRate float64
	// RecoveryFactor for the adaptive layer.
	RecoveryFactor float64
	// TenantTTL is the idle duration after which a tenant limiter is
	// evicted. Zero means the default (5 minutes).
	TenantTTL time.Duration
	// Logger receives diagnostic messages. Nil means discard.
	Logger *slog.Logger
}
