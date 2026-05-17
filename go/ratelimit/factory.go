// SPDX-License-Identifier: Apache-2.0

package ratelimit

import (
	"log/slog"
	"sync"
)

// tenantFactory implements [Factory] with per-tenant limiter isolation.
// Each tenant gets its own token bucket (and optional adaptive wrapper).
type tenantFactory struct {
	mu      sync.Mutex
	tenants map[string]Limiter
	opts    FactoryOptions
	logger  *slog.Logger
}

// NewFactory creates a [Factory] that produces per-tenant limiters.
func NewFactory(opts FactoryOptions) Factory {
	logger := opts.Logger
	if logger == nil {
		logger = slog.New(slog.DiscardHandler)
	}
	return &tenantFactory{
		tenants: make(map[string]Limiter),
		opts:    opts,
		logger:  logger,
	}
}

func (f *tenantFactory) ForTenant(tenantID string) Limiter {
	f.mu.Lock()
	defer f.mu.Unlock()

	if lim, ok := f.tenants[tenantID]; ok {
		return lim
	}

	bucket := NewBucket(BucketOptions{
		MaxTokens:        f.opts.DefaultMaxTokens,
		RefillRatePerSec: f.opts.DefaultRefillRate,
		MaxConcurrency:   f.opts.DefaultMaxConcurrency,
		TenantID:         tenantID,
		Logger:           f.logger,
	})

	var lim Limiter = bucket
	if f.opts.AdaptiveEnabled {
		lim = NewAdaptive(AdaptiveOptions{
			Bucket:         bucket,
			MinRefillRate:  f.opts.MinRefillRate,
			MaxRefillRate:  f.opts.MaxRefillRate,
			RecoveryFactor: f.opts.RecoveryFactor,
			Logger:         f.logger,
		})
	}

	f.tenants[tenantID] = lim
	return lim
}

func (f *tenantFactory) Close() error {
	f.mu.Lock()
	defer f.mu.Unlock()

	var firstErr error
	for id, lim := range f.tenants {
		if err := lim.Close(); err != nil && firstErr == nil {
			firstErr = err
		}
		delete(f.tenants, id)
	}
	return firstErr
}
