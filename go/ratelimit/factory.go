// SPDX-License-Identifier: Apache-2.0

package ratelimit

import (
	"fmt"
	"log/slog"
	"sync"
	"time"
)

// defaultTenantTTL is the idle duration after which a tenant limiter
// is evicted from the factory. Default: 5 minutes.
const defaultTenantTTL = 5 * time.Minute

// evictionInterval is how often the factory sweeps for stale tenants.
const evictionInterval = 60 * time.Second

// tenantEntry holds a limiter and its last access timestamp.
type tenantEntry struct {
	limiter      Limiter
	lastAccessed time.Time
}

// tenantFactory implements [Factory] with per-tenant limiter isolation.
// Each tenant gets its own token bucket (and optional adaptive wrapper).
// Idle tenants are evicted after TenantTTL.
type tenantFactory struct {
	mu      sync.Mutex
	tenants map[string]*tenantEntry
	opts    FactoryOptions
	logger  *slog.Logger
	closed  bool
	stopCh  chan struct{}
}

// NewFactory creates a [Factory] that produces per-tenant limiters.
// Panics if DefaultMaxTokens or DefaultRefillRate are not positive.
func NewFactory(opts FactoryOptions) Factory {
	if opts.DefaultMaxTokens <= 0 {
		panic(fmt.Sprintf("ratelimit: DefaultMaxTokens must be positive, got %d", opts.DefaultMaxTokens))
	}
	if opts.DefaultRefillRate <= 0 {
		panic(fmt.Sprintf("ratelimit: DefaultRefillRate must be positive, got %f", opts.DefaultRefillRate))
	}

	logger := opts.Logger
	if logger == nil {
		logger = slog.New(slog.DiscardHandler)
	}

	ttl := opts.TenantTTL
	if ttl <= 0 {
		ttl = defaultTenantTTL
	}
	opts.TenantTTL = ttl

	f := &tenantFactory{
		tenants: make(map[string]*tenantEntry),
		opts:    opts,
		logger:  logger,
		stopCh:  make(chan struct{}),
	}

	go f.evictionLoop()
	return f
}

// evictionLoop periodically removes idle tenants.
func (f *tenantFactory) evictionLoop() {
	ticker := time.NewTicker(evictionInterval)
	defer ticker.Stop()

	for {
		select {
		case <-f.stopCh:
			return
		case <-ticker.C:
			f.evictStale()
		}
	}
}

// evictStale removes tenants that have not been accessed within the TTL.
func (f *tenantFactory) evictStale() {
	f.mu.Lock()
	defer f.mu.Unlock()

	now := time.Now()
	for id, entry := range f.tenants {
		if now.Sub(entry.lastAccessed) > f.opts.TenantTTL {
			_ = entry.limiter.Close()
			delete(f.tenants, id)
			f.logger.Info("evicted idle tenant rate limiter", "tenantId", id)
		}
	}
}

func (f *tenantFactory) ForTenant(tenantID string) Limiter {
	f.mu.Lock()
	defer f.mu.Unlock()

	if f.closed {
		panic("ratelimit: ForTenant called on closed factory")
	}

	if entry, ok := f.tenants[tenantID]; ok {
		entry.lastAccessed = time.Now()
		return entry.limiter
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

	f.tenants[tenantID] = &tenantEntry{
		limiter:      lim,
		lastAccessed: time.Now(),
	}
	return lim
}

func (f *tenantFactory) Close() error {
	f.mu.Lock()
	defer f.mu.Unlock()

	f.closed = true
	close(f.stopCh)

	var firstErr error
	for id, entry := range f.tenants {
		if err := entry.limiter.Close(); err != nil && firstErr == nil {
			firstErr = err
		}
		delete(f.tenants, id)
	}
	return firstErr
}
