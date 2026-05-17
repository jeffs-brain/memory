// SPDX-License-Identifier: Apache-2.0

// Package connector provides the core connector framework for pulling
// documents from external services (Slack, Google Drive, Notion, etc.)
// into the ingestion pipeline. It defines the Connector interface that
// all connectors implement, a connector registry for lookup by name,
// and shared utilities for OAuth2, rate limiting, pagination, and sync
// state tracking.
package connector

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"

	"github.com/jeffs-brain/memory/go/brain"
)

// Sentinel errors returned by the connector package.
var (
	// ErrConnectorNotFound is returned when a registry lookup finds no
	// connector with the given name.
	ErrConnectorNotFound = errors.New("connector: not found")

	// ErrConnectorExists is returned when attempting to register a
	// connector with a name that is already taken.
	ErrConnectorExists = errors.New("connector: already registered")

	// ErrNotConfigured is returned when a connector method is called
	// before Configure.
	ErrNotConfigured = errors.New("connector: not configured")

	// ErrAlreadyRunning is returned when Start is called on a connector
	// that is already running.
	ErrAlreadyRunning = errors.New("connector: already running")

	// ErrNotRunning is returned when Stop is called on a connector that
	// is not running.
	ErrNotRunning = errors.New("connector: not running")
)

// ConnectorConfig holds shared configuration injected into every
// connector. Connector-specific settings are passed separately via
// Configure.
type ConnectorConfig struct {
	// Name identifies the connector type (e.g. "slack", "gdrive").
	Name string
	// BrainID scopes all stored state (tokens, cursors) to a single brain.
	BrainID string
	// PollInterval is the delay between sync iterations when running in
	// continuous mode via Start. Defaults to 5 minutes when zero.
	PollInterval time.Duration
	// Store is the brain Store used for persisting tokens and sync state.
	Store brain.Store
	// RateLimiter is an optional rate limiter for API calls. When nil,
	// no rate limiting is applied.
	RateLimiter *RateLimiter
}

// DefaultPollInterval is the default interval between sync polls.
const DefaultPollInterval = 5 * time.Minute

// EffectivePollInterval returns PollInterval if set, otherwise the default.
func (c ConnectorConfig) EffectivePollInterval() time.Duration {
	if c.PollInterval > 0 {
		return c.PollInterval
	}
	return DefaultPollInterval
}

// ConnectorDocument represents a single document fetched from an
// external service, ready for ingestion.
type ConnectorDocument struct {
	// ExternalID is the unique identifier in the source system.
	ExternalID string
	// Content holds the raw document bytes.
	Content []byte
	// MIME is the content type (e.g. "text/markdown").
	MIME string
	// Title is the document title.
	Title string
	// URL is an optional link back to the source.
	URL string
	// Metadata holds source-specific key-value pairs.
	Metadata map[string]string
	// ModifiedAt is the last modification time in the source system.
	ModifiedAt time.Time
	// Checksum is an optional content hash from the source for change
	// detection without re-downloading.
	Checksum string
}

// Connector is the interface all external-service connectors implement.
// Implementations handle service-specific API calls, authentication, and
// data transformation, delegating infrastructure concerns (rate limiting,
// pagination, state) to the shared framework.
type Connector interface {
	// Name returns the connector identifier (e.g. "slack", "gdrive").
	Name() string

	// Configure validates and stores connector-specific configuration
	// such as API keys, OAuth credentials, and content filters.
	Configure(config map[string]any) error

	// FetchAll performs a full sync, returning all available documents as
	// a channel. The error channel receives at most one error. Both
	// channels are closed when the fetch completes. Context cancellation
	// aborts the fetch.
	FetchAll(ctx context.Context) (<-chan ConnectorDocument, <-chan error)

	// FetchSince performs an incremental sync, returning documents
	// modified since the given cursor position.
	FetchSince(ctx context.Context, cursor SyncCursor) (<-chan ConnectorDocument, <-chan error)

	// Start begins a continuous sync loop that polls at the configured
	// interval. Returns when the context is cancelled or Stop is called.
	Start(ctx context.Context) error

	// Stop gracefully stops the continuous sync loop started by Start.
	Stop() error
}

// ConnectorFactory is a function that creates a new Connector instance
// with the given shared configuration.
type ConnectorFactory func(cfg ConnectorConfig) Connector

// Registry provides thread-safe registration and lookup of connector
// factories by name.
type Registry struct {
	mu        sync.RWMutex
	factories map[string]ConnectorFactory
}

// NewRegistry creates an empty connector registry.
func NewRegistry() *Registry {
	return &Registry{
		factories: make(map[string]ConnectorFactory),
	}
}

// Register adds a connector factory under the given name. Returns
// ErrConnectorExists if the name is already registered.
func (r *Registry) Register(name string, factory ConnectorFactory) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if _, exists := r.factories[name]; exists {
		return fmt.Errorf("%w: %s", ErrConnectorExists, name)
	}
	r.factories[name] = factory
	return nil
}

// Lookup returns a new Connector instance for the given name, created
// via the registered factory and the provided configuration. Returns
// ErrConnectorNotFound if no factory is registered under that name.
func (r *Registry) Lookup(name string, cfg ConnectorConfig) (Connector, error) {
	r.mu.RLock()
	factory, exists := r.factories[name]
	r.mu.RUnlock()
	if !exists {
		return nil, fmt.Errorf("%w: %s", ErrConnectorNotFound, name)
	}
	cfg.Name = name
	return factory(cfg), nil
}

// Names returns all registered connector names in no particular order.
func (r *Registry) Names() []string {
	r.mu.RLock()
	defer r.mu.RUnlock()
	names := make([]string, 0, len(r.factories))
	for name := range r.factories {
		names = append(names, name)
	}
	return names
}

// Has reports whether a connector factory is registered under the given
// name.
func (r *Registry) Has(name string) bool {
	r.mu.RLock()
	defer r.mu.RUnlock()
	_, exists := r.factories[name]
	return exists
}
