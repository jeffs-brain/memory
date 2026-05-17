// SPDX-License-Identifier: Apache-2.0

// Package connector provides the shared types and interfaces for
// external-service connectors that pull documents into the ingestion
// pipeline. Individual connectors (Slack, Google Drive, Notion, etc.)
// implement the Connector interface defined here.
//
// This package is the dependency target for P5-1 (connector framework).
// Concrete connectors live alongside the framework in the same package
// and will be extracted to separate modules at integration time.
package connector

import (
	"context"
	"time"
)

// ConnectorDocument represents a single document fetched from an
// external service, ready to be fed into the ingestion pipeline.
type ConnectorDocument struct {
	ExternalID string
	Content    []byte
	MIME       string
	Title      string
	URL        string
	Metadata   map[string]string
	ModifiedAt time.Time
	Checksum   string
}

// SyncCursor tracks the position of the last successful sync for a
// connector. The Value field is opaque to the framework -- each
// connector decides what to store (timestamp, cursor token, change
// ID, etc.).
type SyncCursor struct {
	Value     string
	UpdatedAt time.Time
	Metadata  map[string]string
}

// Connector is the interface that all external-service connectors must
// implement. It provides full sync, incremental sync, and continuous
// polling capabilities.
type Connector interface {
	// Name returns the connector identifier (e.g. "slack", "gdrive").
	Name() string

	// Configure validates and stores connector-specific configuration.
	Configure(config map[string]string) error

	// FetchAll performs a full sync. Returns a channel of documents and
	// a channel that will receive at most one error (or be closed on
	// success).
	FetchAll(ctx context.Context) (<-chan ConnectorDocument, <-chan error)

	// FetchSince performs an incremental sync starting from the given
	// cursor position.
	FetchSince(ctx context.Context, cursor SyncCursor) (<-chan ConnectorDocument, <-chan error)

	// Start begins a continuous sync loop that polls at the configured
	// interval and sends documents to the returned channel.
	Start(ctx context.Context) (<-chan ConnectorDocument, <-chan error)

	// Stop gracefully stops the continuous sync loop.
	Stop() error
}

// RateLimiter provides token-bucket-based rate limiting with
// exponential backoff for HTTP API calls.
type RateLimiter struct {
	maxTokens    int
	tokens       float64
	refillRate   float64 // tokens per second
	lastRefill   time.Time
}

// NewRateLimiter creates a rate limiter with the given maximum token
// count and refill rate (tokens per second).
func NewRateLimiter(maxTokens int, refillRate float64) *RateLimiter {
	return &RateLimiter{
		maxTokens:  maxTokens,
		tokens:     float64(maxTokens),
		refillRate: refillRate,
		lastRefill: time.Now(),
	}
}

// Acquire blocks until count tokens are available, then consumes them.
// Returns an error if the context is cancelled while waiting.
func (rl *RateLimiter) Acquire(ctx context.Context, count int) error {
	for {
		rl.refill()
		if rl.tokens >= float64(count) {
			rl.tokens -= float64(count)
			return nil
		}
		deficit := float64(count) - rl.tokens
		waitDuration := time.Duration(deficit / rl.refillRate * float64(time.Second))
		if waitDuration < time.Millisecond {
			waitDuration = time.Millisecond
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(waitDuration):
		}
	}
}

func (rl *RateLimiter) refill() {
	now := time.Now()
	elapsed := now.Sub(rl.lastRefill).Seconds()
	rl.tokens += elapsed * rl.refillRate
	if rl.tokens > float64(rl.maxTokens) {
		rl.tokens = float64(rl.maxTokens)
	}
	rl.lastRefill = now
}
