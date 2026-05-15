// SPDX-License-Identifier: Apache-2.0

// Package trigger provides an in-process event bus for dispatching
// ingestion trigger events, with optional Redis and PostgreSQL bridges
// for cross-process delivery.
package trigger

import "time"

// IngestTriggerEvent represents a request to ingest a document. Events
// are published to the bus and delivered to all registered handlers.
type IngestTriggerEvent struct {
	ID        string            `json:"id"`
	BrainID   string            `json:"brainId"`
	Source    TriggerSource     `json:"source"`
	Payload   TriggerPayload    `json:"payload"`
	Metadata  map[string]string `json:"metadata,omitempty"`
	Timestamp time.Time         `json:"timestamp"`
}

// TriggerSource identifies where a trigger event originated.
type TriggerSource string

const (
	SourceEventBus TriggerSource = "event-bus"
	SourceRedis    TriggerSource = "redis"
	SourcePostgres TriggerSource = "postgres"
	SourceHook     TriggerSource = "hook"
	SourceSchedule TriggerSource = "schedule"
)

// TriggerPayload carries the content reference for an ingestion request.
type TriggerPayload struct {
	Kind    PayloadKind `json:"kind"`
	Path    string      `json:"path,omitempty"`
	URL     string      `json:"url,omitempty"`
	Content string      `json:"content,omitempty"`
	Title   string      `json:"title,omitempty"`
	Mime    string      `json:"mime,omitempty"`
}

// PayloadKind classifies the type of content in a TriggerPayload.
type PayloadKind string

const (
	PayloadFile PayloadKind = "file"
	PayloadURL  PayloadKind = "url"
	PayloadRaw  PayloadKind = "raw"
)

// TriggerHandler processes a single trigger event. Returning an error
// does not remove the handler from the subscription; errors are logged
// by the bus implementation.
type TriggerHandler func(event IngestTriggerEvent) error

// Bus is the core abstraction for publishing and subscribing to
// ingestion trigger events. The default implementation is purely
// in-process; external bridges add Redis or PostgreSQL delivery.
type Bus interface {
	// Publish enqueues an event for delivery to all subscribers. It
	// validates the event and returns an error for malformed inputs.
	Publish(event IngestTriggerEvent) error

	// Subscribe registers a handler that receives all published events.
	// The returned function removes the subscription when called.
	Subscribe(handler TriggerHandler) (unsubscribe func())

	// Close drains pending events and releases resources. Blocks until
	// all inflight deliveries complete or the context deadline expires.
	Close() error
}

// BusOptions configures the in-process event bus.
type BusOptions struct {
	// MaxQueueDepth is the backpressure threshold. When the queue exceeds
	// this depth, the oldest events are dropped and a warning is logged.
	// Defaults to 1000.
	MaxQueueDepth int

	// Logger receives diagnostic messages. When nil, logging is disabled.
	Logger Logger
}

// Logger mirrors the logging contract used across the memory SDK.
type Logger interface {
	Debug(msg string, ctx ...map[string]string)
	Info(msg string, ctx ...map[string]string)
	Warn(msg string, ctx ...map[string]string)
	Error(msg string, ctx ...map[string]string)
}

// EventTransport is the extension point for custom event delivery
// mechanisms. Bridges (Redis, Postgres) implement this interface to
// forward events between processes.
type EventTransport interface {
	// Start begins listening for remote events and publishing them to
	// the provided bus.
	Start(bus Bus) error

	// Close stops the transport and releases connections.
	Close() error
}

// Validate checks that the event has all required fields.
func (e IngestTriggerEvent) Validate() error {
	if e.ID == "" {
		return ErrMissingID
	}
	if e.BrainID == "" {
		return ErrMissingBrainID
	}
	if e.Timestamp.IsZero() {
		return ErrMissingTimestamp
	}
	switch e.Payload.Kind {
	case PayloadFile:
		if e.Payload.Path == "" {
			return ErrMissingPayloadPath
		}
	case PayloadURL:
		if e.Payload.URL == "" {
			return ErrMissingPayloadURL
		}
	case PayloadRaw:
		if e.Payload.Content == "" {
			return ErrMissingPayloadContent
		}
	default:
		return ErrInvalidPayloadKind
	}
	return nil
}
