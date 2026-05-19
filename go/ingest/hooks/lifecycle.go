// SPDX-License-Identifier: Apache-2.0

package hooks

import (
	"context"
)

// DocumentDetectedEvent is fired when the pipeline detects URLs or file
// references worth ingesting. Plugins can inspect the references and
// return false to cancel further processing.
type DocumentDetectedEvent struct {
	BrainID    string
	References []DetectedReference
	SessionID  string
	ActorID    string
}

// DetectedReference represents a single URL or file path found in content.
type DetectedReference struct {
	Kind       string // "url" or "file"
	Value      string
	Confidence float64
}

// IngestHookEvent is the payload for OnIngestStart and OnIngestEnd hooks.
type IngestHookEvent struct {
	BrainID     string
	Path        string
	Source      string
	ContentType string
	Bytes       int
}

// IngestPlugin defines the optional lifecycle hooks for the ingestion
// pipeline. Implementations do not need to provide all three methods;
// use the fire* helpers which perform nil-safe dispatch.
type IngestPlugin interface {
	// Name returns a human-readable label used in log messages.
	Name() string

	// OnDocumentDetected fires when the pipeline detects references
	// worth ingesting. Return false to cancel. Returning an error
	// is logged but treated as non-cancelling (fail open).
	OnDocumentDetected(ctx context.Context, event DocumentDetectedEvent) (bool, error)

	// OnIngestStart fires before the ingest pipeline processes a
	// document. Return false to cancel ingestion.
	OnIngestStart(ctx context.Context, event IngestHookEvent) (bool, error)

	// OnIngestEnd fires after the ingest pipeline completes processing.
	OnIngestEnd(ctx context.Context, event IngestHookEvent) error
}

// BaseIngestPlugin provides no-op defaults for all hooks. Embed this
// struct in concrete plugins to avoid implementing hooks you do not need.
type BaseIngestPlugin struct {
	PluginName string
}

// Name returns the plugin name.
func (b *BaseIngestPlugin) Name() string { return b.PluginName }

// OnDocumentDetected is a no-op that allows processing to continue.
func (b *BaseIngestPlugin) OnDocumentDetected(_ context.Context, _ DocumentDetectedEvent) (bool, error) {
	return true, nil
}

// OnIngestStart is a no-op that allows processing to continue.
func (b *BaseIngestPlugin) OnIngestStart(_ context.Context, _ IngestHookEvent) (bool, error) {
	return true, nil
}

// OnIngestEnd is a no-op.
func (b *BaseIngestPlugin) OnIngestEnd(_ context.Context, _ IngestHookEvent) error { return nil }

// FireDocumentDetected calls OnDocumentDetected on all plugins in order.
// If any plugin returns false, the result is cancelled=true. Errors are
// logged and swallowed (fail open).
func FireDocumentDetected(
	ctx context.Context,
	plugins []IngestPlugin,
	event DocumentDetectedEvent,
	logger Logger,
) (cancelled bool) {
	for _, p := range plugins {
		proceed, err := p.OnDocumentDetected(ctx, event)
		if err != nil {
			if logger != nil {
				logger.Warn("ingest: onDocumentDetected hook threw", map[string]string{
					"plugin": p.Name(),
					"error":  err.Error(),
				})
			}
			continue
		}
		if !proceed {
			if logger != nil {
				logger.Info("ingest: plugin cancelled document detection", map[string]string{
					"plugin": p.Name(),
				})
			}
			return true
		}
	}
	return false
}

// FireIngestStart calls OnIngestStart on all plugins in registration
// order. If any plugin returns false, the result is cancelled=true.
// Errors are logged and swallowed (fail open).
func FireIngestStart(
	ctx context.Context,
	plugins []IngestPlugin,
	event IngestHookEvent,
	logger Logger,
) (cancelled bool) {
	for _, p := range plugins {
		proceed, err := p.OnIngestStart(ctx, event)
		if err != nil {
			if logger != nil {
				logger.Warn("ingest: onIngestStart hook threw", map[string]string{
					"plugin": p.Name(),
					"error":  err.Error(),
				})
			}
			continue
		}
		if !proceed {
			if logger != nil {
				logger.Info("ingest: plugin cancelled ingest start", map[string]string{
					"plugin": p.Name(),
				})
			}
			return true
		}
	}
	return false
}

// FireIngestEnd calls OnIngestEnd on all plugins in reverse registration
// order. Errors are logged and swallowed.
func FireIngestEnd(
	ctx context.Context,
	plugins []IngestPlugin,
	event IngestHookEvent,
	logger Logger,
) {
	for i := len(plugins) - 1; i >= 0; i-- {
		p := plugins[i]
		if err := p.OnIngestEnd(ctx, event); err != nil {
			if logger != nil {
				logger.Warn("ingest: onIngestEnd hook threw", map[string]string{
					"plugin": p.Name(),
					"error":  err.Error(),
				})
			}
		}
	}
}
