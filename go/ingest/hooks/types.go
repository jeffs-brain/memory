// SPDX-License-Identifier: Apache-2.0

// Package hooks watches Store mutation events (ChangeEvent) and dispatches
// ingestion requests for matching file paths. It provides configurable
// path patterns, debouncing for rapid writes, and opt-out via batch reason.
package hooks

import (
	"sync"
	"time"
)

// DispatchFunc is called when a matching change event should trigger
// ingestion. The path is the store path of the changed document.
//
// The function is invoked in a background goroutine after the debounce
// timer fires. Implementations that perform network I/O or long-running
// work should enforce their own context/timeout to avoid goroutine
// leaks. Panics inside DispatchFunc will crash the process.
type DispatchFunc func(brainID string, path string) error

// PathMatcher determines whether a store path should trigger ingestion.
type PathMatcher func(path string) bool

// MutationHookOptions configures the store mutation hook.
type MutationHookOptions struct {
	// BrainID is the brain that owns the store being watched.
	BrainID string

	// Dispatch is called when a debounced matching event fires.
	Dispatch DispatchFunc

	// PathMatchers determines which paths trigger ingestion. When empty,
	// the default matcher (raw/documents/**) is used.
	PathMatchers []PathMatcher

	// OptOutReasons is a set of batch reasons that suppress ingestion.
	// Events with a matching reason are silently ignored. Common values:
	// "pipeline", "ingest", "reconcile".
	OptOutReasons map[string]bool

	// DebounceInterval is the minimum delay between dispatches for the
	// same path. Rapid writes within this window are coalesced into a
	// single dispatch. Defaults to 1000ms.
	DebounceInterval time.Duration

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

// MutationHook subscribes to store change events and dispatches
// ingestion requests.
type MutationHook struct {
	mu     sync.Mutex
	opts   MutationHookOptions
	timers map[string]*time.Timer
	closed chan struct{}
}

// DefaultPathMatcher returns true for paths under raw/documents/.
// The bare prefix "raw/documents/" is rejected; a filename must follow.
func DefaultPathMatcher(path string) bool {
	return len(path) > len("raw/documents/") && path[:len("raw/documents/")] == "raw/documents/"
}
