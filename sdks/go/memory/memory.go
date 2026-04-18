// SPDX-License-Identifier: Apache-2.0

// Package memory implements the memory stages layer on top of a
// [brain.Store]: extract, reflect, consolidate, recall, plus episodic,
// procedural, and heuristic sublayers. Ported from the upstream jeff
// harness with the harness-specific hooks stubbed out or inlined.
package memory

import (
	"os"
	"path/filepath"
	"sync"

	"github.com/jeffs-brain/memory/go/brain"
)

// Memory is the entry point for all memory operations. It holds a
// [brain.Store] that backs the persistent brain state and exposes methods
// for reading, writing, and consolidating memory files.
//
// Callers should hold one Memory per process and inject it into
// [Extractor], [Reflector], [Consolidator], and [EpisodeRecorder] so every
// mutation flows through a single shared store.
type Memory struct {
	store brain.Store
}

// New returns a Memory backed by the given store.
func New(store brain.Store) *Memory {
	return &Memory{store: store}
}

// Store returns the underlying brain store. Useful for callers that need
// to construct sibling components against the same backend.
func (m *Memory) Store() brain.Store { return m.store }

// ---- Default Memory singleton ----

var (
	defaultMemoryMu    sync.Mutex
	defaultMemoryCache = make(map[string]*Memory)
)

// Default returns a process-wide Memory backed by an auto-detected store
// rooted at ~/.config/jeffs-brain. The cache keeps one instance per brain
// root so tests that set $HOME to a temp directory pick up a fresh store.
//
// TODO(integration): the upstream jeff Default() wires in a git-backed
// store when the brain root contains a .git directory. That selection
// logic lives in jeff's autodetect package and has not yet been ported.
// For now Default always stays nil and callers must inject a store via
// [New].
func Default() *Memory {
	root := brainRoot()
	defaultMemoryMu.Lock()
	defer defaultMemoryMu.Unlock()
	if m, ok := defaultMemoryCache[root]; ok {
		return m
	}
	return nil
}

// brainRoot returns the absolute path of the brain root. Honours the
// JEFFS_BRAIN_ROOT env var for tests and tooling; falls back to
// ~/.config/jeffs-brain when unset.
func brainRoot() string {
	if override := os.Getenv("JEFFS_BRAIN_ROOT"); override != "" {
		return override
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return "."
	}
	return filepath.Join(home, ".config", "jeffs-brain")
}
