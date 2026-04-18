// SPDX-License-Identifier: Apache-2.0

// Package brain defines the storage abstraction and lifecycle primitives for
// a Jeffs Brain instance.
//
// The canonical Go implementation is being ported from the upstream jeff
// project. This scaffold defines the public surface; concrete behaviour will
// land in subsequent passes.
package brain

import (
	"context"
	"errors"
)

// Brain is a handle to a single logical brain backed by a [Store].
//
// A Brain owns its store and closes it on [Brain.Close]. Higher-level
// packages (memory, knowledge, retrieval) operate against the embedded
// store.
type Brain struct {
	id    string
	root  string
	store Store
}

// Options configures how a Brain is opened.
type Options struct {
	// ID is the logical brain identifier. Required.
	ID string

	// Root is the on-disk root used by local backends. Ignored by the
	// HTTP store.
	Root string

	// Store lets callers inject a pre-constructed Store (fs, git, mem,
	// http). When nil, Open selects a default backend from Root.
	Store Store
}

// Open returns a Brain configured per opts. When opts.Store is nil the
// caller must wire a backend in the next pass.
func Open(ctx context.Context, opts Options) (*Brain, error) {
	if opts.ID == "" {
		return nil, errors.New("brain: Options.ID is required")
	}
	if opts.Store == nil {
		// TODO: implement default backend selection (fs vs git vs http).
		return nil, errors.New("brain: Options.Store is required in scaffold")
	}
	return &Brain{id: opts.ID, root: opts.Root, store: opts.Store}, nil
}

// ID returns the brain identifier.
func (b *Brain) ID() string { return b.id }

// Root returns the on-disk root if the backend uses one.
func (b *Brain) Root() string { return b.root }

// Store returns the underlying [Store] for advanced callers.
func (b *Brain) Store() Store { return b.store }

// Close releases resources held by the brain. Safe to call multiple times.
func (b *Brain) Close() error {
	if b == nil || b.store == nil {
		return nil
	}
	return b.store.Close()
}
