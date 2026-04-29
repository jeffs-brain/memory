// SPDX-License-Identifier: Apache-2.0

package brain

import (
	"context"
	"time"
)

// Path is a logical path within the brain. Use the helpers in paths.go to
// construct values; never string-concatenate. Path is not an OS filesystem
// path; it is a stable identifier understood by any [Store] implementation.
//
// Conventions:
//   - forward slashes only
//   - no leading slash
//   - no ".." components
//   - lowercase file extensions
type Path string

// String returns the path as a plain string.
func (p Path) String() string { return string(p) }

// FileInfo describes a single brain entry returned by [Store.Stat] or
// [Store.List]. ModTime reflects the last mutation visible to the store;
// for git backends this is the commit time of the last touching commit,
// not the working tree mtime.
type FileInfo struct {
	Path    Path
	Size    int64
	ModTime time.Time
	IsDir   bool
}

// ListOpts tunes a [Store.List] call.
type ListOpts struct {
	Recursive        bool
	Glob             string
	IncludeGenerated bool
}

// BatchOptions controls how a batch commits.
type BatchOptions struct {
	Reason  string
	Message string
	Author  string
	Email   string
}

// Store is the single surface for shared brain state.
//
// Implementations MUST:
//
//   - Treat all write operations as atomic from the caller's perspective.
//   - Return [ErrNotFound] (wrapped, so errors.Is works) for missing paths.
//   - Honour context cancellation for any operation that can block on I/O
//     or network.
//   - Never accept or emit OS-specific path separators.
//   - Be safe for concurrent use.
//
// Implementations SHOULD:
//
//   - Emit [ChangeEvent]s via any registered [EventSink] after a successful
//     mutation. The event is emitted AFTER the change is durable.
//   - Hide backend-specific concepts (commits, temp files, locks) from
//     callers.
type Store interface {
	// Read returns the full contents of a file. Returns [ErrNotFound]
	// (wrapped) if the path does not exist. Binary-safe.
	Read(ctx context.Context, p Path) ([]byte, error)

	// Write performs an atomic full rewrite of p with the given content.
	Write(ctx context.Context, p Path, content []byte) error

	// Append atomically appends content to p. If p does not exist it is
	// created.
	Append(ctx context.Context, p Path, content []byte) error

	// Delete removes the file at p. Returns [ErrNotFound] if missing.
	Delete(ctx context.Context, p Path) error

	// Rename atomically moves src to dst.
	Rename(ctx context.Context, src, dst Path) error

	// Exists reports whether p exists.
	Exists(ctx context.Context, p Path) (bool, error)

	// Stat returns [FileInfo] for p, or [ErrNotFound].
	Stat(ctx context.Context, p Path) (FileInfo, error)

	// List enumerates entries under dir. Results are sorted
	// lexicographically by Path for deterministic output.
	List(ctx context.Context, dir Path, opts ListOpts) ([]FileInfo, error)

	// Batch groups a set of mutations into a single atomic unit.
	Batch(ctx context.Context, opts BatchOptions, fn func(Batch) error) error

	// Subscribe registers a sink that receives [ChangeEvent]s after every
	// successful mutation. Returns an unsubscribe function.
	Subscribe(sink EventSink) (unsubscribe func())

	// LocalPath returns the on-disk path for p if the backend has one.
	LocalPath(p Path) (string, bool)

	// Close releases any resources held by the backend.
	Close() error
}

// Batch is a transactional handle. Calls made against it are buffered and
// flushed as a single unit when the outer fn returns nil.
type Batch interface {
	Read(ctx context.Context, p Path) ([]byte, error)
	Write(ctx context.Context, p Path, content []byte) error
	Append(ctx context.Context, p Path, content []byte) error
	Delete(ctx context.Context, p Path) error
	Rename(ctx context.Context, src, dst Path) error
	Exists(ctx context.Context, p Path) (bool, error)
	Stat(ctx context.Context, p Path) (FileInfo, error)
	List(ctx context.Context, dir Path, opts ListOpts) ([]FileInfo, error)
}
