// SPDX-License-Identifier: Apache-2.0

// Package fs implements [brain.Store] backed by a regular filesystem
// directory. Writes are atomic via temp + fsync + rename. Batches buffer
// mutations in an ordered in-memory journal and replay them on commit; on
// error they are discarded without touching the working tree.
//
// The root directory passed to [New] is the brain root. All logical paths
// ([brain.Path]) resolve to locations under that root.
package fs

import (
	"context"
	"errors"
	"fmt"
	iofs "io/fs"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/jeffs-brain/memory/go/brain"
	"github.com/jeffs-brain/memory/go/brain/storeutil"
)

// Store is a filesystem-backed [brain.Store].
//
// Concurrency model: mu is a read-write mutex guarding open/closed state.
// Read and write operations take mu in read mode; only [Store.Close]
// takes it in write mode. This ensures Close blocks until every in-flight
// operation has unwound, and guarantees that after Close returns every
// subsequent call yields [brain.ErrReadOnly]. sinkMu is an independent
// mutex guarding the subscriber map so Subscribe/Unsubscribe never
// serialise against the I/O path.
type Store struct {
	root string

	mu     sync.RWMutex
	closed bool

	sinkMu sync.Mutex
	sinks  map[uint64]brain.EventSink
	nextID uint64
}

// New creates a Store rooted at root. The directory is created if it does
// not exist. Callers should pass an absolute path; passing a relative path
// is allowed but makes the store sensitive to the process working
// directory.
func New(root string) (*Store, error) {
	abs, err := filepath.Abs(root)
	if err != nil {
		return nil, fmt.Errorf("fsstore: resolve root: %w", err)
	}
	if err := os.MkdirAll(abs, 0o755); err != nil {
		return nil, fmt.Errorf("fsstore: create root %s: %w", abs, err)
	}
	return &Store{root: abs, sinks: make(map[uint64]brain.EventSink)}, nil
}

// Root returns the absolute filesystem path of the store root.
func (s *Store) Root() string { return s.root }

// resolve turns a logical [brain.Path] into an absolute filesystem path
// using the mapping rules shared with the git store via
// [storeutil.Resolve]. It also validates the path shape.
func (s *Store) resolve(p brain.Path) (string, error) {
	return storeutil.Resolve(s.root, p)
}

// wrapNotFound rewraps os.ErrNotExist as brain.ErrNotFound with path context.
func wrapNotFound(p brain.Path, op string, err error) error {
	if errors.Is(err, iofs.ErrNotExist) {
		return fmt.Errorf("fsstore: %s %s: %w", op, p, brain.ErrNotFound)
	}
	return err
}

// --- Read side ---

// Read implements [brain.Store].
func (s *Store) Read(ctx context.Context, p brain.Path) ([]byte, error) {
	release, err := s.acquireRead()
	if err != nil {
		return nil, err
	}
	defer release()
	abs, err := s.resolve(p)
	if err != nil {
		return nil, err
	}
	data, err := os.ReadFile(abs)
	if err != nil {
		return nil, wrapNotFound(p, "read", err)
	}
	return data, nil
}

// Exists implements [brain.Store].
func (s *Store) Exists(ctx context.Context, p brain.Path) (bool, error) {
	release, err := s.acquireRead()
	if err != nil {
		return false, err
	}
	defer release()
	abs, err := s.resolve(p)
	if err != nil {
		return false, err
	}
	_, err = os.Stat(abs)
	if errors.Is(err, iofs.ErrNotExist) {
		return false, nil
	}
	if err != nil {
		return false, err
	}
	return true, nil
}

// Stat implements [brain.Store].
func (s *Store) Stat(ctx context.Context, p brain.Path) (brain.FileInfo, error) {
	release, err := s.acquireRead()
	if err != nil {
		return brain.FileInfo{}, err
	}
	defer release()
	abs, err := s.resolve(p)
	if err != nil {
		return brain.FileInfo{}, err
	}
	info, err := os.Stat(abs)
	if err != nil {
		return brain.FileInfo{}, wrapNotFound(p, "stat", err)
	}
	return brain.FileInfo{
		Path:    p,
		Size:    info.Size(),
		ModTime: info.ModTime(),
		IsDir:   info.IsDir(),
	}, nil
}

// List implements [brain.Store].
func (s *Store) List(ctx context.Context, dir brain.Path, opts brain.ListOpts) ([]brain.FileInfo, error) {
	release, err := s.acquireRead()
	if err != nil {
		return nil, err
	}
	defer release()
	absDir := s.root
	if dir != "" {
		resolved, err := s.resolve(dir)
		if err != nil {
			return nil, err
		}
		absDir = resolved
	}
	entries, err := storeutil.List(s.root, absDir, dir, opts)
	if err != nil {
		return nil, fmt.Errorf("fsstore: %w", err)
	}
	return entries, nil
}

// --- Write side ---
//
// Writes hold the RWMutex in read mode for the duration of their I/O. Only
// [Store.Close] takes the write lock, so Close cannot race with an
// in-flight mutation: Close blocks until every pending operation has
// unwound, and after Close returns every op immediately yields
// [brain.ErrReadOnly].

// Write implements [brain.Store].
func (s *Store) Write(ctx context.Context, p brain.Path, content []byte) error {
	release, err := s.acquireRead()
	if err != nil {
		return err
	}
	defer release()
	abs, err := s.resolve(p)
	if err != nil {
		return err
	}
	_, statErr := os.Stat(abs)
	existedBefore := statErr == nil
	if err := storeutil.AtomicWrite(abs, content, 0o644); err != nil {
		return err
	}
	kind := brain.ChangeCreated
	if existedBefore {
		kind = brain.ChangeUpdated
	}
	s.emitLocked(brain.ChangeEvent{Kind: kind, Path: p, When: time.Now()})
	return nil
}

// Append implements [brain.Store].
func (s *Store) Append(ctx context.Context, p brain.Path, content []byte) error {
	release, err := s.acquireRead()
	if err != nil {
		return err
	}
	defer release()
	abs, err := s.resolve(p)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(abs), 0o755); err != nil {
		return err
	}
	_, statErr := os.Stat(abs)
	existedBefore := statErr == nil
	f, err := os.OpenFile(abs, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return err
	}
	if _, err := f.Write(content); err != nil {
		_ = f.Close()
		return err
	}
	if err := f.Sync(); err != nil {
		_ = f.Close()
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}
	kind := brain.ChangeCreated
	if existedBefore {
		kind = brain.ChangeUpdated
	}
	s.emitLocked(brain.ChangeEvent{Kind: kind, Path: p, When: time.Now()})
	return nil
}

// Delete implements [brain.Store].
func (s *Store) Delete(ctx context.Context, p brain.Path) error {
	release, err := s.acquireRead()
	if err != nil {
		return err
	}
	defer release()
	abs, err := s.resolve(p)
	if err != nil {
		return err
	}
	if err := os.Remove(abs); err != nil {
		return wrapNotFound(p, "delete", err)
	}
	s.emitLocked(brain.ChangeEvent{Kind: brain.ChangeDeleted, Path: p, When: time.Now()})
	return nil
}

// Rename implements [brain.Store].
func (s *Store) Rename(ctx context.Context, src, dst brain.Path) error {
	release, err := s.acquireRead()
	if err != nil {
		return err
	}
	defer release()
	srcAbs, err := s.resolve(src)
	if err != nil {
		return err
	}
	dstAbs, err := s.resolve(dst)
	if err != nil {
		return err
	}
	if _, err := os.Stat(srcAbs); err != nil {
		return wrapNotFound(src, "rename", err)
	}
	if err := os.MkdirAll(filepath.Dir(dstAbs), 0o755); err != nil {
		return err
	}
	if err := os.Rename(srcAbs, dstAbs); err != nil {
		return err
	}
	s.emitLocked(brain.ChangeEvent{Kind: brain.ChangeRenamed, Path: dst, OldPath: src, When: time.Now()})
	return nil
}

// --- Subscribe / Close / LocalPath ---

// Subscribe implements [brain.Store]. The returned function unsubscribes
// the sink; calling it more than once is safe.
func (s *Store) Subscribe(sink brain.EventSink) func() {
	s.sinkMu.Lock()
	id := s.nextID
	s.nextID++
	s.sinks[id] = sink
	s.sinkMu.Unlock()
	return func() {
		s.sinkMu.Lock()
		delete(s.sinks, id)
		s.sinkMu.Unlock()
	}
}

// LocalPath implements [brain.Store]. The filesystem backend always has a
// real path; the bool return is always true.
func (s *Store) LocalPath(p brain.Path) (string, bool) {
	abs, err := s.resolve(p)
	if err != nil {
		return "", false
	}
	return abs, true
}

// Close implements [brain.Store]. After Close, all operations return
// [brain.ErrReadOnly]. Close is idempotent: a second call is a no-op and
// still returns nil.
func (s *Store) Close() error {
	s.mu.Lock()
	s.closed = true
	s.mu.Unlock()
	s.sinkMu.Lock()
	s.sinks = make(map[uint64]brain.EventSink)
	s.sinkMu.Unlock()
	return nil
}

// acquireRead pins the store open for an operation and returns a release
// function. Returns [brain.ErrReadOnly] if the store has already been
// closed. Callers must defer the returned release before using the store.
func (s *Store) acquireRead() (func(), error) {
	s.mu.RLock()
	if s.closed {
		s.mu.RUnlock()
		return nil, brain.ErrReadOnly
	}
	return s.mu.RUnlock, nil
}

// emitLocked dispatches an event to all current sinks. Must be called
// while holding s.mu in read or write mode; it locks sinkMu only for the
// brief snapshot and releases before invoking callbacks.
func (s *Store) emitLocked(evt brain.ChangeEvent) {
	s.sinkMu.Lock()
	sinks := make([]brain.EventSink, 0, len(s.sinks))
	for _, sink := range s.sinks {
		sinks = append(sinks, sink)
	}
	s.sinkMu.Unlock()
	for _, sink := range sinks {
		sink.OnBrainChange(evt)
	}
}

// compile-time interface check.
var _ brain.Store = (*Store)(nil)
