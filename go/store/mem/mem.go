// SPDX-License-Identifier: Apache-2.0

// Package mem provides an in-memory [brain.Store] implementation. It is
// intended for tests: it skips all filesystem interaction in favour of a
// map, which makes memory-package and knowledge-package unit tests an
// order of magnitude faster.
//
// It is safe for concurrent use and honours the full contract including
// batch read-your-own-writes semantics, event subscription, and sentinel
// errors.
package mem

import (
	"bytes"
	"context"
	"fmt"
	"path"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/jeffs-brain/memory/go/brain"
)

// Store is an in-memory [brain.Store].
type Store struct {
	mu     sync.RWMutex
	files  map[brain.Path]*entry
	sinks  map[uint64]brain.EventSink
	nextID uint64
	closed bool
}

type entry struct {
	content []byte
	modTime time.Time
}

// New creates an empty Store.
func New() *Store {
	return &Store{
		files: make(map[brain.Path]*entry),
		sinks: make(map[uint64]brain.EventSink),
	}
}

// --- Read side ---

// Read implements [brain.Store].
func (s *Store) Read(ctx context.Context, p brain.Path) ([]byte, error) {
	if err := s.checkOpen(); err != nil {
		return nil, err
	}
	if err := brain.ValidatePath(p); err != nil {
		return nil, err
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	e, ok := s.files[p]
	if !ok {
		return nil, fmt.Errorf("memstore: read %s: %w", p, brain.ErrNotFound)
	}
	return bytes.Clone(e.content), nil
}

// Exists implements [brain.Store].
func (s *Store) Exists(ctx context.Context, p brain.Path) (bool, error) {
	if err := s.checkOpen(); err != nil {
		return false, err
	}
	if err := brain.ValidatePath(p); err != nil {
		return false, err
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	_, ok := s.files[p]
	if ok {
		return true, nil
	}
	// A path may be present as a directory prefix of another file — the
	// memstore has no explicit directories, so treat any prefix hit as
	// existing.
	prefix := string(p) + "/"
	for k := range s.files {
		if strings.HasPrefix(string(k), prefix) {
			return true, nil
		}
	}
	return false, nil
}

// Stat implements [brain.Store].
func (s *Store) Stat(ctx context.Context, p brain.Path) (brain.FileInfo, error) {
	if err := s.checkOpen(); err != nil {
		return brain.FileInfo{}, err
	}
	if err := brain.ValidatePath(p); err != nil {
		return brain.FileInfo{}, err
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	e, ok := s.files[p]
	if !ok {
		// Check directory-prefix existence for Stat parity with the
		// filesystem backend.
		prefix := string(p) + "/"
		for k := range s.files {
			if strings.HasPrefix(string(k), prefix) {
				return brain.FileInfo{Path: p, IsDir: true}, nil
			}
		}
		return brain.FileInfo{}, fmt.Errorf("memstore: stat %s: %w", p, brain.ErrNotFound)
	}
	return brain.FileInfo{
		Path:    p,
		Size:    int64(len(e.content)),
		ModTime: e.modTime,
	}, nil
}

// List implements [brain.Store].
func (s *Store) List(ctx context.Context, dir brain.Path, opts brain.ListOpts) ([]brain.FileInfo, error) {
	if err := s.checkOpen(); err != nil {
		return nil, err
	}
	s.mu.RLock()
	defer s.mu.RUnlock()

	prefix := string(dir)
	if prefix != "" && !strings.HasSuffix(prefix, "/") {
		prefix += "/"
	}

	// For non-recursive listings we need both files directly under the dir
	// and directory entries that exist only as path components of deeper
	// files.
	seenDirs := make(map[brain.Path]bool)
	result := make([]brain.FileInfo, 0)

	for p, e := range s.files {
		ps := string(p)
		if prefix != "" && !strings.HasPrefix(ps, prefix) {
			continue
		}
		rest := strings.TrimPrefix(ps, prefix)
		if rest == "" {
			continue
		}
		if opts.Recursive {
			if !opts.IncludeGenerated && brain.IsGenerated(p) {
				continue
			}
			if opts.Glob != "" {
				if !globMatch(opts.Glob, lastSegment(rest)) {
					continue
				}
			}
			result = append(result, brain.FileInfo{
				Path:    p,
				Size:    int64(len(e.content)),
				ModTime: e.modTime,
			})
			continue
		}
		// Non-recursive: either a direct child (no further /) or a
		// synthetic dir entry for the first path component.
		slash := strings.IndexByte(rest, '/')
		if slash == -1 {
			if !opts.IncludeGenerated && brain.IsGenerated(p) {
				continue
			}
			if opts.Glob != "" {
				if !globMatch(opts.Glob, rest) {
					continue
				}
			}
			result = append(result, brain.FileInfo{
				Path:    p,
				Size:    int64(len(e.content)),
				ModTime: e.modTime,
			})
		} else {
			childDir := brain.Path(prefix + rest[:slash])
			if !seenDirs[childDir] {
				seenDirs[childDir] = true
				result = append(result, brain.FileInfo{
					Path:  childDir,
					IsDir: true,
				})
			}
		}
	}

	sort.Slice(result, func(i, j int) bool { return result[i].Path < result[j].Path })
	return result, nil
}

// --- Write side ---

// Write implements [brain.Store].
func (s *Store) Write(ctx context.Context, p brain.Path, content []byte) error {
	if err := s.checkOpen(); err != nil {
		return err
	}
	if err := brain.ValidatePath(p); err != nil {
		return err
	}
	s.mu.Lock()
	_, existed := s.files[p]
	s.files[p] = &entry{content: bytes.Clone(content), modTime: time.Now()}
	sinks := s.sinksSnapshot()
	s.mu.Unlock()
	kind := brain.ChangeCreated
	if existed {
		kind = brain.ChangeUpdated
	}
	dispatch(sinks, brain.ChangeEvent{Kind: kind, Path: p, When: time.Now()})
	return nil
}

// Append implements [brain.Store].
func (s *Store) Append(ctx context.Context, p brain.Path, content []byte) error {
	if err := s.checkOpen(); err != nil {
		return err
	}
	if err := brain.ValidatePath(p); err != nil {
		return err
	}
	s.mu.Lock()
	e, existed := s.files[p]
	if !existed {
		e = &entry{}
		s.files[p] = e
	}
	e.content = append(e.content, content...)
	e.modTime = time.Now()
	sinks := s.sinksSnapshot()
	s.mu.Unlock()
	kind := brain.ChangeCreated
	if existed {
		kind = brain.ChangeUpdated
	}
	dispatch(sinks, brain.ChangeEvent{Kind: kind, Path: p, When: time.Now()})
	return nil
}

// Delete implements [brain.Store].
func (s *Store) Delete(ctx context.Context, p brain.Path) error {
	if err := s.checkOpen(); err != nil {
		return err
	}
	if err := brain.ValidatePath(p); err != nil {
		return err
	}
	s.mu.Lock()
	_, ok := s.files[p]
	if !ok {
		s.mu.Unlock()
		return fmt.Errorf("memstore: delete %s: %w", p, brain.ErrNotFound)
	}
	delete(s.files, p)
	sinks := s.sinksSnapshot()
	s.mu.Unlock()
	dispatch(sinks, brain.ChangeEvent{Kind: brain.ChangeDeleted, Path: p, When: time.Now()})
	return nil
}

// Rename implements [brain.Store].
func (s *Store) Rename(ctx context.Context, src, dst brain.Path) error {
	if err := s.checkOpen(); err != nil {
		return err
	}
	if err := brain.ValidatePath(src); err != nil {
		return err
	}
	if err := brain.ValidatePath(dst); err != nil {
		return err
	}
	s.mu.Lock()
	e, ok := s.files[src]
	if !ok {
		s.mu.Unlock()
		return fmt.Errorf("memstore: rename %s: %w", src, brain.ErrNotFound)
	}
	delete(s.files, src)
	s.files[dst] = &entry{content: bytes.Clone(e.content), modTime: time.Now()}
	sinks := s.sinksSnapshot()
	s.mu.Unlock()
	dispatch(sinks, brain.ChangeEvent{Kind: brain.ChangeRenamed, Path: dst, OldPath: src, When: time.Now()})
	return nil
}

// --- Subscribe / Close / LocalPath ---

// Subscribe implements [brain.Store].
func (s *Store) Subscribe(sink brain.EventSink) func() {
	s.mu.Lock()
	id := s.nextID
	s.nextID++
	s.sinks[id] = sink
	s.mu.Unlock()
	return func() {
		s.mu.Lock()
		delete(s.sinks, id)
		s.mu.Unlock()
	}
}

// LocalPath implements [brain.Store]. The memory backend has no on-disk
// path; always returns false.
func (s *Store) LocalPath(p brain.Path) (string, bool) { return "", false }

// Close implements [brain.Store].
func (s *Store) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.closed = true
	s.sinks = make(map[uint64]brain.EventSink)
	return nil
}

func (s *Store) checkOpen() error {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if s.closed {
		return brain.ErrReadOnly
	}
	return nil
}

// sinksSnapshot must be called with s.mu held (write or read).
func (s *Store) sinksSnapshot() []brain.EventSink {
	sinks := make([]brain.EventSink, 0, len(s.sinks))
	for _, sink := range s.sinks {
		sinks = append(sinks, sink)
	}
	return sinks
}

func dispatch(sinks []brain.EventSink, evt brain.ChangeEvent) {
	for _, sink := range sinks {
		sink.OnBrainChange(evt)
	}
}

// --- Batch ---

// Batch implements [brain.Store].
func (s *Store) Batch(ctx context.Context, opts brain.BatchOptions, fn func(brain.Batch) error) error {
	if err := s.checkOpen(); err != nil {
		return err
	}
	// Snapshot the current file map. On commit, we replace the live map
	// with the batch's mutated copy. On rollback, we discard it.
	s.mu.Lock()
	snapshot := make(map[brain.Path]*entry, len(s.files))
	for k, v := range s.files {
		snapshot[k] = &entry{content: bytes.Clone(v.content), modTime: v.modTime}
	}
	s.mu.Unlock()

	b := &memBatch{store: s, files: snapshot}
	if err := fn(b); err != nil {
		return err
	}

	// Commit: swap in the batch map, then emit events for all changes.
	s.mu.Lock()
	old := s.files
	s.files = b.files
	sinks := s.sinksSnapshot()
	s.mu.Unlock()

	events := diffEvents(old, b.files, opts.Reason)
	for _, evt := range events {
		dispatch(sinks, evt)
	}
	return nil
}

type memBatch struct {
	store *Store
	files map[brain.Path]*entry
}

func (b *memBatch) Read(ctx context.Context, p brain.Path) ([]byte, error) {
	if err := brain.ValidatePath(p); err != nil {
		return nil, err
	}
	e, ok := b.files[p]
	if !ok {
		return nil, fmt.Errorf("memstore: read %s: %w", p, brain.ErrNotFound)
	}
	return bytes.Clone(e.content), nil
}

func (b *memBatch) Write(ctx context.Context, p brain.Path, content []byte) error {
	if err := brain.ValidatePath(p); err != nil {
		return err
	}
	b.files[p] = &entry{content: bytes.Clone(content), modTime: time.Now()}
	return nil
}

func (b *memBatch) Append(ctx context.Context, p brain.Path, content []byte) error {
	if err := brain.ValidatePath(p); err != nil {
		return err
	}
	e, ok := b.files[p]
	if !ok {
		e = &entry{}
		b.files[p] = e
	}
	e.content = append(e.content, content...)
	e.modTime = time.Now()
	return nil
}

func (b *memBatch) Delete(ctx context.Context, p brain.Path) error {
	if err := brain.ValidatePath(p); err != nil {
		return err
	}
	if _, ok := b.files[p]; !ok {
		return fmt.Errorf("memstore: delete %s: %w", p, brain.ErrNotFound)
	}
	delete(b.files, p)
	return nil
}

func (b *memBatch) Rename(ctx context.Context, src, dst brain.Path) error {
	if err := brain.ValidatePath(src); err != nil {
		return err
	}
	if err := brain.ValidatePath(dst); err != nil {
		return err
	}
	e, ok := b.files[src]
	if !ok {
		return fmt.Errorf("memstore: rename %s: %w", src, brain.ErrNotFound)
	}
	delete(b.files, src)
	b.files[dst] = &entry{content: bytes.Clone(e.content), modTime: time.Now()}
	return nil
}

func (b *memBatch) Exists(ctx context.Context, p brain.Path) (bool, error) {
	if err := brain.ValidatePath(p); err != nil {
		return false, err
	}
	if _, ok := b.files[p]; ok {
		return true, nil
	}
	prefix := string(p) + "/"
	for k := range b.files {
		if strings.HasPrefix(string(k), prefix) {
			return true, nil
		}
	}
	return false, nil
}

func (b *memBatch) Stat(ctx context.Context, p brain.Path) (brain.FileInfo, error) {
	if err := brain.ValidatePath(p); err != nil {
		return brain.FileInfo{}, err
	}
	e, ok := b.files[p]
	if !ok {
		return brain.FileInfo{}, fmt.Errorf("memstore: stat %s: %w", p, brain.ErrNotFound)
	}
	return brain.FileInfo{
		Path:    p,
		Size:    int64(len(e.content)),
		ModTime: e.modTime,
	}, nil
}

func (b *memBatch) List(ctx context.Context, dir brain.Path, opts brain.ListOpts) ([]brain.FileInfo, error) {
	// Delegate to a synthetic read-only Store view over the batch map.
	tmp := &Store{files: b.files}
	return tmp.List(ctx, dir, opts)
}

// diffEvents produces the events that fire when old becomes new. Used to
// emit events on batch commit.
func diffEvents(old, updated map[brain.Path]*entry, reason string) []brain.ChangeEvent {
	var events []brain.ChangeEvent
	now := time.Now()
	for p, e := range updated {
		prev, existed := old[p]
		if !existed {
			events = append(events, brain.ChangeEvent{Kind: brain.ChangeCreated, Path: p, Reason: reason, When: now})
			continue
		}
		if !bytes.Equal(prev.content, e.content) {
			events = append(events, brain.ChangeEvent{Kind: brain.ChangeUpdated, Path: p, Reason: reason, When: now})
		}
	}
	for p := range old {
		if _, stillThere := updated[p]; !stillThere {
			events = append(events, brain.ChangeEvent{Kind: brain.ChangeDeleted, Path: p, Reason: reason, When: now})
		}
	}
	return events
}

// --- Utility ---

func lastSegment(s string) string {
	if idx := strings.LastIndex(s, "/"); idx >= 0 {
		return s[idx+1:]
	}
	return s
}

// globMatch applies a simple glob using Go's [path.Match] semantics, but
// against the base name only. A pattern with unsupported syntax
// (returning an error from path.Match) is treated as no match.
func globMatch(pattern, name string) bool {
	matched, err := path.Match(pattern, name)
	if err != nil {
		return false
	}
	return matched
}

// compile-time interface check.
var _ brain.Store = (*Store)(nil)
