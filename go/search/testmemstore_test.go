// SPDX-License-Identifier: Apache-2.0

package search

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

// testStore is a lightweight in-memory [brain.Store] used by the
// search test suite only. Mirrors the behaviour of the production
// jeff memstore so the ported tests exercise the exact same
// subscribe / rename / delete semantics as upstream. Lives inside
// the search package so we can keep the port of jeff tests
// self-contained without touching store/mem/ (which is a separate
// scaffold package).
type testStore struct {
	mu     sync.RWMutex
	files  map[brain.Path]*testEntry
	sinks  map[uint64]brain.EventSink
	nextID uint64
	closed bool
}

type testEntry struct {
	content []byte
	modTime time.Time
}

// newTestStore creates an empty store. Callers should schedule a
// cleanup via t.Cleanup to avoid leaking goroutine state across
// tests.
func newTestStore() *testStore {
	return &testStore{
		files: make(map[brain.Path]*testEntry),
		sinks: make(map[uint64]brain.EventSink),
	}
}

func (s *testStore) Read(ctx context.Context, p brain.Path) ([]byte, error) {
	if err := s.checkOpen(); err != nil {
		return nil, err
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	e, ok := s.files[p]
	if !ok {
		return nil, fmt.Errorf("testmemstore: read %s: %w", p, brain.ErrNotFound)
	}
	return bytes.Clone(e.content), nil
}

func (s *testStore) Exists(ctx context.Context, p brain.Path) (bool, error) {
	if err := s.checkOpen(); err != nil {
		return false, err
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	if _, ok := s.files[p]; ok {
		return true, nil
	}
	prefix := string(p) + "/"
	for k := range s.files {
		if strings.HasPrefix(string(k), prefix) {
			return true, nil
		}
	}
	return false, nil
}

func (s *testStore) Stat(ctx context.Context, p brain.Path) (brain.FileInfo, error) {
	if err := s.checkOpen(); err != nil {
		return brain.FileInfo{}, err
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	e, ok := s.files[p]
	if !ok {
		prefix := string(p) + "/"
		for k := range s.files {
			if strings.HasPrefix(string(k), prefix) {
				return brain.FileInfo{Path: p, IsDir: true}, nil
			}
		}
		return brain.FileInfo{}, fmt.Errorf("testmemstore: stat %s: %w", p, brain.ErrNotFound)
	}
	return brain.FileInfo{
		Path:    p,
		Size:    int64(len(e.content)),
		ModTime: e.modTime,
	}, nil
}

func (s *testStore) List(ctx context.Context, dir brain.Path, opts brain.ListOpts) ([]brain.FileInfo, error) {
	if err := s.checkOpen(); err != nil {
		return nil, err
	}
	s.mu.RLock()
	defer s.mu.RUnlock()

	prefix := string(dir)
	if prefix != "" && !strings.HasSuffix(prefix, "/") {
		prefix += "/"
	}

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
			if opts.Glob != "" && !globMatch(opts.Glob, lastSegment(rest)) {
				continue
			}
			result = append(result, brain.FileInfo{
				Path:    p,
				Size:    int64(len(e.content)),
				ModTime: e.modTime,
			})
			continue
		}
		// Non-recursive: either a direct child or a synthetic dir
		// entry for the first path component.
		slash := strings.IndexByte(rest, '/')
		if slash == -1 {
			if !opts.IncludeGenerated && brain.IsGenerated(p) {
				continue
			}
			if opts.Glob != "" && !globMatch(opts.Glob, rest) {
				continue
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

func (s *testStore) Write(ctx context.Context, p brain.Path, content []byte) error {
	if err := s.checkOpen(); err != nil {
		return err
	}
	s.mu.Lock()
	_, existed := s.files[p]
	s.files[p] = &testEntry{content: bytes.Clone(content), modTime: time.Now()}
	sinks := s.snapshotSinks()
	s.mu.Unlock()
	kind := brain.ChangeCreated
	if existed {
		kind = brain.ChangeUpdated
	}
	dispatchTest(sinks, brain.ChangeEvent{Kind: kind, Path: p, When: time.Now()})
	return nil
}

func (s *testStore) Append(ctx context.Context, p brain.Path, content []byte) error {
	if err := s.checkOpen(); err != nil {
		return err
	}
	s.mu.Lock()
	e, existed := s.files[p]
	if !existed {
		e = &testEntry{}
		s.files[p] = e
	}
	e.content = append(e.content, content...)
	e.modTime = time.Now()
	sinks := s.snapshotSinks()
	s.mu.Unlock()
	kind := brain.ChangeCreated
	if existed {
		kind = brain.ChangeUpdated
	}
	dispatchTest(sinks, brain.ChangeEvent{Kind: kind, Path: p, When: time.Now()})
	return nil
}

func (s *testStore) Delete(ctx context.Context, p brain.Path) error {
	if err := s.checkOpen(); err != nil {
		return err
	}
	s.mu.Lock()
	_, ok := s.files[p]
	if !ok {
		s.mu.Unlock()
		return fmt.Errorf("testmemstore: delete %s: %w", p, brain.ErrNotFound)
	}
	delete(s.files, p)
	sinks := s.snapshotSinks()
	s.mu.Unlock()
	dispatchTest(sinks, brain.ChangeEvent{Kind: brain.ChangeDeleted, Path: p, When: time.Now()})
	return nil
}

func (s *testStore) Rename(ctx context.Context, src, dst brain.Path) error {
	if err := s.checkOpen(); err != nil {
		return err
	}
	s.mu.Lock()
	e, ok := s.files[src]
	if !ok {
		s.mu.Unlock()
		return fmt.Errorf("testmemstore: rename %s: %w", src, brain.ErrNotFound)
	}
	delete(s.files, src)
	s.files[dst] = &testEntry{content: bytes.Clone(e.content), modTime: time.Now()}
	sinks := s.snapshotSinks()
	s.mu.Unlock()
	dispatchTest(sinks, brain.ChangeEvent{Kind: brain.ChangeRenamed, Path: dst, OldPath: src, When: time.Now()})
	return nil
}

func (s *testStore) Batch(ctx context.Context, opts brain.BatchOptions, fn func(brain.Batch) error) error {
	// The search package does not exercise Batch during tests; a
	// stub is sufficient to satisfy the brain.Store interface.
	return fmt.Errorf("testmemstore: batch not supported")
}

func (s *testStore) Subscribe(sink brain.EventSink) func() {
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

func (s *testStore) LocalPath(p brain.Path) (string, bool) { return "", false }

func (s *testStore) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.closed = true
	s.sinks = make(map[uint64]brain.EventSink)
	return nil
}

func (s *testStore) checkOpen() error {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if s.closed {
		return brain.ErrReadOnly
	}
	return nil
}

// snapshotSinks must be called with s.mu held (write or read).
func (s *testStore) snapshotSinks() []brain.EventSink {
	sinks := make([]brain.EventSink, 0, len(s.sinks))
	for _, sink := range s.sinks {
		sinks = append(sinks, sink)
	}
	return sinks
}

func dispatchTest(sinks []brain.EventSink, evt brain.ChangeEvent) {
	for _, sink := range sinks {
		sink.OnBrainChange(evt)
	}
}

// lastSegment returns the final slash-delimited segment of a path.
func lastSegment(s string) string {
	if idx := strings.LastIndex(s, "/"); idx >= 0 {
		return s[idx+1:]
	}
	return s
}

// globMatch applies path.Match against the base name, treating
// pattern errors as a non-match.
func globMatch(pattern, name string) bool {
	matched, err := path.Match(pattern, name)
	if err != nil {
		return false
	}
	return matched
}

var _ brain.Store = (*testStore)(nil)
