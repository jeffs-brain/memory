// SPDX-License-Identifier: Apache-2.0

// Package pt implements a [brain.Store] that maps every logical path
// directly onto the on-disk root. Unlike [go/store/fs], no
// memory/global/ → memory/ remapping happens: the on-disk tree is a
// byte-for-byte mirror of the logical tree.
//
// This is the layout the HTTP daemon uses and the layout every SDK
// (Go, TS, Py) reads when opening a brain root. Extraction pipelines
// that want to populate a brain for cross-SDK consumption should write
// through this store so the files surface at the paths each SDK
// expects.
package pt

import (
	"context"
	"errors"
	"fmt"
	iofs "io/fs"
	"os"
	"path"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/jeffs-brain/memory/go/brain"
)

// Store is a filesystem-backed passthrough [brain.Store]. Every
// logical path resolves to root/path with no remapping. Atomic writes
// use rename, a mutex guards open/closed state, and a separate sink
// map tracks subscribers.
type Store struct {
	root string

	mu     sync.RWMutex
	closed bool

	sinkMu sync.Mutex
	sinks  map[uint64]brain.EventSink
	nextID uint64
}

// New creates a passthrough Store rooted at root. The directory is
// created if missing. Relative paths are resolved against the process
// working directory.
func New(root string) (*Store, error) {
	abs, err := filepath.Abs(root)
	if err != nil {
		return nil, fmt.Errorf("pt: resolve root: %w", err)
	}
	if err := os.MkdirAll(abs, 0o755); err != nil {
		return nil, fmt.Errorf("pt: create root %s: %w", abs, err)
	}
	return &Store{
		root:  abs,
		sinks: make(map[uint64]brain.EventSink),
	}, nil
}

func (s *Store) Root() string { return s.root }

func (s *Store) resolve(p brain.Path) (string, error) {
	if err := brain.ValidatePath(p); err != nil {
		return "", err
	}
	return filepath.Join(s.root, filepath.FromSlash(string(p))), nil
}

func (s *Store) acquireRead() (func(), error) {
	s.mu.RLock()
	if s.closed {
		s.mu.RUnlock()
		return nil, brain.ErrReadOnly
	}
	return s.mu.RUnlock, nil
}

func wrapNotFound(p brain.Path, op string, err error) error {
	if errors.Is(err, iofs.ErrNotExist) {
		return fmt.Errorf("pt: %s %s: %w", op, p, brain.ErrNotFound)
	}
	return err
}

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
	return brain.FileInfo{Path: p, Size: info.Size(), ModTime: info.ModTime(), IsDir: info.IsDir()}, nil
}

// List walks the subtree when opts.Recursive is set, otherwise lists
// the immediate children. Generated files (basename starting `_`) are
// dropped unless opts.IncludeGenerated is true.
func (s *Store) List(ctx context.Context, dir brain.Path, opts brain.ListOpts) ([]brain.FileInfo, error) {
	release, err := s.acquireRead()
	if err != nil {
		return nil, err
	}
	defer release()
	absDir := s.root
	if dir != "" {
		if err := brain.ValidatePath(dir); err != nil {
			return nil, err
		}
		absDir = filepath.Join(s.root, filepath.FromSlash(string(dir)))
	}
	info, err := os.Stat(absDir)
	if err != nil {
		if errors.Is(err, iofs.ErrNotExist) {
			return nil, nil
		}
		return nil, err
	}
	if !info.IsDir() {
		return nil, nil
	}

	out := []brain.FileInfo{}
	if opts.Recursive {
		walkErr := filepath.Walk(absDir, func(p string, fi iofs.FileInfo, walkErr error) error {
			if walkErr != nil {
				return walkErr
			}
			if p == absDir {
				return nil
			}
			rel, _ := filepath.Rel(s.root, p)
			rel = filepath.ToSlash(rel)
			base := filepath.Base(p)
			if !opts.IncludeGenerated && strings.HasPrefix(base, "_") {
				if fi.IsDir() {
					return filepath.SkipDir
				}
				return nil
			}
			if opts.Glob != "" {
				if !fi.IsDir() {
					if matched, _ := path.Match(opts.Glob, base); !matched {
						return nil
					}
				}
			}
			out = append(out, brain.FileInfo{
				Path:    brain.Path(rel),
				Size:    fi.Size(),
				ModTime: fi.ModTime(),
				IsDir:   fi.IsDir(),
			})
			return nil
		})
		if walkErr != nil {
			return nil, walkErr
		}
	} else {
		entries, err := os.ReadDir(absDir)
		if err != nil {
			return nil, err
		}
		for _, e := range entries {
			base := e.Name()
			if !opts.IncludeGenerated && strings.HasPrefix(base, "_") {
				continue
			}
			if opts.Glob != "" {
				if !e.IsDir() {
					if matched, _ := path.Match(opts.Glob, base); !matched {
						continue
					}
				}
			}
			fi, err := e.Info()
			if err != nil {
				continue
			}
			rel := base
			if dir != "" {
				rel = path.Join(string(dir), base)
			}
			out = append(out, brain.FileInfo{
				Path:    brain.Path(rel),
				Size:    fi.Size(),
				ModTime: fi.ModTime(),
				IsDir:   e.IsDir(),
			})
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Path < out[j].Path })
	return out, nil
}

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
	if err := os.MkdirAll(filepath.Dir(abs), 0o755); err != nil {
		return err
	}
	_, statErr := os.Stat(abs)
	existedBefore := statErr == nil
	tmp := abs + ".tmp"
	if err := os.WriteFile(tmp, content, 0o644); err != nil {
		return err
	}
	if err := os.Rename(tmp, abs); err != nil {
		_ = os.Remove(tmp)
		return err
	}
	kind := brain.ChangeCreated
	if existedBefore {
		kind = brain.ChangeUpdated
	}
	s.emit(brain.ChangeEvent{Kind: kind, Path: p, When: time.Now()})
	return nil
}

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
	if err := f.Close(); err != nil {
		return err
	}
	kind := brain.ChangeCreated
	if existedBefore {
		kind = brain.ChangeUpdated
	}
	s.emit(brain.ChangeEvent{Kind: kind, Path: p, When: time.Now()})
	return nil
}

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
	s.emit(brain.ChangeEvent{Kind: brain.ChangeDeleted, Path: p, When: time.Now()})
	return nil
}

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
	s.emit(brain.ChangeEvent{Kind: brain.ChangeRenamed, Path: dst, OldPath: src, When: time.Now()})
	return nil
}

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

func (s *Store) LocalPath(p brain.Path) (string, bool) {
	abs, err := s.resolve(p)
	if err != nil {
		return "", false
	}
	return abs, true
}

func (s *Store) Close() error {
	s.mu.Lock()
	s.closed = true
	s.mu.Unlock()
	s.sinkMu.Lock()
	s.sinks = make(map[uint64]brain.EventSink)
	s.sinkMu.Unlock()
	return nil
}

func (s *Store) emit(evt brain.ChangeEvent) {
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

// Batch buffers mutations as a journal and replays them on commit. The
// semantics match store/fs: write+delete cancels, write+write keeps
// the latter, append accumulates.
func (s *Store) Batch(ctx context.Context, _ brain.BatchOptions, fn func(brain.Batch) error) error {
	release, err := s.acquireRead()
	if err != nil {
		return err
	}
	defer release()
	b := &batch{store: s}
	if err := fn(b); err != nil {
		return err
	}
	return b.commit(ctx)
}

type batchOpKind int

const (
	opWrite batchOpKind = iota + 1
	opAppend
	opDelete
	opRename
)

type batchOp struct {
	kind    batchOpKind
	path    brain.Path
	src     brain.Path
	content []byte
}

type batch struct {
	store *Store
	ops   []batchOp
}

func (b *batch) Read(ctx context.Context, p brain.Path) ([]byte, error) {
	content, present, _, err := b.effective(ctx, p, len(b.ops))
	if err != nil {
		return nil, err
	}
	if !present {
		return nil, fmt.Errorf("pt: read %s: %w", p, brain.ErrNotFound)
	}
	return append([]byte(nil), content...), nil
}

func (b *batch) Exists(ctx context.Context, p brain.Path) (bool, error) {
	_, present, _, err := b.effective(ctx, p, len(b.ops))
	return present, err
}

func (b *batch) Stat(ctx context.Context, p brain.Path) (brain.FileInfo, error) {
	content, present, fromStore, err := b.effective(ctx, p, len(b.ops))
	if err != nil {
		return brain.FileInfo{}, err
	}
	if !present {
		return brain.FileInfo{}, fmt.Errorf("pt: stat %s: %w", p, brain.ErrNotFound)
	}
	if fromStore {
		return b.store.Stat(ctx, p)
	}
	return brain.FileInfo{Path: p, Size: int64(len(content)), ModTime: time.Now()}, nil
}

func (b *batch) List(ctx context.Context, dir brain.Path, opts brain.ListOpts) ([]brain.FileInfo, error) {
	return b.store.List(ctx, dir, opts)
}

func (b *batch) Write(ctx context.Context, p brain.Path, content []byte) error {
	if err := brain.ValidatePath(p); err != nil {
		return err
	}
	b.ops = append(b.ops, batchOp{kind: opWrite, path: p, content: append([]byte(nil), content...)})
	return nil
}

func (b *batch) Append(ctx context.Context, p brain.Path, content []byte) error {
	if err := brain.ValidatePath(p); err != nil {
		return err
	}
	b.ops = append(b.ops, batchOp{kind: opAppend, path: p, content: append([]byte(nil), content...)})
	return nil
}

func (b *batch) Delete(ctx context.Context, p brain.Path) error {
	if err := brain.ValidatePath(p); err != nil {
		return err
	}
	_, present, _, err := b.effective(ctx, p, len(b.ops))
	if err != nil {
		return err
	}
	if !present {
		return fmt.Errorf("pt: delete %s: %w", p, brain.ErrNotFound)
	}
	b.ops = append(b.ops, batchOp{kind: opDelete, path: p})
	return nil
}

func (b *batch) Rename(ctx context.Context, src, dst brain.Path) error {
	if err := brain.ValidatePath(src); err != nil {
		return err
	}
	if err := brain.ValidatePath(dst); err != nil {
		return err
	}
	_, present, _, err := b.effective(ctx, src, len(b.ops))
	if err != nil {
		return err
	}
	if !present {
		return fmt.Errorf("pt: rename %s: %w", src, brain.ErrNotFound)
	}
	b.ops = append(b.ops, batchOp{kind: opRename, path: dst, src: src})
	return nil
}

func (b *batch) effective(ctx context.Context, p brain.Path, upto int) (content []byte, present, fromStore bool, err error) {
	if upto > len(b.ops) {
		upto = len(b.ops)
	}
	var have bool
	var buf []byte
	for i := 0; i < upto; i++ {
		op := b.ops[i]
		switch op.kind {
		case opWrite:
			if op.path == p {
				have = true
				buf = append(buf[:0], op.content...)
			}
		case opAppend:
			if op.path == p {
				if !have {
					existing, rerr := b.store.Read(ctx, p)
					if rerr != nil && !errors.Is(rerr, brain.ErrNotFound) {
						return nil, false, false, rerr
					}
					buf = existing
					have = true
				}
				buf = append(buf, op.content...)
			}
		case opDelete:
			if op.path == p {
				return nil, false, false, nil
			}
		case opRename:
			if op.src == p {
				return nil, false, false, nil
			}
			if op.path == p {
				sub, sok, _, serr := b.effective(ctx, op.src, i)
				if serr != nil {
					return nil, false, false, serr
				}
				if sok {
					have = true
					buf = append(buf[:0], sub...)
				} else {
					return nil, false, false, nil
				}
			}
		}
	}
	if have {
		return buf, true, false, nil
	}
	data, rerr := b.store.Read(ctx, p)
	if rerr != nil {
		if errors.Is(rerr, brain.ErrNotFound) {
			return nil, false, true, nil
		}
		return nil, false, true, rerr
	}
	return data, true, true, nil
}

func (b *batch) commit(ctx context.Context) error {
	if len(b.ops) == 0 {
		return nil
	}
	touched := map[brain.Path]struct{}{}
	var order []brain.Path
	for _, op := range b.ops {
		paths := []brain.Path{op.path}
		if op.kind == opRename {
			paths = append(paths, op.src)
		}
		for _, p := range paths {
			if _, ok := touched[p]; !ok {
				touched[p] = struct{}{}
				order = append(order, p)
			}
		}
	}
	for _, p := range order {
		content, present, fromStore, err := b.effective(ctx, p, len(b.ops))
		if err != nil {
			return err
		}
		if present {
			if fromStore {
				continue
			}
			if err := b.store.Write(ctx, p, content); err != nil {
				return err
			}
			continue
		}
		exists, err := b.store.Exists(ctx, p)
		if err != nil {
			return err
		}
		if !exists {
			continue
		}
		if err := b.store.Delete(ctx, p); err != nil {
			if errors.Is(err, brain.ErrNotFound) {
				continue
			}
			return err
		}
	}
	return nil
}

var _ brain.Store = (*Store)(nil)
