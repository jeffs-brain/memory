// SPDX-License-Identifier: Apache-2.0

package http

import (
	"context"
	"errors"
	"fmt"
	"path"
	"sort"
	"strings"
	"time"

	"github.com/jeffs-brain/memory/go/brain"
)

// journalOpKind classifies a single buffered mutation inside a batch.
type journalOpKind int

const (
	jOpWrite journalOpKind = iota + 1
	jOpAppend
	jOpDelete
	jOpRename
)

// journalOp is a single buffered mutation. For rename we record both src
// and dst; other kinds leave Src empty.
type journalOp struct {
	Kind    journalOpKind
	Path    brain.Path
	Src     brain.Path
	Content []byte
}

// batch buffers mutations locally and flushes them as a single
// POST /documents/batch-ops when the caller's fn returns nil. Writes
// observe their own pending mutations via replay.
type batch struct {
	store   *Store
	ctx     context.Context
	journal []journalOp
}

type batchState int

const (
	stateUntouched batchState = iota
	statePresent
	stateDeleted
)

// replay walks the journal and reports the final state visible to a read
// of p under the pending mutations. Mirrors the TS HttpStore journal
// semantics (spec/STORAGE.md Batch contract).
func replay(journal []journalOp, p brain.Path) (batchState, []byte) {
	touched := false
	present := false
	var content []byte
	for _, op := range journal {
		switch op.Kind {
		case jOpWrite:
			if op.Path == p {
				present = true
				content = append([]byte(nil), op.Content...)
				touched = true
			}
		case jOpAppend:
			if op.Path == p {
				var base []byte
				if present {
					base = content
				}
				content = append(base, op.Content...)
				present = true
				touched = true
			}
		case jOpDelete:
			if op.Path == p {
				present = false
				content = nil
				touched = true
			}
		case jOpRename:
			if op.Src == p {
				present = false
				content = nil
				touched = true
			} else if op.Path == p {
				present = true
				touched = true
			}
		}
	}
	if !touched {
		return stateUntouched, nil
	}
	if present {
		return statePresent, content
	}
	return stateDeleted, nil
}

// Read returns the pending content when the journal has touched p,
// otherwise falls through to the server.
func (b *batch) Read(ctx context.Context, p brain.Path) ([]byte, error) {
	if err := brain.ValidatePath(p); err != nil {
		return nil, err
	}
	state, content := replay(b.journal, p)
	if state == statePresent {
		out := make([]byte, len(content))
		copy(out, content)
		return out, nil
	}
	if state == stateDeleted {
		return nil, fmt.Errorf("store/http: read %s: %w", p, brain.ErrNotFound)
	}
	return b.store.Read(ctx, p)
}

// Write buffers a full-rewrite op. The op is never sent individually; it
// lands on the wire as part of the batch commit payload.
func (b *batch) Write(_ context.Context, p brain.Path, content []byte) error {
	if err := brain.ValidatePath(p); err != nil {
		return err
	}
	buf := make([]byte, len(content))
	copy(buf, content)
	b.journal = append(b.journal, journalOp{Kind: jOpWrite, Path: p, Content: buf})
	return nil
}

// Append materialises to a single write of the concatenated content, so
// the server sees a flat sequence independent of append ordering on the
// backend. Matches the TS HttpBatch.append semantics.
func (b *batch) Append(ctx context.Context, p brain.Path, content []byte) error {
	if err := brain.ValidatePath(p); err != nil {
		return err
	}
	state, pending := replay(b.journal, p)
	var base []byte
	switch state {
	case statePresent:
		base = pending
	case stateDeleted:
		base = nil
	default:
		existing, err := b.store.Read(ctx, p)
		if err != nil && !errors.Is(err, brain.ErrNotFound) {
			return err
		}
		base = existing
	}
	buf := make([]byte, 0, len(base)+len(content))
	buf = append(buf, base...)
	buf = append(buf, content...)
	b.journal = append(b.journal, journalOp{Kind: jOpWrite, Path: p, Content: buf})
	return nil
}

// Delete buffers a delete. If the journal shows the path already absent a
// fresh ErrNotFound surfaces; otherwise we go to the server to confirm
// existence before buffering to avoid committing a no-op.
func (b *batch) Delete(ctx context.Context, p brain.Path) error {
	if err := brain.ValidatePath(p); err != nil {
		return err
	}
	state, _ := replay(b.journal, p)
	switch state {
	case statePresent:
		b.journal = append(b.journal, journalOp{Kind: jOpDelete, Path: p})
		return nil
	case stateDeleted:
		return fmt.Errorf("store/http: delete %s: %w", p, brain.ErrNotFound)
	}
	exists, err := b.store.Exists(ctx, p)
	if err != nil {
		return err
	}
	if !exists {
		return fmt.Errorf("store/http: delete %s: %w", p, brain.ErrNotFound)
	}
	b.journal = append(b.journal, journalOp{Kind: jOpDelete, Path: p})
	return nil
}

// Rename materialises as write-to-dst followed by delete-of-src so the
// server sees a flat sequence without having to reason about rename
// ordering on its own.
func (b *batch) Rename(ctx context.Context, src, dst brain.Path) error {
	if err := brain.ValidatePath(src); err != nil {
		return err
	}
	if err := brain.ValidatePath(dst); err != nil {
		return err
	}
	state, pending := replay(b.journal, src)
	var payload []byte
	switch state {
	case statePresent:
		payload = pending
	case stateDeleted:
		return fmt.Errorf("store/http: rename %s: %w", src, brain.ErrNotFound)
	default:
		existing, err := b.store.Read(ctx, src)
		if err != nil {
			return err
		}
		payload = existing
	}
	buf := make([]byte, len(payload))
	copy(buf, payload)
	b.journal = append(b.journal, journalOp{Kind: jOpWrite, Path: dst, Content: buf})
	b.journal = append(b.journal, journalOp{Kind: jOpDelete, Path: src})
	return nil
}

// Exists reports presence under pending mutations.
func (b *batch) Exists(ctx context.Context, p brain.Path) (bool, error) {
	if err := brain.ValidatePath(p); err != nil {
		return false, err
	}
	state, _ := replay(b.journal, p)
	switch state {
	case statePresent:
		return true, nil
	case stateDeleted:
		return false, nil
	}
	return b.store.Exists(ctx, p)
}

// Stat reports the size of pending content; modtime is synthesised. Server
// fallback for untouched paths.
func (b *batch) Stat(ctx context.Context, p brain.Path) (brain.FileInfo, error) {
	if err := brain.ValidatePath(p); err != nil {
		return brain.FileInfo{}, err
	}
	state, content := replay(b.journal, p)
	switch state {
	case statePresent:
		return brain.FileInfo{
			Path:    p,
			Size:    int64(len(content)),
			ModTime: time.Now().UTC(),
			IsDir:   false,
		}, nil
	case stateDeleted:
		return brain.FileInfo{}, fmt.Errorf("store/http: stat %s: %w", p, brain.ErrNotFound)
	}
	return b.store.Stat(ctx, p)
}

// List overlays buffered mutations on top of the server listing. Deleted
// paths vanish, new paths appear, renamed-away sources disappear. Matches
// the TS HttpBatch.list semantics.
func (b *batch) List(ctx context.Context, dir brain.Path, opts brain.ListOpts) ([]brain.FileInfo, error) {
	base, err := b.store.List(ctx, dir, opts)
	if err != nil {
		return nil, err
	}
	byPath := make(map[brain.Path]brain.FileInfo, len(base))
	for _, fi := range base {
		byPath[fi.Path] = fi
	}
	touched := make(map[brain.Path]struct{})
	for _, op := range b.journal {
		if op.Kind == jOpRename {
			touched[op.Src] = struct{}{}
			touched[op.Path] = struct{}{}
		} else {
			touched[op.Path] = struct{}{}
		}
	}
	for p := range touched {
		if !pathUnder(p, dir, opts.Recursive) {
			continue
		}
		state, content := replay(b.journal, p)
		switch state {
		case statePresent:
			if !opts.IncludeGenerated && brain.IsGenerated(p) {
				delete(byPath, p)
				continue
			}
			if opts.Glob != "" {
				matched, err := path.Match(opts.Glob, path.Base(string(p)))
				if err != nil || !matched {
					continue
				}
			}
			byPath[p] = brain.FileInfo{
				Path:    p,
				Size:    int64(len(content)),
				ModTime: time.Now().UTC(),
				IsDir:   false,
			}
		case stateDeleted:
			delete(byPath, p)
		}
	}
	out := make([]brain.FileInfo, 0, len(byPath))
	for _, fi := range byPath {
		out = append(out, fi)
	}
	sort.Slice(out, func(i, j int) bool {
		return out[i].Path < out[j].Path
	})
	return out, nil
}

// pathUnder mirrors the TS pathUnderLocal helper. Reports whether p falls
// within dir (recursive or immediate child depending on opts).
func pathUnder(p, dir brain.Path, recursive bool) bool {
	if dir == "" {
		return true
	}
	ps := string(p)
	ds := string(dir)
	if ps == ds {
		return true
	}
	if !(len(ps) > len(ds) && strings.HasPrefix(ps, ds) && ps[len(ds)] == '/') {
		return false
	}
	if recursive {
		return true
	}
	rest := ps[len(ds)+1:]
	return !strings.Contains(rest, "/")
}

// compile-time guard: *batch satisfies brain.Batch.
var _ brain.Batch = (*batch)(nil)
