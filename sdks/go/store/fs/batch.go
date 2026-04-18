// SPDX-License-Identifier: Apache-2.0

package fs

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"sort"
	"time"

	"github.com/jeffs-brain/memory/go/brain"
)

// Batch implements [brain.Store] by buffering mutations in an ordered
// in-memory journal and replaying them on commit. The journal is an
// append-only slice of pending ops; reads inside the batch walk the
// journal in order and then fall back to the underlying store, so every
// operation observes the effect of every earlier one in the same batch.
//
// Same-path combinations are honoured semantically:
//
//   - Write followed by Delete cancels both — the path is neither written
//     nor deleted on commit, and Delete is a no-op rather than returning
//     ErrNotFound.
//   - Write followed by Write keeps the latter content.
//   - Write followed by Rename(src, dst) moves the pending content to dst
//     on commit; no separate rename against the working tree fires.
//   - Append followed by Rename(src, dst) writes the appended result to
//     dst and removes src in a single commit step.
//
// On commit the journal is replayed top-to-bottom, executing each op
// against the underlying filesystem (via atomic per-file writes). This is
// a practical, not crash-resistant, batch: a process crash mid-commit may
// leave some writes durable and others not. The git-backed store lifts
// this to true transactional semantics via a single commit.
func (s *Store) Batch(ctx context.Context, _ brain.BatchOptions, fn func(brain.Batch) error) error {
	release, err := s.acquireRead()
	if err != nil {
		return err
	}
	defer release()

	b := &fsBatch{store: s}
	if err := fn(b); err != nil {
		return err
	}
	return b.commit(ctx)
}

// batchOp is a single buffered mutation in the journal.
type batchOp struct {
	kind    batchOpKind
	path    brain.Path // target path for write/append/delete, dst for rename
	src     brain.Path // source path for rename only
	content []byte     // payload for write/append
}

type batchOpKind int

const (
	opWrite batchOpKind = iota + 1
	opAppend
	opDelete
	opRename
)

type fsBatch struct {
	store *Store

	// ops is the ordered journal. Append-only during the user callback,
	// replayed top-to-bottom on commit.
	ops []batchOp
}

// --- Batch read side: replay the journal, fall through to the store ---

// effectiveContent returns the content of p as observed at the end of
// the journal. present=false means the path has been deleted or never
// existed. fromStore=true means the caller should consult the store for
// authoritative metadata (we have not mutated this path in the batch).
func (b *fsBatch) effectiveContent(ctx context.Context, p brain.Path) (content []byte, present bool, fromStore bool, err error) {
	return b.effectiveContentUpto(ctx, p, len(b.ops))
}

// effectiveContentUpto replays the journal up to but not including the
// given index, returning the effective content of p at that point. A
// rename entry dispatches recursively against the source's state as of
// the rename's own index, which makes Write/Append + Rename move the
// pending content correctly even when several renames chain in one
// batch.
func (b *fsBatch) effectiveContentUpto(ctx context.Context, p brain.Path, upto int) (content []byte, present bool, fromStore bool, err error) {
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
				have = true
				buf = nil
				return nil, false, false, nil
			}
		case opRename:
			if op.src == p {
				// Source has been renamed away, so p is now absent.
				have = true
				buf = nil
				return nil, false, false, nil
			}
			if op.path == p {
				sub, sok, _, serr := b.effectiveContentUpto(ctx, op.src, i)
				if serr != nil {
					return nil, false, false, serr
				}
				if sok {
					have = true
					buf = append(buf[:0], sub...)
				} else {
					have = true
					buf = nil
					return nil, false, false, nil
				}
			}
		}
	}
	if have {
		return buf, true, false, nil
	}
	// Fall through to the underlying store.
	data, rerr := b.store.Read(ctx, p)
	if rerr != nil {
		if errors.Is(rerr, brain.ErrNotFound) {
			return nil, false, true, nil
		}
		return nil, false, true, rerr
	}
	return data, true, true, nil
}

func (b *fsBatch) Read(ctx context.Context, p brain.Path) ([]byte, error) {
	content, present, _, err := b.effectiveContent(ctx, p)
	if err != nil {
		return nil, err
	}
	if !present {
		return nil, fmt.Errorf("fsstore: read %s: %w", p, brain.ErrNotFound)
	}
	return append([]byte(nil), content...), nil
}

func (b *fsBatch) Exists(ctx context.Context, p brain.Path) (bool, error) {
	_, present, _, err := b.effectiveContent(ctx, p)
	if err != nil {
		return false, err
	}
	return present, nil
}

func (b *fsBatch) Stat(ctx context.Context, p brain.Path) (brain.FileInfo, error) {
	content, present, fromStore, err := b.effectiveContent(ctx, p)
	if err != nil {
		return brain.FileInfo{}, err
	}
	if !present {
		return brain.FileInfo{}, fmt.Errorf("fsstore: stat %s: %w", p, brain.ErrNotFound)
	}
	if fromStore {
		return b.store.Stat(ctx, p)
	}
	return brain.FileInfo{Path: p, Size: int64(len(content)), ModTime: time.Now()}, nil
}

// List inside a batch merges the underlying list with the journal effects.
// We compute per-path effective state via effectiveContent for anything
// the journal touches, overlay that onto the store's view of dir, and
// honour glob/generated filters.
func (b *fsBatch) List(ctx context.Context, dir brain.Path, opts brain.ListOpts) ([]brain.FileInfo, error) {
	base, err := b.store.List(ctx, dir, opts)
	if err != nil {
		return nil, err
	}
	byPath := make(map[brain.Path]brain.FileInfo, len(base))
	for _, fi := range base {
		byPath[fi.Path] = fi
	}

	// Collect every path mentioned by the journal (source and destination).
	touched := make(map[brain.Path]struct{})
	for _, op := range b.ops {
		touched[op.path] = struct{}{}
		if op.kind == opRename {
			touched[op.src] = struct{}{}
		}
	}

	for p := range touched {
		if !pathUnder(p, dir, opts.Recursive) {
			continue
		}
		content, present, _, err := b.effectiveContent(ctx, p)
		if err != nil {
			return nil, err
		}
		if !present {
			delete(byPath, p)
			continue
		}
		byPath[p] = brain.FileInfo{Path: p, Size: int64(len(content)), ModTime: time.Now()}
	}

	result := make([]brain.FileInfo, 0, len(byPath))
	for _, fi := range byPath {
		if !opts.IncludeGenerated && brain.IsGenerated(fi.Path) {
			continue
		}
		result = append(result, fi)
	}
	sort.Slice(result, func(i, j int) bool { return result[i].Path < result[j].Path })
	return result, nil
}

// pathUnder reports whether p is under dir according to recursive/shallow
// semantics.
func pathUnder(p, dir brain.Path, recursive bool) bool {
	if dir == "" {
		return true
	}
	ps := string(p)
	ds := string(dir)
	if !((ps == ds) || (len(ps) > len(ds) && ps[:len(ds)] == ds && ps[len(ds)] == '/')) {
		return false
	}
	if recursive {
		return true
	}
	rest := ps[len(ds)+1:]
	for i := 0; i < len(rest); i++ {
		if rest[i] == '/' {
			return false
		}
	}
	return true
}

// --- Batch write side: append to the ordered journal ---

func (b *fsBatch) Write(ctx context.Context, p brain.Path, content []byte) error {
	if err := brain.ValidatePath(p); err != nil {
		return err
	}
	b.ops = append(b.ops, batchOp{kind: opWrite, path: p, content: append([]byte(nil), content...)})
	return nil
}

func (b *fsBatch) Append(ctx context.Context, p brain.Path, content []byte) error {
	if err := brain.ValidatePath(p); err != nil {
		return err
	}
	b.ops = append(b.ops, batchOp{kind: opAppend, path: p, content: append([]byte(nil), content...)})
	return nil
}

func (b *fsBatch) Delete(ctx context.Context, p brain.Path) error {
	if err := brain.ValidatePath(p); err != nil {
		return err
	}
	// Delete of a path that is effectively absent inside this batch is an
	// error (contract parity with the non-batch path). Delete following a
	// pending Write in the same batch succeeds and both ops collapse at
	// commit time into a no-op.
	_, present, _, err := b.effectiveContent(ctx, p)
	if err != nil {
		return err
	}
	if !present {
		return fmt.Errorf("fsstore: delete %s: %w", p, brain.ErrNotFound)
	}
	b.ops = append(b.ops, batchOp{kind: opDelete, path: p})
	return nil
}

func (b *fsBatch) Rename(ctx context.Context, src, dst brain.Path) error {
	if err := brain.ValidatePath(src); err != nil {
		return err
	}
	if err := brain.ValidatePath(dst); err != nil {
		return err
	}
	_, present, _, err := b.effectiveContent(ctx, src)
	if err != nil {
		return err
	}
	if !present {
		return fmt.Errorf("fsstore: rename %s: %w", src, brain.ErrNotFound)
	}
	b.ops = append(b.ops, batchOp{kind: opRename, path: dst, src: src})
	return nil
}

// --- Commit ---

// commit replays the ordered journal against the underlying store. Same-
// path combinations are merged first so we never push work that would be
// undone by a later op. Within that merged set, writes and renames are
// applied in insertion order; the commit result is equivalent to walking
// the journal top-to-bottom but avoids redundant I/O.
func (b *fsBatch) commit(ctx context.Context) error {
	if len(b.ops) == 0 {
		return nil
	}

	// Compute the net effect per path via effectiveContent (which already
	// understands the full journal semantics). For every path the journal
	// touches we emit at most one terminal op against the store.
	touched := make(map[brain.Path]struct{})
	// Ordered list of paths the journal touches, in first-mention order,
	// so the commit honours insertion-order determinism on top of the
	// semantic merge.
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

	type planStep struct {
		path    brain.Path
		kind    batchOpKind
		content []byte
	}
	var plan []planStep
	for _, p := range order {
		content, present, fromStore, err := b.effectiveContent(ctx, p)
		if err != nil {
			return err
		}
		if present {
			// If the effective content comes from the store unchanged we
			// have nothing to do for this path.
			if fromStore {
				continue
			}
			plan = append(plan, planStep{path: p, kind: opWrite, content: append([]byte(nil), content...)})
			continue
		}
		// Not present: did it exist in the store before the batch? If so
		// we must delete it; if not it was only transient within the
		// batch and we can skip the work.
		exists, err := b.store.Exists(ctx, p)
		if err != nil {
			return err
		}
		if !exists {
			continue
		}
		plan = append(plan, planStep{path: p, kind: opDelete})
	}

	for _, step := range plan {
		switch step.kind {
		case opWrite:
			// Preserve the existing content check: if the underlying file
			// already matches, skip the write to avoid spurious events
			// and mtime churn.
			current, readErr := b.store.Read(ctx, step.path)
			if readErr == nil && bytes.Equal(current, step.content) {
				continue
			}
			if err := b.store.Write(ctx, step.path, step.content); err != nil {
				return err
			}
		case opDelete:
			if err := b.store.Delete(ctx, step.path); err != nil {
				if errors.Is(err, brain.ErrNotFound) {
					continue
				}
				return err
			}
		}
	}
	return nil
}
