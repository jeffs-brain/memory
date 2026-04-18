// SPDX-License-Identifier: Apache-2.0

package git

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	gogit "github.com/go-git/go-git/v5"
	"github.com/go-git/go-git/v5/plumbing"
	"github.com/go-git/go-git/v5/plumbing/object"

	"github.com/jeffs-brain/memory/go/brain"
)

// Batch groups mutations into a single git commit. Operations are
// buffered in an in-memory journal and replayed against the working
// tree when the user callback returns nil. On commit the staged
// changes are written to the git index and committed in one atomic
// step.
//
// Concurrency:
//
//   - s.mu is held across the fn callback, stage, and commit steps. It
//     is released before the push step so a slow network cannot stall
//     concurrent commits.
//   - s.pushMu serialises pushes so only one push runs at a time.
//
// On push rejection with a non-fast-forward error the store rebases
// (via [Store.Freshen]) and retries the push once. If the retry also
// fails, [brain.ErrConflict] is returned. Push errors that are not
// non-fast-forward are logged at warn level and swallowed: the commit
// is durable and a subsequent manual push will catch up.
//
// If the callback returns an error the working tree is rolled back to
// the state before the batch ran. On an empty repo (no HEAD yet) the
// rollback cleans up every touched path explicitly before dropping
// index entries, so behaviour does not depend on go-git's habit of
// removing untracked files during HardReset.
func (s *Store) Batch(ctx context.Context, opts brain.BatchOptions, fn func(brain.Batch) error) error {
	s.mu.Lock()
	if s.closed.Load() {
		s.mu.Unlock()
		return brain.ErrReadOnly
	}

	// Buffer events so the caller only sees the net result of a
	// successful commit. Discard on rollback, flush on commit.
	s.beginBatchEvents()

	headBefore, hadHead := s.currentHead()

	b := &gitBatch{
		store: s,
	}
	if err := fn(b); err != nil {
		s.discardBatchEvents()
		s.mu.Unlock()
		return err
	}

	if len(b.ops) == 0 {
		s.discardBatchEvents()
		s.mu.Unlock()
		return nil
	}

	// Compute the merged plan, collect touched paths (for staging and
	// rollback), and apply the plan to the working tree.
	plan, touched, err := b.plan(ctx)
	if err != nil {
		s.discardBatchEvents()
		s.mu.Unlock()
		return err
	}
	events, applyErr := b.apply(ctx, plan)
	if applyErr != nil {
		rbErr := s.rollbackTo(touched, headBefore, hadHead)
		s.discardBatchEvents()
		s.mu.Unlock()
		if rbErr != nil {
			return fmt.Errorf("%w (rollback failed: %v)", applyErr, rbErr)
		}
		return applyErr
	}

	if len(touched) == 0 {
		s.discardBatchEvents()
		s.mu.Unlock()
		return nil
	}

	w, err := s.repo.Worktree()
	if err != nil {
		err = s.rollbackWithError(touched, headBefore, hadHead, err)
		s.discardBatchEvents()
		s.mu.Unlock()
		return err
	}
	for p := range touched {
		if stageErr := s.stagePath(w, p); stageErr != nil {
			stageErr = s.rollbackWithError(touched, headBefore, hadHead, stageErr)
			s.discardBatchEvents()
			s.mu.Unlock()
			return stageErr
		}
	}

	subject := "[" + firstOr(opts.Reason, "write") + "]"
	if opts.Message != "" {
		subject = subject + " " + opts.Message
	} else {
		subject = subject + " " + summariseTouched(touched, b.renames)
	}

	author := s.author
	if opts.Author != "" {
		author.Name = opts.Author
	}
	if opts.Email != "" {
		author.Email = opts.Email
	}

	commitOpts := &gogit.CommitOptions{
		Author: &object.Signature{
			Name:  author.Name,
			Email: author.Email,
			When:  time.Now(),
		},
		AllowEmptyCommits: false,
	}
	if s.opts.Sign != nil {
		commitOpts.Signer = signerFromSignFn(s.opts.Sign)
	}
	_, commitErr := w.Commit(subject, commitOpts)
	if errors.Is(commitErr, gogit.ErrEmptyCommit) {
		// Net-zero batch — nothing to commit. Not an error.
		s.discardBatchEvents()
		s.mu.Unlock()
		return nil
	}
	if commitErr != nil {
		commitErr = s.rollbackWithError(touched, headBefore, hadHead, fmt.Errorf("gitstore: commit: %w", commitErr))
		s.discardBatchEvents()
		s.mu.Unlock()
		return commitErr
	}

	// Queue the synthesised events inside the batch buffer so flush
	// delivers them in the order the user wrote them.
	s.sinkMu.Lock()
	s.batchPending = append(s.batchPending, events...)
	s.sinkMu.Unlock()

	// Release the commit lock before flushing events and pushing so
	// concurrent batches and event sinks that read the store are not
	// blocked. The commit is durable — flushing after unlock is safe.
	s.mu.Unlock()

	s.flushBatchEvents(opts.Reason)

	if s.opts.AutoPush && s.opts.RemoteURL != "" {
		return s.pushWithRetry(ctx)
	}
	return nil
}

// pushWithRetry performs a synchronous push and, on non-fast-forward
// rejection, rebases via [Store.Freshen] and retries the push once.
// Any push error other than non-fast-forward is logged and swallowed:
// the commit is durable locally and a subsequent manual push will
// deliver it.
func (s *Store) pushWithRetry(ctx context.Context) error {
	s.pushMu.Lock()
	defer s.pushMu.Unlock()

	err := s.push(ctx)
	if err == nil || errors.Is(err, gogit.NoErrAlreadyUpToDate) {
		return nil
	}
	if !isNonFastForward(err) {
		slog.Warn("gitstore: push failed, commit remains local", "err", err)
		return nil
	}
	if freshErr := s.Freshen(ctx); freshErr != nil {
		slog.Warn("gitstore: rebase during push retry failed", "err", freshErr)
		return fmt.Errorf("%w: git push rejected by %s/%s: rebase during retry failed: %v",
			brain.ErrConflict, defaultRemoteName, s.opts.Branch, freshErr)
	}
	if retryErr := s.push(ctx); retryErr != nil && !errors.Is(retryErr, gogit.NoErrAlreadyUpToDate) {
		if isNonFastForward(retryErr) {
			return fmt.Errorf("%w: git push rejected by %s/%s: %v",
				brain.ErrConflict, defaultRemoteName, s.opts.Branch, retryErr)
		}
		slog.Warn("gitstore: push retry failed, commit remains local", "err", retryErr)
	}
	return nil
}

// isNonFastForward reports whether a push error indicates the remote
// has diverged. Prefers the typed error surfaced by go-git; falls back
// to a conservative substring match for the transport error wrappers
// that do not carry the typed sentinel through.
func isNonFastForward(err error) bool {
	if err == nil {
		return false
	}
	if errors.Is(err, gogit.ErrNonFastForwardUpdate) {
		return true
	}
	msg := err.Error()
	return strings.Contains(msg, "non-fast-forward") ||
		strings.Contains(msg, "stale info")
}

// stagePath adds or removes a path from the index depending on whether
// it exists in the working tree.
func (s *Store) stagePath(w *gogit.Worktree, p brain.Path) error {
	abs, ok := s.tree.localPath(p)
	if !ok {
		return fmt.Errorf("gitstore: cannot resolve local path for %s", p)
	}
	rel, err := filepath.Rel(s.opts.Root, abs)
	if err != nil {
		return err
	}
	rel = filepath.ToSlash(rel)

	if _, statErr := os.Stat(abs); statErr != nil {
		// Path is gone from the working tree. Try to remove from the
		// index; swallow errors that mean "already not there".
		if _, rmErr := w.Remove(rel); rmErr != nil &&
			!errors.Is(rmErr, gogit.ErrGlobNoMatches) &&
			!strings.Contains(rmErr.Error(), "entry not found") {
			return fmt.Errorf("gitstore: remove %s: %w", rel, rmErr)
		}
		return nil
	}

	if _, addErr := w.Add(rel); addErr != nil {
		return fmt.Errorf("gitstore: add %s: %w", rel, addErr)
	}
	return nil
}

// currentHead returns the current HEAD commit hash or ("", false) if
// the repo has no commits yet.
func (s *Store) currentHead() (string, bool) {
	head, err := s.repo.Head()
	if err != nil {
		return "", false
	}
	return head.Hash().String(), true
}

// rollbackTo restores the working tree to the given HEAD state. On an
// empty repo (hadHead=false) there is no commit to reset to; instead
// every touched path is deleted from the working tree and any
// corresponding index entries are dropped via w.Remove.
func (s *Store) rollbackTo(touched map[brain.Path]struct{}, head string, hadHead bool) error {
	// Remove the working-tree files first.
	for p := range touched {
		if abs, ok := s.tree.localPath(p); ok {
			_ = os.Remove(abs)
		}
	}

	w, err := s.repo.Worktree()
	if err != nil {
		return err
	}

	if !hadHead {
		for p := range touched {
			abs, ok := s.tree.localPath(p)
			if !ok {
				continue
			}
			rel, relErr := filepath.Rel(s.opts.Root, abs)
			if relErr != nil {
				continue
			}
			rel = filepath.ToSlash(rel)
			if _, rmErr := w.Remove(rel); rmErr != nil &&
				!errors.Is(rmErr, gogit.ErrGlobNoMatches) &&
				!strings.Contains(rmErr.Error(), "entry not found") {
				// Entries that were never in the index are fine.
			}
		}
		return nil
	}
	return w.Reset(&gogit.ResetOptions{
		Mode:   gogit.HardReset,
		Commit: plumbing.NewHash(head),
	})
}

func (s *Store) rollbackWithError(touched map[brain.Path]struct{}, head string, hadHead bool, cause error) error {
	if rbErr := s.rollbackTo(touched, head, hadHead); rbErr != nil {
		return fmt.Errorf("%w (rollback failed: %v)", cause, rbErr)
	}
	return cause
}

// --- gitBatch implements brain.Batch ---

type gitBatchOpKind int

const (
	opWrite gitBatchOpKind = iota + 1
	opAppend
	opDelete
	opRename
)

type gitBatchOp struct {
	kind    gitBatchOpKind
	path    brain.Path // target for write/append/delete, dst for rename
	src     brain.Path // source path for rename only
	content []byte
}

type gitBatch struct {
	store *Store
	ops   []gitBatchOp
	// renames is populated after plan() so the commit subject summary
	// can show src -> dst pairs.
	renames map[brain.Path]brain.Path
}

// effectiveContent walks the journal up to but not including upto and
// returns the effective content of p at that point. Mirrors the
// semantics used by the fsstore port.
func (b *gitBatch) effectiveContent(ctx context.Context, p brain.Path, upto int) (content []byte, present bool, fromStore bool, err error) {
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
					existing, rerr := b.store.tree.read(p)
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
				have = true
				buf = nil
				return nil, false, false, nil
			}
			if op.path == p {
				sub, sok, _, serr := b.effectiveContent(ctx, op.src, i)
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
	data, rerr := b.store.tree.read(p)
	if rerr != nil {
		if errors.Is(rerr, brain.ErrNotFound) {
			return nil, false, true, nil
		}
		return nil, false, true, rerr
	}
	return data, true, true, nil
}

// plan computes the ordered, merged plan for commit. Returns the plan,
// the full set of touched paths (to drive staging and rollback), and
// the rename map (for the commit subject summary).
func (b *gitBatch) plan(ctx context.Context) ([]planStep, map[brain.Path]struct{}, error) {
	touched := make(map[brain.Path]struct{})
	renames := make(map[brain.Path]brain.Path)
	var order []brain.Path
	for _, op := range b.ops {
		paths := []brain.Path{op.path}
		if op.kind == opRename {
			paths = append(paths, op.src)
			renames[op.src] = op.path
		}
		for _, p := range paths {
			if _, ok := touched[p]; !ok {
				touched[p] = struct{}{}
				order = append(order, p)
			}
		}
	}
	b.renames = renames

	var plan []planStep
	for _, p := range order {
		content, present, fromStore, err := b.effectiveContent(ctx, p, len(b.ops))
		if err != nil {
			return nil, nil, err
		}
		if present {
			if fromStore {
				// No net change for this path; leave it alone. Still
				// keep it in touched so staging/rollback covers it.
				continue
			}
			plan = append(plan, planStep{
				path:    p,
				kind:    opWrite,
				content: append([]byte(nil), content...),
				isNew:   !pathExistedInStore(b, p),
			})
			continue
		}
		exists, existErr := b.store.tree.exists(p)
		if existErr != nil {
			return nil, nil, existErr
		}
		if !exists {
			continue
		}
		plan = append(plan, planStep{path: p, kind: opDelete})
	}
	return plan, touched, nil
}

// pathExistedInStore reports whether p was present on disk before the
// batch started. Used to distinguish ChangeCreated from ChangeUpdated
// in the emitted events.
func pathExistedInStore(b *gitBatch, p brain.Path) bool {
	exists, err := b.store.tree.exists(p)
	if err != nil {
		return false
	}
	return exists
}

// apply executes the plan against the working tree and returns the
// synthesised change events in the order the user wrote ops.
func (b *gitBatch) apply(ctx context.Context, plan []planStep) ([]brain.ChangeEvent, error) {
	// Snapshot which write/append/rename targets exist on disk before we
	// touch the working tree, so synthesised events can distinguish
	// creation from update without reading state that the apply step
	// itself has already changed.
	existedBefore := make(map[brain.Path]bool, len(b.ops))
	for _, op := range b.ops {
		switch op.kind {
		case opWrite, opAppend, opRename:
			if _, seen := existedBefore[op.path]; seen {
				continue
			}
			exists, err := b.store.tree.exists(op.path)
			if err != nil {
				return nil, err
			}
			existedBefore[op.path] = exists
		}
	}

	// Execute writes/deletes in order. For deterministic output we also
	// sort the plan by path so tests see stable ordering independent of
	// map iteration.
	sort.SliceStable(plan, func(i, j int) bool {
		return plan[i].path < plan[j].path
	})

	for _, step := range plan {
		switch step.kind {
		case opWrite:
			current, readErr := b.store.tree.read(step.path)
			if readErr == nil && bytes.Equal(current, step.content) {
				continue
			}
			if _, err := b.store.tree.write(ctx, step.path, step.content); err != nil {
				return nil, err
			}
		case opDelete:
			if err := b.store.tree.delete(ctx, step.path); err != nil {
				if errors.Is(err, brain.ErrNotFound) {
					continue
				}
				return nil, err
			}
		}
	}

	// Synthesise events from the user-level op order so subscribers
	// observe the journalled sequence, not the merged plan.
	events := make([]brain.ChangeEvent, 0, len(b.ops))
	for _, op := range b.ops {
		switch op.kind {
		case opWrite, opAppend:
			// Write/Append both emit created-or-updated. The pre-batch
			// snapshot tells us whether the path existed before this
			// batch ran; later ops in the same batch see the mutation
			// from the earlier op as an update.
			kind := brain.ChangeUpdated
			if !existedBefore[op.path] {
				kind = brain.ChangeCreated
				existedBefore[op.path] = true
			}
			events = append(events, brain.ChangeEvent{
				Kind: kind,
				Path: op.path,
				When: now(),
			})
		case opDelete:
			events = append(events, brain.ChangeEvent{
				Kind: brain.ChangeDeleted,
				Path: op.path,
				When: now(),
			})
		case opRename:
			events = append(events, brain.ChangeEvent{
				Kind:    brain.ChangeRenamed,
				Path:    op.path,
				OldPath: op.src,
				When:    now(),
			})
		}
	}
	return events, nil
}

// planStep is a single materialised write or delete produced by the
// merge step.
type planStep struct {
	path    brain.Path
	kind    gitBatchOpKind
	content []byte
	isNew   bool
}

func (b *gitBatch) Read(ctx context.Context, p brain.Path) ([]byte, error) {
	content, present, _, err := b.effectiveContent(ctx, p, len(b.ops))
	if err != nil {
		return nil, err
	}
	if !present {
		return nil, fmt.Errorf("gitstore: read %s: %w", p, brain.ErrNotFound)
	}
	return append([]byte(nil), content...), nil
}

func (b *gitBatch) Write(ctx context.Context, p brain.Path, content []byte) error {
	if err := brain.ValidatePath(p); err != nil {
		return err
	}
	b.ops = append(b.ops, gitBatchOp{kind: opWrite, path: p, content: append([]byte(nil), content...)})
	_ = ctx
	return nil
}

func (b *gitBatch) Append(ctx context.Context, p brain.Path, content []byte) error {
	if err := brain.ValidatePath(p); err != nil {
		return err
	}
	b.ops = append(b.ops, gitBatchOp{kind: opAppend, path: p, content: append([]byte(nil), content...)})
	_ = ctx
	return nil
}

func (b *gitBatch) Delete(ctx context.Context, p brain.Path) error {
	if err := brain.ValidatePath(p); err != nil {
		return err
	}
	_, present, _, err := b.effectiveContent(ctx, p, len(b.ops))
	if err != nil {
		return err
	}
	if !present {
		return fmt.Errorf("gitstore: delete %s: %w", p, brain.ErrNotFound)
	}
	b.ops = append(b.ops, gitBatchOp{kind: opDelete, path: p})
	return nil
}

func (b *gitBatch) Rename(ctx context.Context, src, dst brain.Path) error {
	if err := brain.ValidatePath(src); err != nil {
		return err
	}
	if err := brain.ValidatePath(dst); err != nil {
		return err
	}
	_, present, _, err := b.effectiveContent(ctx, src, len(b.ops))
	if err != nil {
		return err
	}
	if !present {
		return fmt.Errorf("gitstore: rename %s: %w", src, brain.ErrNotFound)
	}
	b.ops = append(b.ops, gitBatchOp{kind: opRename, path: dst, src: src})
	return nil
}

func (b *gitBatch) Exists(ctx context.Context, p brain.Path) (bool, error) {
	_, present, _, err := b.effectiveContent(ctx, p, len(b.ops))
	if err != nil {
		return false, err
	}
	return present, nil
}

func (b *gitBatch) Stat(ctx context.Context, p brain.Path) (brain.FileInfo, error) {
	content, present, fromStore, err := b.effectiveContent(ctx, p, len(b.ops))
	if err != nil {
		return brain.FileInfo{}, err
	}
	if !present {
		return brain.FileInfo{}, fmt.Errorf("gitstore: stat %s: %w", p, brain.ErrNotFound)
	}
	if fromStore {
		return b.store.tree.stat(p)
	}
	return brain.FileInfo{Path: p, Size: int64(len(content)), ModTime: now()}, nil
}

func (b *gitBatch) List(ctx context.Context, dir brain.Path, opts brain.ListOpts) ([]brain.FileInfo, error) {
	base, err := b.store.tree.list(dir, opts)
	if err != nil {
		return nil, err
	}
	byPath := make(map[brain.Path]brain.FileInfo, len(base))
	for _, fi := range base {
		byPath[fi.Path] = fi
	}

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
		content, present, _, err := b.effectiveContent(ctx, p, len(b.ops))
		if err != nil {
			return nil, err
		}
		if !present {
			delete(byPath, p)
			continue
		}
		byPath[p] = brain.FileInfo{Path: p, Size: int64(len(content)), ModTime: now()}
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

// pathUnder reports whether p is under dir according to
// recursive/shallow semantics.
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

// --- helpers ---

func firstOr(s, fallback string) string {
	if s == "" {
		return fallback
	}
	return s
}

// summariseTouched produces a short description for the commit subject
// when BatchOptions.Message is empty.
func summariseTouched(touched map[brain.Path]struct{}, renames map[brain.Path]brain.Path) string {
	if len(renames) > 0 {
		parts := make([]string, 0, len(renames))
		for src, dst := range renames {
			parts = append(parts, string(src)+" -> "+string(dst))
		}
		sort.Strings(parts)
		return strings.Join(parts, ", ")
	}
	if len(touched) == 1 {
		for p := range touched {
			return string(p)
		}
	}
	return fmt.Sprintf("%d files", len(touched))
}
