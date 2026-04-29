// SPDX-License-Identifier: Apache-2.0

// Package git implements [brain.Store] backed by a git working tree.
//
// Writes go to the working tree via an internal filesystem helper and
// are staged and committed inside the batch. Each [brain.Batch]
// produces exactly one commit whose subject line is derived from
// [brain.BatchOptions]. When a remote is configured the commit is
// pushed synchronously as part of the batch; on non-fast-forward
// rejection the store rebases once (by shelling out to the git binary)
// and retries the push. Push errors that are not non-fast-forward are
// logged and left local — the commit is durable, the user can retry
// later.
//
// On [New] the store opens (or initialises/clones) the repository.
// When a remote is configured [New] attempts an opportunistic
// fetch+rebase so the first batch sees an up-to-date working tree, then
// checks for local commits ahead of origin and pushes them. Both steps
// log on failure and return successfully — the brain is always usable
// offline.
//
// A pluggable commit signing callback ([GitSignFn]) can be supplied via
// [Options.Sign]. When present it is invoked on every batch commit and
// on the init commit produced when the gitstore bootstraps a brand-new
// repository. It is NOT invoked on tag creation or on [Store.Push]
// (transport auth is out of scope in v1.0).
package git

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"os"
	"os/exec"
	"path/filepath"
	"sync"
	"sync/atomic"
	"time"

	gogit "github.com/go-git/go-git/v5"
	"github.com/go-git/go-git/v5/config"
	"github.com/go-git/go-git/v5/plumbing"
	"github.com/go-git/go-git/v5/plumbing/object"
	"github.com/go-git/go-git/v5/plumbing/storer"

	"github.com/jeffs-brain/memory/go/brain"
)

// Defaults used when [Options] omits a field.
const (
	defaultBranch = "main"
	defaultAuthor = "jeffs-brain"
	defaultEmail  = "noreply@jeffsbrain.com"

	autostashMarker      = "memory-sync-autostash"
	initCommitMessage    = "[init] memory gitstore initialised"
	defaultRemoteName    = "origin"
	defaultCommitLogTag  = "memory: gitstore"
	readmeForEmptyInit   = "# memory\n\nGitstore-initialised brain.\n"
)

// Options controls how a [Store] is created.
type Options struct {
	// Root is the absolute path of the git working tree (the brain
	// root). If the directory does not contain a .git entry, New will
	// either clone from RemoteURL (if set) or git-init in place.
	Root string

	// RemoteURL is the origin URL (e.g.
	// git@github.com:user/memory-brain.git). Empty for local-only
	// repos.
	RemoteURL string

	// Branch is the working branch. Defaults to "main".
	Branch string

	// Author is the commit author name. Defaults to "jeffs-brain".
	Author string

	// Email is the commit author email. Defaults to
	// "noreply@jeffsbrain.com".
	Email string

	// AutoPush enables a synchronous push after each successful commit,
	// rebasing and retrying once on non-fast-forward rejection.
	AutoPush bool

	// Sign is an optional commit signing callback. When non-nil it is
	// invoked on every batch commit and on the init commit. See
	// [GitSignFn] and [GitSignPayload].
	Sign GitSignFn
}

// Store is a git-backed [brain.Store]. Mutations land in the working
// tree and are committed on batch close.
type Store struct {
	tree *workingTree
	repo *gogit.Repository
	opts Options

	// mu serialises commits (fn callback + stage + commit). Released
	// before push so a slow network cannot block concurrent commits.
	// pushMu serialises pushes on top of that.
	mu     sync.Mutex
	pushMu sync.Mutex

	// sinkMu guards the user-facing sink list and the batch event
	// buffer.
	sinkMu       sync.Mutex
	sinks        map[uint64]brain.EventSink
	nextSinkID   uint64
	batchDepth   int
	batchPending []brain.ChangeEvent

	author object.Signature
	closed atomic.Bool
}

// New opens or initialises a git-backed store at opts.Root.
//
// Behaviour:
//
//   - If .git is present, open the existing repository.
//   - If the directory is empty and opts.RemoteURL is set, clone.
//   - Otherwise, git-init in place (with an optional remote config)
//     and produce the init commit.
//
// When a remote is configured, New attempts an opportunistic
// fetch+rebase so the first batch sees an up-to-date working tree, then
// pushes any local commits that are ahead of origin. Both steps log on
// failure and return successfully — the caller is never prevented from
// using the brain offline.
func New(ctx context.Context, opts Options) (*Store, error) {
	if opts.Root == "" {
		return nil, errors.New("gitstore: Root is required")
	}
	if opts.Branch == "" {
		opts.Branch = defaultBranch
	}
	if opts.Author == "" {
		opts.Author = defaultAuthor
	}
	if opts.Email == "" {
		opts.Email = defaultEmail
	}

	abs, err := filepath.Abs(opts.Root)
	if err != nil {
		return nil, fmt.Errorf("gitstore: resolve root: %w", err)
	}
	opts.Root = abs

	tree, err := newWorkingTree(opts.Root)
	if err != nil {
		return nil, err
	}

	repo, created, err := openOrInit(ctx, opts)
	if err != nil {
		return nil, err
	}

	s := &Store{
		tree:  tree,
		repo:  repo,
		opts:  opts,
		sinks: make(map[uint64]brain.EventSink),
		author: object.Signature{
			Name:  opts.Author,
			Email: opts.Email,
		},
	}

	// Bootstrap init commit for a newly initialised repository. The
	// commit is empty so the HEAD exists from the start, letting
	// rollback and ahead/behind detection use a consistent baseline.
	if created {
		if err := s.initCommit(ctx); err != nil {
			return nil, fmt.Errorf("gitstore: init commit: %w", err)
		}
	}

	if opts.RemoteURL != "" {
		if err := s.Freshen(ctx); err != nil {
			slog.Warn("gitstore: freshen on startup failed, continuing offline", "err", err)
		}
		if err := s.recoverPendingPush(ctx); err != nil {
			slog.Warn("gitstore: recover pending push on startup failed", "err", err)
		}
	}

	return s, nil
}

// openOrInit opens an existing git repo at opts.Root or initialises a
// new one. Clones from the remote if the directory is empty and a
// remote URL is provided. Returns created=true only when a brand-new
// repository was initialised (so the caller can stamp an init commit).
func openOrInit(ctx context.Context, opts Options) (repo *gogit.Repository, created bool, err error) {
	gitEntry := filepath.Join(opts.Root, ".git")
	if _, statErr := os.Stat(gitEntry); statErr == nil {
		r, openErr := gogit.PlainOpen(opts.Root)
		return r, false, openErr
	}
	if entries, readErr := os.ReadDir(opts.Root); readErr == nil {
		if dirIsEmpty(entries) && opts.RemoteURL != "" {
			r, cloneErr := gogit.PlainCloneContext(ctx, opts.Root, false, &gogit.CloneOptions{
				URL:           opts.RemoteURL,
				ReferenceName: plumbing.NewBranchReferenceName(opts.Branch),
				SingleBranch:  true,
			})
			return r, false, cloneErr
		}
	}
	if err := os.MkdirAll(opts.Root, 0o755); err != nil {
		return nil, false, fmt.Errorf("gitstore: mkdir root: %w", err)
	}
	r, initErr := gogit.PlainInit(opts.Root, false)
	if initErr != nil {
		return nil, false, fmt.Errorf("gitstore: init repo: %w", initErr)
	}
	if opts.RemoteURL != "" {
		if _, err := r.CreateRemote(&config.RemoteConfig{
			Name: defaultRemoteName,
			URLs: []string{opts.RemoteURL},
		}); err != nil {
			return nil, false, fmt.Errorf("gitstore: add remote: %w", err)
		}
	}
	// Point HEAD at the requested branch. PlainInit defaults to master.
	ref := plumbing.NewSymbolicReference(plumbing.HEAD, plumbing.NewBranchReferenceName(opts.Branch))
	if err := r.Storer.SetReference(ref); err != nil {
		return nil, false, fmt.Errorf("gitstore: set HEAD: %w", err)
	}
	return r, true, nil
}

// dirIsEmpty reports whether a directory contains nothing, ignoring a
// lone .git entry (handled separately).
func dirIsEmpty(entries []os.DirEntry) bool {
	for _, e := range entries {
		if e.Name() == ".git" {
			continue
		}
		return false
	}
	return true
}

// initCommit creates the bootstrap commit on a brand-new repository so
// HEAD exists from the start. The commit writes a minimal README so the
// commit has a tree, which lets signing engines that require a non-
// empty tree still work. When Options.Sign is set the callback is
// invoked for this commit.
func (s *Store) initCommit(ctx context.Context) error {
	readmePath := filepath.Join(s.opts.Root, "README.md")
	if _, err := os.Stat(readmePath); errors.Is(err, os.ErrNotExist) {
		if err := os.WriteFile(readmePath, []byte(readmeForEmptyInit), 0o644); err != nil {
			return fmt.Errorf("gitstore: write init README: %w", err)
		}
	}

	w, err := s.repo.Worktree()
	if err != nil {
		return err
	}
	if _, err := w.Add("README.md"); err != nil {
		return fmt.Errorf("gitstore: stage init README: %w", err)
	}
	opts := &gogit.CommitOptions{
		Author: &object.Signature{
			Name:  s.author.Name,
			Email: s.author.Email,
			When:  time.Now(),
		},
		AllowEmptyCommits: false,
	}
	if s.opts.Sign != nil {
		opts.Signer = signerFromSignFn(s.opts.Sign)
	}
	if _, err := w.Commit(initCommitMessage, opts); err != nil {
		return err
	}
	_ = ctx
	return nil
}

// Close releases resources. After Close, all operations return
// [brain.ErrReadOnly]. Close is idempotent. Any pending push work is
// flushed synchronously.
func (s *Store) Close() error {
	s.mu.Lock()
	s.closed.Store(true)
	s.mu.Unlock()
	// Serialise with any in-flight push so the caller does not return
	// before the network I/O has settled.
	s.pushMu.Lock()
	s.pushMu.Unlock()
	return nil
}

// LocalPath returns the on-disk path for p. The git backend always has
// a working tree, so LocalPath always succeeds for valid paths.
func (s *Store) LocalPath(p brain.Path) (string, bool) {
	return s.tree.localPath(p)
}

// Subscribe registers an event sink. Events are delivered synchronously
// on the write path for standalone mutations and after commit for
// batched mutations. Within a batch, events are buffered; they flush
// once the commit step succeeds, or are discarded if the batch rolls
// back.
func (s *Store) Subscribe(sink brain.EventSink) func() {
	s.sinkMu.Lock()
	id := s.nextSinkID
	s.nextSinkID++
	s.sinks[id] = sink
	s.sinkMu.Unlock()
	return func() {
		s.sinkMu.Lock()
		delete(s.sinks, id)
		s.sinkMu.Unlock()
	}
}

// emit dispatches an event immediately to all sinks. Used for
// standalone mutations. snapshotSinksLocked returns a copy of the sink
// set while sinkMu is held; callbacks run outside the lock.
func (s *Store) emit(evt brain.ChangeEvent) {
	s.sinkMu.Lock()
	if s.batchDepth > 0 {
		s.batchPending = append(s.batchPending, evt)
		s.sinkMu.Unlock()
		return
	}
	sinks := s.snapshotSinksLocked()
	s.sinkMu.Unlock()
	for _, sink := range sinks {
		sink.OnBrainChange(evt)
	}
}

// snapshotSinksLocked must be called with sinkMu held.
func (s *Store) snapshotSinksLocked() []brain.EventSink {
	out := make([]brain.EventSink, 0, len(s.sinks))
	for _, sink := range s.sinks {
		out = append(out, sink)
	}
	return out
}

// beginBatchEvents increments the batch-event counter so any events
// emitted by the batch are buffered rather than delivered.
func (s *Store) beginBatchEvents() {
	s.sinkMu.Lock()
	s.batchDepth++
	s.sinkMu.Unlock()
}

// discardBatchEvents drops the buffered events after a rollback.
func (s *Store) discardBatchEvents() {
	s.sinkMu.Lock()
	s.batchDepth--
	if s.batchDepth == 0 {
		s.batchPending = nil
	}
	s.sinkMu.Unlock()
}

// flushBatchEvents dispatches the buffered events to all sinks.
func (s *Store) flushBatchEvents(reason string) {
	s.sinkMu.Lock()
	s.batchDepth--
	var events []brain.ChangeEvent
	if s.batchDepth == 0 {
		events = s.batchPending
		s.batchPending = nil
	}
	sinks := s.snapshotSinksLocked()
	s.sinkMu.Unlock()
	for _, evt := range events {
		if evt.Reason == "" {
			evt.Reason = reason
		}
		for _, sink := range sinks {
			sink.OnBrainChange(evt)
		}
	}
}

// ---- Read-only operations delegate directly to the working tree ----

// Read implements [brain.Store].
func (s *Store) Read(ctx context.Context, p brain.Path) ([]byte, error) {
	if err := s.checkOpen(); err != nil {
		return nil, err
	}
	return s.tree.read(p)
}

// Exists implements [brain.Store].
func (s *Store) Exists(ctx context.Context, p brain.Path) (bool, error) {
	if err := s.checkOpen(); err != nil {
		return false, err
	}
	return s.tree.exists(p)
}

// Stat implements [brain.Store].
func (s *Store) Stat(ctx context.Context, p brain.Path) (brain.FileInfo, error) {
	if err := s.checkOpen(); err != nil {
		return brain.FileInfo{}, err
	}
	return s.tree.stat(p)
}

// List implements [brain.Store].
func (s *Store) List(ctx context.Context, dir brain.Path, opts brain.ListOpts) ([]brain.FileInfo, error) {
	if err := s.checkOpen(); err != nil {
		return nil, err
	}
	return s.tree.list(dir, opts)
}

// ---- Mutations commit immediately ----

// Write implements [brain.Store]. Each standalone Write commits as its
// own git commit with subject "write <path>". Callers wanting multiple
// writes in one commit should use [Store.Batch].
func (s *Store) Write(ctx context.Context, p brain.Path, content []byte) error {
	return s.Batch(ctx, brain.BatchOptions{Reason: "write", Message: "write " + string(p)}, func(b brain.Batch) error {
		return b.Write(ctx, p, content)
	})
}

// Append implements [brain.Store].
func (s *Store) Append(ctx context.Context, p brain.Path, content []byte) error {
	return s.Batch(ctx, brain.BatchOptions{Reason: "append", Message: "append " + string(p)}, func(b brain.Batch) error {
		return b.Append(ctx, p, content)
	})
}

// Delete implements [brain.Store].
func (s *Store) Delete(ctx context.Context, p brain.Path) error {
	return s.Batch(ctx, brain.BatchOptions{Reason: "delete", Message: "delete " + string(p)}, func(b brain.Batch) error {
		return b.Delete(ctx, p)
	})
}

// Rename implements [brain.Store].
func (s *Store) Rename(ctx context.Context, src, dst brain.Path) error {
	return s.Batch(ctx, brain.BatchOptions{Reason: "rename", Message: "rename " + string(src) + " -> " + string(dst)}, func(b brain.Batch) error {
		return b.Rename(ctx, src, dst)
	})
}

// checkOpen returns [brain.ErrReadOnly] if the store has been closed.
func (s *Store) checkOpen() error {
	if s.closed.Load() {
		return brain.ErrReadOnly
	}
	return nil
}

// Freshen fetches from the remote and rebases the local branch on top
// of it. A no-op when no remote is configured.
//
// go-git's PullContext is fast-forward only, so it cannot honour the
// doc promise of "rebases on divergence". Freshen shells out to the git
// binary: `git fetch origin <branch>` then `git rebase
// origin/<branch>`. A dirty working tree (modified tracked files or
// untracked files) is stashed automatically before the rebase and
// popped afterwards.
//
// Returns [brain.ErrConflict] when the rebase aborts on a merge
// conflict or when a stash-pop after a successful rebase conflicts
// with incoming changes.
func (s *Store) Freshen(ctx context.Context) error {
	if s.opts.RemoteURL == "" {
		return nil
	}
	if _, err := exec.LookPath("git"); err != nil {
		return fmt.Errorf("gitstore: git binary not available for rebase: %w", err)
	}

	if _, err := s.runGit(ctx, "fetch", defaultRemoteName, s.opts.Branch); err != nil {
		return fmt.Errorf("gitstore: fetch: %w", err)
	}

	stashed, err := s.autoStashIfDirty(ctx)
	if err != nil {
		return fmt.Errorf("gitstore: autostash: %w", err)
	}

	if _, rebaseErr := s.runGit(ctx, "rebase", defaultRemoteName+"/"+s.opts.Branch); rebaseErr != nil {
		// Roll back so the user is never left in a rebase-in-progress
		// state they didn't ask for.
		_, _ = s.runGit(ctx, "rebase", "--abort")
		if stashed {
			if _, popErr := s.runGit(ctx, "stash", "pop"); popErr != nil {
				slog.Warn("gitstore: stash pop failed after rebase abort; stashed state preserved in `git stash list`", "err", popErr)
			}
		}
		return fmt.Errorf("%w: git pull conflicted while rebasing on %s/%s: %v",
			brain.ErrConflict, defaultRemoteName, s.opts.Branch, rebaseErr)
	}

	if stashed {
		if _, popErr := s.runGit(ctx, "stash", "pop"); popErr != nil {
			// Rebase succeeded but pop conflicted. Keep the stash
			// around for manual recovery.
			return fmt.Errorf("%w: git pull restored remote changes but local stash pop conflicted (stashed state preserved in `git stash list`): %v",
				brain.ErrConflict, popErr)
		}
	}
	return nil
}

// autoStashIfDirty stashes tracked modifications and untracked files so
// a subsequent rebase does not fail on "You have unstaged changes".
// Ignored files are never stashed. Returns (true, nil) when a stash
// entry was created.
func (s *Store) autoStashIfDirty(ctx context.Context) (bool, error) {
	out, err := s.runGit(ctx, "status", "--porcelain")
	if err != nil {
		return false, fmt.Errorf("status: %w", err)
	}
	if len(bytes.TrimSpace(out)) == 0 {
		return false, nil
	}
	if _, err := s.runGit(ctx, "stash", "push", "--include-untracked", "-m", autostashMarker); err != nil {
		return false, fmt.Errorf("stash push: %w", err)
	}
	return true, nil
}

// runGit runs a git subcommand inside the store root and returns its
// combined stdout. Errors carry the captured stderr so diagnostics are
// readable when surfaced to the operator.
func (s *Store) runGit(ctx context.Context, args ...string) ([]byte, error) {
	full := append([]string{"-C", s.opts.Root}, args...)
	var stdout, stderr bytes.Buffer
	cmd := exec.CommandContext(ctx, "git", full...)
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		return stdout.Bytes(), fmt.Errorf("%w: %s", err, bytes.TrimSpace(stderr.Bytes()))
	}
	return stdout.Bytes(), nil
}

// recoverPendingPush checks whether HEAD is ahead of origin/<branch>
// and attempts a push if so. Failure is logged but non-fatal — the
// commit is durable locally and the next successful push will catch
// up.
func (s *Store) recoverPendingPush(ctx context.Context) error {
	ahead, err := s.hasLocalCommitsAhead()
	if err != nil {
		return err
	}
	if !ahead {
		return nil
	}
	if err := s.push(ctx); err != nil {
		if errors.Is(err, gogit.NoErrAlreadyUpToDate) {
			return nil
		}
		return err
	}
	return nil
}

// hasLocalCommitsAhead reports whether the local branch has commits
// not present on origin/<branch>. Returns (false, nil) when either
// reference is missing — a freshly initialised repo legitimately has
// no origin ref yet.
func (s *Store) hasLocalCommitsAhead() (bool, error) {
	head, err := s.repo.Head()
	if err != nil {
		return false, nil
	}
	remoteRef, err := s.repo.Reference(plumbing.NewRemoteReferenceName(defaultRemoteName, s.opts.Branch), true)
	if err != nil {
		return false, nil
	}
	if head.Hash() == remoteRef.Hash() {
		return false, nil
	}
	iter, err := s.repo.Log(&gogit.LogOptions{From: head.Hash()})
	if err != nil {
		return false, err
	}
	defer iter.Close()
	var ahead bool
	err = iter.ForEach(func(c *object.Commit) error {
		if c.Hash == remoteRef.Hash() {
			return storer.ErrStop
		}
		ahead = true
		return nil
	})
	if err != nil {
		return false, err
	}
	return ahead, nil
}

// signerAdapter adapts a [GitSignFn] to go-git's Signer interface.
type signerAdapter struct {
	fn GitSignFn
}

// Sign implements go-git's Signer. The encoded commit bytes are handed
// to the user-supplied callback and the returned armored signature is
// embedded verbatim in the commit header.
func (a *signerAdapter) Sign(message io.Reader) ([]byte, error) {
	payload, err := io.ReadAll(message)
	if err != nil {
		return nil, err
	}
	sig, err := a.fn(GitSignPayload{Payload: string(payload)})
	if err != nil {
		return nil, err
	}
	return []byte(sig), nil
}

// signerFromSignFn is a small helper so callers do not have to know
// about signerAdapter.
func signerFromSignFn(fn GitSignFn) gogit.Signer {
	return &signerAdapter{fn: fn}
}
