// SPDX-License-Identifier: Apache-2.0

package git_test

import (
	"bytes"
	"context"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/jeffs-brain/memory/go/brain"
	gitstore "github.com/jeffs-brain/memory/go/store/git"
)

// requireGit skips a test if the git binary is not on PATH. Tests that
// manipulate a bare remote or call [Store.Freshen] need the real binary
// because go-git cannot rebase.
func requireGit(t *testing.T) {
	t.Helper()
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git binary not available on PATH")
	}
}

// initBareRemote creates a bare git remote in a fresh temp dir and
// returns its absolute path.
func initBareRemote(t *testing.T) string {
	t.Helper()
	dir := filepath.Join(t.TempDir(), "remote.git")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	runGit(t, dir, "init", "--bare", "-b", "main")
	return dir
}

// seedRemote makes an initial empty commit on the bare remote so the
// "main" branch ref exists. Without this, PlainClone cannot resolve
// refs/heads/main.
func seedRemote(t *testing.T, remote string) {
	t.Helper()
	seed := filepath.Join(t.TempDir(), "seed")
	if err := os.MkdirAll(seed, 0o755); err != nil {
		t.Fatalf("mkdir seed: %v", err)
	}
	runGit(t, seed, "init", "-b", "main")
	runGit(t, seed, "config", "user.email", "seed@example.com")
	runGit(t, seed, "config", "user.name", "seed")
	runGit(t, seed, "commit", "--allow-empty", "-m", "seed")
	runGit(t, seed, "remote", "add", "origin", remote)
	runGit(t, seed, "push", "origin", "main")
}

func runGit(t *testing.T, dir string, args ...string) string {
	t.Helper()
	cmd := exec.Command("git", args...)
	cmd.Dir = dir
	var out bytes.Buffer
	cmd.Stdout = &out
	cmd.Stderr = &out
	if err := cmd.Run(); err != nil {
		t.Fatalf("git %v in %s: %v\n%s", args, dir, err, out.String())
	}
	return out.String()
}

// newLocalStore spins up a local-only gitstore for contract-style
// tests. Uses t.TempDir so cleanup is automatic.
func newLocalStore(t *testing.T) *gitstore.Store {
	t.Helper()
	ctx := context.Background()
	store, err := gitstore.New(ctx, gitstore.Options{
		Root:   t.TempDir(),
		Branch: "main",
	})
	if err != nil {
		t.Fatalf("gitstore.New: %v", err)
	}
	t.Cleanup(func() { _ = store.Close() })
	return store
}

// --- Contract-style tests ---

func TestGitStore_WriteReadRoundTrip(t *testing.T) {
	requireGit(t)
	ctx := context.Background()
	store := newLocalStore(t)

	p := brain.Path("memory/global/user-role.md")
	if err := store.Write(ctx, p, []byte("Alex is a lead engineer")); err != nil {
		t.Fatalf("Write: %v", err)
	}
	got, err := store.Read(ctx, p)
	if err != nil {
		t.Fatalf("Read: %v", err)
	}
	if string(got) != "Alex is a lead engineer" {
		t.Fatalf("Read = %q, want %q", got, "Alex is a lead engineer")
	}
}

func TestGitStore_ReadMissingIsErrNotFound(t *testing.T) {
	requireGit(t)
	ctx := context.Background()
	store := newLocalStore(t)

	_, err := store.Read(ctx, "memory/global/missing.md")
	if !errors.Is(err, brain.ErrNotFound) {
		t.Fatalf("Read missing: want ErrNotFound, got %v", err)
	}
}

func TestGitStore_WriteRejectsInvalidPath(t *testing.T) {
	requireGit(t)
	ctx := context.Background()
	store := newLocalStore(t)

	err := store.Write(ctx, "../etc/passwd", []byte("nope"))
	if !errors.Is(err, brain.ErrInvalidPath) {
		t.Fatalf("Write invalid: want ErrInvalidPath, got %v", err)
	}
}

func TestGitStore_AppendCreatesThenExtends(t *testing.T) {
	requireGit(t)
	ctx := context.Background()
	store := newLocalStore(t)

	p := brain.Path("wiki/_log.md")
	if err := store.Append(ctx, p, []byte("line one\n")); err != nil {
		t.Fatalf("Append (create): %v", err)
	}
	if err := store.Append(ctx, p, []byte("line two\n")); err != nil {
		t.Fatalf("Append (extend): %v", err)
	}
	got, err := store.Read(ctx, p)
	if err != nil {
		t.Fatalf("Read: %v", err)
	}
	if string(got) != "line one\nline two\n" {
		t.Fatalf("Read = %q", got)
	}
}

func TestGitStore_DeleteMissingIsErrNotFound(t *testing.T) {
	requireGit(t)
	ctx := context.Background()
	store := newLocalStore(t)

	err := store.Delete(ctx, "memory/global/nope.md")
	if !errors.Is(err, brain.ErrNotFound) {
		t.Fatalf("Delete missing: want ErrNotFound, got %v", err)
	}
}

func TestGitStore_RenameMovesContent(t *testing.T) {
	requireGit(t)
	ctx := context.Background()
	store := newLocalStore(t)

	src := brain.Path("raw/web/old.md")
	dst := brain.Path("raw/.sources/web/old.md")
	if err := store.Write(ctx, src, []byte("content")); err != nil {
		t.Fatalf("Write src: %v", err)
	}
	if err := store.Rename(ctx, src, dst); err != nil {
		t.Fatalf("Rename: %v", err)
	}
	exists, _ := store.Exists(ctx, src)
	if exists {
		t.Error("src still exists after rename")
	}
	got, err := store.Read(ctx, dst)
	if err != nil {
		t.Fatalf("Read dst: %v", err)
	}
	if string(got) != "content" {
		t.Fatalf("dst = %q", got)
	}
}

func TestGitStore_ListSorted(t *testing.T) {
	requireGit(t)
	ctx := context.Background()
	store := newLocalStore(t)

	for _, p := range []brain.Path{
		"memory/global/charlie.md",
		"memory/global/alpha.md",
		"memory/global/bravo.md",
	} {
		if err := store.Write(ctx, p, []byte(string(p))); err != nil {
			t.Fatalf("Write %s: %v", p, err)
		}
	}
	entries, err := store.List(ctx, "memory/global", brain.ListOpts{})
	if err != nil {
		t.Fatalf("List: %v", err)
	}
	want := []brain.Path{
		"memory/global/alpha.md",
		"memory/global/bravo.md",
		"memory/global/charlie.md",
	}
	if len(entries) != len(want) {
		t.Fatalf("List = %d entries, want %d", len(entries), len(want))
	}
	for i := range want {
		if entries[i].Path != want[i] {
			t.Errorf("entries[%d] = %q, want %q", i, entries[i].Path, want[i])
		}
	}
}

func TestGitStore_ListHidesGenerated(t *testing.T) {
	requireGit(t)
	ctx := context.Background()
	store := newLocalStore(t)

	if err := store.Write(ctx, "wiki/_index.md", []byte("master")); err != nil {
		t.Fatalf("Write generated: %v", err)
	}
	if err := store.Write(ctx, "wiki/article.md", []byte("content")); err != nil {
		t.Fatalf("Write article: %v", err)
	}
	entries, err := store.List(ctx, "wiki", brain.ListOpts{})
	if err != nil {
		t.Fatalf("List: %v", err)
	}
	for _, e := range entries {
		if e.Path == "wiki/_index.md" {
			t.Fatal("default listing returned generated file")
		}
	}
}

func TestGitStore_BatchCommitsOneOp(t *testing.T) {
	requireGit(t)
	ctx := context.Background()
	store := newLocalStore(t)

	err := store.Batch(ctx, brain.BatchOptions{Reason: "test"}, func(b brain.Batch) error {
		if err := b.Write(ctx, "memory/global/a.md", []byte("one")); err != nil {
			return err
		}
		return b.Write(ctx, "memory/global/b.md", []byte("two"))
	})
	if err != nil {
		t.Fatalf("Batch: %v", err)
	}
	got, _ := store.Read(ctx, "memory/global/a.md")
	if string(got) != "one" {
		t.Errorf("a.md = %q", got)
	}
	got, _ = store.Read(ctx, "memory/global/b.md")
	if string(got) != "two" {
		t.Errorf("b.md = %q", got)
	}
}

func TestGitStore_BatchRollsBackOnError(t *testing.T) {
	requireGit(t)
	ctx := context.Background()
	store := newLocalStore(t)

	if err := store.Write(ctx, "memory/global/existing.md", []byte("original")); err != nil {
		t.Fatalf("seed: %v", err)
	}

	sentinel := errors.New("abort")
	err := store.Batch(ctx, brain.BatchOptions{Reason: "test"}, func(b brain.Batch) error {
		if err := b.Write(ctx, "memory/global/new.md", []byte("should not survive")); err != nil {
			return err
		}
		if err := b.Write(ctx, "memory/global/existing.md", []byte("should not overwrite")); err != nil {
			return err
		}
		return sentinel
	})
	if !errors.Is(err, sentinel) {
		t.Fatalf("want sentinel, got %v", err)
	}
	exists, _ := store.Exists(ctx, "memory/global/new.md")
	if exists {
		t.Error("new file survived rolled-back batch")
	}
	got, _ := store.Read(ctx, "memory/global/existing.md")
	if string(got) != "original" {
		t.Errorf("existing.md = %q, want 'original'", got)
	}
}

func TestGitStore_BatchEventsFireAfterCommit(t *testing.T) {
	requireGit(t)
	ctx := context.Background()
	store := newLocalStore(t)

	var duringCallback int
	var finalCount int
	unsub := store.Subscribe(brain.EventSinkFunc(func(evt brain.ChangeEvent) {
		finalCount++
	}))
	defer unsub()

	err := store.Batch(ctx, brain.BatchOptions{Reason: "test"}, func(b brain.Batch) error {
		if err := b.Write(ctx, "memory/global/e1.md", []byte("one")); err != nil {
			return err
		}
		if err := b.Write(ctx, "memory/global/e2.md", []byte("two")); err != nil {
			return err
		}
		duringCallback = finalCount
		return nil
	})
	if err != nil {
		t.Fatalf("Batch: %v", err)
	}
	if duringCallback != 0 {
		t.Errorf("events fired during callback: %d", duringCallback)
	}
	if finalCount < 2 {
		t.Errorf("finalCount = %d, want >= 2", finalCount)
	}
}

func TestGitStore_LocalPathAlwaysExists(t *testing.T) {
	requireGit(t)
	store := newLocalStore(t)
	p := brain.Path("memory/global/user.md")
	abs, ok := store.LocalPath(p)
	if !ok {
		t.Fatal("LocalPath(ok=false) for git-backed store")
	}
	if !filepath.IsAbs(abs) {
		t.Errorf("LocalPath non-absolute: %q", abs)
	}
}

func TestGitStore_CloseMakesStoreUnusable(t *testing.T) {
	requireGit(t)
	ctx := context.Background()
	store := newLocalStore(t)

	if err := store.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	err := store.Write(ctx, "memory/global/after.md", []byte("nope"))
	if !errors.Is(err, brain.ErrReadOnly) {
		t.Fatalf("Write after close: want ErrReadOnly, got %v", err)
	}
}

// --- Remote-backed integration tests ported from jeff ---

func TestGitStore_FreshenAppliesRemoteChanges(t *testing.T) {
	requireGit(t)
	remote := initBareRemote(t)
	seedRemote(t, remote)

	ctx := context.Background()
	localRoot := t.TempDir()
	runGit(t, localRoot, "clone", remote, ".")
	runGit(t, localRoot, "config", "user.email", "local@example.com")
	runGit(t, localRoot, "config", "user.name", "local")

	store, err := gitstore.New(ctx, gitstore.Options{
		Root:      localRoot,
		RemoteURL: remote,
		Branch:    "main",
	})
	if err != nil {
		t.Fatalf("gitstore.New: %v", err)
	}
	defer func() { _ = store.Close() }()

	otherRoot := t.TempDir()
	runGit(t, otherRoot, "clone", remote, ".")
	runGit(t, otherRoot, "config", "user.email", "other@example.com")
	runGit(t, otherRoot, "config", "user.name", "other")
	wikiDir := filepath.Join(otherRoot, "wiki")
	if err := os.MkdirAll(wikiDir, 0o755); err != nil {
		t.Fatalf("mkdir wiki: %v", err)
	}
	if err := os.WriteFile(filepath.Join(wikiDir, "from-remote.md"), []byte("remote content"), 0o644); err != nil {
		t.Fatalf("write: %v", err)
	}
	runGit(t, otherRoot, "add", "wiki/from-remote.md")
	runGit(t, otherRoot, "commit", "-m", "from other")
	runGit(t, otherRoot, "push", "origin", "main")

	if err := store.Freshen(ctx); err != nil {
		t.Fatalf("Freshen: %v", err)
	}
	got, err := store.Read(ctx, "wiki/from-remote.md")
	if err != nil {
		t.Fatalf("Read after Freshen: %v", err)
	}
	if string(got) != "remote content" {
		t.Errorf("content after Freshen = %q", got)
	}
}

// TestGitStore_FreshenSurvivesDirtyWorkingTree covers the real-world
// case where background tools leave the working tree dirty at startup.
// Freshen must auto-stash both the tracked modification and the
// untracked file, rebase cleanly, and pop the stash so the dirty state
// is restored afterwards.
func TestGitStore_FreshenSurvivesDirtyWorkingTree(t *testing.T) {
	requireGit(t)
	remote := initBareRemote(t)
	seedRemote(t, remote)

	ctx := context.Background()
	localRoot := t.TempDir()
	runGit(t, localRoot, "clone", remote, ".")
	runGit(t, localRoot, "config", "user.email", "local@example.com")
	runGit(t, localRoot, "config", "user.name", "local")

	trackedDir := filepath.Join(localRoot, "wiki", ".obsidian")
	if err := os.MkdirAll(trackedDir, 0o755); err != nil {
		t.Fatalf("mkdir obsidian: %v", err)
	}
	trackedFile := filepath.Join(trackedDir, "graph.json")
	if err := os.WriteFile(trackedFile, []byte(`{"version":1}`), 0o644); err != nil {
		t.Fatalf("write tracked: %v", err)
	}
	runGit(t, localRoot, "add", "wiki/.obsidian/graph.json")
	runGit(t, localRoot, "commit", "-m", "seed tracked")
	runGit(t, localRoot, "push", "origin", "main")

	store, err := gitstore.New(ctx, gitstore.Options{
		Root:      localRoot,
		RemoteURL: remote,
		Branch:    "main",
	})
	if err != nil {
		t.Fatalf("gitstore.New: %v", err)
	}
	defer func() { _ = store.Close() }()

	if err := os.WriteFile(trackedFile, []byte(`{"version":2}`), 0o644); err != nil {
		t.Fatalf("dirty tracked: %v", err)
	}
	untrackedDir := filepath.Join(localRoot, "linear")
	if err := os.MkdirAll(untrackedDir, 0o700); err != nil {
		t.Fatalf("mkdir untracked: %v", err)
	}
	untrackedFile := filepath.Join(untrackedDir, "personal.json")
	if err := os.WriteFile(untrackedFile, []byte(`{"api_key":"xxx"}`), 0o600); err != nil {
		t.Fatalf("write untracked: %v", err)
	}

	otherRoot := t.TempDir()
	runGit(t, otherRoot, "clone", remote, ".")
	runGit(t, otherRoot, "config", "user.email", "other@example.com")
	runGit(t, otherRoot, "config", "user.name", "other")
	otherFile := filepath.Join(otherRoot, "wiki", "from-remote.md")
	if err := os.MkdirAll(filepath.Dir(otherFile), 0o755); err != nil {
		t.Fatalf("mkdir other wiki: %v", err)
	}
	if err := os.WriteFile(otherFile, []byte("remote"), 0o644); err != nil {
		t.Fatalf("write other: %v", err)
	}
	runGit(t, otherRoot, "add", "wiki/from-remote.md")
	runGit(t, otherRoot, "commit", "-m", "from other")
	runGit(t, otherRoot, "push", "origin", "main")

	if err := store.Freshen(ctx); err != nil {
		t.Fatalf("Freshen with dirty tree: %v", err)
	}
	if _, err := os.Stat(filepath.Join(localRoot, "wiki", "from-remote.md")); err != nil {
		t.Errorf("remote change missing after Freshen: %v", err)
	}
	got, err := os.ReadFile(trackedFile)
	if err != nil {
		t.Fatalf("read tracked after Freshen: %v", err)
	}
	if string(got) != `{"version":2}` {
		t.Errorf("tracked modification not restored: got %q", got)
	}
	if _, err := os.Stat(untrackedFile); err != nil {
		t.Errorf("untracked file missing after Freshen: %v", err)
	}
}

// TestGitStore_BatchIdempotentWriteWithUntrackedWorkingTree covers the
// case where a batch writes content that is byte-identical to HEAD
// while unrelated untracked files are present in the working tree.
// The batch should be treated as a no-op rather than failing.
func TestGitStore_BatchIdempotentWriteWithUntrackedWorkingTree(t *testing.T) {
	requireGit(t)
	ctx := context.Background()
	root := t.TempDir()

	store, err := gitstore.New(ctx, gitstore.Options{
		Root:   root,
		Branch: "main",
	})
	if err != nil {
		t.Fatalf("gitstore.New: %v", err)
	}
	defer func() { _ = store.Close() }()

	content := []byte("dimensions-v1")
	if err := store.Write(ctx, "raw/dims.json", content); err != nil {
		t.Fatalf("initial write: %v", err)
	}

	untracked := filepath.Join(root, "linear", "personal.json")
	if err := os.MkdirAll(filepath.Dir(untracked), 0o700); err != nil {
		t.Fatalf("mkdir untracked: %v", err)
	}
	if err := os.WriteFile(untracked, []byte(`{"api_key":"xxx"}`), 0o600); err != nil {
		t.Fatalf("write untracked: %v", err)
	}

	err = store.Batch(ctx, brain.BatchOptions{Reason: "test"}, func(b brain.Batch) error {
		return b.Write(ctx, "raw/dims.json", content)
	})
	if err != nil {
		t.Fatalf("idempotent batch: %v", err)
	}

	// HEAD should still only have the init commit and the original
	// write — exactly two commits, no spurious third one.
	head := runGit(t, root, "rev-list", "--count", "HEAD")
	if strings.TrimSpace(head) != "2" {
		t.Errorf("commit count after no-op batch = %q, want 2 (init + initial write)", strings.TrimSpace(head))
	}
	if _, err := os.Stat(untracked); err != nil {
		t.Errorf("untracked file disappeared: %v", err)
	}
}

func TestGitStore_PushRetryOnNonFastForward(t *testing.T) {
	requireGit(t)
	remote := initBareRemote(t)
	seedRemote(t, remote)

	ctx := context.Background()

	rootA := t.TempDir()
	runGit(t, rootA, "clone", remote, ".")
	runGit(t, rootA, "config", "user.email", "a@example.com")
	runGit(t, rootA, "config", "user.name", "a")

	storeA, err := gitstore.New(ctx, gitstore.Options{
		Root:      rootA,
		RemoteURL: remote,
		Branch:    "main",
		AutoPush:  true,
		Author:    "A",
		Email:     "a@example.com",
	})
	if err != nil {
		t.Fatalf("gitstore.New A: %v", err)
	}
	defer func() { _ = storeA.Close() }()

	rootB := t.TempDir()
	runGit(t, rootB, "clone", remote, ".")
	runGit(t, rootB, "config", "user.email", "b@example.com")
	runGit(t, rootB, "config", "user.name", "b")
	bFile := filepath.Join(rootB, "wiki", "b.md")
	if err := os.MkdirAll(filepath.Dir(bFile), 0o755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	if err := os.WriteFile(bFile, []byte("b wrote"), 0o644); err != nil {
		t.Fatalf("write b: %v", err)
	}
	runGit(t, rootB, "add", "wiki/b.md")
	runGit(t, rootB, "commit", "-m", "from B")
	runGit(t, rootB, "push", "origin", "main")

	if err := storeA.Write(ctx, "wiki/a.md", []byte("a wrote")); err != nil {
		t.Fatalf("Write A: %v", err)
	}

	check := t.TempDir()
	runGit(t, check, "clone", remote, ".")
	if _, err := os.Stat(filepath.Join(check, "wiki", "a.md")); err != nil {
		t.Errorf("remote missing a.md: %v", err)
	}
	if _, err := os.Stat(filepath.Join(check, "wiki", "b.md")); err != nil {
		t.Errorf("remote missing b.md: %v", err)
	}
}

func TestGitStore_EmptyRepoBatchRollback(t *testing.T) {
	requireGit(t)
	ctx := context.Background()
	root := t.TempDir()
	store, err := gitstore.New(ctx, gitstore.Options{
		Root:   root,
		Branch: "main",
	})
	if err != nil {
		t.Fatalf("gitstore.New: %v", err)
	}
	defer func() { _ = store.Close() }()

	sentinel := errors.New("abort")
	err = store.Batch(ctx, brain.BatchOptions{Reason: "test"}, func(b brain.Batch) error {
		if err := b.Write(ctx, "wiki/first.md", []byte("no")); err != nil {
			return err
		}
		return sentinel
	})
	if !errors.Is(err, sentinel) {
		t.Fatalf("want sentinel, got %v", err)
	}
	if _, err := os.Stat(filepath.Join(root, "wiki", "first.md")); !os.IsNotExist(err) {
		t.Errorf("file survived rollback: err=%v", err)
	}
}

func TestGitStore_RecoverPendingPushOnStartup(t *testing.T) {
	requireGit(t)
	remote := initBareRemote(t)
	seedRemote(t, remote)

	ctx := context.Background()
	root := t.TempDir()
	runGit(t, root, "clone", remote, ".")
	runGit(t, root, "config", "user.email", "p@example.com")
	runGit(t, root, "config", "user.name", "p")

	storeOffline, err := gitstore.New(ctx, gitstore.Options{
		Root:      root,
		RemoteURL: remote,
		Branch:    "main",
	})
	if err != nil {
		t.Fatalf("gitstore.New offline: %v", err)
	}
	if err := storeOffline.Write(ctx, "wiki/pending.md", []byte("offline work")); err != nil {
		t.Fatalf("Write: %v", err)
	}
	if err := storeOffline.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	check := t.TempDir()
	runGit(t, check, "clone", remote, ".")
	if _, err := os.Stat(filepath.Join(check, "wiki", "pending.md")); err == nil {
		t.Fatal("remote already has pending.md before recovery")
	}

	storeOnline, err := gitstore.New(ctx, gitstore.Options{
		Root:      root,
		RemoteURL: remote,
		Branch:    "main",
	})
	if err != nil {
		t.Fatalf("gitstore.New online: %v", err)
	}
	defer func() { _ = storeOnline.Close() }()

	check2 := t.TempDir()
	runGit(t, check2, "clone", remote, ".")
	if _, err := os.Stat(filepath.Join(check2, "wiki", "pending.md")); err != nil {
		t.Errorf("remote missing pending.md after recovery: %v", err)
	}
}

// --- Signing callback tests ---

func TestGitStore_SignCallbackInvokedOnBatchCommit(t *testing.T) {
	requireGit(t)
	ctx := context.Background()
	root := t.TempDir()

	var invocations int
	var lastPayload string
	sign := func(p gitstore.GitSignPayload) (string, error) {
		invocations++
		lastPayload = p.Payload
		return "-----BEGIN PGP SIGNATURE-----\ntest\n-----END PGP SIGNATURE-----", nil
	}
	store, err := gitstore.New(ctx, gitstore.Options{
		Root:   root,
		Branch: "main",
		Sign:   sign,
	})
	if err != nil {
		t.Fatalf("gitstore.New: %v", err)
	}
	defer func() { _ = store.Close() }()

	// New() already ran the init commit — that should have invoked the
	// signer once.
	if invocations < 1 {
		t.Fatalf("sign not invoked on init commit: invocations=%d", invocations)
	}
	initInvocations := invocations

	if err := store.Write(ctx, "memory/global/a.md", []byte("x")); err != nil {
		t.Fatalf("Write: %v", err)
	}
	if invocations != initInvocations+1 {
		t.Errorf("sign invocations after batch commit = %d, want %d",
			invocations, initInvocations+1)
	}
	if lastPayload == "" {
		t.Error("last payload empty; sign should receive encoded commit bytes")
	}
}

func TestGitStore_SignCallbackNotInvokedOnPush(t *testing.T) {
	requireGit(t)
	remote := initBareRemote(t)
	seedRemote(t, remote)

	ctx := context.Background()
	root := t.TempDir()
	runGit(t, root, "clone", remote, ".")
	runGit(t, root, "config", "user.email", "s@example.com")
	runGit(t, root, "config", "user.name", "s")

	var invocations int
	sign := func(p gitstore.GitSignPayload) (string, error) {
		invocations++
		return "-----BEGIN PGP SIGNATURE-----\ntest\n-----END PGP SIGNATURE-----", nil
	}
	store, err := gitstore.New(ctx, gitstore.Options{
		Root:      root,
		RemoteURL: remote,
		Branch:    "main",
		AutoPush:  true,
		Sign:      sign,
	})
	if err != nil {
		t.Fatalf("gitstore.New: %v", err)
	}
	defer func() { _ = store.Close() }()

	before := invocations
	if err := store.Write(ctx, "memory/global/pushed.md", []byte("x")); err != nil {
		t.Fatalf("Write: %v", err)
	}
	// Exactly one additional sign call for the batch commit; the push
	// itself must not trigger the callback.
	if invocations != before+1 {
		t.Errorf("sign invocations across write+push = %d, want %d", invocations, before+1)
	}
}
