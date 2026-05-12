// SPDX-License-Identifier: Apache-2.0

package pt

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/jeffs-brain/memory/go/brain"
)

// TestPassthroughLayout is the load-bearing assertion for this
// package: a write to logical `memory/global/foo.md` must land at
// on-disk `memory/global/foo.md`, NOT `memory/foo.md`. The HTTP daemon
// and the TS/Py SDKs depend on this 1:1 mapping; see the eval/lme
// runner's BrainCache handling for context.
func TestPassthroughLayout(t *testing.T) {
	t.Parallel()

	root := t.TempDir()
	store, err := New(root)
	if err != nil {
		t.Fatalf("pt.New: %v", err)
	}
	t.Cleanup(func() { _ = store.Close() })

	ctx := context.Background()
	logical := brain.Path("memory/global/hedgehogs.md")
	body := []byte("---\ntitle: Hedgehogs\n---\n\nThey like gardens.\n")
	if err := store.Write(ctx, logical, body); err != nil {
		t.Fatalf("Write: %v", err)
	}

	onDisk := filepath.Join(root, "memory", "global", "hedgehogs.md")
	if _, err := os.Stat(onDisk); err != nil {
		t.Fatalf("expected file at %s: %v", onDisk, err)
	}
	// Negative: fs.Store would remap to memory/hedgehogs.md. Confirm
	// that did NOT happen.
	remapped := filepath.Join(root, "memory", "hedgehogs.md")
	if _, err := os.Stat(remapped); err == nil {
		t.Fatalf("unexpected file at remapped path %s; passthrough must not remap", remapped)
	}

	got, err := store.Read(ctx, logical)
	if err != nil {
		t.Fatalf("Read: %v", err)
	}
	if string(got) != string(body) {
		t.Fatalf("Read got %q, want %q", got, body)
	}
}

// TestBatchCommitsAtLogicalPath verifies batched writes also use the
// 1:1 layout.
func TestBatchCommitsAtLogicalPath(t *testing.T) {
	t.Parallel()

	root := t.TempDir()
	store, err := New(root)
	if err != nil {
		t.Fatalf("pt.New: %v", err)
	}
	t.Cleanup(func() { _ = store.Close() })

	ctx := context.Background()
	paths := []brain.Path{
		"memory/global/a.md",
		"memory/project/eval-lme/b.md",
		"raw/lme/session.md",
		"wiki/note.md",
	}
	err = store.Batch(ctx, brain.BatchOptions{Reason: "test"}, func(b brain.Batch) error {
		for _, p := range paths {
			if err := b.Write(ctx, p, []byte("body")); err != nil {
				return err
			}
		}
		return nil
	})
	if err != nil {
		t.Fatalf("Batch: %v", err)
	}
	for _, p := range paths {
		full := filepath.Join(root, filepath.FromSlash(string(p)))
		if _, err := os.Stat(full); err != nil {
			t.Fatalf("expected %s: %v", full, err)
		}
	}
}

// TestListRecursiveReturnsLogicalPaths confirms List returns paths in
// the logical form callers expect.
func TestListRecursiveReturnsLogicalPaths(t *testing.T) {
	t.Parallel()

	root := t.TempDir()
	store, err := New(root)
	if err != nil {
		t.Fatalf("pt.New: %v", err)
	}
	t.Cleanup(func() { _ = store.Close() })

	ctx := context.Background()
	if err := store.Write(ctx, "memory/global/a.md", []byte("x")); err != nil {
		t.Fatal(err)
	}
	if err := store.Write(ctx, "memory/global/deep/b.md", []byte("y")); err != nil {
		t.Fatal(err)
	}
	entries, err := store.List(ctx, brain.Path("memory/global"), brain.ListOpts{Recursive: true})
	if err != nil {
		t.Fatalf("List: %v", err)
	}
	got := map[string]bool{}
	for _, e := range entries {
		if e.IsDir {
			continue
		}
		got[string(e.Path)] = true
	}
	want := []string{"memory/global/a.md", "memory/global/deep/b.md"}
	for _, w := range want {
		if !got[w] {
			t.Errorf("missing %s in list: %v", w, got)
		}
	}
}

// TestBatchListOverlaysJournalState verifies that Batch.List merges
// journal state (pending writes, deletes, renames) with the base store.
func TestBatchListOverlaysJournalState(t *testing.T) {
	t.Parallel()

	type storeSetup struct {
		path    brain.Path
		content []byte
	}
	type batchAction struct {
		op      string // "write", "delete", "rename", "append"
		path    brain.Path
		dst     brain.Path // rename destination
		content []byte
	}

	tests := []struct {
		name      string
		setup     []storeSetup
		actions   []batchAction
		listDir   brain.Path
		listOpts  brain.ListOpts
		wantPaths []string
	}{
		{
			name:  "write in batch is visible before commit",
			setup: nil,
			actions: []batchAction{
				{op: "write", path: "docs/new.md", content: []byte("hello")},
			},
			listDir:   "docs",
			listOpts:  brain.ListOpts{Recursive: true},
			wantPaths: []string{"docs/new.md"},
		},
		{
			name:  "multiple writes all visible",
			setup: nil,
			actions: []batchAction{
				{op: "write", path: "docs/a.md", content: []byte("a")},
				{op: "write", path: "docs/b.md", content: []byte("b")},
				{op: "write", path: "docs/c.md", content: []byte("c")},
			},
			listDir:   "docs",
			listOpts:  brain.ListOpts{Recursive: true},
			wantPaths: []string{"docs/a.md", "docs/b.md", "docs/c.md"},
		},
		{
			name: "prefix filtering works",
			actions: []batchAction{
				{op: "write", path: "docs/a.md", content: []byte("a")},
				{op: "write", path: "other/b.md", content: []byte("b")},
				{op: "write", path: "docs/sub/c.md", content: []byte("c")},
			},
			listDir:   "docs",
			listOpts:  brain.ListOpts{Recursive: true},
			wantPaths: []string{"docs/a.md", "docs/sub/c.md"},
		},
		{
			name: "delete in batch hides path from list",
			setup: []storeSetup{
				{path: "docs/existing.md", content: []byte("old")},
			},
			actions: []batchAction{
				{op: "delete", path: "docs/existing.md"},
			},
			listDir:   "docs",
			listOpts:  brain.ListOpts{Recursive: true},
			wantPaths: []string{},
		},
		{
			name: "delete base store path not shown",
			setup: []storeSetup{
				{path: "notes/a.md", content: []byte("a")},
				{path: "notes/b.md", content: []byte("b")},
			},
			actions: []batchAction{
				{op: "delete", path: "notes/a.md"},
			},
			listDir:   "notes",
			listOpts:  brain.ListOpts{Recursive: true},
			wantPaths: []string{"notes/b.md"},
		},
		{
			name: "write then delete same path not in list",
			actions: []batchAction{
				{op: "write", path: "docs/temp.md", content: []byte("temp")},
				{op: "delete", path: "docs/temp.md"},
			},
			listDir:   "docs",
			listOpts:  brain.ListOpts{Recursive: true},
			wantPaths: []string{},
		},
		{
			name: "empty prefix lists everything",
			setup: []storeSetup{
				{path: "a/x.md", content: []byte("x")},
			},
			actions: []batchAction{
				{op: "write", path: "b/y.md", content: []byte("y")},
			},
			listDir:   "",
			listOpts:  brain.ListOpts{Recursive: true},
			wantPaths: []string{"a/x.md", "b/y.md"},
		},
		{
			name:      "prefix with no matches returns empty slice",
			setup:     nil,
			actions:   []batchAction{},
			listDir:   "nonexistent",
			listOpts:  brain.ListOpts{Recursive: true},
			wantPaths: []string{},
		},
		{
			name: "rename moves path in listing",
			setup: []storeSetup{
				{path: "docs/old.md", content: []byte("content")},
			},
			actions: []batchAction{
				{op: "rename", path: "docs/old.md", dst: "docs/new.md"},
			},
			listDir:   "docs",
			listOpts:  brain.ListOpts{Recursive: true},
			wantPaths: []string{"docs/new.md"},
		},
		{
			name: "non-recursive list only shows immediate children",
			setup: []storeSetup{
				{path: "docs/top.md", content: []byte("top")},
				{path: "docs/sub/deep.md", content: []byte("deep")},
			},
			actions: []batchAction{
				{op: "write", path: "docs/added.md", content: []byte("added")},
				{op: "write", path: "docs/nested/other.md", content: []byte("nested")},
			},
			listDir:   "docs",
			listOpts:  brain.ListOpts{Recursive: false},
			wantPaths: []string{"docs/added.md", "docs/top.md"},
		},
		{
			name: "batch write overlays base store entry",
			setup: []storeSetup{
				{path: "docs/file.md", content: []byte("original")},
			},
			actions: []batchAction{
				{op: "write", path: "docs/file.md", content: []byte("updated")},
			},
			listDir:   "docs",
			listOpts:  brain.ListOpts{Recursive: true},
			wantPaths: []string{"docs/file.md"},
		},
		{
			name: "append in batch visible in list",
			setup: []storeSetup{
				{path: "docs/file.md", content: []byte("start")},
			},
			actions: []batchAction{
				{op: "append", path: "docs/file.md", content: []byte(" end")},
			},
			listDir:   "docs",
			listOpts:  brain.ListOpts{Recursive: true},
			wantPaths: []string{"docs/file.md"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			root := t.TempDir()
			store, err := New(root)
			if err != nil {
				t.Fatalf("pt.New: %v", err)
			}
			t.Cleanup(func() { _ = store.Close() })

			ctx := context.Background()

			// Pre-populate the store with setup data.
			for _, s := range tt.setup {
				if err := store.Write(ctx, s.path, s.content); err != nil {
					t.Fatalf("setup Write(%s): %v", s.path, err)
				}
			}

			// Execute the batch and validate List inside the callback.
			err = store.Batch(ctx, brain.BatchOptions{Reason: "test"}, func(b brain.Batch) error {
				for _, a := range tt.actions {
					switch a.op {
					case "write":
						if werr := b.Write(ctx, a.path, a.content); werr != nil {
							return werr
						}
					case "delete":
						if derr := b.Delete(ctx, a.path); derr != nil {
							return derr
						}
					case "rename":
						if rerr := b.Rename(ctx, a.path, a.dst); rerr != nil {
							return rerr
						}
					case "append":
						if aerr := b.Append(ctx, a.path, a.content); aerr != nil {
							return aerr
						}
					}
				}

				entries, lerr := b.List(ctx, tt.listDir, tt.listOpts)
				if lerr != nil {
					return lerr
				}

				gotPaths := make([]string, 0, len(entries))
				for _, e := range entries {
					if e.IsDir {
						continue
					}
					gotPaths = append(gotPaths, string(e.Path))
				}

				if len(gotPaths) != len(tt.wantPaths) {
					t.Errorf("List returned %d paths, want %d\ngot:  %v\nwant: %v",
						len(gotPaths), len(tt.wantPaths), gotPaths, tt.wantPaths)
					return nil
				}

				wantSet := make(map[string]struct{}, len(tt.wantPaths))
				for _, w := range tt.wantPaths {
					wantSet[w] = struct{}{}
				}
				for _, g := range gotPaths {
					if _, ok := wantSet[g]; !ok {
						t.Errorf("unexpected path in list: %s\ngot:  %v\nwant: %v",
							g, gotPaths, tt.wantPaths)
					}
				}
				return nil
			})
			if err != nil {
				t.Fatalf("Batch: %v", err)
			}
		})
	}
}

// TestBatchListReturnsNonNilSlice verifies that an empty result is a
// non-nil empty slice rather than nil.
func TestBatchListReturnsNonNilSlice(t *testing.T) {
	t.Parallel()

	root := t.TempDir()
	store, err := New(root)
	if err != nil {
		t.Fatalf("pt.New: %v", err)
	}
	t.Cleanup(func() { _ = store.Close() })

	ctx := context.Background()
	err = store.Batch(ctx, brain.BatchOptions{Reason: "test"}, func(b brain.Batch) error {
		entries, lerr := b.List(ctx, "nonexistent", brain.ListOpts{Recursive: true})
		if lerr != nil {
			return lerr
		}
		if entries == nil {
			t.Errorf("expected non-nil empty slice, got nil")
		}
		return nil
	})
	if err != nil {
		t.Fatalf("Batch: %v", err)
	}
}
