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
