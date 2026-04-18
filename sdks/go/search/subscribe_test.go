// SPDX-License-Identifier: Apache-2.0

package search

import (
	"context"
	"testing"

	"github.com/jeffs-brain/memory/go/brain"

	_ "modernc.org/sqlite"
)

func TestClassifyPath(t *testing.T) {
	cases := []struct {
		in        brain.Path
		wantScope string
		wantSlug  string
		wantOK    bool
	}{
		{"memory/global/go.md", "global_memory", "", true},
		{"memory/global/deep/topic.md", "global_memory", "", true},
		{"memory/project/jeff/api.md", "project_memory", "jeff", true},
		{"memory/project/jeff/deep/notes.md", "project_memory", "jeff", true},
		{"wiki/topic/article.md", "wiki", "", true},
		{"wiki/_index.md", "wiki", "", true},
		{"raw/documents/hedgehogs.md", "raw_document", "", true},
		{"raw/documents/deep/nested.md", "raw_document", "", true},
		{"raw/.sources/web/foo.md", "sources", "", true},
		{"raw/web/foo.md", "", "", false},
		{"memory/global/not-md.txt", "", "", false},
		{"memory/project/", "", "", false},
	}
	for _, tc := range cases {
		gotScope, gotSlug, gotOK := classifyPath(tc.in)
		if gotScope != tc.wantScope || gotSlug != tc.wantSlug || gotOK != tc.wantOK {
			t.Errorf("classifyPath(%q) = (%q, %q, %v), want (%q, %q, %v)",
				tc.in, gotScope, gotSlug, gotOK, tc.wantScope, tc.wantSlug, tc.wantOK)
		}
	}
}

// TestSubscribe_IndexesBrainMutations drives a test store through
// Write/Delete/Rename and verifies the FTS index mirrors the
// changes via the event subscription.
func TestSubscribe_IndexesBrainMutations(t *testing.T) {
	store := newTestStore()
	t.Cleanup(func() { _ = store.Close() })

	db := openTestDB(t)
	idx, err := NewIndex(db, store)
	if err != nil {
		t.Fatalf("NewIndex: %v", err)
	}

	unsub := idx.Subscribe(store)
	t.Cleanup(unsub)

	ctx := context.Background()
	if err := store.Write(ctx, "wiki/go/channels.md", []byte(`---
title: Channels
summary: Channel fundamentals
---
Channels are goroutine-safe pipes.`)); err != nil {
		t.Fatalf("write: %v", err)
	}

	results, err := idx.Search("Channels", SearchOpts{})
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(results) == 0 {
		t.Fatal("expected at least one result after brain write")
	}
	if results[0].Title != "Channels" {
		t.Errorf("title = %q, want %q", results[0].Title, "Channels")
	}

	if err := store.Rename(ctx, "wiki/go/channels.md", "wiki/go/chans.md"); err != nil {
		t.Fatalf("rename: %v", err)
	}
	results, err = idx.Search("Channels", SearchOpts{})
	if err != nil {
		t.Fatalf("Search after rename: %v", err)
	}
	if len(results) == 0 {
		t.Fatal("expected result at new path after rename")
	}
	if results[0].Path != "wiki/go/chans.md" {
		t.Errorf("path = %q, want %q", results[0].Path, "wiki/go/chans.md")
	}

	if err := store.Delete(ctx, "wiki/go/chans.md"); err != nil {
		t.Fatalf("delete: %v", err)
	}
	results, err = idx.Search("Channels", SearchOpts{})
	if err != nil {
		t.Fatalf("Search after delete: %v", err)
	}
	if len(results) != 0 {
		t.Errorf("expected zero results after delete, got %d", len(results))
	}
}

// TestRebuildIfEmpty_SeedsFromStore exercises the startup-time
// rebuild path that populates a fresh FTS index from an
// already-populated store.
func TestRebuildIfEmpty_SeedsFromStore(t *testing.T) {
	store := newTestStore()
	t.Cleanup(func() { _ = store.Close() })

	ctx := context.Background()
	if err := store.Write(ctx, "memory/global/go.md", []byte(`---
name: Go Patterns
description: Common Go idioms
---
Go favours composition over inheritance.`)); err != nil {
		t.Fatalf("write: %v", err)
	}
	if err := store.Write(ctx, "wiki/platform/auth.md", []byte(`---
title: Auth
summary: OAuth flow summary
---
OAuth flows in the platform start at /login.`)); err != nil {
		t.Fatalf("write: %v", err)
	}

	db := openTestDB(t)
	idx, err := NewIndex(db, store)
	if err != nil {
		t.Fatalf("NewIndex: %v", err)
	}
	if err := idx.RebuildIfEmpty(ctx, store); err != nil {
		t.Fatalf("RebuildIfEmpty: %v", err)
	}

	results, err := idx.Search("OAuth", SearchOpts{})
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(results) == 0 {
		t.Fatal("expected a wiki result after RebuildIfEmpty")
	}

	if err := idx.RebuildIfEmpty(ctx, store); err != nil {
		t.Fatalf("RebuildIfEmpty second call: %v", err)
	}
}
