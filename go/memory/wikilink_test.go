// SPDX-License-Identifier: Apache-2.0

package memory

import (
	"context"
	"testing"

	"github.com/jeffs-brain/memory/go/brain"
)

func TestExtractWikilinks_Multiple(t *testing.T) {
	content := "See [[architecture]] and [[deployment]] for details. Also check [[tooling]]."
	links := ExtractWikilinks(content)

	if len(links) != 3 {
		t.Fatalf("expected 3 links, got %d: %v", len(links), links)
	}
	expected := []string{"architecture", "deployment", "tooling"}
	for i, want := range expected {
		if links[i] != want {
			t.Errorf("links[%d] = %q, want %q", i, links[i], want)
		}
	}
}

func TestExtractWikilinks_WithDisplayText(t *testing.T) {
	content := "Refer to [[auth-migration|Auth Migration Notes]] for context."
	links := ExtractWikilinks(content)

	if len(links) != 1 {
		t.Fatalf("expected 1 link, got %d: %v", len(links), links)
	}
	if links[0] != "auth-migration|Auth Migration Notes" {
		t.Errorf("link = %q, want %q", links[0], "auth-migration|Auth Migration Notes")
	}
}

func TestExtractWikilinks_NoLinks(t *testing.T) {
	content := "Plain text with no wikilinks at all."
	links := ExtractWikilinks(content)

	if len(links) != 0 {
		t.Errorf("expected 0 links, got %d: %v", len(links), links)
	}
}

func TestExtractWikilinks_GlobalPrefix(t *testing.T) {
	content := "Check [[global:coding-style]] for shared preferences."
	links := ExtractWikilinks(content)

	if len(links) != 1 {
		t.Fatalf("expected 1 link, got %d: %v", len(links), links)
	}
	if links[0] != "global:coding-style" {
		t.Errorf("link = %q, want %q", links[0], "global:coding-style")
	}
}

func TestResolveWikilink_ProjectBeforeGlobal(t *testing.T) {
	mem, store := newTestMemory(t)
	projectPath := "/example/project"
	slug := ProjectSlug(projectPath)

	writeTopic(t, store, brain.MemoryProjectTopic(slug, "architecture"), "project body")
	writeTopic(t, store, brain.MemoryGlobalTopic("architecture"), "global body")

	resolved := mem.ResolveWikilink(context.Background(), "architecture", projectPath)
	if resolved != brain.MemoryProjectTopic(slug, "architecture") {
		t.Errorf("expected project resolution, got %q", resolved)
	}
}

func TestResolveWikilink_GlobalPrefixBypassesProject(t *testing.T) {
	mem, store := newTestMemory(t)
	projectPath := "/example/project"
	slug := ProjectSlug(projectPath)

	writeTopic(t, store, brain.MemoryProjectTopic(slug, "coding-style"), "project body")
	writeTopic(t, store, brain.MemoryGlobalTopic("coding-style"), "global body")

	resolved := mem.ResolveWikilink(context.Background(), "global:coding-style", projectPath)
	if resolved != brain.MemoryGlobalTopic("coding-style") {
		t.Errorf("expected global resolution, got %q", resolved)
	}
}

func TestResolveWikilink_FallsBackToGlobal(t *testing.T) {
	mem, store := newTestMemory(t)
	projectPath := "/example/project"
	writeTopic(t, store, brain.MemoryGlobalTopic("user-prefs"), "global body")

	resolved := mem.ResolveWikilink(context.Background(), "user-prefs", projectPath)
	if resolved != brain.MemoryGlobalTopic("user-prefs") {
		t.Errorf("expected global fallback, got %q", resolved)
	}
}

func TestResolveWikilink_Missing(t *testing.T) {
	mem, _ := newTestMemory(t)
	resolved := mem.ResolveWikilink(context.Background(), "nowhere", "/example/project")
	if resolved != "" {
		t.Errorf("expected empty path for missing target, got %q", resolved)
	}
}

func TestResolveAllWikilinks_Dedup(t *testing.T) {
	mem, store := newTestMemory(t)
	projectPath := "/example/project"
	slug := ProjectSlug(projectPath)

	writeTopic(t, store, brain.MemoryProjectTopic(slug, "auth"), "body")

	content := "See [[auth]] and later [[auth]] again."
	paths := mem.ResolveAllWikilinks(context.Background(), content, projectPath)

	if len(paths) != 1 {
		t.Fatalf("expected 1 resolved path, got %d: %v", len(paths), paths)
	}
}

func TestResolveAllWikilinks_SkipsMissing(t *testing.T) {
	mem, store := newTestMemory(t)
	projectPath := "/example/project"
	slug := ProjectSlug(projectPath)

	writeTopic(t, store, brain.MemoryProjectTopic(slug, "exists"), "body")

	content := "See [[exists]] and [[missing]]."
	paths := mem.ResolveAllWikilinks(context.Background(), content, projectPath)

	if len(paths) != 1 {
		t.Fatalf("expected 1 resolved path (missing skipped), got %d: %v", len(paths), paths)
	}
}

func TestNormaliseTopic_SpacesToHyphens(t *testing.T) {
	got := normaliseTopic("Some Topic Name")
	if got != "some-topic-name" {
		t.Errorf("got %q, want %q", got, "some-topic-name")
	}
}

func TestNormaliseTopic_AlreadyNormalised(t *testing.T) {
	got := normaliseTopic("already-kebab")
	if got != "already-kebab" {
		t.Errorf("got %q, want %q", got, "already-kebab")
	}
}

func TestNormaliseTopic_TrimsWhitespace(t *testing.T) {
	got := normaliseTopic("  padded  ")
	if got != "padded" {
		t.Errorf("got %q, want %q", got, "padded")
	}
}

func TestNormaliseTopic_Empty(t *testing.T) {
	got := normaliseTopic("")
	if got != "" {
		t.Errorf("got %q, want empty", got)
	}
}
