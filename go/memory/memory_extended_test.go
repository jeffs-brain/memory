// SPDX-License-Identifier: Apache-2.0

package memory

import (
	"context"
	"path/filepath"
	"strings"
	"testing"

	"github.com/jeffs-brain/memory/go/brain"
)

func TestProjectSlug_GitRepo(t *testing.T) {
	restore := SetSlugMapForTest(filepath.Join(t.TempDir(), "slug-map.yaml"))
	t.Cleanup(restore)

	slug := ProjectSlug(".")
	if slug == "" {
		t.Error("slug should not be empty for current directory")
	}
	if strings.Contains(slug, "/") {
		t.Errorf("slug should not contain slashes: %q", slug)
	}
}

func TestProjectSlug_EmptyPath(t *testing.T) {
	restore := SetSlugMapForTest(filepath.Join(t.TempDir(), "slug-map.yaml"))
	t.Cleanup(restore)

	slug := ProjectSlug("")
	if slug == "" {
		t.Error("slug should not be empty even for empty path")
	}
}

func TestListProjectTopics_MultipleTopics(t *testing.T) {
	mem, store := newTestMemory(t)
	projectPath := "/example/project"
	slug := ProjectSlug(projectPath)

	files := map[string]string{
		"architecture": `---
name: Architecture
description: System design decisions
type: project
---

Key decisions here.`,
		"preferences": `---
name: User Preferences
description: Coding style and preferences
type: user
---

Prefers British English.`,
		"debugging-notes": `Just raw notes without frontmatter.`,
	}

	for name, content := range files {
		writeTopic(t, store, brain.MemoryProjectTopic(slug, name), content)
	}

	topics, err := mem.ListProjectTopics(context.Background(), projectPath)
	if err != nil {
		t.Fatal(err)
	}
	if len(topics) != 3 {
		t.Fatalf("expected 3 topics, got %d", len(topics))
	}

	byName := make(map[string]TopicFile)
	for _, topic := range topics {
		byName[topic.Name] = topic
	}

	arch, ok := byName["Architecture"]
	if !ok {
		t.Fatal("missing Architecture topic")
	}
	if arch.Type != "project" {
		t.Errorf("Architecture type = %q, want %q", arch.Type, "project")
	}
	if arch.Description != "System design decisions" {
		t.Errorf("Architecture description = %q", arch.Description)
	}

	prefs, ok := byName["User Preferences"]
	if !ok {
		t.Fatal("missing User Preferences topic")
	}
	if prefs.Type != "user" {
		t.Errorf("User Preferences type = %q, want %q", prefs.Type, "user")
	}

	notes, ok := byName["debugging-notes"]
	if !ok {
		t.Fatal("missing debugging-notes topic")
	}
	if notes.Description != "" {
		t.Errorf("debugging-notes description = %q, want empty", notes.Description)
	}
}

func TestListProjectTopics_SkipsMemoryMDCaseInsensitive(t *testing.T) {
	mem, store := newTestMemory(t)
	projectPath := "/example/project"
	slug := ProjectSlug(projectPath)

	writeTopic(t, store, brain.MemoryProjectIndex(slug), "# Index")
	writeTopic(t, store, brain.MemoryProjectTopic(slug, "topic"), "Content")

	topics, err := mem.ListProjectTopics(context.Background(), projectPath)
	if err != nil {
		t.Fatal(err)
	}
	if len(topics) != 1 {
		t.Fatalf("expected 1 topic (MEMORY.md skipped), got %d", len(topics))
	}
	if topics[0].Name != "topic" {
		t.Errorf("topic name = %q, want %q", topics[0].Name, "topic")
	}
}

func TestListProjectTopics_PathFieldPopulated(t *testing.T) {
	mem, store := newTestMemory(t)
	projectPath := "/example/project"
	slug := ProjectSlug(projectPath)

	writeTopic(t, store, brain.MemoryProjectTopic(slug, "my-topic"), "Content")

	topics, err := mem.ListProjectTopics(context.Background(), projectPath)
	if err != nil {
		t.Fatal(err)
	}
	if len(topics) != 1 {
		t.Fatalf("expected 1 topic, got %d", len(topics))
	}
	if !strings.HasSuffix(string(topics[0].Path), "my-topic.md") {
		t.Errorf("topic Path = %q, expected suffix 'my-topic.md'", topics[0].Path)
	}
}

func TestReadTopic_ExactContent(t *testing.T) {
	mem, store := newTestMemory(t)
	content := "---\nname: Test\n---\n\nBody with special chars: £ € ñ\nLine 2."
	writeTopic(t, store, brain.MemoryGlobalTopic("exact"), content)

	got, err := mem.ReadTopic(context.Background(), brain.MemoryGlobalTopic("exact"))
	if err != nil {
		t.Fatal(err)
	}
	if got != content {
		t.Errorf("ReadTopic content mismatch:\ngot:  %q\nwant: %q", got, content)
	}
}

func TestLoadIndex_Exactly200Lines(t *testing.T) {
	mem, store := newTestMemory(t)
	projectPath := "/example/project"
	slug := ProjectSlug(projectPath)

	var lines []string
	for i := 0; i < 200; i++ {
		lines = append(lines, "line")
	}
	content := strings.Join(lines, "\n")
	writeTopic(t, store, brain.MemoryProjectIndex(slug), content)

	got := mem.LoadProjectIndex(context.Background(), projectPath)
	gotLines := strings.Split(got, "\n")
	if len(gotLines) != 200 {
		t.Errorf("expected 200 lines (no truncation), got %d", len(gotLines))
	}
	if strings.Contains(got, "[...truncated]") {
		t.Error("200 lines should not trigger truncation")
	}
}

func TestLoadIndex_201Lines(t *testing.T) {
	mem, store := newTestMemory(t)
	projectPath := "/example/project"
	slug := ProjectSlug(projectPath)

	var lines []string
	for i := 0; i < 201; i++ {
		lines = append(lines, "line")
	}
	content := strings.Join(lines, "\n")
	writeTopic(t, store, brain.MemoryProjectIndex(slug), content)

	got := mem.LoadProjectIndex(context.Background(), projectPath)
	if !strings.Contains(got, "[...truncated]") {
		t.Error("201 lines should trigger truncation")
	}
}

func TestLoadIndex_WhitespaceOnly(t *testing.T) {
	mem, store := newTestMemory(t)
	projectPath := "/example/project"
	slug := ProjectSlug(projectPath)

	writeTopic(t, store, brain.MemoryProjectIndex(slug), "   \n  \n  ")

	got := mem.LoadProjectIndex(context.Background(), projectPath)
	if got != "" {
		t.Errorf("expected empty string for whitespace-only file, got %q", got)
	}
}

func TestParseFrontmatter_ExtraFieldsIgnored(t *testing.T) {
	content := `---
name: Test
description: A test topic
type: project
author: Alex
version: 2
---

Body here.`

	fm, body := ParseFrontmatter(content)
	if fm.Name != "Test" {
		t.Errorf("name = %q, want %q", fm.Name, "Test")
	}
	if fm.Description != "A test topic" {
		t.Errorf("description = %q, want %q", fm.Description, "A test topic")
	}
	if fm.Type != "project" {
		t.Errorf("type = %q, want %q", fm.Type, "project")
	}
	if body != "Body here." {
		t.Errorf("body = %q, want %q", body, "Body here.")
	}
}

func TestParseFrontmatter_PartialFields(t *testing.T) {
	content := `---
name: Partial
---

Only name is set.`

	fm, body := ParseFrontmatter(content)
	if fm.Name != "Partial" {
		t.Errorf("name = %q, want %q", fm.Name, "Partial")
	}
	if fm.Description != "" {
		t.Errorf("description = %q, want empty", fm.Description)
	}
	if fm.Type != "" {
		t.Errorf("type = %q, want empty", fm.Type)
	}
	if body != "Only name is set." {
		t.Errorf("body = %q", body)
	}
}

func TestParseFrontmatter_MultilineBody(t *testing.T) {
	content := `---
name: Multi
---

Line 1.
Line 2.
Line 3.`

	_, body := ParseFrontmatter(content)
	lines := strings.Split(body, "\n")
	if len(lines) != 3 {
		t.Errorf("body lines = %d, want 3", len(lines))
	}
}

func TestParseFrontmatter_ExtendedFields(t *testing.T) {
	content := `---
name: Go Testing Patterns
description: Learned patterns for testing
type: feedback
created: 2026-04-04T15:00:00Z
modified: 2026-04-04T16:00:00Z
confidence: high
source: reflection
tags:
  - heuristic
  - testing
  - go
---

Body content.`

	fm, body := ParseFrontmatter(content)
	if fm.Name != "Go Testing Patterns" {
		t.Errorf("name = %q", fm.Name)
	}
	if fm.Created != "2026-04-04T15:00:00Z" {
		t.Errorf("created = %q", fm.Created)
	}
	if fm.Modified != "2026-04-04T16:00:00Z" {
		t.Errorf("modified = %q", fm.Modified)
	}
	if fm.Confidence != "high" {
		t.Errorf("confidence = %q", fm.Confidence)
	}
	if fm.Source != "reflection" {
		t.Errorf("source = %q", fm.Source)
	}
	if len(fm.Tags) != 3 {
		t.Fatalf("expected 3 tags, got %d: %v", len(fm.Tags), fm.Tags)
	}
	if fm.Tags[0] != "heuristic" || fm.Tags[1] != "testing" || fm.Tags[2] != "go" {
		t.Errorf("tags = %v", fm.Tags)
	}
	if body != "Body content." {
		t.Errorf("body = %q", body)
	}
}

func TestParseFrontmatter_InlineCommaTags(t *testing.T) {
	content := `---
name: Test
tags: foo, bar, baz
---

Body.`

	fm, _ := ParseFrontmatter(content)
	if len(fm.Tags) != 3 {
		t.Fatalf("expected 3 tags, got %d: %v", len(fm.Tags), fm.Tags)
	}
	if fm.Tags[0] != "foo" || fm.Tags[1] != "bar" || fm.Tags[2] != "baz" {
		t.Errorf("tags = %v", fm.Tags)
	}
}

func TestParseFrontmatter_BackwardsCompatible(t *testing.T) {
	content := `---
name: Old Format
description: Only three fields
type: project
---

Old body.`

	fm, body := ParseFrontmatter(content)
	if fm.Name != "Old Format" {
		t.Errorf("name = %q", fm.Name)
	}
	if fm.Description != "Only three fields" {
		t.Errorf("description = %q", fm.Description)
	}
	if fm.Type != "project" {
		t.Errorf("type = %q", fm.Type)
	}
	if fm.Created != "" || fm.Modified != "" || fm.Confidence != "" || fm.Source != "" {
		t.Error("extended fields should be empty for old-format files")
	}
	if len(fm.Tags) != 0 {
		t.Errorf("tags should be empty, got %v", fm.Tags)
	}
	if body != "Old body." {
		t.Errorf("body = %q", body)
	}
}

func TestBuildMemoryPrompt_IncludesLogicalLabel(t *testing.T) {
	mem, store := newTestMemory(t)
	projectPath := "/example/project"
	slug := ProjectSlug(projectPath)

	writeTopic(t, store, brain.MemoryProjectIndex(slug), "Some content")

	got := mem.BuildMemoryPromptFor(projectPath)
	wantLogical := "memory/project/" + slug
	if !strings.Contains(got, wantLogical) {
		t.Errorf("BuildMemoryPromptFor should include logical label %q, got:\n%s", wantLogical, got)
	}
}

func TestBuildMemoryPrompt_EmptyFile(t *testing.T) {
	mem, store := newTestMemory(t)
	projectPath := "/example/project"
	slug := ProjectSlug(projectPath)

	writeTopic(t, store, brain.MemoryProjectIndex(slug), "")

	got := mem.BuildMemoryPromptFor(projectPath)
	if got != "" {
		t.Errorf("expected empty prompt for empty index file, got %q", got)
	}
}

func TestParseKV_ValueWithColon(t *testing.T) {
	key, val, ok := parseKV("url: https://example.com:8080")
	if !ok {
		t.Fatal("expected ok")
	}
	if key != "url" {
		t.Errorf("key = %q, want %q", key, "url")
	}
	if val != "https://example.com:8080" {
		t.Errorf("val = %q, want %q", val, "https://example.com:8080")
	}
}

func TestParseKV_WhitespaceHandling(t *testing.T) {
	key, val, ok := parseKV("  name  :  some value  ")
	if !ok {
		t.Fatal("expected ok")
	}
	if key != "name" {
		t.Errorf("key = %q, want %q", key, "name")
	}
	if val != "some value" {
		t.Errorf("val = %q, want %q", val, "some value")
	}
}

func TestMemoryInstructions_ContainsKeyContent(t *testing.T) {
	if !strings.Contains(MemoryInstructions, "Persistent Memory") {
		t.Error("MemoryInstructions missing 'Persistent Memory' header")
	}
	if !strings.Contains(MemoryInstructions, "MEMORY.md") {
		t.Error("MemoryInstructions missing reference to MEMORY.md")
	}
	if !strings.Contains(MemoryInstructions, "Topic files") {
		t.Error("MemoryInstructions missing reference to topic files")
	}
	if !strings.Contains(MemoryInstructions, "frontmatter") {
		t.Error("MemoryInstructions missing reference to frontmatter")
	}
	if !strings.Contains(MemoryInstructions, "Global memory") {
		t.Error("MemoryInstructions missing reference to global memory scope")
	}
	if !strings.Contains(MemoryInstructions, "Project memory") {
		t.Error("MemoryInstructions missing reference to project memory scope")
	}
	if !strings.Contains(MemoryInstructions, "two-tier") {
		t.Error("MemoryInstructions missing reference to two-tier system")
	}
}
