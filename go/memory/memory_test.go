// SPDX-License-Identifier: Apache-2.0

package memory

import (
	"context"
	"path/filepath"
	"strings"
	"testing"

	"github.com/jeffs-brain/memory/go/brain"
)

func TestBrainRoot_EnvOverride(t *testing.T) {
	t.Setenv("JEFFS_BRAIN_ROOT", "/tmp/test-brain")
	got := brainRoot()
	if got != "/tmp/test-brain" {
		t.Fatalf("brainRoot() = %q, want %q", got, "/tmp/test-brain")
	}
}

func TestBrainRoot_DefaultFallback(t *testing.T) {
	t.Setenv("JEFFS_BRAIN_ROOT", "")
	got := brainRoot()
	if got == "" || got == "." {
		t.Skipf("brainRoot() = %q, skipping (no home dir?)", got)
	}
	if !strings.HasSuffix(got, ".config/jeffs-brain") {
		t.Fatalf("brainRoot() = %q, want suffix .config/jeffs-brain", got)
	}
}

// --- ProjectSlug tests ---

func TestProjectSlug_AbsolutePath(t *testing.T) {
	restore := SetSlugMapForTest(filepath.Join(t.TempDir(), "slug-map.yaml"))
	t.Cleanup(restore)

	tmp := t.TempDir()
	slug := ProjectSlug(tmp)

	if strings.Contains(slug, "/") {
		t.Errorf("slug contains /: %s", slug)
	}
	if strings.HasPrefix(slug, "-") {
		t.Errorf("slug starts with -: %s", slug)
	}
	if slug == "" {
		t.Error("slug is empty")
	}
}

func TestProjectSlug_Deterministic(t *testing.T) {
	restore := SetSlugMapForTest(filepath.Join(t.TempDir(), "slug-map.yaml"))
	t.Cleanup(restore)

	tmp := t.TempDir()
	a := ProjectSlug(tmp)
	b := ProjectSlug(tmp)
	if a != b {
		t.Errorf("non-deterministic: %q != %q", a, b)
	}
}

func TestProjectSlug_DifferentPaths(t *testing.T) {
	restore := SetSlugMapForTest(filepath.Join(t.TempDir(), "slug-map.yaml"))
	t.Cleanup(restore)

	a := t.TempDir()
	b := t.TempDir()
	if ProjectSlug(a) == ProjectSlug(b) {
		t.Error("different paths produced same slug")
	}
}

// --- ParseFrontmatter tests ---

func TestParseFrontmatter_Complete(t *testing.T) {
	content := `---
name: Architecture
description: System architecture notes
type: project
---

Some body content here.`

	fm, body := ParseFrontmatter(content)

	if fm.Name != "Architecture" {
		t.Errorf("name = %q, want %q", fm.Name, "Architecture")
	}
	if fm.Description != "System architecture notes" {
		t.Errorf("description = %q, want %q", fm.Description, "System architecture notes")
	}
	if fm.Type != "project" {
		t.Errorf("type = %q, want %q", fm.Type, "project")
	}
	if body != "Some body content here." {
		t.Errorf("body = %q, want %q", body, "Some body content here.")
	}
}

func TestParseFrontmatter_QuotedValues(t *testing.T) {
	content := `---
name: "My Topic"
description: 'A description'
type: user
---

Body.`

	fm, _ := ParseFrontmatter(content)

	if fm.Name != "My Topic" {
		t.Errorf("name = %q, want %q", fm.Name, "My Topic")
	}
	if fm.Description != "A description" {
		t.Errorf("description = %q, want %q", fm.Description, "A description")
	}
	if fm.Type != "user" {
		t.Errorf("type = %q, want %q", fm.Type, "user")
	}
}

func TestParseFrontmatter_NoFrontmatter(t *testing.T) {
	content := "Just some plain text."
	fm, body := ParseFrontmatter(content)

	if fm.Name != "" || fm.Description != "" || fm.Type != "" {
		t.Errorf("expected empty fields, got name=%q desc=%q type=%q", fm.Name, fm.Description, fm.Type)
	}
	if body != content {
		t.Errorf("body = %q, want %q", body, content)
	}
}

func TestParseFrontmatter_UnclosedBlock(t *testing.T) {
	content := `---
name: Broken
type: project
No closing delimiter here.`

	fm, body := ParseFrontmatter(content)
	if fm.Name != "" {
		t.Errorf("expected empty name for unclosed block, got %q", fm.Name)
	}
	if body != content {
		t.Errorf("body should be full content for unclosed block")
	}
}

func TestParseFrontmatter_EmptyBody(t *testing.T) {
	content := `---
name: Empty
type: reference
---`

	fm, body := ParseFrontmatter(content)
	if fm.Name != "Empty" {
		t.Errorf("name = %q, want %q", fm.Name, "Empty")
	}
	if fm.Type != "reference" {
		t.Errorf("type = %q, want %q", fm.Type, "reference")
	}
	if body != "" {
		t.Errorf("body = %q, want empty", body)
	}
}

// --- LoadProjectIndex tests ---

func TestLoadProjectIndex_NoFile(t *testing.T) {
	mem, _ := newTestMemory(t)
	result := mem.LoadProjectIndex(context.Background(), "/some/project")
	if result != "" {
		t.Errorf("expected empty string for missing MEMORY.md, got %q", result)
	}
}

func TestLoadProjectIndex_ReadsContent(t *testing.T) {
	mem, store := newTestMemory(t)
	projectPath := "/example/project"
	slug := ProjectSlug(projectPath)

	content := "# My Memory\n\nSome important notes."
	writeTopic(t, store, brain.MemoryProjectIndex(slug), content)

	got := mem.LoadProjectIndex(context.Background(), projectPath)
	if got != content {
		t.Errorf("LoadProjectIndex = %q, want %q", got, content)
	}
}

func TestLoadProjectIndex_CapsAt200Lines(t *testing.T) {
	mem, store := newTestMemory(t)
	projectPath := "/example/project"
	slug := ProjectSlug(projectPath)

	var lines []string
	for i := 0; i < 250; i++ {
		lines = append(lines, "line")
	}
	content := strings.Join(lines, "\n")
	writeTopic(t, store, brain.MemoryProjectIndex(slug), content)

	got := mem.LoadProjectIndex(context.Background(), projectPath)
	gotLines := strings.Split(got, "\n")
	if len(gotLines) != 201 {
		t.Errorf("expected 201 lines (200 + truncation), got %d", len(gotLines))
	}
	if !strings.Contains(got, "[...truncated]") {
		t.Error("expected truncation notice")
	}
}

// --- ListProjectTopics tests ---

func TestListProjectTopics_EmptyScope(t *testing.T) {
	mem, _ := newTestMemory(t)
	topics, err := mem.ListProjectTopics(context.Background(), "/no/such/project")
	if err != nil {
		t.Fatal(err)
	}
	if len(topics) != 0 {
		t.Errorf("expected no topics, got %d", len(topics))
	}
}

func TestListProjectTopics_SkipsMemoryMD(t *testing.T) {
	mem, store := newTestMemory(t)
	projectPath := "/example/project"
	slug := ProjectSlug(projectPath)

	writeTopic(t, store, brain.MemoryProjectIndex(slug), "# Index")
	writeTopic(t, store, brain.MemoryProjectTopic(slug, "architecture"), `---
name: Architecture
description: System design
type: project
---

Content here.`)

	topics, err := mem.ListProjectTopics(context.Background(), projectPath)
	if err != nil {
		t.Fatal(err)
	}
	if len(topics) != 1 {
		t.Fatalf("expected 1 topic, got %d", len(topics))
	}
	if topics[0].Name != "Architecture" {
		t.Errorf("topic name = %q, want %q", topics[0].Name, "Architecture")
	}
	if topics[0].Description != "System design" {
		t.Errorf("topic description = %q, want %q", topics[0].Description, "System design")
	}
	if topics[0].Type != "project" {
		t.Errorf("topic type = %q, want %q", topics[0].Type, "project")
	}
}

func TestListProjectTopics_FallbackToFilename(t *testing.T) {
	mem, store := newTestMemory(t)
	projectPath := "/example/project"
	slug := ProjectSlug(projectPath)

	writeTopic(t, store, brain.MemoryProjectTopic(slug, "notes"), "Just plain text.")

	topics, err := mem.ListProjectTopics(context.Background(), projectPath)
	if err != nil {
		t.Fatal(err)
	}
	if len(topics) != 1 {
		t.Fatalf("expected 1 topic, got %d", len(topics))
	}
	if topics[0].Name != "notes" {
		t.Errorf("topic name = %q, want %q", topics[0].Name, "notes")
	}
}

// --- ReadTopic tests ---

func TestReadTopic_Success(t *testing.T) {
	mem, store := newTestMemory(t)
	expected := "# Test\n\nContent."
	writeTopic(t, store, brain.MemoryGlobalTopic("test"), expected)

	got, err := mem.ReadTopic(context.Background(), brain.MemoryGlobalTopic("test"))
	if err != nil {
		t.Fatal(err)
	}
	if got != expected {
		t.Errorf("ReadTopic = %q, want %q", got, expected)
	}
}

func TestReadTopic_NotFound(t *testing.T) {
	mem, _ := newTestMemory(t)
	_, err := mem.ReadTopic(context.Background(), brain.MemoryGlobalTopic("nonexistent"))
	if err == nil {
		t.Error("expected error for missing file")
	}
}

// --- BuildMemoryPromptFor tests ---

func TestBuildMemoryPrompt_NoIndex(t *testing.T) {
	mem, _ := newTestMemory(t)
	got := mem.BuildMemoryPromptFor("/no/such/project")
	if got != "" {
		t.Errorf("expected empty string for missing index, got %q", got)
	}
}

func TestBuildMemoryPrompt_WithProjectIndex(t *testing.T) {
	mem, store := newTestMemory(t)
	projectPath := "/example/project"
	slug := ProjectSlug(projectPath)

	content := "# Memory Index\n\n- Architecture decisions\n- User preferences"
	writeTopic(t, store, brain.MemoryProjectIndex(slug), content)

	got := mem.BuildMemoryPromptFor(projectPath)
	if !strings.Contains(got, "# Project Memory") {
		t.Error("missing Project Memory header")
	}
	if !strings.Contains(got, content) {
		t.Error("missing index content")
	}
	if !strings.Contains(got, "Project memory directory:") {
		t.Error("missing project memory directory path")
	}
}

func TestBuildMemoryPrompt_GlobalOnly(t *testing.T) {
	mem, store := newTestMemory(t)
	writeTopic(t, store, brain.MemoryGlobalIndex(), "- User prefers British English")

	got := mem.BuildMemoryPromptFor("/example/project")
	if !strings.Contains(got, "# Global Memory") {
		t.Error("missing Global Memory header")
	}
	if !strings.Contains(got, "British English") {
		t.Error("missing global index content")
	}
	if !strings.Contains(got, "Global memory directory:") {
		t.Error("missing global memory directory path")
	}
	if strings.Contains(got, "# Project Memory") {
		t.Error("should not contain Project Memory header when no project index")
	}
}

func TestBuildMemoryPrompt_ProjectOnly(t *testing.T) {
	mem, store := newTestMemory(t)
	projectPath := "/example/project"
	slug := ProjectSlug(projectPath)
	writeTopic(t, store, brain.MemoryProjectIndex(slug), "- Architecture notes")

	got := mem.BuildMemoryPromptFor(projectPath)
	if !strings.Contains(got, "# Project Memory") {
		t.Error("missing Project Memory header")
	}
	if strings.Contains(got, "# Global Memory") {
		t.Error("should not contain Global Memory header when no global index")
	}
}

func TestBuildMemoryPrompt_BothScopes(t *testing.T) {
	mem, store := newTestMemory(t)
	projectPath := "/example/project"
	slug := ProjectSlug(projectPath)

	writeTopic(t, store, brain.MemoryGlobalIndex(), "- User prefers British English")
	writeTopic(t, store, brain.MemoryProjectIndex(slug), "- Architecture notes")

	got := mem.BuildMemoryPromptFor(projectPath)
	if !strings.Contains(got, "# Global Memory") {
		t.Error("missing Global Memory header")
	}
	if !strings.Contains(got, "# Project Memory") {
		t.Error("missing Project Memory header")
	}
	if !strings.Contains(got, "British English") {
		t.Error("missing global index content")
	}
	if !strings.Contains(got, "Architecture notes") {
		t.Error("missing project index content")
	}

	globalPos := strings.Index(got, "# Global Memory")
	projectPos := strings.Index(got, "# Project Memory")
	if globalPos >= projectPos {
		t.Error("Global Memory should appear before Project Memory")
	}
}

// --- parseKV tests ---

func TestParseKV_Simple(t *testing.T) {
	key, val, ok := parseKV("name: test")
	if !ok {
		t.Fatal("expected ok")
	}
	if key != "name" || val != "test" {
		t.Errorf("got key=%q val=%q", key, val)
	}
}

func TestParseKV_NoColon(t *testing.T) {
	_, _, ok := parseKV("no colon here")
	if ok {
		t.Error("expected not ok for line without colon")
	}
}

func TestParseKV_DoubleQuoted(t *testing.T) {
	_, val, ok := parseKV(`name: "quoted value"`)
	if !ok {
		t.Fatal("expected ok")
	}
	if val != "quoted value" {
		t.Errorf("val = %q, want %q", val, "quoted value")
	}
}

func TestParseKV_SingleQuoted(t *testing.T) {
	_, val, ok := parseKV("name: 'single quoted'")
	if !ok {
		t.Fatal("expected ok")
	}
	if val != "single quoted" {
		t.Errorf("val = %q, want %q", val, "single quoted")
	}
}

func TestParseKV_EmptyValue(t *testing.T) {
	key, val, ok := parseKV("name:")
	if !ok {
		t.Fatal("expected ok")
	}
	if key != "name" || val != "" {
		t.Errorf("got key=%q val=%q", key, val)
	}
}
