// SPDX-License-Identifier: Apache-2.0

package memory

import (
	"context"
	"strings"
	"testing"

	"github.com/jeffs-brain/memory/go/brain"
)

func TestHeuristicFilename_FromCategory(t *testing.T) {
	h := Heuristic{
		Rule:     "Always run Go tests before committing",
		Category: "testing",
	}

	got := heuristicFilename(h)
	want := "heuristic-testing-always-run.md"
	if got != want {
		t.Errorf("heuristicFilename = %q, want %q", got, want)
	}
}

func TestHeuristicFilename_AntiPattern(t *testing.T) {
	h := Heuristic{
		Rule:        "Using SQL string concatenation for queries",
		Category:    "debugging",
		AntiPattern: true,
	}

	got := heuristicFilename(h)
	if !strings.HasPrefix(got, "heuristic-anti-") {
		t.Errorf("anti-pattern filename should start with 'heuristic-anti-', got %q", got)
	}
	if !strings.HasSuffix(got, ".md") {
		t.Errorf("filename should end with .md, got %q", got)
	}
}

func TestHeuristicFilename_LongRule(t *testing.T) {
	h := Heuristic{
		Rule:     "Always ensure that your database migrations are thoroughly tested against production-like data volumes before deploying to staging environments",
		Category: "architecture",
	}

	got := heuristicFilename(h)
	parts := strings.Split(strings.TrimSuffix(got, ".md"), "-")
	if len(parts) > 4 {
		t.Errorf("filename has too many parts (%d): %q", len(parts), got)
	}
}

func TestBuildHeuristicContent_Standard(t *testing.T) {
	h := Heuristic{
		Rule:       "Run tests in parallel where possible",
		Context:    "Go test suites with independent cases",
		Confidence: "medium",
		Category:   "testing",
	}

	content := buildHeuristicContent(h)

	if !strings.HasPrefix(content, "---\n") {
		t.Error("content should start with frontmatter delimiter")
	}
	if !strings.Contains(content, "type: feedback") {
		t.Error("content should have type: feedback")
	}
	if !strings.Contains(content, "confidence: medium") {
		t.Error("content should have confidence: medium")
	}
	if !strings.Contains(content, "source: reflection") {
		t.Error("content should have source: reflection")
	}
	if !strings.Contains(content, "## Run tests in parallel where") {
		t.Error("content should contain a ## heading from the rule")
	}
	if !strings.Contains(content, "**Context:** Go test suites") {
		t.Error("content should contain the context line")
	}
	if !strings.Contains(content, "**Why:** Observed during reflection") {
		t.Error("content should contain the why line")
	}
}

func TestBuildHeuristicContent_AntiPattern(t *testing.T) {
	h := Heuristic{
		Rule:        "Never use string concatenation for SQL queries, use parameterised queries instead",
		Context:     "Database access layer",
		Confidence:  "high",
		Category:    "debugging",
		AntiPattern: true,
	}

	content := buildHeuristicContent(h)

	if !strings.Contains(content, "## Anti-pattern:") {
		t.Error("anti-pattern should have 'Anti-pattern:' heading")
	}
	if !strings.Contains(content, "**Don't:**") {
		t.Error("anti-pattern should have 'Don't:' line")
	}
	if !strings.Contains(content, "anti-pattern") {
		t.Error("anti-pattern should appear in tags")
	}
}

func TestBuildHeuristicContent_HasCorrectTags(t *testing.T) {
	h := Heuristic{
		Rule:       "Check error returns in Go",
		Confidence: "low",
		Category:   "approach",
	}

	content := buildHeuristicContent(h)

	if !strings.Contains(content, "  - heuristic\n") {
		t.Error("content should have 'heuristic' tag")
	}
	if !strings.Contains(content, "  - approach\n") {
		t.Error("content should have category tag")
	}
}

func TestFindExistingHeuristic_MatchesByNameAndCategory(t *testing.T) {
	mem, store := newTestMemory(t)
	projectPath := "/example/project"
	slug := ProjectSlug(projectPath)
	prefix := brain.MemoryProjectPrefix(slug)

	existingFilename := "heuristic-testing-always-run.md"
	existing := `---
name: "Testing: always run tests"
description: "Always run tests"
type: feedback
confidence: low
source: reflection
tags:
  - heuristic
  - testing
---

## Always run tests

Run tests before committing.`
	writeTopic(t, store, brain.MemoryProjectTopic(slug, strings.TrimSuffix(existingFilename, ".md")), existing)

	candidate := Heuristic{
		Rule:     "Always run tests before pushing",
		Category: "testing",
	}

	path, content, found := mem.findExistingHeuristic(context.Background(), candidate, prefix)
	if !found {
		t.Fatal("expected to find existing heuristic")
	}
	if !strings.HasSuffix(string(path), existingFilename) {
		t.Errorf("matched the wrong file: %s", path)
	}
	if !strings.Contains(content, "Always run tests") {
		t.Error("returned content should include the existing body")
	}
}

func TestFindExistingHeuristic_NoMatch(t *testing.T) {
	mem, store := newTestMemory(t)
	projectPath := "/example/project"
	slug := ProjectSlug(projectPath)
	prefix := brain.MemoryProjectPrefix(slug)

	writeTopic(t, store, brain.MemoryProjectTopic(slug, "heuristic-testing-always-run"), `---
name: "Testing: always run"
type: feedback
tags:
  - heuristic
  - testing
---

## Body
Content.`)

	candidate := Heuristic{
		Rule:     "Never concatenate SQL strings",
		Category: "debugging",
	}

	_, _, found := mem.findExistingHeuristic(context.Background(), candidate, prefix)
	if found {
		t.Error("expected no match for unrelated candidate")
	}
}

func TestMergeHeuristic_AddsNewSection(t *testing.T) {
	existing := `---
name: "Testing: always run tests"
description: "Always run tests"
type: feedback
created: 2025-01-01T00:00:00Z
modified: 2025-01-01T00:00:00Z
confidence: low
source: reflection
tags:
  - heuristic
  - testing
---

## Always run tests before

Always run tests before committing.

**Why:** Observed during reflection

**Confidence:** low (1 observation)`

	h := Heuristic{
		Rule:       "Run tests in CI as well as locally",
		Context:    "CI pipeline",
		Confidence: "medium",
		Category:   "testing",
	}

	result := mergeHeuristic(existing, h)

	sections := countSections(strings.SplitN(result, "---", 3)[2])
	if sections != 2 {
		t.Errorf("expected 2 sections after merge, got %d", sections)
	}

	if !strings.Contains(result, "Run tests in CI") {
		t.Error("merged content should contain new rule text")
	}
}

func TestMergeHeuristic_UpdatesModified(t *testing.T) {
	existing := `---
name: "Testing: run tests"
type: feedback
created: 2025-01-01T00:00:00Z
modified: 2025-01-01T00:00:00Z
confidence: low
source: reflection
tags:
  - heuristic
  - testing
---

## Run tests

Content.`

	h := Heuristic{
		Rule:       "Run tests often",
		Confidence: "low",
		Category:   "testing",
	}

	result := mergeHeuristic(existing, h)

	if strings.Contains(result, "modified: 2025-01-01T00:00:00Z") {
		t.Error("modified date should have been updated")
	}
	if !strings.Contains(result, "modified: ") {
		t.Error("merged content should contain a modified field")
	}
}

func TestMergeHeuristic_UpdatesConfidence(t *testing.T) {
	existing := `---
name: "Testing: run tests"
type: feedback
created: 2025-01-01T00:00:00Z
modified: 2025-01-01T00:00:00Z
confidence: low
source: reflection
tags:
  - heuristic
  - testing
---

## Run tests

Content.`

	h := Heuristic{
		Rule:       "Keep running tests",
		Confidence: "low",
		Category:   "testing",
	}

	result := mergeHeuristic(existing, h)

	if !strings.Contains(result, "confidence: medium") {
		t.Errorf("expected confidence: medium after 2 observations, got:\n%s", result)
	}
}

func TestConfidenceFromObservations(t *testing.T) {
	tests := []struct {
		count int
		want  string
	}{
		{1, "low"},
		{2, "medium"},
		{3, "medium"},
		{4, "high"},
		{10, "high"},
	}

	for _, tc := range tests {
		got := confidenceFromObservations(tc.count)
		if got != tc.want {
			t.Errorf("confidenceFromObservations(%d) = %q, want %q", tc.count, got, tc.want)
		}
	}
}

func TestApplyHeuristics_CreatesNewFile(t *testing.T) {
	mem, store := newTestMemory(t)
	projectPath := "/example/project"
	slug := ProjectSlug(projectPath)

	heuristics := []Heuristic{
		{
			Rule:       "Always check error returns in Go",
			Context:    "Go error handling",
			Confidence: "low",
			Category:   "approach",
			Scope:      "project",
		},
	}

	if err := mem.ApplyHeuristics(context.Background(), slug, heuristics); err != nil {
		t.Fatalf("ApplyHeuristics: %v", err)
	}

	entries, err := store.List(context.Background(), brain.MemoryProjectPrefix(slug), brain.ListOpts{IncludeGenerated: true})
	if err != nil {
		t.Fatalf("list: %v", err)
	}

	var mdFiles []brain.FileInfo
	for _, e := range entries {
		if e.IsDir {
			continue
		}
		name := baseName(string(e.Path))
		if strings.HasSuffix(name, ".md") && name != "MEMORY.md" {
			mdFiles = append(mdFiles, e)
		}
	}

	if len(mdFiles) != 1 {
		t.Fatalf("expected 1 heuristic file, got %d", len(mdFiles))
	}

	data, _ := store.Read(context.Background(), mdFiles[0].Path)
	content := string(data)

	if !strings.Contains(content, "heuristic") {
		t.Error("file should contain 'heuristic' tag")
	}
	if !strings.Contains(content, "approach") {
		t.Error("file should contain category tag")
	}
}

func TestApplyHeuristics_MergesExisting(t *testing.T) {
	mem, store := newTestMemory(t)
	projectPath := "/example/project"
	slug := ProjectSlug(projectPath)

	first := []Heuristic{
		{
			Rule:       "Check error returns carefully",
			Confidence: "low",
			Category:   "approach",
			Scope:      "project",
		},
	}
	if err := mem.ApplyHeuristics(context.Background(), slug, first); err != nil {
		t.Fatalf("first ApplyHeuristics: %v", err)
	}

	second := []Heuristic{
		{
			Rule:       "Check error returns and handle gracefully",
			Confidence: "low",
			Category:   "approach",
			Scope:      "project",
		},
	}
	if err := mem.ApplyHeuristics(context.Background(), slug, second); err != nil {
		t.Fatalf("second ApplyHeuristics: %v", err)
	}

	entries, _ := store.List(context.Background(), brain.MemoryProjectPrefix(slug), brain.ListOpts{IncludeGenerated: true})

	var mdFiles []brain.FileInfo
	for _, e := range entries {
		if e.IsDir {
			continue
		}
		name := baseName(string(e.Path))
		if strings.HasSuffix(name, ".md") && name != "MEMORY.md" {
			mdFiles = append(mdFiles, e)
		}
	}

	if len(mdFiles) != 1 {
		t.Fatalf("expected 1 merged file, got %d", len(mdFiles))
	}

	data, _ := store.Read(context.Background(), mdFiles[0].Path)
	content := string(data)

	sections := countSections(content)
	if sections < 2 {
		t.Errorf("expected at least 2 ## sections after merge, got %d", sections)
	}
}

func TestApplyHeuristics_RoutesGlobalCorrectly(t *testing.T) {
	mem, store := newTestMemory(t)
	projectPath := "/example/project"
	slug := ProjectSlug(projectPath)

	heuristics := []Heuristic{
		{
			Rule:       "Prefer composition over inheritance",
			Confidence: "high",
			Category:   "architecture",
			Scope:      "global",
		},
	}

	if err := mem.ApplyHeuristics(context.Background(), slug, heuristics); err != nil {
		t.Fatalf("ApplyHeuristics: %v", err)
	}

	globalEntries, _ := store.List(context.Background(), brain.MemoryGlobalPrefix(), brain.ListOpts{IncludeGenerated: true})
	var globalFiles []brain.FileInfo
	for _, e := range globalEntries {
		if e.IsDir {
			continue
		}
		name := baseName(string(e.Path))
		if strings.HasSuffix(name, ".md") && name != "MEMORY.md" {
			globalFiles = append(globalFiles, e)
		}
	}
	if len(globalFiles) != 1 {
		t.Fatalf("expected 1 global heuristic file, got %d", len(globalFiles))
	}

	projectEntries, _ := store.List(context.Background(), brain.MemoryProjectPrefix(slug), brain.ListOpts{IncludeGenerated: true})
	for _, e := range projectEntries {
		name := baseName(string(e.Path))
		if !e.IsDir && strings.HasSuffix(name, ".md") && name != "MEMORY.md" {
			t.Error("no heuristic files should be in project memory for global-scoped heuristic")
		}
	}
}

func TestListHeuristicsIn_BothScopes(t *testing.T) {
	mem, _ := newTestMemory(t)
	projectPath := "/example/project"
	slug := ProjectSlug(projectPath)

	if err := mem.ApplyHeuristics(context.Background(), slug, []Heuristic{
		{
			Rule:       "Use table-driven tests in Go",
			Confidence: "medium",
			Category:   "testing",
			Scope:      "project",
		},
	}); err != nil {
		t.Fatalf("project ApplyHeuristics: %v", err)
	}

	if err := mem.ApplyHeuristics(context.Background(), slug, []Heuristic{
		{
			Rule:       "Always write British English",
			Confidence: "high",
			Category:   "communication",
			Scope:      "global",
		},
	}); err != nil {
		t.Fatalf("global ApplyHeuristics: %v", err)
	}

	summaries := mem.ListHeuristicsIn(context.Background(), projectPath)
	if len(summaries) != 2 {
		t.Fatalf("expected 2 heuristics (1 project + 1 global), got %d", len(summaries))
	}

	var hasProject, hasGlobal bool
	for _, s := range summaries {
		if s.Scope == "project" {
			hasProject = true
		}
		if s.Scope == "global" {
			hasGlobal = true
		}
	}

	if !hasProject {
		t.Error("expected a project-scoped heuristic")
	}
	if !hasGlobal {
		t.Error("expected a global-scoped heuristic")
	}
}
