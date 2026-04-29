// SPDX-License-Identifier: Apache-2.0

package memory

import (
	"context"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/jeffs-brain/memory/go/brain"
)

func TestNewConsolidator(t *testing.T) {
	mem, _ := newTestMemory(t)
	c := NewConsolidator(nil, "", mem)
	if c == nil {
		t.Fatal("expected non-nil Consolidator")
	}
}

func TestRunQuick_EmptyMemory(t *testing.T) {
	mem, _ := newTestMemory(t)
	c := NewConsolidator(nil, "", mem)

	report, err := c.RunQuick(context.Background())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if report == nil {
		t.Fatal("expected non-nil report")
	}
	if report.Duration <= 0 {
		t.Error("expected positive duration")
	}
	if len(report.Errors) > 0 {
		t.Errorf("unexpected errors: %v", report.Errors)
	}
}

func TestRegenerateIndexes_RebuildFromFiles(t *testing.T) {
	mem, store := newTestMemory(t)
	projectPath := "/example/project"
	slug := ProjectSlug(projectPath)

	writeTopic(t, store, brain.MemoryProjectTopic(slug, "architecture"), `---
name: Architecture Overview
description: High-level system architecture
type: project
---

The system uses microservices.
`)
	writeTopic(t, store, brain.MemoryProjectTopic(slug, "testing-patterns"), `---
name: Testing Patterns
description: How we write tests
type: reference
---

Use table-driven tests.
`)
	writeTopic(t, store, brain.MemoryProjectIndex(slug), "- [Old Entry](old.md) — outdated\n")

	c := NewConsolidator(nil, "", mem)
	report, err := c.RunQuick(context.Background())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if report.IndexesRebuilt < 1 {
		t.Errorf("expected at least 1 index rebuilt, got %d", report.IndexesRebuilt)
	}

	data, err := store.Read(context.Background(), brain.MemoryProjectIndex(slug))
	if err != nil {
		t.Fatalf("reading rebuilt MEMORY.md: %v", err)
	}
	content := string(data)

	if !strings.Contains(content, "architecture.md") {
		t.Error("rebuilt index should contain architecture.md")
	}
	if !strings.Contains(content, "testing-patterns.md") {
		t.Error("rebuilt index should contain testing-patterns.md")
	}
	if strings.Contains(content, "old.md") {
		t.Error("rebuilt index should not contain old.md (file doesn't exist)")
	}
	if !strings.Contains(content, "Architecture Overview") {
		t.Error("rebuilt index should contain the topic name")
	}
}

func TestDetectStaleness_RecentFile(t *testing.T) {
	mem, store := newTestMemory(t)
	projectPath := "/example/project"
	slug := ProjectSlug(projectPath)

	now := time.Now().UTC().Format(time.RFC3339)
	writeTopic(t, store, brain.MemoryProjectTopic(slug, "recent"), fmt.Sprintf(`---
name: Recent Topic
description: Just created
type: project
modified: %s
---

Fresh content.
`, now))

	c := NewConsolidator(nil, "", mem)
	report, err := c.RunQuick(context.Background())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if report.StaleMemoriesFlagged != 0 {
		t.Errorf("expected 0 stale memories, got %d", report.StaleMemoriesFlagged)
	}
}

func TestDetectStaleness_OldFile(t *testing.T) {
	mem, store := newTestMemory(t)
	projectPath := "/example/project"
	slug := ProjectSlug(projectPath)

	oldDate := time.Now().AddDate(0, 0, -(stalenessThresholdDays + 10)).UTC().Format(time.RFC3339)
	writeTopic(t, store, brain.MemoryProjectTopic(slug, "ancient"), fmt.Sprintf(`---
name: Ancient Topic
description: Very old
type: project
modified: %s
---

Old content.
`, oldDate))

	c := NewConsolidator(nil, "", mem)
	report, err := c.RunQuick(context.Background())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if report.StaleMemoriesFlagged != 1 {
		t.Errorf("expected 1 stale memory, got %d", report.StaleMemoriesFlagged)
	}
}

func TestReinforceHeuristics_UpdatesConfidence(t *testing.T) {
	mem, store := newTestMemory(t)
	projectPath := "/example/project"
	slug := ProjectSlug(projectPath)

	writeTopic(t, store, brain.MemoryProjectTopic(slug, "heuristic-testing-go"), `---
name: "Testing: Go patterns"
description: "Use table-driven tests"
type: feedback
confidence: low
source: reflection
tags:
  - heuristic
  - testing
---

## Observation 1

First pattern observed.

## Observation 2

Second pattern observed.

## Observation 3

Third pattern observed.

## Observation 4

Fourth pattern observed.
`)

	c := NewConsolidator(nil, "", mem)
	report, err := c.RunQuick(context.Background())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if report.HeuristicsUpdated != 1 {
		t.Errorf("expected 1 heuristic updated, got %d", report.HeuristicsUpdated)
	}

	data, err := store.Read(context.Background(), brain.MemoryProjectTopic(slug, "heuristic-testing-go"))
	if err != nil {
		t.Fatalf("reading updated heuristic: %v", err)
	}
	content := string(data)

	if !strings.Contains(content, "confidence: high") {
		t.Errorf("expected confidence to be updated to high, got:\n%s", content)
	}
}

func TestReinforceHeuristics_NoChangeWhenCorrect(t *testing.T) {
	mem, store := newTestMemory(t)
	projectPath := "/example/project"
	slug := ProjectSlug(projectPath)

	writeTopic(t, store, brain.MemoryProjectTopic(slug, "heuristic-style-code"), `---
name: "Style: Code formatting"
description: "Use consistent formatting"
type: feedback
confidence: low
source: reflection
tags:
  - heuristic
  - style
---

## Observation 1

Format code consistently.
`)

	c := NewConsolidator(nil, "", mem)
	report, err := c.RunQuick(context.Background())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if report.HeuristicsUpdated != 0 {
		t.Errorf("expected 0 heuristics updated, got %d", report.HeuristicsUpdated)
	}
}

func TestConsolidator_ConcurrentRunPrevented(t *testing.T) {
	mem, _ := newTestMemory(t)
	c := NewConsolidator(nil, "", mem)

	c.mu.Lock()
	c.inProgress = true
	c.mu.Unlock()

	_, err := c.RunQuick(context.Background())
	if err == nil {
		t.Fatal("expected error for concurrent run")
	}
	if !strings.Contains(err.Error(), "already in progress") {
		t.Errorf("expected 'already in progress' error, got: %v", err)
	}

	c.mu.Lock()
	c.inProgress = false
	c.mu.Unlock()
}

func TestParseDeduplicationResult_ValidJSON(t *testing.T) {
	result := parseDeduplicationResult(`{"verdict": "merge", "reason": "overlapping content"}`)
	if result.Verdict != "merge" {
		t.Errorf("expected merge, got %s", result.Verdict)
	}
}

func TestParseDeduplicationResult_InvalidJSON(t *testing.T) {
	result := parseDeduplicationResult("not json at all")
	if result.Verdict != "distinct" {
		t.Errorf("expected distinct fallback, got %s", result.Verdict)
	}
}

func TestScopePrefixes_EmptyBrain(t *testing.T) {
	mem, _ := newTestMemory(t)
	c := NewConsolidator(nil, "", mem)

	prefixes := c.scopePrefixes(context.Background())
	if len(prefixes) != 1 {
		t.Fatalf("expected 1 prefix (global only), got %d: %v", len(prefixes), prefixes)
	}
	if prefixes[0] != brain.MemoryGlobalPrefix() {
		t.Errorf("expected global prefix, got %q", prefixes[0])
	}
}

func TestScopePrefixes_WithProjects(t *testing.T) {
	mem, store := newTestMemory(t)
	writeTopic(t, store, brain.MemoryProjectTopic("alpha", "seed"), "a")
	writeTopic(t, store, brain.MemoryProjectTopic("beta", "seed"), "b")

	c := NewConsolidator(nil, "", mem)
	prefixes := c.scopePrefixes(context.Background())

	if len(prefixes) != 3 {
		t.Fatalf("expected 3 prefixes, got %d: %v", len(prefixes), prefixes)
	}
	found := map[string]bool{}
	for _, p := range prefixes {
		found[string(p)] = true
	}
	if !found[string(brain.MemoryGlobalPrefix())] {
		t.Error("missing global prefix")
	}
	if !found[string(brain.MemoryProjectPrefix("alpha"))] {
		t.Error("missing alpha project prefix")
	}
	if !found[string(brain.MemoryProjectPrefix("beta"))] {
		t.Error("missing beta project prefix")
	}
}

func TestRebuildIndex_EmptyScope(t *testing.T) {
	mem, store := newTestMemory(t)
	c := NewConsolidator(nil, "", mem)
	projectPath := "/example/project"
	slug := ProjectSlug(projectPath)

	writeTopic(t, store, brain.MemoryProjectTopic(slug, "seed"), "body")
	if err := store.Delete(context.Background(), brain.MemoryProjectTopic(slug, "seed")); err != nil {
		t.Fatalf("delete seed: %v", err)
	}

	err := store.Batch(context.Background(), brain.BatchOptions{Reason: "test"}, func(b brain.Batch) error {
		return c.rebuildIndexInBatch(context.Background(), b, brain.MemoryProjectPrefix(slug))
	})
	if err != nil {
		t.Fatalf("rebuildIndex empty scope: %v", err)
	}

	if data, err := store.Read(context.Background(), brain.MemoryProjectIndex(slug)); err == nil {
		if strings.Contains(string(data), "](") {
			t.Errorf("expected empty index for empty scope, got:\n%s", data)
		}
	}
}

func TestModifiedTime_ExistingFile(t *testing.T) {
	mem, store := newTestMemory(t)
	c := NewConsolidator(nil, "", mem)

	p := brain.MemoryGlobalTopic("dated")
	writeTopic(t, store, p, `---
name: Dated
modified: 2025-06-01T12:00:00Z
---

body`)

	got := c.modifiedTime(context.Background(), p)
	want, _ := time.Parse(time.RFC3339, "2025-06-01T12:00:00Z")
	if !got.Equal(want) {
		t.Errorf("modifiedTime = %v, want %v", got, want)
	}
}

func TestModifiedTime_MissingFile(t *testing.T) {
	mem, _ := newTestMemory(t)
	c := NewConsolidator(nil, "", mem)

	got := c.modifiedTime(context.Background(), brain.MemoryGlobalTopic("nowhere"))
	if !got.IsZero() {
		t.Errorf("expected zero time for missing file, got %v", got)
	}
}
