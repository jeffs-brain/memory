// SPDX-License-Identifier: Apache-2.0

package buffer

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/jeffs-brain/memory/go/brain"
	"github.com/jeffs-brain/memory/go/store/mem"
)

func newTestBuffer(store brain.Store, scope Scope, cfg ...Config) *Buffer {
	c := DefaultConfig()
	if len(cfg) > 0 {
		c = cfg[0]
	}
	return New(store, scope, c)
}

func TestAppendAndRenderRoundTrip(t *testing.T) {
	store := mem.New()
	buf := newTestBuffer(store, Scope{Kind: ScopeGlobal})
	ctx := context.Background()

	obs := Observation{
		At:       time.Date(2026, 4, 14, 10, 30, 0, 0, time.UTC),
		Intent:   "edit",
		Entities: []string{"brain/paths.go"},
		Outcome:  "ok",
		Summary:  "Added buffer path helpers",
	}

	if err := buf.Append(ctx, obs); err != nil {
		t.Fatalf("Append: %v", err)
	}

	got, err := buf.Render(ctx)
	if err != nil {
		t.Fatalf("Render: %v", err)
	}

	if !strings.Contains(got, "Added buffer path helpers") {
		t.Fatalf("Render output missing summary: %q", got)
	}
	if !strings.Contains(got, "(edit)") {
		t.Fatalf("Render output missing intent: %q", got)
	}
	if !strings.Contains(got, "{brain/paths.go}") {
		t.Fatalf("Render output missing entities: %q", got)
	}
	if strings.Contains(got, "[ok]") {
		t.Fatalf("Render output should not contain [ok]: %q", got)
	}
}

func TestMultipleAppendsAccumulate(t *testing.T) {
	store := mem.New()
	buf := newTestBuffer(store, Scope{Kind: ScopeGlobal})
	ctx := context.Background()

	now := time.Date(2026, 4, 14, 10, 0, 0, 0, time.UTC)
	for i := range 5 {
		obs := Observation{
			At:      now.Add(time.Duration(i) * time.Minute),
			Intent:  "chat",
			Summary: strings.Repeat("x", 10),
		}
		if err := buf.Append(ctx, obs); err != nil {
			t.Fatalf("Append %d: %v", i, err)
		}
	}

	got, err := buf.Render(ctx)
	if err != nil {
		t.Fatalf("Render: %v", err)
	}

	lines := strings.Split(strings.TrimSpace(got), "\n")
	if len(lines) != 5 {
		t.Fatalf("expected 5 lines, got %d: %q", len(lines), got)
	}
}

func TestRenderEmptyBuffer(t *testing.T) {
	store := mem.New()
	buf := newTestBuffer(store, Scope{Kind: ScopeGlobal})
	ctx := context.Background()

	got, err := buf.Render(ctx)
	if err != nil {
		t.Fatalf("Render: %v", err)
	}
	if got != "" {
		t.Fatalf("expected empty string, got %q", got)
	}
}

func TestTokenCountApproximation(t *testing.T) {
	store := mem.New()
	buf := newTestBuffer(store, Scope{Kind: ScopeGlobal})
	ctx := context.Background()

	tokens, err := buf.TokenCount(ctx)
	if err != nil {
		t.Fatalf("TokenCount: %v", err)
	}
	if tokens != 0 {
		t.Fatalf("expected 0 tokens for empty buffer, got %d", tokens)
	}

	content := strings.Repeat("abcd", 100)
	if err := store.Write(ctx, brain.MemoryBufferGlobal(), []byte(content)); err != nil {
		t.Fatalf("Write: %v", err)
	}

	tokens, err = buf.TokenCount(ctx)
	if err != nil {
		t.Fatalf("TokenCount: %v", err)
	}
	if tokens != 100 {
		t.Fatalf("expected 100 tokens, got %d", tokens)
	}
}

func TestNeedsCompactionTriggersAtThreshold(t *testing.T) {
	store := mem.New()
	cfg := Config{
		TokenBudget:       10,
		CompactThreshold:  100,
		KeepRecentPercent: 50,
		MaxObservationLen: 160,
	}
	buf := newTestBuffer(store, Scope{Kind: ScopeGlobal}, cfg)
	ctx := context.Background()

	needs, err := buf.NeedsCompaction(ctx)
	if err != nil {
		t.Fatalf("NeedsCompaction: %v", err)
	}
	if needs {
		t.Fatal("should not need compaction on empty buffer")
	}

	content := strings.Repeat("a", 44)
	if err := store.Write(ctx, brain.MemoryBufferGlobal(), []byte(content)); err != nil {
		t.Fatalf("Write: %v", err)
	}

	needs, err = buf.NeedsCompaction(ctx)
	if err != nil {
		t.Fatalf("NeedsCompaction: %v", err)
	}
	if !needs {
		t.Fatal("should need compaction after exceeding budget")
	}
}

func TestCompactKeepsRecentPercentage(t *testing.T) {
	store := mem.New()
	cfg := Config{
		TokenBudget:       8192,
		CompactThreshold:  100,
		KeepRecentPercent: 50,
		MaxObservationLen: 160,
	}
	buf := newTestBuffer(store, Scope{Kind: ScopeGlobal}, cfg)
	ctx := context.Background()

	now := time.Date(2026, 4, 14, 10, 0, 0, 0, time.UTC)
	for i := range 10 {
		obs := Observation{
			At:      now.Add(time.Duration(i) * time.Minute),
			Intent:  "chat",
			Summary: "line " + strings.Repeat("x", 5),
		}
		if err := buf.Append(ctx, obs); err != nil {
			t.Fatalf("Append %d: %v", i, err)
		}
	}

	removed, err := buf.Compact(ctx)
	if err != nil {
		t.Fatalf("Compact: %v", err)
	}
	if removed != 5 {
		t.Fatalf("expected 5 removed, got %d", removed)
	}

	got, err := buf.Render(ctx)
	if err != nil {
		t.Fatalf("Render: %v", err)
	}

	lines := strings.Split(strings.TrimSpace(got), "\n")
	if len(lines) != 5 {
		t.Fatalf("expected 5 remaining lines, got %d", len(lines))
	}

	if !strings.Contains(lines[0], "10:05:00") {
		t.Fatalf("first kept line should be 10:05, got %q", lines[0])
	}
	if !strings.Contains(lines[4], "10:09:00") {
		t.Fatalf("last kept line should be 10:09, got %q", lines[4])
	}
}

func TestCompactEmptyBufferIsNoop(t *testing.T) {
	store := mem.New()
	buf := newTestBuffer(store, Scope{Kind: ScopeGlobal})
	ctx := context.Background()

	removed, err := buf.Compact(ctx)
	if err != nil {
		t.Fatalf("Compact: %v", err)
	}
	if removed != 0 {
		t.Fatalf("expected 0 removed on empty buffer, got %d", removed)
	}
}

func TestObservationSummaryTruncation(t *testing.T) {
	store := mem.New()
	cfg := Config{
		TokenBudget:       8192,
		CompactThreshold:  100,
		KeepRecentPercent: 50,
		MaxObservationLen: 20,
	}
	buf := newTestBuffer(store, Scope{Kind: ScopeGlobal}, cfg)
	ctx := context.Background()

	longSummary := strings.Repeat("z", 50)
	obs := Observation{
		At:      time.Date(2026, 4, 14, 12, 0, 0, 0, time.UTC),
		Intent:  "chat",
		Summary: longSummary,
	}

	if err := buf.Append(ctx, obs); err != nil {
		t.Fatalf("Append: %v", err)
	}

	got, err := buf.Render(ctx)
	if err != nil {
		t.Fatalf("Render: %v", err)
	}

	if strings.Contains(got, longSummary) {
		t.Fatal("rendered output should not contain the full 50-char summary")
	}
	truncated := strings.Repeat("z", 20)
	if !strings.Contains(got, truncated) {
		t.Fatalf("rendered output should contain truncated summary: %q", got)
	}
}

func TestGlobalVsProjectScopeUsesCorrectPaths(t *testing.T) {
	store := mem.New()
	ctx := context.Background()

	globalBuf := newTestBuffer(store, Scope{Kind: ScopeGlobal})
	projectBuf := newTestBuffer(store, Scope{Kind: ScopeProject, Slug: "my-project"})

	obs := Observation{
		At:      time.Date(2026, 4, 14, 9, 0, 0, 0, time.UTC),
		Intent:  "plan",
		Summary: "global observation",
	}
	if err := globalBuf.Append(ctx, obs); err != nil {
		t.Fatalf("global Append: %v", err)
	}

	obs.Summary = "project observation"
	if err := projectBuf.Append(ctx, obs); err != nil {
		t.Fatalf("project Append: %v", err)
	}

	globalContent, err := globalBuf.Render(ctx)
	if err != nil {
		t.Fatalf("global Render: %v", err)
	}
	if !strings.Contains(globalContent, "global observation") {
		t.Fatal("global buffer missing its observation")
	}
	if strings.Contains(globalContent, "project observation") {
		t.Fatal("global buffer should not contain project observation")
	}

	projectContent, err := projectBuf.Render(ctx)
	if err != nil {
		t.Fatalf("project Render: %v", err)
	}
	if !strings.Contains(projectContent, "project observation") {
		t.Fatal("project buffer missing its observation")
	}
	if strings.Contains(projectContent, "global observation") {
		t.Fatal("project buffer should not contain global observation")
	}

	globalPath := brain.MemoryBufferGlobal()
	projectPath := brain.MemoryBufferProject("my-project")

	exists, _ := store.Exists(ctx, globalPath)
	if !exists {
		t.Fatalf("expected file at %s", globalPath)
	}
	exists, _ = store.Exists(ctx, projectPath)
	if !exists {
		t.Fatalf("expected file at %s", projectPath)
	}
}

func TestFormatObservationOutputFormat(t *testing.T) {
	at := time.Date(2026, 4, 14, 14, 30, 45, 0, time.UTC)

	cases := []struct {
		name     string
		intent   string
		outcome  string
		summary  string
		entities []string
		want     string
	}{
		{
			name:     "full observation",
			intent:   "edit",
			outcome:  "error",
			summary:  "failed to write file",
			entities: []string{"paths.go", "store.go"},
			want:     "- [14:30:45] (edit) [error] failed to write file {paths.go, store.go}",
		},
		{
			name:    "ok outcome omitted",
			intent:  "read",
			outcome: "ok",
			summary: "read config",
			want:    "- [14:30:45] (read) read config",
		},
		{
			name:    "empty outcome omitted",
			intent:  "chat",
			outcome: "",
			summary: "discussed architecture",
			want:    "- [14:30:45] (chat) discussed architecture",
		},
		{
			name:    "no intent",
			intent:  "",
			outcome: "",
			summary: "bare observation",
			want:    "- [14:30:45] bare observation",
		},
		{
			name:     "entities only",
			intent:   "plan",
			outcome:  "partial",
			summary:  "planned refactor",
			entities: []string{"memory"},
			want:     "- [14:30:45] (plan) [partial] planned refactor {memory}",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := formatObservation(at, tc.intent, tc.outcome, tc.summary, tc.entities)
			if got != tc.want {
				t.Fatalf("got:  %q\nwant: %q", got, tc.want)
			}
		})
	}
}
