// SPDX-License-Identifier: Apache-2.0

package memory

import (
	"context"
	"strings"
	"testing"

	"github.com/jeffs-brain/memory/go/brain"
	"github.com/jeffs-brain/memory/go/store/mem"
)

func TestApplyExtractions_StampsSupersededByOnOldFile(t *testing.T) {
	store := mem.New()
	m := New(store)
	ctx := context.Background()

	oldFact := ExtractedMemory{
		Action:      "create",
		Filename:    "gym_time.md",
		Name:        "gym time",
		Description: "when I go to the gym",
		Type:        "project",
		Scope:       "project",
		Content:     "I go to the gym at 7pm on Tuesdays.",
		IndexEntry:  "- gym_time.md — evening gym routine",
	}
	if err := m.ApplyExtractions(ctx, "eval-lme", []ExtractedMemory{oldFact}); err != nil {
		t.Fatalf("initial apply: %v", err)
	}

	newFact := ExtractedMemory{
		Action:      "create",
		Filename:    "gym_time_v2.md",
		Name:        "gym time",
		Description: "updated gym time",
		Type:        "project",
		Scope:       "project",
		Content:     "Actually, I now go to the gym at 6pm on Wednesdays.",
		IndexEntry:  "- gym_time_v2.md — updated",
		Supersedes:  "gym_time.md",
	}
	if err := m.ApplyExtractions(ctx, "eval-lme", []ExtractedMemory{newFact}); err != nil {
		t.Fatalf("supersede apply: %v", err)
	}

	newBody, err := store.Read(ctx, brain.MemoryProjectTopic("eval-lme", "gym_time_v2"))
	if err != nil {
		t.Fatalf("read new file: %v", err)
	}
	if !strings.Contains(string(newBody), "supersedes: gym_time.md") {
		t.Errorf("new file missing supersedes frontmatter, got:\n%s", string(newBody))
	}

	oldBody, err := store.Read(ctx, brain.MemoryProjectTopic("eval-lme", "gym_time"))
	if err != nil {
		t.Fatalf("read old file: %v", err)
	}
	if !strings.Contains(string(oldBody), "superseded_by: gym_time_v2.md") {
		t.Errorf("old file missing superseded_by frontmatter, got:\n%s", string(oldBody))
	}

	if !strings.Contains(string(oldBody), "7pm on Tuesdays") {
		t.Errorf("old body content was destroyed, got:\n%s", string(oldBody))
	}
}

func TestApplyExtractions_SupersedesMissingOldFileIsSafe(t *testing.T) {
	store := mem.New()
	m := New(store)
	ctx := context.Background()

	em := ExtractedMemory{
		Action:     "create",
		Filename:   "new.md",
		Name:       "new",
		Type:       "project",
		Scope:      "project",
		Content:    "fresh fact",
		Supersedes: "does-not-exist.md",
	}
	if err := m.ApplyExtractions(ctx, "eval-lme", []ExtractedMemory{em}); err != nil {
		t.Fatalf("apply must not fail when supersede target is missing: %v", err)
	}

	got, err := store.Read(ctx, brain.MemoryProjectTopic("eval-lme", "new"))
	if err != nil {
		t.Fatalf("new file must be written, got: %v", err)
	}
	if !strings.Contains(string(got), "supersedes: does-not-exist.md") {
		t.Errorf("new file must still record the lineage pointer, got:\n%s", string(got))
	}
}

func TestParseFrontmatter_ReadsSupersessionFields(t *testing.T) {
	raw := "---\n" +
		"name: gym time\n" +
		"type: project\n" +
		"modified: 2024-03-25T00:00:00Z\n" +
		"supersedes: gym_time_v1.md\n" +
		"superseded_by: gym_time_v3.md\n" +
		"---\n\nBody here.\n"
	fm, body := ParseFrontmatter(raw)
	if fm.Supersedes != "gym_time_v1.md" {
		t.Errorf("Supersedes = %q, want gym_time_v1.md", fm.Supersedes)
	}
	if fm.SupersededBy != "gym_time_v3.md" {
		t.Errorf("SupersededBy = %q, want gym_time_v3.md", fm.SupersededBy)
	}
	if !strings.Contains(body, "Body here.") {
		t.Errorf("body must survive parse, got %q", body)
	}
}
