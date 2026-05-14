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

func writeMemoryFile(t *testing.T, store brain.Store, p brain.Path, fm map[string]string, body string) {
	t.Helper()
	var sb strings.Builder
	sb.WriteString("---\n")
	for k, v := range fm {
		fmt.Fprintf(&sb, "%s: %s\n", k, v)
	}
	sb.WriteString("---\n\n")
	sb.WriteString(body)
	writeTopic(t, store, p, sb.String())
}

func TestRunHygieneDetectsContradictionByName(t *testing.T) {
	mem, store := newTestMemory(t)
	cons := NewConsolidator(nil, "", mem)

	now := time.Date(2026, 5, 15, 10, 0, 0, 0, time.UTC)
	writeMemoryFile(t, store, brain.MemoryGlobalTopic("gym_time_a"),
		map[string]string{
			"name":       "Gym time",
			"modified":   now.Format(time.RFC3339),
			"confidence": "medium",
		}, "Trains at 8am.")
	writeMemoryFile(t, store, brain.MemoryGlobalTopic("gym_time_b"),
		map[string]string{
			"name":       "Gym time",
			"modified":   now.Format(time.RFC3339),
			"confidence": "high",
		}, "Trains at 7am.")

	report, err := cons.RunHygiene(context.Background(), HygieneOptions{Now: now})
	if err != nil {
		t.Fatalf("RunHygiene: %v", err)
	}
	if len(report.Contradictions) != 1 {
		t.Fatalf("expected 1 contradiction group, got %d", len(report.Contradictions))
	}
	g := report.Contradictions[0]
	if g.KeyReason != "name" {
		t.Errorf("expected key reason name, got %q", g.KeyReason)
	}
	if len(g.Members) != 2 {
		t.Errorf("expected 2 members, got %d", len(g.Members))
	}
	if g.Canonical != "" {
		t.Errorf("dry-run should not pick canonical, got %s", g.Canonical)
	}
}

func TestRunHygieneApplyStampsSupersededBy(t *testing.T) {
	mem, store := newTestMemory(t)
	cons := NewConsolidator(nil, "", mem)

	now := time.Date(2026, 5, 15, 10, 0, 0, 0, time.UTC)
	writeMemoryFile(t, store, brain.MemoryGlobalTopic("gym_time_a"),
		map[string]string{
			"name":       "Gym time",
			"modified":   now.Add(-time.Hour).Format(time.RFC3339),
			"confidence": "medium",
		}, "Trains at 8am.")
	writeMemoryFile(t, store, brain.MemoryGlobalTopic("gym_time_b"),
		map[string]string{
			"name":       "Gym time",
			"modified":   now.Format(time.RFC3339),
			"confidence": "high",
		}, "Trains at 7am.")

	report, err := cons.RunHygiene(context.Background(), HygieneOptions{Apply: true, Now: now})
	if err != nil {
		t.Fatalf("RunHygiene: %v", err)
	}
	if len(report.Contradictions) != 1 {
		t.Fatalf("expected 1 contradiction group, got %d", len(report.Contradictions))
	}
	if report.Contradictions[0].Canonical != brain.MemoryGlobalTopic("gym_time_b") {
		t.Errorf("expected high-confidence and newer to win, got canonical %s", report.Contradictions[0].Canonical)
	}

	oldData, err := store.Read(context.Background(), brain.MemoryGlobalTopic("gym_time_a"))
	if err != nil {
		t.Fatalf("read old: %v", err)
	}
	if !strings.Contains(string(oldData), "superseded_by: gym_time_b.md") {
		t.Errorf("expected old file stamped with superseded_by, got:\n%s", oldData)
	}
}

func TestRunHygieneGroupsByClaimKey(t *testing.T) {
	mem, store := newTestMemory(t)
	cons := NewConsolidator(nil, "", mem)

	now := time.Date(2026, 5, 15, 10, 0, 0, 0, time.UTC).Format(time.RFC3339)
	writeMemoryFile(t, store, brain.MemoryGlobalTopic("a"),
		map[string]string{"name": "A", "modified": now, "claim_key": "weather_forecast"},
		"sunny")
	writeMemoryFile(t, store, brain.MemoryGlobalTopic("b"),
		map[string]string{"name": "B", "modified": now, "claim_key": "weather_forecast"},
		"rainy")

	report, err := cons.RunHygiene(context.Background(), HygieneOptions{})
	if err != nil {
		t.Fatalf("RunHygiene: %v", err)
	}
	if len(report.Contradictions) != 1 {
		t.Fatalf("expected 1 contradiction group, got %d", len(report.Contradictions))
	}
	if report.Contradictions[0].KeyReason != "claim_key" {
		t.Errorf("expected key reason claim_key, got %q", report.Contradictions[0].KeyReason)
	}
}

func TestRunHygieneAgesOutSupersededFile(t *testing.T) {
	mem, store := newTestMemory(t)
	cons := NewConsolidator(nil, "", mem)

	now := time.Date(2026, 5, 15, 10, 0, 0, 0, time.UTC)
	old := now.AddDate(0, 0, -45).Format(time.RFC3339)
	writeMemoryFile(t, store, brain.MemoryGlobalTopic("old_fact"),
		map[string]string{
			"name":          "Old",
			"modified":      old,
			"superseded_by": "new_fact.md",
		}, "old body")

	report, err := cons.RunHygiene(context.Background(), HygieneOptions{
		Apply:          true,
		RetiredAgeDays: 30,
		Now:            now,
	})
	if err != nil {
		t.Fatalf("RunHygiene: %v", err)
	}
	if len(report.AgingRetired) != 1 {
		t.Fatalf("expected 1 aging-retired, got %d", len(report.AgingRetired))
	}

	target := brain.MemoryGlobalTopic("old_fact")
	if ok, _ := store.Exists(context.Background(), target); !ok {
		t.Fatalf("expected old file to remain at %s", target)
	}
	data, err := store.Read(context.Background(), target)
	if err != nil {
		t.Fatalf("read after retire: %v", err)
	}
	content := string(data)
	if !strings.Contains(content, "retired: true") {
		t.Errorf("expected retired: true frontmatter, got:\n%s", data)
	}
	if !strings.Contains(content, "retired_on: 2026-05-15") {
		t.Errorf("expected pinned retired_on date, got:\n%s", data)
	}
}

func TestRunHygieneRetainsRecentSupersededFile(t *testing.T) {
	mem, store := newTestMemory(t)
	cons := NewConsolidator(nil, "", mem)

	now := time.Date(2026, 5, 15, 10, 0, 0, 0, time.UTC)
	recent := now.AddDate(0, 0, -5).Format(time.RFC3339)
	writeMemoryFile(t, store, brain.MemoryGlobalTopic("recent_super"),
		map[string]string{
			"name":          "Recent",
			"modified":      recent,
			"superseded_by": "new.md",
		}, "body")

	report, err := cons.RunHygiene(context.Background(), HygieneOptions{
		Apply:          true,
		RetiredAgeDays: 30,
		Now:            now,
	})
	if err != nil {
		t.Fatalf("RunHygiene: %v", err)
	}
	if len(report.AgingRetired) != 0 {
		t.Errorf("expected no aging retirements, got %d", len(report.AgingRetired))
	}
}

func TestRunHygieneIgnoresAlreadyRetired(t *testing.T) {
	mem, store := newTestMemory(t)
	cons := NewConsolidator(nil, "", mem)

	now := time.Date(2026, 5, 15, 10, 0, 0, 0, time.UTC).Format(time.RFC3339)
	writeMemoryFile(t, store, brain.MemoryGlobalTopic("retired_a"),
		map[string]string{
			"name":     "Same",
			"modified": now,
			"retired":  "true",
		}, "")
	writeMemoryFile(t, store, brain.MemoryGlobalTopic("live_b"),
		map[string]string{
			"name":     "Same",
			"modified": now,
		}, "")

	report, err := cons.RunHygiene(context.Background(), HygieneOptions{})
	if err != nil {
		t.Fatalf("RunHygiene: %v", err)
	}
	if len(report.Contradictions) != 0 {
		t.Errorf("retired file should not contribute to contradictions, got %d groups", len(report.Contradictions))
	}
}

func TestPickCanonicalConfidenceWinsOverRecency(t *testing.T) {
	older := time.Date(2026, 5, 15, 8, 0, 0, 0, time.UTC).Format(time.RFC3339)
	newer := time.Date(2026, 5, 15, 10, 0, 0, 0, time.UTC).Format(time.RFC3339)

	got := pickCanonical([]TopicFile{
		{Path: "memory/global/low.md", Confidence: "low", Modified: newer},
		{Path: "memory/global/high.md", Confidence: "high", Modified: older},
	})
	if got.Path != "memory/global/high.md" {
		t.Errorf("expected high-confidence winner, got %s", got.Path)
	}
}

func TestPickCanonicalRecencyAsTieBreaker(t *testing.T) {
	older := time.Date(2026, 5, 15, 8, 0, 0, 0, time.UTC).Format(time.RFC3339)
	newer := time.Date(2026, 5, 15, 10, 0, 0, 0, time.UTC).Format(time.RFC3339)

	got := pickCanonical([]TopicFile{
		{Path: "memory/global/older.md", Confidence: "medium", Modified: older},
		{Path: "memory/global/newer.md", Confidence: "medium", Modified: newer},
	})
	if got.Path != "memory/global/newer.md" {
		t.Errorf("expected newer winner on tie, got %s", got.Path)
	}
}

func TestRunHygieneDistinctSubjectsWithSameStateKeyAreNotContradictions(t *testing.T) {
	mem, store := newTestMemory(t)
	cons := NewConsolidator(nil, "", mem)

	now := time.Date(2026, 5, 15, 10, 0, 0, 0, time.UTC).Format(time.RFC3339)
	writeMemoryFile(t, store, brain.MemoryGlobalTopic("alex"),
		map[string]string{"name": "Alex context", "modified": now, "state_key": "state.owned.item.set.context", "state_subject": "alex"},
		"alex body")
	writeMemoryFile(t, store, brain.MemoryGlobalTopic("boudewijn"),
		map[string]string{"name": "Boudewijn context", "modified": now, "state_key": "state.owned.item.set.context", "state_subject": "boudewijn"},
		"boudewijn body")

	report, err := cons.RunHygiene(context.Background(), HygieneOptions{})
	if err != nil {
		t.Fatalf("RunHygiene: %v", err)
	}
	if len(report.Contradictions) != 0 {
		t.Errorf("distinct subjects sharing a schema should not contradict, got %d groups", len(report.Contradictions))
	}
}
