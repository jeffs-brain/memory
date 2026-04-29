// SPDX-License-Identifier: Apache-2.0

package lme

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func fixture() []Question {
	return []Question{
		{
			ID:         "q1",
			Category:   "single-session",
			Question:   "What colour was the car?",
			Answer:     "red",
			SessionIDs: []string{"s1"},
		},
		{
			ID:         "q2",
			Category:   "single-session",
			Question:   "Where did they meet?",
			Answer:     "London",
			SessionIDs: []string{"s1", "s2"},
		},
		{
			ID:               "q3",
			Category:         "temporal",
			Question:         "When was the appointment moved to?",
			Answer:           "Tuesday",
			SessionIDs:       []string{"s3"},
			HaystackSessions: [][]SessionMessage{{{Role: "user", Content: "The appointment was rescheduled to Tuesday."}}},
		},
	}
}

func writeFixture(t *testing.T, qs []Question) string {
	t.Helper()
	data, err := json.Marshal(qs)
	if err != nil {
		t.Fatalf("marshal fixture: %v", err)
	}
	path := filepath.Join(t.TempDir(), "lme.json")
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatalf("write fixture: %v", err)
	}
	return path
}

func TestDataset_LoadAndParse(t *testing.T) {
	path := writeFixture(t, fixture())
	ds, err := LoadDataset(path)
	if err != nil {
		t.Fatalf("LoadDataset: %v", err)
	}

	if len(ds.Questions) != 3 {
		t.Fatalf("Questions: got %d, want 3", len(ds.Questions))
	}

	q := ds.Questions[0]
	if q.ID != "q1" {
		t.Errorf("ID = %q, want %q", q.ID, "q1")
	}
	if q.Category != "single-session" {
		t.Errorf("Category = %q, want %q", q.Category, "single-session")
	}
	if q.Question != "What colour was the car?" {
		t.Errorf("Question = %q, want %q", q.Question, "What colour was the car?")
	}
	if q.Answer != "red" {
		t.Errorf("Answer = %q, want %q", q.Answer, "red")
	}

	if ds.Questions[2].HaystackText() == "" {
		t.Error("HaystackText on q3 should be populated")
	}

	if len(ds.Categories) != 2 {
		t.Fatalf("Categories: got %d, want 2", len(ds.Categories))
	}
	if ds.Categories[0] != "single-session" || ds.Categories[1] != "temporal" {
		t.Errorf("Categories = %v, want [single-session temporal]", ds.Categories)
	}
}

func TestDataset_SHA256Stable(t *testing.T) {
	qs := fixture()
	path := writeFixture(t, qs)

	ds1, err := LoadDataset(path)
	if err != nil {
		t.Fatalf("first load: %v", err)
	}

	ds2, err := LoadDataset(path)
	if err != nil {
		t.Fatalf("second load: %v", err)
	}

	if ds1.SHA256 != ds2.SHA256 {
		t.Errorf("SHA256 not stable: %s vs %s", ds1.SHA256, ds2.SHA256)
	}
	if ds1.SHA256 == "" {
		t.Error("SHA256 should not be empty")
	}
}

func TestDataset_ByCategory(t *testing.T) {
	path := writeFixture(t, fixture())
	ds, err := LoadDataset(path)
	if err != nil {
		t.Fatalf("LoadDataset: %v", err)
	}

	bc := ds.ByCategory()
	if len(bc) != 2 {
		t.Fatalf("ByCategory: got %d groups, want 2", len(bc))
	}
	if len(bc["single-session"]) != 2 {
		t.Errorf("single-session: got %d, want 2", len(bc["single-session"]))
	}
	if len(bc["temporal"]) != 1 {
		t.Errorf("temporal: got %d, want 1", len(bc["temporal"]))
	}
}

func TestDataset_SampleStratified(t *testing.T) {
	path := writeFixture(t, fixture())
	ds, err := LoadDataset(path)
	if err != nil {
		t.Fatalf("LoadDataset: %v", err)
	}

	sampled := ds.Sample(2, 42)
	if len(sampled) != 2 {
		t.Fatalf("Sample(2): got %d, want 2", len(sampled))
	}

	cats := make(map[string]int)
	for _, q := range sampled {
		cats[q.Category]++
	}
	if cats["single-session"] == 0 {
		t.Error("expected at least one single-session question in sample")
	}
	if cats["temporal"] == 0 {
		t.Error("expected at least one temporal question in sample")
	}
}

func TestDataset_SampleDeterministic(t *testing.T) {
	path := writeFixture(t, fixture())
	ds, err := LoadDataset(path)
	if err != nil {
		t.Fatalf("LoadDataset: %v", err)
	}

	s1 := ds.Sample(2, 99)
	s2 := ds.Sample(2, 99)

	if len(s1) != len(s2) {
		t.Fatalf("lengths differ: %d vs %d", len(s1), len(s2))
	}
	for i := range s1 {
		if s1[i].ID != s2[i].ID {
			t.Errorf("index %d: ID %q vs %q", i, s1[i].ID, s2[i].ID)
		}
	}
}

func TestDataset_SampleAllWhenNLarge(t *testing.T) {
	path := writeFixture(t, fixture())
	ds, err := LoadDataset(path)
	if err != nil {
		t.Fatalf("LoadDataset: %v", err)
	}

	sampled := ds.Sample(100, 1)
	if len(sampled) != len(ds.Questions) {
		t.Errorf("Sample(100): got %d, want %d", len(sampled), len(ds.Questions))
	}
}

func TestDataset_VerifySHA(t *testing.T) {
	path := writeFixture(t, fixture())
	ds, err := LoadDataset(path)
	if err != nil {
		t.Fatalf("LoadDataset: %v", err)
	}

	if err := ds.VerifySHA(ds.SHA256); err != nil {
		t.Errorf("VerifySHA with correct hash: %v", err)
	}

	err = ds.VerifySHA("0000000000000000000000000000000000000000000000000000000000000000")
	if err == nil {
		t.Fatal("VerifySHA should reject wrong hash")
	}
	if !strings.Contains(err.Error(), "mismatch") {
		t.Errorf("error should mention mismatch: %v", err)
	}
}

func TestDataset_ValidationEmptyQuestions(t *testing.T) {
	path := writeFixture(t, []Question{})

	_, err := LoadDataset(path)
	if err == nil {
		t.Fatal("expected error for empty dataset")
	}
	if !strings.Contains(err.Error(), "no questions") {
		t.Errorf("error should mention no questions: %v", err)
	}
}

func TestDataset_ValidationMissingID(t *testing.T) {
	qs := []Question{{
		Category: "single-session",
		Question: "What?",
		Answer:   "Something",
	}}
	path := writeFixture(t, qs)

	_, err := LoadDataset(path)
	if err == nil {
		t.Fatal("expected error for missing ID")
	}
	if !strings.Contains(err.Error(), "empty ID") {
		t.Errorf("error should mention empty ID: %v", err)
	}
}

func TestDataset_ValidationMissingCategory(t *testing.T) {
	qs := []Question{{
		ID:       "q1",
		Question: "What?",
		Answer:   "Something",
	}}
	path := writeFixture(t, qs)

	_, err := LoadDataset(path)
	if err == nil {
		t.Fatal("expected error for missing category")
	}
	if !strings.Contains(err.Error(), "empty category") {
		t.Errorf("error should mention empty category: %v", err)
	}
}

func TestDataset_ValidationMissingQuestion(t *testing.T) {
	qs := []Question{{
		ID:       "q1",
		Category: "temporal",
		Answer:   "Something",
	}}
	path := writeFixture(t, qs)

	_, err := LoadDataset(path)
	if err == nil {
		t.Fatal("expected error for missing question text")
	}
	if !strings.Contains(err.Error(), "empty question text") {
		t.Errorf("error should mention empty question text: %v", err)
	}
}

func TestDataset_ValidationMissingAnswer(t *testing.T) {
	qs := []Question{{
		ID:       "q1",
		Category: "temporal",
		Question: "When?",
	}}
	path := writeFixture(t, qs)

	_, err := LoadDataset(path)
	if err == nil {
		t.Fatal("expected error for missing answer")
	}
	if !strings.Contains(err.Error(), "empty answer") {
		t.Errorf("error should mention empty answer: %v", err)
	}
}

func TestDataset_LoadMissingFile(t *testing.T) {
	_, err := LoadDataset(filepath.Join(t.TempDir(), "nonexistent.json"))
	if err == nil {
		t.Fatal("expected error for missing file")
	}
	if !strings.Contains(err.Error(), "read dataset") {
		t.Errorf("error should mention read dataset: %v", err)
	}
}

func TestDataset_LoadBadJSON(t *testing.T) {
	path := filepath.Join(t.TempDir(), "bad.json")
	if err := os.WriteFile(path, []byte("{not json array}"), 0o644); err != nil {
		t.Fatalf("write: %v", err)
	}

	_, err := LoadDataset(path)
	if err == nil {
		t.Fatal("expected error for bad JSON")
	}
	if !strings.Contains(err.Error(), "parse dataset") {
		t.Errorf("error should mention parse dataset: %v", err)
	}
}
