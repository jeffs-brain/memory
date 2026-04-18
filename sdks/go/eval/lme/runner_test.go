// SPDX-License-Identifier: Apache-2.0

package lme

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestRun_EndToEnd(t *testing.T) {
	questions := []Question{
		{
			ID:               "q1",
			Category:         "single-session",
			Question:         "What colour was the car?",
			Answer:           "red",
			SessionIDs:       []string{"sess-001"},
			HaystackSessions: [][]SessionMessage{{{Role: "user", Content: "The car was red and parked outside the building."}}},
		},
		{
			ID:               "q2",
			Category:         "single-session",
			Question:         "What was the dog's name?",
			Answer:           "Rex",
			SessionIDs:       []string{"sess-002"},
			HaystackSessions: [][]SessionMessage{{{Role: "user", Content: "The dog called Rex was friendly and well-trained."}}},
		},
		{
			ID:               "q3",
			Category:         "temporal",
			Question:         "When did the meeting happen?",
			Answer:           "Tuesday",
			SessionIDs:       []string{"sess-003"},
			HaystackSessions: [][]SessionMessage{{{Role: "user", Content: "The meeting happened on Tuesday morning at 9am."}}},
		},
	}

	dir := t.TempDir()
	dsPath := filepath.Join(dir, "test-dataset.json")
	data, err := json.Marshal(questions)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if err := os.WriteFile(dsPath, data, 0o644); err != nil {
		t.Fatalf("write: %v", err)
	}

	result, err := Run(context.Background(), RunConfig{
		DatasetPath: dsPath,
		Seed:        42,
	})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}

	if result.QuestionsRun != 3 {
		t.Errorf("QuestionsRun = %d, want 3", result.QuestionsRun)
	}
	if result.IngestMode != "bulk" {
		t.Errorf("IngestMode = %q, want %q", result.IngestMode, "bulk")
	}

	if result.OverallScore == 0 {
		t.Errorf("OverallScore = 0, expected some correct answers")
	}
	if len(result.ByCategory) == 0 {
		t.Error("ByCategory is empty")
	}
}

func TestRun_WithSampling(t *testing.T) {
	questions := []Question{
		{ID: "q1", Category: "single-session", Question: "Q1?", Answer: "A1", SessionIDs: []string{"s1"}, HaystackSessions: [][]SessionMessage{{{Role: "user", Content: "A1 content"}}}},
		{ID: "q2", Category: "single-session", Question: "Q2?", Answer: "A2", SessionIDs: []string{"s2"}, HaystackSessions: [][]SessionMessage{{{Role: "user", Content: "A2 content"}}}},
		{ID: "q3", Category: "temporal", Question: "Q3?", Answer: "A3", SessionIDs: []string{"s3"}, HaystackSessions: [][]SessionMessage{{{Role: "user", Content: "A3 content"}}}},
		{ID: "q4", Category: "temporal", Question: "Q4?", Answer: "A4", SessionIDs: []string{"s4"}, HaystackSessions: [][]SessionMessage{{{Role: "user", Content: "A4 content"}}}},
	}

	dir := t.TempDir()
	dsPath := filepath.Join(dir, "test-dataset.json")
	data, _ := json.Marshal(questions)
	_ = os.WriteFile(dsPath, data, 0o644)

	result, err := Run(context.Background(), RunConfig{
		DatasetPath: dsPath,
		SampleSize:  2,
		Seed:        42,
	})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}

	if result.QuestionsRun != 2 {
		t.Errorf("QuestionsRun = %d, want 2", result.QuestionsRun)
	}
}

func TestRun_SHAVerification(t *testing.T) {
	questions := []Question{
		{ID: "q1", Category: "single-session", Question: "Q?", Answer: "A", SessionIDs: []string{"s1"}, HaystackSessions: [][]SessionMessage{{{Role: "user", Content: "content"}}}},
	}
	dir := t.TempDir()
	dsPath := filepath.Join(dir, "test.json")
	data, _ := json.Marshal(questions)
	_ = os.WriteFile(dsPath, data, 0o644)

	_, err := Run(context.Background(), RunConfig{
		DatasetPath: dsPath,
		ExpectedSHA: "wrong-sha",
	})
	if err == nil {
		t.Fatal("expected SHA mismatch error")
	}
}

func TestRun_MissingDataset(t *testing.T) {
	_, err := Run(context.Background(), RunConfig{
		DatasetPath: "/nonexistent/path.json",
	})
	if err == nil {
		t.Fatal("expected error for missing dataset")
	}
}
