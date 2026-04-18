// SPDX-License-Identifier: Apache-2.0

package lme

import (
	"encoding/json"
	"testing"
)

func TestLMEResult_RoundTrip(t *testing.T) {
	r := &LMEResult{
		DatasetSHA:      "abc123",
		IngestMode:      "bulk",
		RunSeed:         42,
		QuestionsRun:    100,
		OverallScore:    0.72,
		ExactMatchScore: 0.65,
		ByCategory: map[string]Category{
			"single-session": {Run: 40, Correct: 30, Partial: 5, Incorrect: 5, Score: 0.75},
			"temporal":       {Run: 20, Correct: 14, Partial: 3, Incorrect: 3, Score: 0.70},
		},
		Toggles:        FeatureToggles{},
		CostAccounting: CostAccounting{},
		Questions: []QuestionOutcome{
			{
				ID:          "q1",
				Category:    "single-session",
				Question:    "What colour was the car?",
				GroundTruth: "red",
				AgentAnswer: "red",
			},
		},
	}

	data, err := json.Marshal(r)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}

	var got LMEResult
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	if got.DatasetSHA != r.DatasetSHA {
		t.Errorf("DatasetSHA = %q, want %q", got.DatasetSHA, r.DatasetSHA)
	}
	if got.OverallScore != r.OverallScore {
		t.Errorf("OverallScore = %v, want %v", got.OverallScore, r.OverallScore)
	}
	if len(got.ByCategory) != 2 {
		t.Errorf("ByCategory has %d entries, want 2", len(got.ByCategory))
	}
	if len(got.Questions) != 1 {
		t.Errorf("Questions has %d entries, want 1", len(got.Questions))
	}
	if got.Questions[0].Question != "What colour was the car?" {
		t.Errorf("Question = %q, want %q", got.Questions[0].Question, "What colour was the car?")
	}
}
