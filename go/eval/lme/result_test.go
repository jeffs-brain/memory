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
		SampleIDs:       []string{"q1"},
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
				RetrievalDiagnostics: &RetrievalDiagnostics{
					Request: RetrievalRequestDiagnostics{
						EndpointStyle: "retrieve-only",
						BrainID:       "eval-lme",
						Mode:          "hybrid-rerank",
						TopK:          20,
						QueryHash:     "abc",
						QueryPreview:  "What colour was the car?",
					},
					Response: RetrievalResponseDiagnostics{HTTPStatus: 200, TookMs: 12},
					Evidence: RetrievalEvidenceSummary{
						ReturnedCount:    1,
						RenderedBytes:    128,
						ApproxTokens:     24,
						UniquePaths:      1,
						UniqueSessionIDs: 1,
					},
					Returned: []RetrievedPassageDiagnostic{
						{
							Rank:         1,
							Path:         "memory/project/eval-lme/fact.md",
							SessionID:    "s1",
							Date:         "2024-03-01",
							Score:        0.8,
							TextSHA256:   "def",
							Preview:      "The car was red.",
							ApproxTokens: 4,
						},
					},
					Trace: &RetrievalTraceDiagnostics{
						RequestedMode: "hybrid-rerank",
						EffectiveMode: "hybrid-rerank",
						Reranked:      true,
					},
					Attempts: []RetrievalAttemptDiagnostic{
						{Rung: 0, Mode: "hybrid-rerank", TopK: 20, Chunks: 1, QueryHash: "abc"},
					},
				},
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
	if len(got.SampleIDs) != 1 || got.SampleIDs[0] != "q1" {
		t.Errorf("SampleIDs = %#v, want q1", got.SampleIDs)
	}
	if got.Questions[0].Question != "What colour was the car?" {
		t.Errorf("Question = %q, want %q", got.Questions[0].Question, "What colour was the car?")
	}
	if got.Questions[0].RetrievalDiagnostics == nil {
		t.Fatal("RetrievalDiagnostics = nil, want populated")
	}
	if got.Questions[0].RetrievalDiagnostics.Evidence.ReturnedCount != 1 {
		t.Errorf("diagnostic returned count = %d, want 1", got.Questions[0].RetrievalDiagnostics.Evidence.ReturnedCount)
	}
	if got.Questions[0].RetrievalDiagnostics.Returned[0].Preview != "The car was red." {
		t.Errorf("diagnostic preview = %q", got.Questions[0].RetrievalDiagnostics.Returned[0].Preview)
	}
}
