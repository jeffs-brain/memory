// SPDX-License-Identifier: Apache-2.0

package lme

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestInferDatasetBenchmarkMode(t *testing.T) {
	t.Parallel()

	ds := &Dataset{
		Questions: []Question{
			{
				ID:               "q1",
				Category:         "multi-session",
				Question:         "Where did we go after lunch?",
				Answer:           "the park",
				SessionIDs:       []string{"sess-001", "sess-002"},
				AnswerSessionIDs: []string{"sess-002"},
			},
		},
	}

	if got := inferDatasetBenchmarkMode(ds); got != BenchmarkModeFullContext {
		t.Fatalf("inferDatasetBenchmarkMode() = %q, want %q", got, BenchmarkModeFullContext)
	}
}

func TestRun_BenchmarkModeMismatchFails(t *testing.T) {
	t.Parallel()

	dsPath := writeBenchmarkDataset(t, []map[string]any{
		{
			"question_id":          "q1",
			"question_type":        "multi-session",
			"question":             "Where did we go after lunch?",
			"answer":               "the park",
			"haystack_session_ids": []string{"sess-001", "sess-002"},
			"answer_session_ids":   []string{"sess-002"},
			"haystack_sessions": [][]map[string]any{
				{{"role": "user", "content": "We had lunch in town."}},
				{{"role": "assistant", "content": "After lunch we went to the park."}},
			},
		},
	})

	_, err := Run(context.Background(), RunConfig{
		DatasetPath:       dsPath,
		BenchmarkMode:     BenchmarkModeOracle,
		Concurrency:       1,
		ReplayConcurrency: 1,
	})
	if err == nil {
		t.Fatal("expected benchmark mode mismatch error")
	}
	if !strings.Contains(err.Error(), "does not match dataset context") {
		t.Fatalf("error should mention dataset context mismatch, got: %v", err)
	}
}

func TestRun_RealRetrievalBenchmarkModeRequiresActorEndpoint(t *testing.T) {
	t.Parallel()

	dsPath := writeBenchmarkDataset(t, []map[string]any{
		{
			"question_id":          "q1",
			"question_type":        "single-session",
			"question":             "What colour was the car?",
			"answer":               "red",
			"haystack_session_ids": []string{"sess-001"},
			"haystack_sessions": [][]map[string]any{
				{{"role": "user", "content": "The car was red."}},
			},
		},
	})

	_, err := Run(context.Background(), RunConfig{
		DatasetPath:       dsPath,
		BenchmarkMode:     BenchmarkModeRealRetrieval,
		Concurrency:       1,
		ReplayConcurrency: 1,
	})
	if err == nil {
		t.Fatal("expected missing actor endpoint error")
	}
	if !strings.Contains(err.Error(), "--actor-endpoint") {
		t.Fatalf("error should mention actor-endpoint, got: %v", err)
	}
}

func TestInferBenchmarkSpec_FullActorStyleIsNotLabelledRealRetrieval(t *testing.T) {
	t.Parallel()

	spec, err := inferBenchmarkSpec(nil, RunConfig{
		ActorEndpoint:      "http://127.0.0.1:18850",
		ActorEndpointStyle: "full",
	})
	if err != nil {
		t.Fatalf("inferBenchmarkSpec: %v", err)
	}
	if spec.Mode != BenchmarkModeDaemonRead {
		t.Fatalf("Mode = %q, want %q", spec.Mode, BenchmarkModeDaemonRead)
	}
	if spec.ContextSource != ContextSourceActorAsk {
		t.Fatalf("ContextSource = %q, want %q", spec.ContextSource, ContextSourceActorAsk)
	}
}

func TestInferBenchmarkSpec_AgenticModeUsesDedicatedLabel(t *testing.T) {
	t.Parallel()

	spec, err := inferBenchmarkSpec(nil, RunConfig{AgenticMode: true})
	if err != nil {
		t.Fatalf("inferBenchmarkSpec: %v", err)
	}
	if spec.Mode != BenchmarkModeAgentic {
		t.Fatalf("Mode = %q, want %q", spec.Mode, BenchmarkModeAgentic)
	}
	if spec.ContextSource != ContextSourceAgenticSearch {
		t.Fatalf("ContextSource = %q, want %q", spec.ContextSource, ContextSourceAgenticSearch)
	}
}

func TestSelectQuestionsByID_PreservesExplicitSampleOrder(t *testing.T) {
	t.Parallel()

	selected, err := selectQuestionsByID([]Question{
		{ID: "q1"},
		{ID: "q2"},
		{ID: "q3"},
	}, []string{"q3", "q1"})
	if err != nil {
		t.Fatalf("selectQuestionsByID: %v", err)
	}
	if len(selected) != 2 {
		t.Fatalf("len(selected) = %d, want 2", len(selected))
	}
	if selected[0].ID != "q3" || selected[1].ID != "q1" {
		t.Fatalf("selected order = [%s %s], want [q3 q1]", selected[0].ID, selected[1].ID)
	}
}

func TestSelectQuestionsByID_DuplicateSampleIDsFail(t *testing.T) {
	t.Parallel()

	_, err := selectQuestionsByID([]Question{
		{ID: "q1"},
		{ID: "q2"},
	}, []string{"q1", "q1"})
	if err == nil {
		t.Fatal("expected duplicate sample id error")
	}
	if !strings.Contains(err.Error(), "duplicate sample id") {
		t.Fatalf("error should mention duplicate sample id, got: %v", err)
	}
}

func writeBenchmarkDataset(t *testing.T, questions []map[string]any) string {
	t.Helper()

	dir := t.TempDir()
	path := filepath.Join(dir, "dataset.json")
	raw, err := json.Marshal(questions)
	if err != nil {
		t.Fatalf("marshal dataset: %v", err)
	}
	if err := os.WriteFile(path, raw, 0o644); err != nil {
		t.Fatalf("write dataset: %v", err)
	}
	return path
}
