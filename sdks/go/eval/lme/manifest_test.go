// SPDX-License-Identifier: Apache-2.0

package lme

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/jeffs-brain/memory/go/retrieval"
)

func TestBuildRunManifest_IncludesActorSettings(t *testing.T) {
	manifest := BuildRunManifest(
		&LMEResult{
			DatasetSHA: "dataset-sha",
			IngestMode: "none",
		},
		RunConfig{
			Seed:                        7,
			SampleSize:                  5,
			SampleIDs:                   []string{"q1", "q2"},
			ActorEndpoint:               "http://127.0.0.1:18850",
			ActorEndpointStyle:          " retrieve-only ",
			ActorRetrievalMode:          retrieval.ModeBM25,
			ActorTopK:                   0,
			ActorCandidateK:             120,
			ActorRerankTopN:             0,
			ActorFilterQuestionSessions: true,
			ActorFilters: retrieval.Filters{
				Scope:      " project ",
				Project:    " eval-lme ",
				PathPrefix: " memory/project/eval-lme/ ",
			},
		},
		"judge-m",
	)

	if manifest.ActorEndpointStyle != "retrieve-only" {
		t.Fatalf("ActorEndpointStyle = %q, want retrieve-only", manifest.ActorEndpointStyle)
	}
	if manifest.ReaderFailureMode != "question-error" {
		t.Fatalf("ReaderFailureMode = %q, want question-error", manifest.ReaderFailureMode)
	}
	if manifest.JudgeFailureMode != "question-error" {
		t.Fatalf("JudgeFailureMode = %q, want question-error", manifest.JudgeFailureMode)
	}
	if manifest.ReaderModel != "" {
		t.Fatalf("ReaderModel = %q, want empty", manifest.ReaderModel)
	}
	if manifest.ExtractModel != "" {
		t.Fatalf("ExtractModel = %q, want empty", manifest.ExtractModel)
	}
	if manifest.SampleSignature == "" {
		t.Fatal("SampleSignature = empty, want derived signature")
	}
	if manifest.BenchmarkMode != BenchmarkModeRealRetrieval {
		t.Fatalf("BenchmarkMode = %q, want %q", manifest.BenchmarkMode, BenchmarkModeRealRetrieval)
	}
	if manifest.ContextSource != ContextSourceActorRetrieve {
		t.Fatalf("ContextSource = %q, want %q", manifest.ContextSource, ContextSourceActorRetrieve)
	}
	if manifest.ActorBrain != "eval-lme" {
		t.Fatalf("ActorBrain = %q, want eval-lme", manifest.ActorBrain)
	}
	if manifest.ActorRetrievalMode != string(retrieval.ModeBM25) {
		t.Fatalf("ActorRetrievalMode = %q, want %q", manifest.ActorRetrievalMode, retrieval.ModeBM25)
	}
	if manifest.ActorTopK == nil || *manifest.ActorTopK != 20 {
		t.Fatalf("ActorTopK = %v, want 20", manifest.ActorTopK)
	}
	if manifest.ActorCandidateK == nil || *manifest.ActorCandidateK != 120 {
		t.Fatalf("ActorCandidateK = %v, want 120", manifest.ActorCandidateK)
	}
	if manifest.ActorRerankTopN == nil || *manifest.ActorRerankTopN != 0 {
		t.Fatalf("ActorRerankTopN = %v, want 0", manifest.ActorRerankTopN)
	}
	if manifest.ActorScope != "project" {
		t.Fatalf("ActorScope = %q, want project", manifest.ActorScope)
	}
	if manifest.ActorProject != "eval-lme" {
		t.Fatalf("ActorProject = %q, want eval-lme", manifest.ActorProject)
	}
	if manifest.ActorPathPrefix != "memory/project/eval-lme/" {
		t.Fatalf("ActorPathPrefix = %q, want memory/project/eval-lme/", manifest.ActorPathPrefix)
	}
	if !manifest.ActorQuestionSessions {
		t.Fatal("ActorQuestionSessions = false, want true")
	}
}

func TestBuildRunManifest_InfersFullContextFromDataset(t *testing.T) {
	dir := t.TempDir()
	dsPath := filepath.Join(dir, "full-context.json")
	raw, err := json.Marshal([]map[string]any{
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
	if err != nil {
		t.Fatalf("marshal dataset: %v", err)
	}
	if err := os.WriteFile(dsPath, raw, 0o644); err != nil {
		t.Fatalf("write dataset: %v", err)
	}

	manifest := BuildRunManifest(
		&LMEResult{
			DatasetSHA: "dataset-sha",
			IngestMode: "bulk",
		},
		RunConfig{
			DatasetPath: dsPath,
		},
		"judge-m",
	)

	if manifest.BenchmarkMode != BenchmarkModeFullContext {
		t.Fatalf("BenchmarkMode = %q, want %q", manifest.BenchmarkMode, BenchmarkModeFullContext)
	}
	if manifest.ContextSource != ContextSourceDatasetFull {
		t.Fatalf("ContextSource = %q, want %q", manifest.ContextSource, ContextSourceDatasetFull)
	}
}

func TestBuildRunManifest_ExtractOnlyUsesPrepBenchmarkMode(t *testing.T) {
	t.Setenv("JB_EXTRACT_HEURISTICS", "fallback")

	manifest := BuildRunManifest(
		&LMEResult{
			DatasetSHA: "dataset-sha",
			IngestMode: "replay",
			SampleIDs:  []string{"q1", "q2"},
		},
		RunConfig{
			IngestMode:         "replay",
			ReplayExtractModel: "extract-m",
			ReplayConcurrency:  17,
			ExtractOnly:        true,
		},
		"judge-m",
	)

	if manifest.BenchmarkMode != BenchmarkModeExtractPrep {
		t.Fatalf("BenchmarkMode = %q, want %q", manifest.BenchmarkMode, BenchmarkModeExtractPrep)
	}
	if manifest.ContextSource != ContextSourceExtractPrep {
		t.Fatalf("ContextSource = %q, want %q", manifest.ContextSource, ContextSourceExtractPrep)
	}
	if !manifest.ExtractOnly {
		t.Fatal("ExtractOnly = false, want true")
	}
	if manifest.ExtractModel != "extract-m" {
		t.Fatalf("ExtractModel = %q, want extract-m", manifest.ExtractModel)
	}
	if manifest.ExtractHeuristics != "fallback" {
		t.Fatalf("ExtractHeuristics = %q, want fallback", manifest.ExtractHeuristics)
	}
	if manifest.ExtractionPipeline != ReplayExtractionPipelineVersion {
		t.Fatalf("ExtractionPipeline = %d, want %d", manifest.ExtractionPipeline, ReplayExtractionPipelineVersion)
	}
	if manifest.ReplayConcurrency != 17 {
		t.Fatalf("ReplayConcurrency = %d, want 17", manifest.ReplayConcurrency)
	}
	if manifest.CacheSignature == "" {
		t.Fatal("CacheSignature = empty, want populated")
	}
	if manifest.CacheSignatureInputs == nil {
		t.Fatal("CacheSignatureInputs = nil, want populated")
	}
	if manifest.CacheSignatureInputs.SampleSignature == "" {
		t.Fatal("CacheSignatureInputs.SampleSignature = empty, want populated")
	}
	if got := manifest.CacheSignatureInputs.SampleIDs; len(got) != 2 || got[0] != "q1" || got[1] != "q2" {
		t.Fatalf("CacheSignatureInputs.SampleIDs = %#v, want q1/q2", got)
	}
	if manifest.CacheSignatureInputs.ExtractHeuristics != "fallback" {
		t.Fatalf("CacheSignatureInputs.ExtractHeuristics = %q, want fallback", manifest.CacheSignatureInputs.ExtractHeuristics)
	}
}

func TestRunManifest_CacheSignatureChangesForSampleAndIngestConfig(t *testing.T) {
	t.Setenv("JB_EXTRACT_HEURISTICS", "")

	base := BuildRunManifest(
		&LMEResult{
			DatasetSHA: "dataset-sha",
			IngestMode: "replay",
			SampleIDs:  []string{"q1", "q2"},
		},
		RunConfig{
			IngestMode:         "replay",
			ReplayExtractModel: "extract-m",
			ReplayConcurrency:  17,
			ExtractOnly:        true,
		},
		"judge-m",
	)
	if base.CacheSignature == "" {
		t.Fatal("base cache signature is empty")
	}

	changedSample := BuildRunManifest(
		&LMEResult{
			DatasetSHA: "dataset-sha",
			IngestMode: "replay",
			SampleIDs:  []string{"q1", "q3"},
		},
		RunConfig{
			IngestMode:         "replay",
			ReplayExtractModel: "extract-m",
			ReplayConcurrency:  17,
			ExtractOnly:        true,
		},
		"judge-m",
	)
	if base.CacheSignature == changedSample.CacheSignature {
		t.Fatal("expected changed sample IDs to change cache signature")
	}

	changedModel := BuildRunManifest(
		&LMEResult{
			DatasetSHA: "dataset-sha",
			IngestMode: "replay",
			SampleIDs:  []string{"q1", "q2"},
		},
		RunConfig{
			IngestMode:         "replay",
			ReplayExtractModel: "extract-other",
			ReplayConcurrency:  17,
			ExtractOnly:        true,
		},
		"judge-m",
	)
	if base.CacheSignature == changedModel.CacheSignature {
		t.Fatal("expected changed extract model to change cache signature")
	}

	t.Setenv("JB_EXTRACT_HEURISTICS", "fallback")
	changedHeuristics := BuildRunManifest(
		&LMEResult{
			DatasetSHA: "dataset-sha",
			IngestMode: "replay",
			SampleIDs:  []string{"q1", "q2"},
		},
		RunConfig{
			IngestMode:         "replay",
			ReplayExtractModel: "extract-m",
			ReplayConcurrency:  17,
			ExtractOnly:        true,
		},
		"judge-m",
	)
	if base.CacheSignature == changedHeuristics.CacheSignature {
		t.Fatal("expected changed extract heuristics to change cache signature")
	}

	t.Setenv("JB_EXTRACT_HEURISTICS", "")
	changedConcurrency := BuildRunManifest(
		&LMEResult{
			DatasetSHA: "dataset-sha",
			IngestMode: "replay",
			SampleIDs:  []string{"q1", "q2"},
		},
		RunConfig{
			IngestMode:         "replay",
			ReplayExtractModel: "extract-m",
			ReplayConcurrency:  64,
			ExtractOnly:        true,
		},
		"judge-m",
	)
	if base.CacheSignature != changedConcurrency.CacheSignature {
		t.Fatal("expected changed replay concurrency to keep cache signature")
	}
}

func TestBuildRunManifest_FullActorStyleRecordsActualRequestShape(t *testing.T) {
	manifest := BuildRunManifest(
		&LMEResult{
			DatasetSHA: "dataset-sha",
			IngestMode: "none",
		},
		RunConfig{
			ActorEndpoint:      "http://127.0.0.1:18850",
			ActorEndpointStyle: "full",
		},
		"judge-m",
	)

	if manifest.BenchmarkMode != BenchmarkModeDaemonRead {
		t.Fatalf("BenchmarkMode = %q, want %q", manifest.BenchmarkMode, BenchmarkModeDaemonRead)
	}
	if manifest.ContextSource != ContextSourceActorAsk {
		t.Fatalf("ContextSource = %q, want %q", manifest.ContextSource, ContextSourceActorAsk)
	}
	if manifest.ActorRetrievalMode != string(retrieval.ModeHybridRerank) {
		t.Fatalf("ActorRetrievalMode = %q, want %q", manifest.ActorRetrievalMode, retrieval.ModeHybridRerank)
	}
	if manifest.ActorTopK == nil || *manifest.ActorTopK != 5 {
		t.Fatalf("ActorTopK = %v, want 5", manifest.ActorTopK)
	}
	if manifest.ActorCandidateK != nil {
		t.Fatalf("ActorCandidateK = %v, want nil", manifest.ActorCandidateK)
	}
	if manifest.ActorRerankTopN != nil {
		t.Fatalf("ActorRerankTopN = %v, want nil", manifest.ActorRerankTopN)
	}
}

func TestRunManifest_IsComparableIncludesActorSettings(t *testing.T) {
	base := RunManifest{
		DatasetSHA:            "dataset-sha",
		JudgeModel:            "judge-m",
		JudgePromptVersion:    6,
		JudgeFailureMode:      "question-error",
		ReaderFailureMode:     "question-error",
		ReaderModel:           "reader-m",
		ExtractModel:          "extract-m",
		SampleSignature:       "sample-sha",
		Contextualise:         true,
		ExtractOnly:           false,
		BenchmarkMode:         BenchmarkModeRealRetrieval,
		ContextSource:         ContextSourceActorRetrieve,
		ActorEndpointStyle:    "retrieve-only",
		ActorBrain:            "eval-lme",
		ActorRetrievalMode:    string(retrieval.ModeHybridRerank),
		ActorTopK:             intPtr(20),
		ActorCandidateK:       intPtr(100),
		ActorRerankTopN:       intPtr(60),
		ActorScope:            "project",
		ActorProject:          "eval-lme",
		ActorQuestionSessions: true,
	}
	if !base.IsComparable(base) {
		t.Fatal("expected identical manifests to be comparable")
	}

	changed := base
	changed.ActorRetrievalMode = string(retrieval.ModeBM25)
	if base.IsComparable(changed) {
		t.Fatal("expected changed actor settings to break comparability")
	}

	changed = base
	changed.SampleSignature = "other-sample"
	if base.IsComparable(changed) {
		t.Fatal("expected changed sample signature to break comparability")
	}
}
