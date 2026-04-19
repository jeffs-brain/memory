// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/jeffs-brain/memory/go/eval/lme"
	"github.com/jeffs-brain/memory/go/retrieval"
)

// TestEvalLmeRunFlags_HelpAdvertisesNewFlags verifies the new flags show
// up in the help text so operators and automation can discover them.
func TestEvalLmeRunFlags_HelpAdvertisesNewFlags(t *testing.T) {
	cmd := rootCmd()
	var buf bytes.Buffer
	cmd.SetArgs([]string{"eval", "lme", "run", "--help"})
	cmd.SetOut(&buf)
	cmd.SetErr(&buf)
	if err := cmd.Execute(); err != nil {
		t.Fatalf("help: %v", err)
	}
	out := buf.String()
	for _, flag := range []string{
		"--benchmark-mode",
		"--extract-only",
		"--brain-cache",
		"--actor-endpoint",
		"--actor-brain",
		"--retrieval-mode",
		"--actor-scope",
		"--actor-project",
		"--actor-path-prefix",
		"--replay-concurrency",
		"--concurrency",
		"--judge-cache-dir",
	} {
		if !strings.Contains(out, flag) {
			t.Errorf("help output missing flag %q", flag)
		}
	}
}

func TestEvalLmeRunFlags_RealRetrievalRequiresActorEndpoint(t *testing.T) {
	dir := t.TempDir()
	dsPath := writeTinyDataset(t, filepath.Join(dir, "ds.json"))

	cmd := rootCmd()
	cmd.SetArgs([]string{
		"eval", "lme", "run",
		"--dataset", dsPath,
		"--benchmark-mode", "real-retrieval",
		"--ingest-mode", "none",
		"--judge", "",
		"--no-reader",
	})
	cmd.SetOut(io.Discard)
	cmd.SetErr(io.Discard)

	err := cmd.Execute()
	if err == nil {
		t.Fatal("expected error when real-retrieval runs without an actor endpoint")
	}
	if !strings.Contains(err.Error(), "actor-endpoint") {
		t.Fatalf("error should mention actor-endpoint, got: %v", err)
	}
}

// TestEvalLmeRunFlags_ExtractOnlyRequiresBrainCache verifies the guard
// that refuses --extract-only without --brain-cache, so operators never
// run a throwaway extraction that a downstream daemon cannot attach to.
func TestEvalLmeRunFlags_ExtractOnlyRequiresBrainCache(t *testing.T) {
	dir := t.TempDir()
	dsPath := writeTinyDataset(t, filepath.Join(dir, "ds.json"))

	cmd := rootCmd()
	cmd.SetArgs([]string{
		"eval", "lme", "run",
		"--dataset", dsPath,
		"--extract-only",
		"--ingest-mode", "bulk",
		"--judge", "",
		"--no-reader",
	})
	cmd.SetOut(io.Discard)
	cmd.SetErr(io.Discard)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	cmd.SetContext(ctx)

	err := cmd.Execute()
	if err == nil {
		t.Fatal("expected error when --extract-only is set without --brain-cache")
	}
	if !strings.Contains(err.Error(), "brain-cache") {
		t.Fatalf("error should mention brain-cache, got: %v", err)
	}
}

// TestEvalLmeRunFlags_ExtractOnlyShortCircuits verifies an extract-only
// bulk run exits before the judge phase and leaves a populated brain
// cache on disk plus a manifest.
func TestEvalLmeRunFlags_ExtractOnlyShortCircuits(t *testing.T) {
	dir := t.TempDir()
	dsPath := writeTinyDataset(t, filepath.Join(dir, "ds.json"))
	cachePath := filepath.Join(dir, "brain")
	outPath := filepath.Join(dir, "out.json")
	manifestPath := filepath.Join(dir, "manifest.json")

	cmd := rootCmd()
	cmd.SetArgs([]string{
		"eval", "lme", "run",
		"--dataset", dsPath,
		"--extract-only",
		"--brain-cache", cachePath,
		"--ingest-mode", "bulk",
		"--judge", "",
		"--no-reader",
		"--output", outPath,
		"--manifest", manifestPath,
	})
	cmd.SetOut(io.Discard)
	cmd.SetErr(io.Discard)

	if err := cmd.Execute(); err != nil {
		t.Fatalf("execute: %v", err)
	}

	// The brain cache directory should exist.
	if _, err := os.Stat(cachePath); err != nil {
		t.Fatalf("brain cache missing: %v", err)
	}

	// The ingest-only result JSON should carry QuestionsRun but no
	// per-question trace.
	raw, err := os.ReadFile(outPath)
	if err != nil {
		t.Fatalf("read output: %v", err)
	}
	var result struct {
		QuestionsRun int `json:"questions_run"`
		Questions    []struct {
			ID string `json:"id"`
		} `json:"questions"`
	}
	if err := json.Unmarshal(raw, &result); err != nil {
		t.Fatalf("decode output: %v", err)
	}
	if result.QuestionsRun != 2 {
		t.Fatalf("QuestionsRun = %d, want 2", result.QuestionsRun)
	}
	if len(result.Questions) != 0 {
		t.Fatalf("Questions = %d, want 0 in extract-only mode", len(result.Questions))
	}

	// Manifest should record the ingest mode.
	manifestBytes, err := os.ReadFile(manifestPath)
	if err != nil {
		t.Fatalf("read manifest: %v", err)
	}
	var manifest map[string]any
	if err := json.Unmarshal(manifestBytes, &manifest); err != nil {
		t.Fatalf("decode manifest: %v", err)
	}
	if got, _ := manifest["ingest_mode"].(string); got != "bulk" {
		t.Fatalf("manifest ingest_mode = %q, want bulk", got)
	}
	if got, _ := manifest["benchmark_mode"].(string); got != lme.BenchmarkModeExtractPrep {
		t.Fatalf("manifest benchmark_mode = %q, want %q", got, lme.BenchmarkModeExtractPrep)
	}
	if got, _ := manifest["context_source"].(string); got != lme.ContextSourceExtractPrep {
		t.Fatalf("manifest context_source = %q, want %q", got, lme.ContextSourceExtractPrep)
	}
}

// TestEvalLmeRunFlags_ActorEndpointFlag verifies --actor-endpoint is
// parsed and forwarded to the runner config. Using a non-routable URL
// the call still sets up the config and then errors on the HTTP call,
// which is enough to prove wiring.
func TestEvalLmeRunFlags_ActorEndpointFlag(t *testing.T) {
	dir := t.TempDir()
	dsPath := writeTinyDataset(t, filepath.Join(dir, "ds.json"))
	outPath := filepath.Join(dir, "out.json")

	cmd := rootCmd()
	cmd.SetArgs([]string{
		"eval", "lme", "run",
		"--dataset", dsPath,
		"--ingest-mode", "none",
		"--actor-endpoint", "http://127.0.0.1:1",
		"--actor-brain", "eval-lme",
		"--judge", "",
		"--no-reader",
		"--concurrency", "2",
		"--output", outPath,
	})
	cmd.SetOut(io.Discard)
	cmd.SetErr(io.Discard)

	// The call is expected to fail softly because the actor endpoint is
	// unreachable; individual question outcomes will carry Error
	// strings but the run as a whole should return a result.
	err := cmd.Execute()
	if err != nil {
		t.Fatalf("execute: %v", err)
	}
	raw, err := os.ReadFile(outPath)
	if err != nil {
		t.Fatalf("read output: %v", err)
	}
	var result struct {
		QuestionsRun int `json:"questions_run"`
		Questions    []struct {
			ID    string `json:"id"`
			Error string `json:"error"`
		} `json:"questions"`
	}
	if err := json.Unmarshal(raw, &result); err != nil {
		t.Fatalf("decode output: %v", err)
	}
	if result.QuestionsRun == 0 {
		t.Fatal("expected questions_run > 0")
	}
	hasError := false
	for _, q := range result.Questions {
		if q.Error != "" {
			hasError = true
			break
		}
	}
	if !hasError {
		t.Fatal("expected at least one question to carry an error from the unreachable actor endpoint")
	}
}

func TestEvalLmeRunFlags_ActorSettingsPersistedInManifest(t *testing.T) {
	dir := t.TempDir()
	dsPath := writeTinyDataset(t, filepath.Join(dir, "ds.json"))
	outPath := filepath.Join(dir, "out.json")
	manifestPath := filepath.Join(dir, "manifest.json")

	cmd := rootCmd()
	cmd.SetArgs([]string{
		"eval", "lme", "run",
		"--dataset", dsPath,
		"--ingest-mode", "none",
		"--actor-endpoint", "http://127.0.0.1:1",
		"--actor-endpoint-style", "retrieve-only",
		"--actor-brain", "eval-lme",
		"--retrieval-mode", "bm25",
		"--actor-topk", "40",
		"--actor-candidatek", "120",
		"--actor-rerank-topn", "80",
		"--actor-scope", "project",
		"--actor-project", "eval-lme",
		"--actor-path-prefix", "memory/project/eval-lme/",
		"--judge", "",
		"--no-reader",
		"--output", outPath,
		"--manifest", manifestPath,
	})
	cmd.SetOut(io.Discard)
	cmd.SetErr(io.Discard)

	if err := cmd.Execute(); err != nil {
		t.Fatalf("execute: %v", err)
	}

	manifest, err := lme.LoadManifest(manifestPath)
	if err != nil {
		t.Fatalf("load manifest: %v", err)
	}
	if manifest.ActorEndpointStyle != "retrieve-only" {
		t.Fatalf("ActorEndpointStyle = %q, want retrieve-only", manifest.ActorEndpointStyle)
	}
	if manifest.BenchmarkMode != lme.BenchmarkModeRealRetrieval {
		t.Fatalf("BenchmarkMode = %q, want %q", manifest.BenchmarkMode, lme.BenchmarkModeRealRetrieval)
	}
	if manifest.ContextSource != lme.ContextSourceActorRetrieve {
		t.Fatalf("ContextSource = %q, want %q", manifest.ContextSource, lme.ContextSourceActorRetrieve)
	}
	if manifest.ActorBrain != "eval-lme" {
		t.Fatalf("ActorBrain = %q, want eval-lme", manifest.ActorBrain)
	}
	if manifest.ActorRetrievalMode != string(retrieval.ModeBM25) {
		t.Fatalf("ActorRetrievalMode = %q, want %q", manifest.ActorRetrievalMode, retrieval.ModeBM25)
	}
	if manifest.ActorTopK == nil || *manifest.ActorTopK != 40 {
		t.Fatalf("ActorTopK = %v, want 40", manifest.ActorTopK)
	}
	if manifest.ActorCandidateK == nil || *manifest.ActorCandidateK != 120 {
		t.Fatalf("ActorCandidateK = %v, want 120", manifest.ActorCandidateK)
	}
	if manifest.ActorRerankTopN == nil || *manifest.ActorRerankTopN != 80 {
		t.Fatalf("ActorRerankTopN = %v, want 80", manifest.ActorRerankTopN)
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
}

// writeTinyDataset writes a two-question fixture to path and returns
// the path. Shared by every CLI-level test in this file.
func writeTinyDataset(t *testing.T, path string) string {
	t.Helper()
	questions := []map[string]any{
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
		{
			"question_id":          "q2",
			"question_type":        "single-session",
			"question":             "What was the dog's name?",
			"answer":               "Rex",
			"haystack_session_ids": []string{"sess-002"},
			"haystack_sessions": [][]map[string]any{
				{{"role": "user", "content": "The dog was called Rex."}},
			},
		},
	}
	data, err := json.Marshal(questions)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatalf("write: %v", err)
	}
	return path
}
