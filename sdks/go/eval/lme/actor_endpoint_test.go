// SPDX-License-Identifier: Apache-2.0

package lme

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
)

// TestCallActorEndpoint_StitchesAnswerDeltas verifies the SSE consumer
// concatenates answer_delta frames into a single answer and preserves
// usage totals from the done frame.
func TestCallActorEndpoint_StitchesAnswerDeltas(t *testing.T) {
	t.Parallel()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Errorf("method = %q, want POST", r.Method)
		}
		if !strings.HasPrefix(r.URL.Path, "/v1/brains/") {
			t.Errorf("path = %q, want /v1/brains/...", r.URL.Path)
		}
		if ct := r.Header.Get("Content-Type"); ct != "application/json" {
			t.Errorf("content-type = %q, want application/json", ct)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, ok := w.(http.Flusher)
		if !ok {
			t.Fatal("flusher unavailable")
		}
		frames := []struct {
			event string
			data  string
		}{
			{"retrieve", `{"chunks":[],"topK":5,"mode":"hybrid"}`},
			{"answer_delta", `{"text":"The "}`},
			{"answer_delta", `{"text":"car was "}`},
			{"answer_delta", `{"text":"red."}`},
			{"done", `{"ok":true,"usage":{"input_tokens":42,"output_tokens":8}}`},
		}
		for _, f := range frames {
			fmt.Fprintf(w, "event: %s\n", f.event)
			fmt.Fprintf(w, "data: %s\n\n", f.data)
			flusher.Flush()
		}
	}))
	defer srv.Close()

	answer, usage, err := callActorEndpoint(context.Background(), srv.URL, "eval-lme", "What colour was the car?", "2024-01-01")
	if err != nil {
		t.Fatalf("callActorEndpoint: %v", err)
	}
	if answer != "The car was red." {
		t.Fatalf("answer = %q, want %q", answer, "The car was red.")
	}
	if usage.InputTokens != 42 || usage.OutputTokens != 8 {
		t.Fatalf("usage = %+v, want input=42 output=8", usage)
	}
}

// TestCallActorEndpoint_PropagatesErrorFrames surfaces upstream errors
// that arrive as SSE error events rather than HTTP status codes.
func TestCallActorEndpoint_PropagatesErrorFrames(t *testing.T) {
	t.Parallel()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)
		fmt.Fprint(w, "event: error\ndata: {\"message\":\"upstream blew up\"}\n\n")
		if flusher != nil {
			flusher.Flush()
		}
	}))
	defer srv.Close()

	_, _, err := callActorEndpoint(context.Background(), srv.URL, "eval-lme", "Q", "")
	if err == nil {
		t.Fatal("expected error from SSE error frame")
	}
	if !strings.Contains(err.Error(), "upstream blew up") {
		t.Fatalf("error should carry upstream message, got: %v", err)
	}
}

// TestCallActorEndpoint_RejectsHTTPError short-circuits on 4xx/5xx
// before SSE parsing so misconfigured daemons fail fast.
func TestCallActorEndpoint_RejectsHTTPError(t *testing.T) {
	t.Parallel()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		fmt.Fprint(w, `{"type":"validation_error","detail":"question required"}`)
	}))
	defer srv.Close()

	_, _, err := callActorEndpoint(context.Background(), srv.URL, "eval-lme", "Q", "")
	if err == nil {
		t.Fatal("expected error on HTTP 400")
	}
	if !strings.Contains(err.Error(), "400") {
		t.Fatalf("error should include status code, got: %v", err)
	}
}

// TestRun_ExtractOnlyExitsBeforeJudge verifies Run short-circuits before
// the answer / judge phase when ExtractOnly is set. The result carries
// the ingest-side bookkeeping but no per-question traces.
func TestRun_ExtractOnlyExitsBeforeJudge(t *testing.T) {
	t.Parallel()

	questions := []Question{
		{
			ID:               "q1",
			Category:         "single-session",
			Question:         "Q1?",
			Answer:           "A1",
			SessionIDs:       []string{"s1"},
			HaystackSessions: [][]SessionMessage{{{Role: "user", Content: "A1 content"}}},
		},
		{
			ID:               "q2",
			Category:         "temporal",
			Question:         "Q2?",
			Answer:           "A2",
			SessionIDs:       []string{"s2"},
			HaystackSessions: [][]SessionMessage{{{Role: "user", Content: "A2 content"}}},
		},
	}
	dir := t.TempDir()
	dsPath := dir + "/ds.json"
	data, _ := json.Marshal(questions)
	if err := os.WriteFile(dsPath, data, 0o644); err != nil {
		t.Fatalf("write dataset: %v", err)
	}

	cachePath := dir + "/brain"
	result, err := Run(context.Background(), RunConfig{
		DatasetPath: dsPath,
		IngestMode:  "bulk",
		ExtractOnly: true,
		BrainCache:  cachePath,
		Concurrency: 2,
	})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if result.QuestionsRun != 2 {
		t.Fatalf("QuestionsRun = %d, want 2", result.QuestionsRun)
	}
	if len(result.Questions) != 0 {
		t.Fatalf("Questions = %d, want 0 in extract-only mode", len(result.Questions))
	}
	if result.IngestMode != "bulk" {
		t.Fatalf("IngestMode = %q, want bulk", result.IngestMode)
	}
}

// TestRun_ExtractOnlyRequiresBrainCache guards against the foot-gun of
// running extract-only against an ephemeral in-memory store that would
// be garbage-collected the moment Run returns.
func TestRun_ExtractOnlyRequiresBrainCache(t *testing.T) {
	t.Parallel()

	questions := []Question{
		{ID: "q1", Category: "x", Question: "Q?", Answer: "A", SessionIDs: []string{"s1"}, HaystackSessions: [][]SessionMessage{{{Role: "user", Content: "x"}}}},
	}
	dir := t.TempDir()
	dsPath := dir + "/ds.json"
	data, _ := json.Marshal(questions)
	if err := os.WriteFile(dsPath, data, 0o644); err != nil {
		t.Fatalf("write dataset: %v", err)
	}

	_, err := Run(context.Background(), RunConfig{
		DatasetPath: dsPath,
		ExtractOnly: true,
	})
	if err == nil {
		t.Fatal("expected error without BrainCache")
	}
	if !strings.Contains(err.Error(), "brain-cache") {
		t.Fatalf("error should mention brain-cache, got: %v", err)
	}
}

// TestRun_ExtractOnlyBrainCacheUsesPassthroughLayout guards the
// tri-SDK contract: after extract-only finishes, a downstream daemon
// (Go, TS, Py) must be able to open the cache directory with a
// passthrough store and see the seeded raw sessions at their literal
// logical paths (raw/lme/<sid>.md). The bulk ingest uses this prefix;
// if the store silently remapped it, every SDK would search an empty
// tree at runtime.
func TestRun_ExtractOnlyBrainCacheUsesPassthroughLayout(t *testing.T) {
	t.Parallel()

	questions := []Question{
		{
			ID:               "q1",
			Category:         "single-session",
			Question:         "Q1?",
			Answer:           "A1",
			SessionIDs:       []string{"s-one"},
			HaystackSessions: [][]SessionMessage{{{Role: "user", Content: "hedgehogs love gardens"}}},
		},
	}
	dir := t.TempDir()
	dsPath := dir + "/ds.json"
	data, _ := json.Marshal(questions)
	if err := os.WriteFile(dsPath, data, 0o644); err != nil {
		t.Fatalf("write dataset: %v", err)
	}

	cachePath := dir + "/brain"
	_, err := Run(context.Background(), RunConfig{
		DatasetPath: dsPath,
		IngestMode:  "bulk",
		ExtractOnly: true,
		BrainCache:  cachePath,
	})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}

	// Assert the bulk-ingested raw session sits at its logical path on
	// disk. fs.Store would have remapped raw/ -> raw/; passthrough
	// leaves the literal path. The multi-root writer in ingestBulk
	// ensures the sanitised session id lands verbatim.
	expected := cachePath + "/raw/lme/s-one.md"
	body, err := os.ReadFile(expected)
	if err != nil {
		t.Fatalf("expected raw session at %s, got: %v", expected, err)
	}
	if !strings.Contains(string(body), "hedgehogs") {
		t.Fatalf("raw session missing fixture text: %q", body)
	}
}
