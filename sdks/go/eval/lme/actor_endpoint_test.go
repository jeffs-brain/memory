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

	"github.com/jeffs-brain/memory/go/retrieval"
)

// TestCallActorEndpoint_StitchesAnswerDeltas verifies the SSE consumer
// concatenates answer_delta frames into a single answer and preserves
// usage totals from the done frame.
func TestCallActorEndpoint_StitchesAnswerDeltas(t *testing.T) {
	t.Parallel()

	var captured struct {
		Question     string `json:"question"`
		TopK         int    `json:"topK"`
		Mode         string `json:"mode"`
		ReaderMode   string `json:"readerMode"`
		QuestionDate string `json:"questionDate"`
	}
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
		if err := json.NewDecoder(r.Body).Decode(&captured); err != nil {
			t.Fatalf("decode request: %v", err)
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
	if captured.Mode != "hybrid-rerank" {
		t.Fatalf("mode = %q, want hybrid-rerank", captured.Mode)
	}
	if captured.ReaderMode != "augmented" {
		t.Fatalf("readerMode = %q, want augmented", captured.ReaderMode)
	}
	if captured.QuestionDate != "2024-01-01" {
		t.Fatalf("questionDate = %q, want 2024-01-01", captured.QuestionDate)
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

func TestCallActorRetrieve_RendersStructuredEvidenceAndThreadsQuestionDate(t *testing.T) {
	t.Parallel()

	var captured struct {
		Query        string `json:"query"`
		QuestionDate string `json:"questionDate"`
		Mode         string `json:"mode"`
		CandidateK   int    `json:"candidateK"`
		RerankTopN   int    `json:"rerankTopN"`
		Filters      struct {
			Scope      string `json:"scope"`
			Project    string `json:"project"`
			PathPrefix string `json:"pathPrefix"`
		} `json:"filters"`
	}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Errorf("method = %q, want POST", r.Method)
		}
		if !strings.HasSuffix(r.URL.Path, "/search") {
			t.Errorf("path = %q, want /search suffix", r.URL.Path)
		}
		if err := json.NewDecoder(r.Body).Decode(&captured); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"chunks": []map[string]any{
				{
					"path":  "raw/lme/s1-a.md",
					"text":  "---\nsession_id: s1\nsession_date: 2024-03-01\n---\n[user]: Alpha.\n",
					"score": 0.9,
				},
				{
					"path":  "raw/lme/s2.md",
					"text":  "---\nsession_id: s2\nsession_date: 2024-03-02\n---\n[user]: Bravo.\n",
					"score": 0.8,
				},
				{
					"path":  "raw/lme/s1-b.md",
					"text":  "---\nsession_id: s1\nsession_date: 2024-03-01\n---\n[user]: Charlie.\n",
					"score": 0.7,
				},
			},
		})
	}))
	defer srv.Close()

	content, err := callActorRetrieve(
		context.Background(),
		srv.URL,
		"eval-lme",
		"What happened last Friday?",
		"2024/03/13 (Wed) 10:00",
		3,
		retrieval.ModeHybridRerank,
		80,
		40,
		retrieval.Filters{
			Scope:      "project",
			Project:    "eval-lme",
			PathPrefix: "memory/project/eval-lme/",
		},
	)
	if err != nil {
		t.Fatalf("callActorRetrieve: %v", err)
	}
	if captured.QuestionDate != "2024/03/13 (Wed) 10:00" {
		t.Fatalf("questionDate = %q, want threaded request value", captured.QuestionDate)
	}
	if captured.Mode != "hybrid-rerank" {
		t.Fatalf("mode = %q, want hybrid-rerank", captured.Mode)
	}
	if captured.CandidateK != 80 {
		t.Fatalf("candidateK = %d, want 80", captured.CandidateK)
	}
	if captured.RerankTopN != 40 {
		t.Fatalf("rerankTopN = %d, want 40", captured.RerankTopN)
	}
	if captured.Filters.Scope != "project" {
		t.Fatalf("filters.scope = %q, want project", captured.Filters.Scope)
	}
	if captured.Filters.Project != "eval-lme" {
		t.Fatalf("filters.project = %q, want eval-lme", captured.Filters.Project)
	}
	if captured.Filters.PathPrefix != "memory/project/eval-lme/" {
		t.Fatalf("filters.pathPrefix = %q, want memory/project/eval-lme/", captured.Filters.PathPrefix)
	}
	if !strings.Contains(content, "Retrieved facts (3):") {
		t.Fatalf("rendered evidence missing header:\n%s", content)
	}
	if !strings.Contains(content, "[2024-03-01] [session=s1] [s1-a]") {
		t.Fatalf("rendered evidence missing date-tagged first hit:\n%s", content)
	}
	first := strings.Index(content, "[session=s1]")
	second := strings.LastIndex(content, "[session=s1]")
	other := strings.Index(content, "[session=s2]")
	if first < 0 || second < 0 || other < 0 {
		t.Fatalf("rendered evidence missing session labels:\n%s", content)
	}
	if second <= first {
		t.Fatalf("session clustering did not keep the repeated s1 passages together:\n%s", content)
	}
	if !strings.Contains(content, "[Resolved temporal references: 2024/03/08]") {
		t.Fatalf("rendered evidence missing temporal hint:\n%s", content)
	}
	if strings.Contains(content, "session_id:") || strings.Contains(content, "session_date:") {
		t.Fatalf("rendered evidence should strip frontmatter from excerpt bodies:\n%s", content)
	}
	if !strings.Contains(content, "[user]: Alpha.") || !strings.Contains(content, "[user]: Charlie.") {
		t.Fatalf("rendered evidence missing stripped body content:\n%s", content)
	}
}

func TestCallActorRetrieve_AcceptsLegacyCapitalisedChunkKeys(t *testing.T) {
	t.Parallel()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{
			"chunks": []map[string]any{
				{
					"Path":  "raw/lme/s1.md",
					"Text":  "---\nsession_id: s1\nsession_date: 2024-03-01\n---\n[user]: Legacy chunk.\n",
					"Score": 0.9,
				},
			},
		})
	}))
	defer srv.Close()

	content, err := callActorRetrieve(
		context.Background(),
		srv.URL,
		"eval-lme",
		"What happened?",
		"",
		3,
		retrieval.ModeHybridRerank,
		0,
		0,
		retrieval.Filters{},
	)
	if err != nil {
		t.Fatalf("callActorRetrieve: %v", err)
	}
	if !strings.Contains(content, "Legacy chunk.") {
		t.Fatalf("rendered evidence missing legacy chunk body:\n%s", content)
	}
	if !strings.Contains(content, "[session=s1]") {
		t.Fatalf("rendered evidence missing legacy session id:\n%s", content)
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
