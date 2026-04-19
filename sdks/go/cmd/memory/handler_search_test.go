// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"database/sql"
	"encoding/json"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"

	"github.com/jeffs-brain/memory/go/brain"
	"github.com/jeffs-brain/memory/go/retrieval"
	"github.com/jeffs-brain/memory/go/search"
	"github.com/jeffs-brain/memory/go/store/mem"

	_ "modernc.org/sqlite"
)

type captureRetriever struct {
	req retrieval.Request
}

func (c *captureRetriever) Retrieve(_ context.Context, req retrieval.Request) (retrieval.Response, error) {
	c.req = req
	return retrieval.Response{
		Chunks: []retrieval.RetrievedChunk{{Path: "raw/lme/s1.md"}},
		Trace:  retrieval.Trace{EffectiveMode: retrieval.ModeHybrid},
	}, nil
}

func setupFallbackSearchBrain(t *testing.T, path string, content string) *BrainResources {
	return setupFallbackSearchBrainDocs(t, map[string]string{path: content})
}

func setupFallbackSearchBrainDocs(t *testing.T, docs map[string]string) *BrainResources {
	t.Helper()

	ctx := context.Background()
	store := mem.New()
	t.Cleanup(func() { _ = store.Close() })
	for path, content := range docs {
		if err := store.Write(ctx, brain.Path(path), []byte(content)); err != nil {
			t.Fatalf("store.Write(%s): %v", path, err)
		}
	}

	db, err := sql.Open("sqlite", ":memory:")
	if err != nil {
		t.Fatalf("sql.Open: %v", err)
	}
	t.Cleanup(func() { _ = db.Close() })

	idx, err := search.NewIndex(db, store)
	if err != nil {
		t.Fatalf("search.NewIndex: %v", err)
	}
	if err := idx.Rebuild(ctx); err != nil {
		t.Fatalf("idx.Rebuild: %v", err)
	}

	return &BrainResources{
		ID:     "eval-lme",
		Store:  store,
		Search: idx,
	}
}

func TestSearchRequest_UnmarshalJSON_FilterAliases(t *testing.T) {
	t.Parallel()

	var req searchRequest
	err := json.Unmarshal([]byte(`{
		"query": "apples",
		"question_date": "2024/03/13 (Wed) 10:00",
		"candidate_k": 80,
		"rerank_top_n": 40,
		"document_paths": ["raw/documents/allowed.md", " raw/documents/allowed.md ", " raw/documents/other.md "]
	}`), &req)
	if err != nil {
		t.Fatalf("Unmarshal: %v", err)
	}
	if req.QuestionDate != "2024/03/13 (Wed) 10:00" {
		t.Fatalf("QuestionDate = %q, want 2024/03/13 (Wed) 10:00", req.QuestionDate)
	}
	if req.CandidateK != 80 {
		t.Fatalf("CandidateK = %d, want 80", req.CandidateK)
	}
	if req.RerankTopN != 40 {
		t.Fatalf("RerankTopN = %d, want 40", req.RerankTopN)
	}
	want := []string{"raw/documents/allowed.md", "raw/documents/other.md"}
	if !reflect.DeepEqual(req.Filters.Paths, want) {
		t.Fatalf("Filters.Paths = %v, want %v", req.Filters.Paths, want)
	}
}

func TestRunSearchPipeline_PassesQuestionDateToRetriever(t *testing.T) {
	t.Parallel()

	retr := &captureRetriever{}
	br := &BrainResources{
		ID:        "eval-lme",
		Retriever: retr,
	}

	req := searchRequest{
		Query:        "What happened last Friday?",
		QuestionDate: "2024/03/13 (Wed) 10:00",
		TopK:         5,
		CandidateK:   80,
		RerankTopN:   40,
		Mode:         string(retrieval.ModeHybridRerank),
		Filters: retrieval.Filters{
			Paths: []string{"raw/lme/s1.md"},
		},
	}

	chunks, _, _, _ := (&Daemon{}).runSearchPipeline(httptest.NewRequest("POST", "/search", nil), br, req)

	if len(chunks) != 1 {
		t.Fatalf("chunks = %d, want 1", len(chunks))
	}
	if retr.req.QuestionDate != req.QuestionDate {
		t.Fatalf("QuestionDate = %q, want %q", retr.req.QuestionDate, req.QuestionDate)
	}
	if retr.req.Mode != retrieval.ModeHybridRerank {
		t.Fatalf("Mode = %q, want hybrid-rerank", retr.req.Mode)
	}
	if retr.req.CandidateK != req.CandidateK {
		t.Fatalf("CandidateK = %d, want %d", retr.req.CandidateK, req.CandidateK)
	}
	if retr.req.RerankTopN != req.RerankTopN {
		t.Fatalf("RerankTopN = %d, want %d", retr.req.RerankTopN, req.RerankTopN)
	}
	if !reflect.DeepEqual(retr.req.Filters.Paths, req.Filters.Paths) {
		t.Fatalf("Filters.Paths = %v, want %v", retr.req.Filters.Paths, req.Filters.Paths)
	}
}

func TestRunSearchPipeline_NormalisesInvalidModeToAuto(t *testing.T) {
	t.Parallel()

	retr := &captureRetriever{}
	br := &BrainResources{
		ID:        "eval-lme",
		Retriever: retr,
	}

	chunks, _, _, _ := (&Daemon{}).runSearchPipeline(
		httptest.NewRequest("POST", "/search", nil),
		br,
		searchRequest{
			Query: "What happened last Friday?",
			TopK:  5,
			Mode:  "definitely-not-a-mode",
		},
	)

	if len(chunks) != 1 {
		t.Fatalf("chunks = %d, want 1", len(chunks))
	}
	if retr.req.Mode != retrieval.ModeAuto {
		t.Fatalf("Mode = %q, want auto", retr.req.Mode)
	}
}

func TestRunSearchPipeline_FallbackHydratesFullBodyAndMetadata(t *testing.T) {
	t.Parallel()

	br := setupFallbackSearchBrain(t, "raw/lme/session.md", "---\nsession_id: s1\nsession_date: 2024/03/08\n---\n[user]: I bought a red bike.\n")

	chunks, _, _, _ := (&Daemon{}).runSearchPipeline(
		httptest.NewRequest("POST", "/search", nil),
		br,
		searchRequest{Query: "bike", TopK: 5},
	)

	if len(chunks) == 0 {
		t.Fatal("expected fallback search hit")
	}
	if strings.Contains(chunks[0].Text, "session_id:") {
		t.Fatalf("fallback chunk leaked frontmatter:\n%s", chunks[0].Text)
	}
	if chunks[0].Text != "[user]: I bought a red bike." {
		t.Fatalf("chunk text = %q, want stripped full body", chunks[0].Text)
	}
	if got := chunks[0].Metadata["session_id"]; got != "s1" {
		t.Fatalf("session_id = %#v, want s1", got)
	}
	if got := chunks[0].Metadata["session_date"]; got != "2024/03/08" {
		t.Fatalf("session_date = %#v, want 2024/03/08", got)
	}
}

func TestRunSearchPipeline_FallbackUsesTemporalDateHints(t *testing.T) {
	t.Parallel()

	br := setupFallbackSearchBrain(t, "memory/global/weekly-note.md", "---\nsession_date: 2024/03/08\n---\nMet the supplier and agreed the timeline.\n")

	chunks, _, _, _ := (&Daemon{}).runSearchPipeline(
		httptest.NewRequest("POST", "/search", nil),
		br,
		searchRequest{
			Query:        "What happened last Friday?",
			QuestionDate: "2024/03/13 (Wed) 10:00",
			TopK:         5,
		},
	)

	if len(chunks) == 0 {
		t.Fatal("expected temporal fallback search hit")
	}
	if chunks[0].Path != "memory/global/weekly-note.md" {
		t.Fatalf("top path = %q, want weekly note", chunks[0].Path)
	}
}

func TestRunSearchPipeline_FallbackRespectsExactPathFilters(t *testing.T) {
	t.Parallel()

	br := setupFallbackSearchBrainDocs(t, map[string]string{
		"raw/documents/allowed.md": "---\n---\nA note about apples.\n",
		"raw/documents/blocked.md": "---\n---\nAnother note about apples.\n",
	})

	chunks, _, _, _ := (&Daemon{}).runSearchPipeline(
		httptest.NewRequest("POST", "/search", nil),
		br,
		searchRequest{
			Query: "apples",
			TopK:  10,
			Filters: retrieval.Filters{
				Paths: []string{"raw/documents/allowed.md"},
			},
		},
	)

	if len(chunks) != 1 {
		t.Fatalf("chunks = %d, want 1", len(chunks))
	}
	if chunks[0].Path != "raw/documents/allowed.md" {
		t.Fatalf("Path = %q, want raw/documents/allowed.md", chunks[0].Path)
	}
}

func TestRunSearchPipeline_FallbackRespectsRawLMEScope(t *testing.T) {
	t.Parallel()

	br := setupFallbackSearchBrainDocs(t, map[string]string{
		"raw/lme/session.md":          "---\nsession_id: s1\n---\n[user]: apples\n",
		"raw/documents/hedgehogs.md":  "---\n---\nApples in a raw document.\n",
		"memory/global/remembered.md": "---\n---\nApples in memory.\n",
	})

	chunks, _, _, _ := (&Daemon{}).runSearchPipeline(
		httptest.NewRequest("POST", "/search", nil),
		br,
		searchRequest{
			Query: "apples",
			TopK:  10,
			Filters: retrieval.Filters{
				Scope: "raw_lme",
			},
		},
	)

	if len(chunks) != 1 {
		t.Fatalf("chunks = %d, want 1", len(chunks))
	}
	if chunks[0].Path != "raw/lme/session.md" {
		t.Fatalf("Path = %q, want raw/lme/session.md", chunks[0].Path)
	}
}
