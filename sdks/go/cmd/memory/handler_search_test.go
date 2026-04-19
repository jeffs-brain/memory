// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"database/sql"
	"net/http/httptest"
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
	t.Helper()

	ctx := context.Background()
	store := mem.New()
	t.Cleanup(func() { _ = store.Close() })
	if err := store.Write(ctx, brain.Path(path), []byte(content)); err != nil {
		t.Fatalf("store.Write: %v", err)
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
