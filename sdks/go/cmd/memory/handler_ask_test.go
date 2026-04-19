// SPDX-License-Identifier: Apache-2.0

package main

import (
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/jeffs-brain/memory/go/llm"
	"github.com/jeffs-brain/memory/go/retrieval"
)

// TestBuildAskCompleteRequest_BasicPreservesOriginalShape locks in the
// pre-augmented behaviour: system prompt, ## Evidence framing, temp 0.2,
// max tokens 1024.
func TestBuildAskCompleteRequest_BasicPreservesOriginalShape(t *testing.T) {
	chunks := []retrieval.RetrievedChunk{
		{Path: "memory/global/cat.md", Title: "Cat note", Text: "Cats sleep a lot."},
	}
	for _, mode := range []string{"", askReaderModeBasic, "BASIC", "  basic  "} {
		req := askRequest{Question: "What do cats do?", ReaderMode: mode}
		got := buildAskCompleteRequest(req, chunks)

		if got.Temperature != 0.2 {
			t.Errorf("mode=%q temperature = %v, want 0.2", mode, got.Temperature)
		}
		if got.MaxTokens != 1024 {
			t.Errorf("mode=%q maxTokens = %d, want 1024", mode, got.MaxTokens)
		}
		if len(got.Messages) != 2 {
			t.Fatalf("mode=%q messages = %d, want 2", mode, len(got.Messages))
		}
		if got.Messages[0].Role != llm.RoleSystem || got.Messages[0].Content != askSystemPrompt {
			t.Errorf("mode=%q first message = %+v, want system askSystemPrompt", mode, got.Messages[0])
		}
		body := got.Messages[1].Content
		if !strings.Contains(body, "## Evidence") {
			t.Errorf("mode=%q user message missing '## Evidence':\n%s", mode, body)
		}
		if !strings.Contains(body, "## Question") {
			t.Errorf("mode=%q user message missing '## Question':\n%s", mode, body)
		}
		if !strings.Contains(body, "Cats sleep a lot.") {
			t.Errorf("mode=%q user message missing chunk body:\n%s", mode, body)
		}
	}
}

// TestBuildAskCompleteRequest_AugmentedInjectsLMEGuidance verifies that
// readerMode=augmented produces the LME CoT reader prompt with all the
// guidance phrases the eval harness relies on.
func TestBuildAskCompleteRequest_AugmentedInjectsLMEGuidance(t *testing.T) {
	chunks := []retrieval.RetrievedChunk{
		{Path: "memory/global/run.md", Title: "Morning run", Text: "Ran 5km on Tuesday."},
		{Path: "memory/global/run-2.md", Title: "Evening run", Text: "Ran 8km on Friday."},
	}
	req := askRequest{
		Question:     "How many kilometres did I run in total?",
		ReaderMode:   askReaderModeAugmented,
		QuestionDate: "2024-03-15",
	}

	got := buildAskCompleteRequest(req, chunks)

	if got.Temperature != 0.0 {
		t.Errorf("augmented temperature = %v, want 0.0", got.Temperature)
	}
	if got.MaxTokens != 800 {
		t.Errorf("augmented maxTokens = %d, want 800", got.MaxTokens)
	}
	if len(got.Messages) != 1 {
		t.Fatalf("augmented messages = %d, want 1 (user only)", len(got.Messages))
	}
	if got.Messages[0].Role != llm.RoleUser {
		t.Errorf("augmented role = %s, want user", got.Messages[0].Role)
	}

	body := got.Messages[0].Content
	wantPhrases := []string{
		"step by step",
		"prefer the value from the most",
		"list, count, enumerate",
		"one per line",
		"Today is 2024-03-15",
		"Friday",
		"Current Date: 2024-03-15",
		"Question: How many kilometres",
		"Answer (step by step):",
		"Ran 5km on Tuesday.",
		"Ran 8km on Friday.",
		"Retrieved facts (2):",
		"[run]",
	}
	for _, want := range wantPhrases {
		if !strings.Contains(body, want) {
			t.Errorf("augmented prompt missing %q\n--- prompt ---\n%s\n--- end ---", want, body)
		}
	}

	// The basic system prompt must NOT leak into augmented mode.
	if strings.Contains(body, askSystemPrompt) {
		t.Errorf("augmented prompt should not include the basic system prompt")
	}
	if strings.Contains(body, "## Evidence") || strings.Contains(body, "## Question") {
		t.Errorf("augmented prompt should not use the basic ## Evidence / ## Question framing")
	}
}

// TestBuildAskCompleteRequest_AugmentedSessionAware verifies that the
// augmented builder uses the shared numbered/date-tagged renderer.
func TestBuildAskCompleteRequest_AugmentedSessionAware(t *testing.T) {
	chunkA := "---\nsession_id: s1\nsession_date: 2024-01-10\n---\n[user]: I bought a red bike.\n"
	chunkB := "---\nsession_id: s2\nsession_date: 2024-02-20\n---\n[user]: Actually the bike is blue now.\n"
	chunks := []retrieval.RetrievedChunk{
		{Path: "raw/lme/s2.md", Text: chunkB},
		{Path: "raw/lme/s1.md", Text: chunkA},
	}
	req := askRequest{
		Question:   "What colour is the bike?",
		ReaderMode: askReaderModeAugmented,
	}

	got := buildAskCompleteRequest(req, chunks)
	body := got.Messages[0].Content

	if !strings.Contains(body, "Retrieved facts (2):") {
		t.Fatalf("augmented prompt missing numbered retrieval header:\n%s", body)
	}
	first := strings.Index(body, "[session=s2]")
	second := strings.Index(body, "[session=s1]")
	if first < 0 || second < 0 {
		t.Fatalf("augmented prompt missing session labels:\n%s", body)
	}
	if first > second {
		t.Errorf("session-aware augmented prompt should place the newer raw session first; s2@%d s1@%d", first, second)
	}
	if strings.Contains(body, "### ") {
		t.Errorf("session-aware augmented prompt should not use markdown title/path framing:\n%s", body)
	}
}

// TestBuildAskCompleteRequest_AugmentedTemporalExpansion verifies that
// resolved date hints from query.ExpandTemporal are injected ahead of
// the retrieved content so the reader can ground "last Friday" style
// questions.
func TestBuildAskCompleteRequest_AugmentedTemporalExpansion(t *testing.T) {
	chunks := []retrieval.RetrievedChunk{
		{Path: "memory/global/note.md", Title: "Note", Text: "Body of the note."},
	}
	// Anchor on a Wednesday; "last Friday" should resolve to the prior week.
	req := askRequest{
		Question:     "What did I do last Friday?",
		ReaderMode:   askReaderModeAugmented,
		QuestionDate: "2024/03/13 (Wed) 10:00",
	}

	got := buildAskCompleteRequest(req, chunks)
	body := got.Messages[0].Content
	if !strings.Contains(body, "[Resolved temporal references:") {
		t.Errorf("augmented prompt missing resolved date hints:\n%s", body)
	}
	if !strings.Contains(body, "2024/03/08") {
		t.Errorf("augmented prompt should resolve 'last Friday' to 2024/03/08:\n%s", body)
	}
}

// TestBuildAskCompleteRequest_AugmentedNoChunksOmitsSessionPath ensures
// the augmented builder does not panic and produces an empty content
// section when there is nothing retrieved.
func TestBuildAskCompleteRequest_AugmentedNoChunksOmitsSessionPath(t *testing.T) {
	req := askRequest{
		Question:   "Anything to say?",
		ReaderMode: askReaderModeAugmented,
	}
	got := buildAskCompleteRequest(req, nil)
	body := got.Messages[0].Content
	if !strings.Contains(body, "Question: Anything to say?") {
		t.Errorf("augmented prompt missing question even when no chunks supplied:\n%s", body)
	}
}

func TestRetrieveForAsk_PassesCandidateKnobsToRetriever(t *testing.T) {
	t.Parallel()

	retr := &captureRetriever{}
	br := &BrainResources{
		ID:        "eval-lme",
		Retriever: retr,
	}
	req := askRequest{
		Question:     "Where is the bike?",
		QuestionDate: "2024/03/13 (Wed) 10:00",
		TopK:         5,
		CandidateK:   80,
		RerankTopN:   40,
		Mode:         string(retrieval.ModeHybridRerank),
	}

	chunks := (&Daemon{}).retrieveForAsk(httptest.NewRequest("POST", "/ask", nil), br, req)

	if len(chunks) != 1 {
		t.Fatalf("chunks = %d, want 1", len(chunks))
	}
	if retr.req.CandidateK != req.CandidateK {
		t.Fatalf("CandidateK = %d, want %d", retr.req.CandidateK, req.CandidateK)
	}
	if retr.req.RerankTopN != req.RerankTopN {
		t.Fatalf("RerankTopN = %d, want %d", retr.req.RerankTopN, req.RerankTopN)
	}
}

func TestRetrieveForAsk_FallbackHydratesFullBodyAndMetadata(t *testing.T) {
	t.Parallel()

	br := setupFallbackSearchBrain(t, "raw/lme/session.md", "---\nsession_id: s1\nsession_date: 2024/03/08\n---\n[user]: I bought a red bike.\n")

	chunks := (&Daemon{}).retrieveForAsk(
		httptest.NewRequest("POST", "/ask", nil),
		br,
		askRequest{Question: "bike", TopK: 5},
	)

	if len(chunks) == 0 {
		t.Fatal("expected fallback ask hit")
	}
	if strings.Contains(chunks[0].Text, "session_id:") {
		t.Fatalf("fallback ask chunk leaked frontmatter:\n%s", chunks[0].Text)
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
