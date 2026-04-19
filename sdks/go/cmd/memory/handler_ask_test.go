// SPDX-License-Identifier: Apache-2.0

package main

import (
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
		"### Morning run (memory/global/run.md)",
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

// TestBuildAskCompleteRequest_AugmentedSessionAware verifies that when
// chunks carry session_id frontmatter the augmented builder pipes them
// through the lme session-block preprocessor instead of the markdown
// title/path framing.
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

	// Sessions must be sorted chronologically in the rendered prompt.
	posJan := strings.Index(body, "2024-01-10")
	posFeb := strings.Index(body, "2024-02-20")
	if posJan < 0 || posFeb < 0 {
		t.Fatalf("augmented prompt missing session date headers:\n%s", body)
	}
	if posJan > posFeb {
		t.Errorf("session-aware augmented prompt should sort chronologically; jan@%d feb@%d", posJan, posFeb)
	}
	// The markdown framing should NOT be used when session blocks are
	// detected; that is the whole point of the session-aware path.
	if strings.Contains(body, "### ") && strings.Contains(body, "(raw/lme/") {
		t.Errorf("session-aware augmented prompt should not use ### title (path) framing:\n%s", body)
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
