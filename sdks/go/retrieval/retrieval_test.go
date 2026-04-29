// SPDX-License-Identifier: Apache-2.0

package retrieval

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/jeffs-brain/memory/go/llm"
)

func newTestCorpus() []fakeChunk {
	return []fakeChunk{
		{
			ID:      "c1",
			Path:    "wiki/invoice-processing.md",
			Title:   "Invoice Processing",
			Summary: "How we automate supplier invoice ingestion",
			Content: "Invoice automation workflow extracts line items from PDFs.",
		},
		{
			ID:      "c2",
			Path:    "wiki/order-processing.md",
			Title:   "Order Processing Pipeline",
			Summary: "Sales order ingestion for retailers",
			Content: "Automated document processing for orders captured via email.",
		},
		{
			ID:      "c3",
			Path:    "wiki/contact-centre.md",
			Title:   "Contact Centre",
			Summary: "Inbound voice routing",
			Content: "Telephony stack routes calls via SIP.",
		},
		{
			ID:      "c4",
			Path:    "memory/global/user-preference-coffee.md",
			Title:   "Coffee preferences",
			Summary: "Alex prefers flat whites",
			Content: "Alex likes flat whites with oat milk.",
		},
		{
			ID:      "c5",
			Path:    "memory/global/user-fact-birthday.md",
			Title:   "User fact: birthday",
			Summary: "Observed on 1986-08-14",
			Content: "[observed on: 1986-08-14] Alex was born on 14 August 1986.",
		},
		{
			ID:      "c6",
			Path:    "wiki/rollup/invoice-summary.md",
			Title:   "Invoice recap",
			Summary: "Roll-up summary across invoice workflows",
			Content: "Overview and summary of invoice workflow totals.",
		},
	}
}

func TestRetrieve_BM25Mode(t *testing.T) {
	t.Parallel()
	src := newFakeSource(newTestCorpus())
	r, err := New(Config{Source: src})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	resp, err := r.Retrieve(context.Background(), Request{
		Query: "invoice automation",
		TopK:  3,
		Mode:  ModeBM25,
	})
	if err != nil {
		t.Fatalf("Retrieve: %v", err)
	}
	if len(resp.Chunks) == 0 {
		t.Fatalf("expected hits")
	}
	if resp.Trace.EffectiveMode != ModeBM25 {
		t.Fatalf("effective mode %q, want bm25", resp.Trace.EffectiveMode)
	}
	if resp.Trace.EmbedderUsed {
		t.Fatalf("embedder should not fire in bm25 mode")
	}
	if resp.Chunks[0].Path != "wiki/invoice-processing.md" {
		t.Fatalf("top path %q, want wiki/invoice-processing.md", resp.Chunks[0].Path)
	}
}

func TestRetrieve_SemanticMode(t *testing.T) {
	t.Parallel()
	src := newFakeSource(newTestCorpus())
	embedder := llm.NewFakeEmbedder(src.embedDim)
	r, err := New(Config{Source: src, Embedder: embedder})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	resp, err := r.Retrieve(context.Background(), Request{
		Query: "invoice automation workflow",
		TopK:  3,
		Mode:  ModeSemantic,
	})
	if err != nil {
		t.Fatalf("Retrieve: %v", err)
	}
	if resp.Trace.EffectiveMode != ModeSemantic {
		t.Fatalf("effective mode %q, want semantic", resp.Trace.EffectiveMode)
	}
	if !resp.Trace.EmbedderUsed {
		t.Fatalf("embedder should have fired")
	}
	if len(resp.Chunks) == 0 {
		t.Fatalf("expected semantic hits")
	}
	for _, chunk := range resp.Chunks {
		if chunk.BM25Rank != 0 {
			t.Fatalf("semantic vector hit carried BM25 rank: %+v", chunk)
		}
	}
}

func TestRetrieve_HybridMode(t *testing.T) {
	t.Parallel()
	src := newFakeSource(newTestCorpus())
	embedder := llm.NewFakeEmbedder(src.embedDim)
	r, err := New(Config{Source: src, Embedder: embedder})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	resp, err := r.Retrieve(context.Background(), Request{
		Query: "invoice automation",
		TopK:  3,
		Mode:  ModeHybrid,
	})
	if err != nil {
		t.Fatalf("Retrieve: %v", err)
	}
	if resp.Trace.EffectiveMode != ModeHybrid {
		t.Fatalf("effective mode %q, want hybrid", resp.Trace.EffectiveMode)
	}
	if resp.Trace.BM25Hits == 0 {
		t.Fatalf("hybrid mode: bm25 leg should find hits")
	}
	if resp.Trace.VectorHits == 0 {
		t.Fatalf("hybrid mode: vector leg should find hits")
	}
	if resp.Trace.FusedHits == 0 {
		t.Fatalf("hybrid mode: fusion should produce hits")
	}
	// Invoice-related docs should dominate.
	top := resp.Chunks[0].Path
	if top != "wiki/invoice-processing.md" && top != "wiki/order-processing.md" {
		t.Fatalf("unexpected top result %q", top)
	}
}

func TestRetrieve_HydratesSessionMetadataFromBodyAndIndex(t *testing.T) {
	t.Parallel()
	src := newFakeSource([]fakeChunk{
		{
			ID:    "m1",
			Path:  "memory/project/eval-lme/user-fact-commute.md",
			Title: "Commute fact",
			Content: strings.Join([]string{
				"---",
				"session_id: session-42",
				"observed_on: 2023-05-22T08:00:00Z",
				"modified: 2023-05-23T09:30:00Z",
				"---",
				"The daily commute takes 45 minutes each way.",
			}, "\n"),
			Scope:   "project_memory",
			Project: "eval-lme",
			Session: "2023-05-22",
		},
	})
	r, err := New(Config{Source: src})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	resp, err := r.Retrieve(context.Background(), Request{
		Query: "daily commute",
		TopK:  1,
		Mode:  ModeBM25,
	})
	if err != nil {
		t.Fatalf("Retrieve: %v", err)
	}
	if len(resp.Chunks) != 1 {
		t.Fatalf("chunks = %d, want 1", len(resp.Chunks))
	}
	if resp.Chunks[0].Text != "The daily commute takes 45 minutes each way." {
		t.Fatalf("chunk text = %q, want stripped body", resp.Chunks[0].Text)
	}
	meta := resp.Chunks[0].Metadata
	if meta["scope"] != "project_memory" {
		t.Fatalf("scope metadata = %v, want project_memory", meta["scope"])
	}
	if meta["project"] != "eval-lme" {
		t.Fatalf("project metadata = %v, want eval-lme", meta["project"])
	}
	if meta["projectSlug"] != "eval-lme" {
		t.Fatalf("projectSlug metadata = %v, want eval-lme", meta["projectSlug"])
	}
	if meta["sessionDate"] != "2023-05-22" || meta["session_date"] != "2023-05-22" {
		t.Fatalf("session date metadata = %+v, want both aliases", meta)
	}
	if meta["sessionId"] != "session-42" || meta["session_id"] != "session-42" {
		t.Fatalf("session id metadata = %+v, want both aliases", meta)
	}
	if meta["observedOn"] != "2023-05-22T08:00:00Z" || meta["observed_on"] != "2023-05-22T08:00:00Z" {
		t.Fatalf("observed_on metadata = %+v, want both aliases", meta)
	}
	if meta["modified"] != "2023-05-23T09:30:00Z" {
		t.Fatalf("modified metadata = %v, want 2023-05-23T09:30:00Z", meta["modified"])
	}
}

func TestRetrieve_DedupesNearDuplicateChunksBeforeTopK(t *testing.T) {
	t.Parallel()
	src := newFakeSource([]fakeChunk{
		{
			ID:    "global-chair",
			Path:  "memory/global/chair.md",
			Title: "Chair preference",
			Content: strings.Join([]string{
				"---",
				"session_id: session-chair",
				"---",
				"The office chair should have adjustable arms and strong lumbar support.",
			}, "\n"),
			Scope:   "global_memory",
			Session: "2024-03-01",
		},
		{
			ID:    "project-chair",
			Path:  "memory/project/work/chair.md",
			Title: "Chair requirements",
			Content: strings.Join([]string{
				"---",
				"session_id: session-chair",
				"---",
				"The office chair should have adjustable arms and strong lumbar support.",
			}, "\n"),
			Scope:   "project_memory",
			Project: "work",
			Session: "2024-03-01",
		},
		{
			ID:      "desk",
			Path:    "memory/project/work/desk.md",
			Title:   "Desk requirement",
			Content: "The desk should be one hundred and forty centimetres wide.",
			Scope:   "project_memory",
			Project: "work",
			Session: "2024-03-02",
		},
		{
			ID:      "monitor",
			Path:    "memory/project/work/monitor.md",
			Title:   "Monitor requirement",
			Content: "The monitor should be matte and support USB C charging.",
			Scope:   "project_memory",
			Project: "work",
			Session: "2024-03-03",
		},
	})
	r, err := New(Config{Source: src})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	resp, err := r.Retrieve(context.Background(), Request{
		Query:      "office chair adjustable arms lumbar support desk monitor",
		TopK:       3,
		CandidateK: 4,
		Mode:       ModeBM25,
		Filters: Filters{
			Scope:   "memory",
			Project: "work",
		},
	})
	if err != nil {
		t.Fatalf("Retrieve: %v", err)
	}
	paths := chunkPaths(resp.Chunks)
	if containsPath(paths, "memory/global/chair.md") && containsPath(paths, "memory/project/work/chair.md") {
		t.Fatalf("near-duplicate chair facts both survived: %v", paths)
	}
	if !containsPath(paths, "memory/project/work/desk.md") || !containsPath(paths, "memory/project/work/monitor.md") {
		t.Fatalf("dedupe did not make room for distinct evidence, got %v", paths)
	}
}

func TestRetrieve_DedupesSameSessionParaphrasesBeforeTopK(t *testing.T) {
	t.Parallel()
	src := newFakeSource([]fakeChunk{
		{
			ID:    "shelter-primary",
			Path:  "memory/project/eval/shelter-primary.md",
			Title: "Shelter fundraiser",
			Content: strings.Join([]string{
				"---",
				"session_id: session-charity",
				"---",
				"The local shelter charity event raised 1000 dollars for families in March.",
			}, "\n"),
			Scope:   "project_memory",
			Project: "eval",
			Session: "2024-03-01",
		},
		{
			ID:    "shelter-paraphrase",
			Path:  "memory/project/eval/shelter-paraphrase.md",
			Title: "Shelter fundraiser follow-up",
			Content: strings.Join([]string{
				"---",
				"session_id: session-charity",
				"---",
				"Later the local shelter charity event raised 1000 dollars for families in March.",
			}, "\n"),
			Scope:   "project_memory",
			Project: "eval",
			Session: "2024-03-01",
		},
		{
			ID:      "cancer-society",
			Path:    "memory/project/eval/cancer-society.md",
			Title:   "Cancer society fundraiser",
			Content: "The cancer society charity event raised 500 dollars for families in April.",
			Scope:   "project_memory",
			Project: "eval",
			Session: "2024-04-01",
		},
		{
			ID:      "school-fair",
			Path:    "memory/project/eval/school-fair.md",
			Title:   "School fair fundraiser",
			Content: "The school fair charity event raised 250 dollars for families in May.",
			Scope:   "project_memory",
			Project: "eval",
			Session: "2024-05-01",
		},
	})
	r, err := New(Config{Source: src})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	resp, err := r.Retrieve(context.Background(), Request{
		Query:      "How much money did the charity events raise for families?",
		TopK:       3,
		CandidateK: 4,
		Mode:       ModeBM25,
		Filters: Filters{
			Scope:   "memory",
			Project: "eval",
		},
	})
	if err != nil {
		t.Fatalf("Retrieve: %v", err)
	}
	paths := chunkPaths(resp.Chunks)
	if containsPath(paths, "memory/project/eval/shelter-primary.md") && containsPath(paths, "memory/project/eval/shelter-paraphrase.md") {
		t.Fatalf("same-session paraphrases both survived: %v", paths)
	}
	if !containsPath(paths, "memory/project/eval/cancer-society.md") || !containsPath(paths, "memory/project/eval/school-fair.md") {
		t.Fatalf("dedupe did not make room for distinct charity evidence, got %v", paths)
	}
}

func TestRetrieve_ExpandsSameSessionNeighbours(t *testing.T) {
	t.Parallel()
	src := newFakeSource([]fakeChunk{
		{
			ID:    "kitchen-seed",
			Path:  "memory/project/home/kitchen-seed.md",
			Title: "Kitchen organisation",
			Content: strings.Join([]string{
				"---",
				"session_id: session-kitchen",
				"---",
				"The user asked for kitchen organisation and cleaning advice after a previous conversation.",
			}, "\n"),
			Scope:   "project_memory",
			Project: "home",
			Session: "2024-03-01",
		},
		{
			ID:    "kitchen-neighbour",
			Path:  "memory/project/home/kitchen-neighbour.md",
			Title: "Kitchen context",
			Content: strings.Join([]string{
				"---",
				"session_id: session-kitchen",
				"---",
				"The user owns a granite sink and keeps a utensil holder on the counter.",
			}, "\n"),
			Scope:   "project_memory",
			Project: "home",
			Session: "2024-03-01",
		},
		{
			ID:      "garden",
			Path:    "memory/project/home/garden.md",
			Title:   "Garden",
			Content: "The user asked about garden storage and patio cleaning.",
			Scope:   "project_memory",
			Project: "home",
			Session: "2024-03-02",
		},
	})
	r, err := New(Config{Source: src})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	resp, err := r.Retrieve(context.Background(), Request{
		Query:      "previous conversation kitchen organisation cleaning advice",
		TopK:       2,
		CandidateK: 1,
		Mode:       ModeBM25,
		Filters: Filters{
			Scope:   "memory",
			Project: "home",
		},
	})
	if err != nil {
		t.Fatalf("Retrieve: %v", err)
	}
	paths := chunkPaths(resp.Chunks)
	if !containsPath(paths, "memory/project/home/kitchen-seed.md") {
		t.Fatalf("seed missing from results: %v", paths)
	}
	if !containsPath(paths, "memory/project/home/kitchen-neighbour.md") {
		t.Fatalf("same-session neighbour missing from results: %v", paths)
	}
	if resp.Trace.SessionExpansions != 1 {
		t.Fatalf("session expansions = %d, want 1", resp.Trace.SessionExpansions)
	}
}

func TestRetrieve_SameSessionExpansionSkipsExactPathFilters(t *testing.T) {
	t.Parallel()
	src := newFakeSource([]fakeChunk{
		{
			ID:    "seed",
			Path:  "memory/project/home/seed.md",
			Title: "Kitchen organisation",
			Content: strings.Join([]string{
				"---",
				"session_id: session-kitchen",
				"---",
				"The user asked for kitchen organisation and cleaning advice.",
			}, "\n"),
			Scope:   "project_memory",
			Project: "home",
			Session: "2024-03-01",
		},
		{
			ID:    "neighbour",
			Path:  "memory/project/home/neighbour.md",
			Title: "Kitchen context",
			Content: strings.Join([]string{
				"---",
				"session_id: session-kitchen",
				"---",
				"The user keeps a utensil holder on the counter.",
			}, "\n"),
			Scope:   "project_memory",
			Project: "home",
			Session: "2024-03-01",
		},
	})
	r, err := New(Config{Source: src})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	resp, err := r.Retrieve(context.Background(), Request{
		Query:      "kitchen organisation cleaning advice",
		TopK:       2,
		CandidateK: 1,
		Mode:       ModeBM25,
		Filters: Filters{
			Scope:   "memory",
			Project: "home",
			Paths:   []string{"memory/project/home/seed.md"},
		},
	})
	if err != nil {
		t.Fatalf("Retrieve: %v", err)
	}
	paths := chunkPaths(resp.Chunks)
	if containsPath(paths, "memory/project/home/neighbour.md") {
		t.Fatalf("same-session expansion ignored exact path filter: %v", paths)
	}
	if resp.Trace.SessionExpansions != 0 {
		t.Fatalf("session expansions = %d, want 0", resp.Trace.SessionExpansions)
	}
}

func TestRetrieve_SameSessionExpansionSkipsOrdinaryQueries(t *testing.T) {
	t.Parallel()
	src := newFakeSource([]fakeChunk{
		{
			ID:    "seed",
			Path:  "memory/project/home/seed.md",
			Title: "Kitchen organisation",
			Content: strings.Join([]string{
				"---",
				"session_id: session-kitchen",
				"---",
				"The user asked for kitchen organisation and cleaning advice.",
			}, "\n"),
			Scope:   "project_memory",
			Project: "home",
			Session: "2024-03-01",
		},
		{
			ID:    "neighbour",
			Path:  "memory/project/home/neighbour.md",
			Title: "Kitchen context",
			Content: strings.Join([]string{
				"---",
				"session_id: session-kitchen",
				"---",
				"The user keeps a utensil holder on the counter.",
			}, "\n"),
			Scope:   "project_memory",
			Project: "home",
			Session: "2024-03-01",
		},
	})
	r, err := New(Config{Source: src})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	resp, err := r.Retrieve(context.Background(), Request{
		Query:      "kitchen organisation cleaning advice",
		TopK:       2,
		CandidateK: 1,
		Mode:       ModeBM25,
		Filters: Filters{
			Scope:   "memory",
			Project: "home",
		},
	})
	if err != nil {
		t.Fatalf("Retrieve: %v", err)
	}
	paths := chunkPaths(resp.Chunks)
	if containsPath(paths, "memory/project/home/neighbour.md") {
		t.Fatalf("ordinary query unexpectedly expanded same-session neighbour: %v", paths)
	}
	if resp.Trace.SessionExpansions != 0 {
		t.Fatalf("session expansions = %d, want 0", resp.Trace.SessionExpansions)
	}
}

func TestRetrieve_SameSessionExpansionDoesNotDuplicateExistingHits(t *testing.T) {
	t.Parallel()
	src := newFakeSource([]fakeChunk{
		{
			ID:    "seed",
			Path:  "memory/project/home/seed.md",
			Title: "Kitchen organisation",
			Content: strings.Join([]string{
				"---",
				"session_id: session-kitchen",
				"---",
				"The user asked for kitchen organisation and cleaning advice.",
			}, "\n"),
			Scope:   "project_memory",
			Project: "home",
			Session: "2024-03-01",
		},
		{
			ID:    "duplicate-neighbour",
			Path:  "memory/project/home/z-duplicate-neighbour.md",
			Title: "Kitchen organisation duplicate",
			Content: strings.Join([]string{
				"---",
				"session_id: session-kitchen",
				"---",
				"The user asked for kitchen organisation and cleaning advice.",
			}, "\n"),
			Scope:   "project_memory",
			Project: "home",
			Session: "2024-03-01",
		},
		{
			ID:    "distinct-neighbour",
			Path:  "memory/project/home/distinct-neighbour.md",
			Title: "Kitchen context",
			Content: strings.Join([]string{
				"---",
				"session_id: session-kitchen",
				"---",
				"The user owns a granite sink and keeps a utensil holder on the counter.",
			}, "\n"),
			Scope:   "project_memory",
			Project: "home",
			Session: "2024-03-01",
		},
	})
	r, err := New(Config{Source: src})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	resp, err := r.Retrieve(context.Background(), Request{
		Query:      "previous conversation kitchen organisation cleaning advice",
		TopK:       3,
		CandidateK: 1,
		Mode:       ModeBM25,
		Filters: Filters{
			Scope:   "memory",
			Project: "home",
		},
	})
	if err != nil {
		t.Fatalf("Retrieve: %v", err)
	}
	paths := chunkPaths(resp.Chunks)
	if containsPath(paths, "memory/project/home/z-duplicate-neighbour.md") {
		t.Fatalf("duplicate neighbour survived expansion: %v", paths)
	}
	if !containsPath(paths, "memory/project/home/distinct-neighbour.md") {
		t.Fatalf("distinct neighbour missing after duplicate suppression: %v", paths)
	}
}

func TestRetrieve_EpisodicRecallAddsTrimmedRawArtefactSection(t *testing.T) {
	t.Parallel()
	src := newFakeSource([]fakeChunk{
		{
			ID:      "wrong-memory",
			Path:    "memory/project/eval/song-summary.md",
			Title:   "Sad song summary",
			Content: "The previous conversation included one sad song. Its chorus used Fmaj7, G7, Cmaj7, Am7.",
			Scope:   "project_memory",
			Project: "eval",
			Session: "2023-05-28",
		},
		{
			ID:    "raw-song",
			Path:  "raw/lme/answer_sharegpt_song_0.md",
			Title: "Raw conversation",
			Content: strings.Join([]string{
				"---",
				"session_id: answer_sharegpt_song_0",
				"session_date: 2023-05-28",
				"---",
				"[user]: Create a sad song with notes",
				"",
				"[assistant]: Here's a sad song with notes for you:",
				"",
				"Chorus:",
				"G G G G A G F",
				"Why did you have to go?",
				"",
				"[user]: be more romantic and heart-felt",
				"",
				"[assistant]: Sure, here's a more romantic and heart-felt song for you:",
				"",
				"Verse 1:",
				"G A B C D E D C B A G",
				"When I first saw you, my heart skipped a beat",
				"",
				"Chorus:",
				"C D E F G A B A G F E D C",
				"You're the one I want, the one I need",
				"C D E F G A B A G F E D C",
				"In your eyes, I see my destiny",
				"",
				"Verse 2:",
				"G A B C D E D C B A G",
			}, "\n"),
			Scope:   "raw_lme",
			Session: "2023-05-28",
		},
	})
	r, err := New(Config{Source: src})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	resp, err := r.Retrieve(context.Background(), Request{
		Query:        "I'm looking back at our previous conversation where you created two sad songs for me. Can you remind me what was the chord progression for the chorus in the second song?",
		QuestionDate: "2023-05-29",
		TopK:         3,
		CandidateK:   2,
		Mode:         ModeBM25,
		Filters: Filters{
			Scope:   "memory",
			Project: "eval",
		},
	})
	if err != nil {
		t.Fatalf("Retrieve: %v", err)
	}
	paths := chunkPaths(resp.Chunks)
	if !containsPath(paths, "raw/lme/answer_sharegpt_song_0.md") {
		t.Fatalf("episodic recall raw artefact missing from results: %v", paths)
	}
	if !resp.Trace.EpisodicRecall || resp.Trace.EpisodicRecallHits != 1 {
		t.Fatalf("episodic recall trace = %+v, want one hit", resp.Trace)
	}
	var rawChunk RetrievedChunk
	for _, chunk := range resp.Chunks {
		if chunk.Path == "raw/lme/answer_sharegpt_song_0.md" {
			rawChunk = chunk
			break
		}
	}
	if !strings.Contains(rawChunk.Text, "C D E F G A B A G F E D C") {
		t.Fatalf("raw recall text did not preserve the requested chorus notes: %q", rawChunk.Text)
	}
	if strings.Contains(rawChunk.Text, "Verse 2:") {
		t.Fatalf("raw recall snippet was not trimmed to the labelled section: %q", rawChunk.Text)
	}
	if rawChunk.Metadata["expansion"] != "episodic_recall" {
		t.Fatalf("raw recall metadata = %+v, want episodic_recall expansion", rawChunk.Metadata)
	}
}

func TestRetrieve_EpisodicRecallSkipsOrdinaryQueries(t *testing.T) {
	t.Parallel()
	src := newFakeSource([]fakeChunk{
		{
			ID:      "memory",
			Path:    "memory/project/eval/song-summary.md",
			Title:   "Song summary",
			Content: "The previous songwriting session included a chorus.",
			Scope:   "project_memory",
			Project: "eval",
			Session: "2023-05-28",
		},
		{
			ID:      "raw",
			Path:    "raw/lme/answer_sharegpt_song_0.md",
			Title:   "Raw conversation",
			Content: "[assistant]: Chorus:\nC D E F G A B A G F E D C",
			Scope:   "raw_lme",
			Session: "2023-05-28",
		},
	})
	r, err := New(Config{Source: src})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	resp, err := r.Retrieve(context.Background(), Request{
		Query:      "songwriting chorus",
		TopK:       3,
		CandidateK: 2,
		Mode:       ModeBM25,
		Filters: Filters{
			Scope:   "memory",
			Project: "eval",
		},
	})
	if err != nil {
		t.Fatalf("Retrieve: %v", err)
	}
	paths := chunkPaths(resp.Chunks)
	if containsPath(paths, "raw/lme/answer_sharegpt_song_0.md") {
		t.Fatalf("ordinary query unexpectedly used episodic recall: %v", paths)
	}
	if resp.Trace.EpisodicRecall {
		t.Fatalf("episodic recall trace should be false: %+v", resp.Trace)
	}
}

func TestRetrieve_EpisodicRecallSkipsExactPathFilters(t *testing.T) {
	t.Parallel()
	src := newFakeSource([]fakeChunk{
		{
			ID:      "memory",
			Path:    "memory/project/eval/song-summary.md",
			Title:   "Song summary",
			Content: "The previous conversation included one sad song with a chorus.",
			Scope:   "project_memory",
			Project: "eval",
			Session: "2023-05-28",
		},
		{
			ID:      "raw",
			Path:    "raw/lme/answer_sharegpt_song_0.md",
			Title:   "Raw conversation",
			Content: "[assistant]: Chorus:\nC D E F G A B A G F E D C",
			Scope:   "raw_lme",
			Session: "2023-05-28",
		},
	})
	r, err := New(Config{Source: src})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	resp, err := r.Retrieve(context.Background(), Request{
		Query: "previous conversation second song chorus",
		TopK:  3,
		Mode:  ModeBM25,
		Filters: Filters{
			Scope:   "memory",
			Project: "eval",
			Paths:   []string{"memory/project/eval/song-summary.md"},
		},
	})
	if err != nil {
		t.Fatalf("Retrieve: %v", err)
	}
	paths := chunkPaths(resp.Chunks)
	if containsPath(paths, "raw/lme/answer_sharegpt_song_0.md") {
		t.Fatalf("episodic recall ignored exact path filter: %v", paths)
	}
	if resp.Trace.EpisodicRecall {
		t.Fatalf("episodic recall trace should be false: %+v", resp.Trace)
	}
}

func TestRetrieve_HybridRerank_AppliesReranker(t *testing.T) {
	t.Parallel()
	src := newFakeSource(newTestCorpus())
	embedder := llm.NewFakeEmbedder(src.embedDim)
	// Reranker reverses whatever head it receives so we can assert
	// it actually ran by checking the new top slot.
	rr := rerankerFn(func(ctx context.Context, query string, chunks []RetrievedChunk) ([]RetrievedChunk, error) {
		out := make([]RetrievedChunk, len(chunks))
		for i := range chunks {
			out[i] = chunks[len(chunks)-1-i]
			out[i].RerankScore = float64(len(chunks) - i)
		}
		return out, nil
	})
	r, err := New(Config{Source: src, Embedder: embedder, Reranker: rr})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	resp, err := r.Retrieve(context.Background(), Request{
		Query: "invoice automation",
		TopK:  5,
		Mode:  ModeHybridRerank,
	})
	if err != nil {
		t.Fatalf("Retrieve: %v", err)
	}
	if !resp.Trace.Reranked {
		t.Fatalf("rerank did not fire; trace: %+v", resp.Trace)
	}
	if resp.Trace.RerankSkipReason != "" {
		t.Fatalf("rerank skip reason %q should be empty", resp.Trace.RerankSkipReason)
	}
}

func TestRetrieve_ModeFallback_WhenEmbedderMissing(t *testing.T) {
	t.Parallel()
	src := newFakeSource(newTestCorpus())
	r, err := New(Config{Source: src})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	resp, err := r.Retrieve(context.Background(), Request{
		Query: "invoice",
		Mode:  ModeHybrid,
	})
	if err != nil {
		t.Fatalf("Retrieve: %v", err)
	}
	if resp.Trace.EffectiveMode != ModeBM25 {
		t.Fatalf("effective %q, want fallback bm25", resp.Trace.EffectiveMode)
	}
	if !resp.Trace.FellBackToBM25 {
		t.Fatalf("FellBackToBM25 should be true")
	}
}

func TestRetrieve_AutoWithoutEmbedder_FallsBackSilently(t *testing.T) {
	t.Parallel()
	src := newFakeSource(newTestCorpus())
	r, err := New(Config{Source: src})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	resp, err := r.Retrieve(context.Background(), Request{
		Query: "invoice",
		Mode:  ModeAuto,
	})
	if err != nil {
		t.Fatalf("Retrieve: %v", err)
	}
	if resp.Trace.EffectiveMode != ModeBM25 {
		t.Fatalf("effective %q, want bm25", resp.Trace.EffectiveMode)
	}
	if resp.Trace.FellBackToBM25 {
		t.Fatalf("auto fallback should not flag FellBackToBM25")
	}
}

func TestRetrieve_AutoWithVectorFailure_ReportsBM25(t *testing.T) {
	t.Parallel()
	src := newFakeSource(newTestCorpus())
	src.vectorFail = errors.New("vector unavailable")
	embedder := llm.NewFakeEmbedder(src.embedDim)
	r, err := New(Config{Source: src, Embedder: embedder})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	resp, err := r.Retrieve(context.Background(), Request{
		Query: "invoice",
		Mode:  ModeAuto,
	})
	if err != nil {
		t.Fatalf("Retrieve: %v", err)
	}
	if resp.Trace.EffectiveMode != ModeBM25 {
		t.Fatalf("effective %q, want bm25 when auto vector leg fails", resp.Trace.EffectiveMode)
	}
	if resp.Trace.VectorSkipReason != "vector_error" {
		t.Fatalf("vector skip reason = %q, want vector_error", resp.Trace.VectorSkipReason)
	}
	if resp.Trace.FellBackToBM25 {
		t.Fatalf("auto vector fallback should not flag explicit BM25 fallback")
	}
}

func TestRetrieve_UnanimityShortcut_SkipsRerank(t *testing.T) {
	t.Parallel()
	// Craft a corpus where the BM25 and vector top-3 agree. We use
	// a narrow corpus with strong overlap so both legs return the
	// same three paths.
	chunks := []fakeChunk{
		{ID: "a", Path: "a.md", Title: "Alpha", Content: "alpha bravo"},
		{ID: "b", Path: "b.md", Title: "Bravo", Content: "alpha bravo"},
		{ID: "c", Path: "c.md", Title: "Charlie", Content: "alpha bravo"},
	}
	src := newFakeSource(chunks)
	embedder := llm.NewFakeEmbedder(src.embedDim)
	rerankerCalls := 0
	rr := rerankerFn(func(ctx context.Context, q string, rs []RetrievedChunk) ([]RetrievedChunk, error) {
		rerankerCalls++
		return rs, nil
	})
	r, err := New(Config{Source: src, Embedder: embedder, Reranker: rr})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	resp, err := r.Retrieve(context.Background(), Request{
		Query: "alpha bravo",
		TopK:  3,
		Mode:  ModeHybridRerank,
	})
	if err != nil {
		t.Fatalf("Retrieve: %v", err)
	}
	if resp.Trace.UnanimitySkipped {
		if rerankerCalls != 0 {
			t.Fatalf("rerank ran despite unanimity shortcut")
		}
		if resp.Trace.RerankSkipReason != "unanimity" {
			t.Fatalf("skip reason %q, want unanimity", resp.Trace.RerankSkipReason)
		}
	} else {
		// If agreements fell below the threshold, the reranker
		// should have run.
		if rerankerCalls == 0 {
			t.Fatalf("reranker never called and unanimity not flagged")
		}
	}
}

func TestRetrieve_NilSource_Errors(t *testing.T) {
	t.Parallel()
	if _, err := New(Config{}); err == nil {
		t.Fatalf("expected error for nil source")
	}
}

func TestRetrieve_BM25LegError_Surfaces(t *testing.T) {
	t.Parallel()
	src := newFakeSource(newTestCorpus())
	src.bm25Fail = errors.New("boom")
	r, err := New(Config{Source: src})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if _, err := r.Retrieve(context.Background(), Request{Query: "anything", Mode: ModeBM25}); err == nil {
		t.Fatalf("expected bm25 error to propagate")
	}
}

func TestRetrieve_Filters_NarrowCorpus(t *testing.T) {
	t.Parallel()
	corpus := newTestCorpus()
	corpus = append(corpus, fakeChunk{
		ID:    "scoped",
		Path:  "memory/project/billing/invoice.md",
		Title: "Billing invoice workflow",
	})
	src := newFakeSource(corpus)
	r, err := New(Config{Source: src})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	resp, err := r.Retrieve(context.Background(), Request{
		Query:   "invoice",
		Mode:    ModeBM25,
		Filters: Filters{PathPrefix: "memory/project/"},
		TopK:    5,
	})
	if err != nil {
		t.Fatalf("Retrieve: %v", err)
	}
	for _, c := range resp.Chunks {
		if c.Path != "memory/project/billing/invoice.md" {
			t.Fatalf("filter leak: %q", c.Path)
		}
	}
}

func TestRetrieve_BM25Fanout_MergesTemporalQueryVariants(t *testing.T) {
	t.Parallel()

	src := newFakeSource(newTestCorpus())
	src.bm25Override = func(expr string) ([]BM25Hit, bool) {
		switch {
		case strings.Contains(expr, "2024") || strings.Contains(expr, "03") || strings.Contains(expr, "08"):
			return []BM25Hit{{
				ID:      "dated",
				Path:    "raw/lme/dated.md",
				Title:   "Dated hit",
				Summary: "Temporal match",
				Content: "This happened on 2024/03/08.",
			}}, true
		case strings.Contains(strings.ToLower(expr), "friday"):
			return []BM25Hit{{
				ID:      "plain",
				Path:    "raw/lme/plain.md",
				Title:   "Plain hit",
				Summary: "Lexical match",
				Content: "We met on Friday.",
			}}, true
		default:
			return nil, false
		}
	}

	r, err := New(Config{Source: src})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	resp, err := r.Retrieve(context.Background(), Request{
		Query:        "What happened last Friday?",
		QuestionDate: "2024/03/13 (Wed) 10:00",
		TopK:         5,
		Mode:         ModeBM25,
	})
	if err != nil {
		t.Fatalf("Retrieve: %v", err)
	}
	if len(resp.Chunks) < 2 {
		t.Fatalf("expected fanout to merge at least two hits, got %d", len(resp.Chunks))
	}
	got := []string{resp.Chunks[0].Path, resp.Chunks[1].Path}
	if !containsPath(got, "raw/lme/plain.md") || !containsPath(got, "raw/lme/dated.md") {
		t.Fatalf("fanout merged paths = %v, want plain + dated hits", got)
	}
	if len(resp.Attempts) == 0 || !strings.Contains(resp.Attempts[0].Query, "||") {
		t.Fatalf("initial attempt should record fanout queries, got %+v", resp.Attempts)
	}
}

func TestRetrieve_BM25_ReweightsMostRecentDatedHits(t *testing.T) {
	t.Parallel()

	src := newFakeSource([]fakeChunk{
		{
			ID:      "older",
			Path:    "memory/global/a-older.md",
			Title:   "Market visit",
			Content: "[Observed on 2024/03/01 (Fri) 09:00]\nEarned $220 at the Downtown Farmers Market.",
		},
		{
			ID:      "newer",
			Path:    "memory/global/z-newer.md",
			Title:   "Market visit",
			Content: "[Observed on 2024/03/08 (Fri) 09:00]\nEarned $420 at the Downtown Farmers Market.",
		},
	})

	r, err := New(Config{Source: src})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	resp, err := r.Retrieve(context.Background(), Request{
		Query: "How much did I earn at the Downtown Farmers Market on my most recent visit?",
		TopK:  5,
		Mode:  ModeBM25,
	})
	if err != nil {
		t.Fatalf("Retrieve: %v", err)
	}
	if len(resp.Chunks) == 0 || resp.Chunks[0].ChunkID != "newer" {
		t.Fatalf("top chunk = %+v, want newer", resp.Chunks)
	}
}

func TestRetrieve_BM25_ReweightsClosestTemporalHintDate(t *testing.T) {
	t.Parallel()

	src := newFakeSource([]fakeChunk{
		{
			ID:      "far",
			Path:    "memory/global/a-far.md",
			Title:   "Weekly note",
			Content: "[Observed on 2024/02/02 (Fri) 10:00]\nMet the supplier and agreed the timeline.",
		},
		{
			ID:      "near",
			Path:    "memory/global/z-near.md",
			Title:   "Weekly note",
			Content: "[Observed on 2024/03/08 (Fri) 10:00]\nMet the supplier and agreed the timeline.",
		},
	})

	r, err := New(Config{Source: src})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	resp, err := r.Retrieve(context.Background(), Request{
		Query:        "What happened with the supplier last Friday?",
		QuestionDate: "2024/03/15 (Fri) 09:00",
		TopK:         5,
		Mode:         ModeBM25,
	})
	if err != nil {
		t.Fatalf("Retrieve: %v", err)
	}
	if len(resp.Chunks) == 0 || resp.Chunks[0].ChunkID != "near" {
		t.Fatalf("top chunk = %+v, want near", resp.Chunks)
	}
}

func TestRetrieve_BM25_TemporalRankingUsesEventDateMetadata(t *testing.T) {
	t.Parallel()

	src := newFakeSource([]fakeChunk{
		{
			ID:        "far",
			Path:      "memory/global/a-far.md",
			Title:     "Supplier note",
			Content:   "Met the supplier and agreed the timeline.",
			Session:   "2024-03-12",
			EventDate: "2024-03-01",
		},
		{
			ID:        "near",
			Path:      "memory/global/z-near.md",
			Title:     "Supplier note",
			Content:   "Met the supplier and agreed the timeline.",
			Session:   "2024-03-12",
			EventDate: "2024-03-08",
		},
	})

	r, err := New(Config{Source: src})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	resp, err := r.Retrieve(context.Background(), Request{
		Query:        "What happened with the supplier last Friday?",
		QuestionDate: "2024/03/15 (Fri) 09:00",
		TopK:         5,
		Mode:         ModeBM25,
	})
	if err != nil {
		t.Fatalf("Retrieve: %v", err)
	}
	if len(resp.Chunks) == 0 || resp.Chunks[0].ChunkID != "near" {
		t.Fatalf("top chunk = %+v, want near event date", resp.Chunks)
	}
}

func TestRetrieve_BM25_DropsFutureDatedHitsRelativeToQuestionDate(t *testing.T) {
	t.Parallel()

	src := newFakeSource([]fakeChunk{
		{
			ID:      "past",
			Path:    "memory/global/past.md",
			Title:   "Supplier visit",
			Content: "[Observed on 2024/03/10 (Sun) 09:00]\nMet the supplier and agreed the next steps.",
		},
		{
			ID:      "future",
			Path:    "memory/global/future.md",
			Title:   "Supplier visit",
			Content: "[Observed on 2024/03/20 (Wed) 09:00]\nMet the supplier and agreed the next steps.",
		},
		{
			ID:      "undated",
			Path:    "memory/global/undated.md",
			Title:   "Supplier visit",
			Content: "Met the supplier and agreed the next steps.",
		},
	})

	r, err := New(Config{Source: src})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	resp, err := r.Retrieve(context.Background(), Request{
		Query:        "What is the most recent supplier visit?",
		QuestionDate: "2024/03/15 (Fri) 09:00",
		TopK:         5,
		Mode:         ModeBM25,
	})
	if err != nil {
		t.Fatalf("Retrieve: %v", err)
	}
	if len(resp.Chunks) == 0 || resp.Chunks[0].ChunkID != "past" {
		t.Fatalf("top chunk = %+v, want past", resp.Chunks)
	}
	for _, chunk := range resp.Chunks {
		if chunk.ChunkID == "future" {
			t.Fatalf("future-dated chunk leaked into results: %+v", resp.Chunks)
		}
	}
}

func TestRetrieve_BM25_KeepsSameDayEvidenceRelativeToQuestionTime(t *testing.T) {
	t.Parallel()

	src := newFakeSource([]fakeChunk{
		{
			ID:      "same-day",
			Path:    "memory/global/same-day.md",
			Title:   "Supplier visit",
			Content: "[Observed on 2024/03/15 (Fri) 18:00]\nMet the supplier and agreed the same-day status status.",
			Session: "2024-03-15",
		},
		{
			ID:      "next-day",
			Path:    "memory/global/next-day.md",
			Title:   "Supplier visit",
			Content: "[Observed on 2024/03/16 (Sat) 08:00]\nMet the supplier and agreed the next-day status status status.",
			Session: "2024-03-16",
		},
	})

	r, err := New(Config{Source: src})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	resp, err := r.Retrieve(context.Background(), Request{
		Query:        "What is the supplier status?",
		QuestionDate: "2024/03/15 (Fri) 09:00",
		TopK:         5,
		Mode:         ModeBM25,
	})
	if err != nil {
		t.Fatalf("Retrieve: %v", err)
	}
	if len(resp.Chunks) == 0 || resp.Chunks[0].ChunkID != "same-day" {
		t.Fatalf("top chunk = %+v, want same-day", resp.Chunks)
	}
	for _, chunk := range resp.Chunks {
		if chunk.ChunkID == "next-day" {
			t.Fatalf("next-day chunk leaked into results: %+v", resp.Chunks)
		}
	}
}

func TestRetrieve_BM25Fanout_DropsDriftedTokenProbes(t *testing.T) {
	t.Parallel()

	src := newFakeSource(newTestCorpus())
	src.bm25Override = func(expr string) ([]BM25Hit, bool) {
		switch expr {
		case "conversation":
			return []BM25Hit{{
				ID:      "noise-conversation",
				Path:    "memory/project/noise-conversation.md",
				Title:   "Conversation note",
				Summary: "Off-topic conversation metadata",
				Content: "Conversation metadata and follow-up notes.",
			}}, true
		case "remembered":
			return []BM25Hit{{
				ID:      "noise-remembered",
				Path:    "memory/project/noise-remembered.md",
				Title:   "Remembered note",
				Summary: "Off-topic remembered preference",
				Content: "Remembered preferences and recap notes.",
			}}, true
		}
		if strings.Contains(expr, "radiation") && strings.Contains(expr, "amplified") && strings.Contains(expr, "zombie") {
			return []BM25Hit{{
				ID:      "target",
				Path:    "raw/lme/answer_sharegpt_hChsWOp_97.md",
				Title:   "",
				Summary: "",
				Content: "We finally named the Radiation Amplified zombie Fissionator.",
			}}, true
		}
		return nil, false
	}

	r, err := New(Config{Source: src})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	resp, err := r.Retrieve(context.Background(), Request{
		Query: "I was thinking back to our previous conversation about the Radiation Amplified zombie, and I was wondering if you remembered what we finally decided to name it?",
		TopK:  5,
		Mode:  ModeBM25,
	})
	if err != nil {
		t.Fatalf("Retrieve: %v", err)
	}
	if len(resp.Chunks) == 0 {
		t.Fatal("expected at least one fanout result")
	}
	if got := resp.Chunks[0].Path; got != "raw/lme/answer_sharegpt_hChsWOp_97.md" {
		t.Fatalf("top fanout hit = %q, want target transcript", got)
	}
}

func TestBuildBM25FanoutQueries_UsesPhraseProbesForCompoundQuestions(t *testing.T) {
	t.Parallel()

	got := buildBM25FanoutQueries(
		"What is the total amount I spent on the designer handbag and high-end skincare products?",
		"",
	)

	for _, want := range []string{
		"handbag cost",
		"high-end products",
		"designer handbag",
		"high-end skincare products",
	} {
		if !containsString(got, want) {
			t.Fatalf("fanout queries = %v, want query %q present", got, want)
		}
	}
}

func TestBuildBM25FanoutQueries_AddsSpecificBackEndLanguageProbe(t *testing.T) {
	t.Parallel()

	got := buildBM25FanoutQueries(
		"I wanted to follow up on our previous conversation about front-end and back-end development. Can you remind me of the specific back-end programming languages you recommended I learn?",
		"",
	)

	if len(got) < 1 {
		t.Fatalf("fanout queries = %v, want focused back-end language probe", got)
	}
	found := false
	for _, query := range got {
		if strings.Contains(query, "back-end") && strings.Contains(query, "language") {
			found = true
			break
		}
	}
	if !found {
		t.Fatalf("fanout queries = %v, want query-derived back-end language phrase present", got)
	}
}

func TestBuildBM25FanoutQueries_DoesNotPromoteSingleAdjectiveEntity(t *testing.T) {
	t.Parallel()

	got := buildBM25FanoutQueries(
		"How many Italian restaurants have I tried in my city?",
		"",
	)

	if containsString(got, "italian") {
		t.Fatalf("fanout queries = %v, do not want standalone adjective entity", got)
	}
}

func TestBuildBM25FanoutQueries_AddsTypeContextProbes(t *testing.T) {
	t.Parallel()

	got := buildBM25FanoutQueries(
		"How many different types of citrus fruits have I used in my cocktail recipes?",
		"",
	)

	for _, want := range []string{
		"cocktail citrus",
		"citrus cocktail",
	} {
		if !containsString(got, want) {
			t.Fatalf("fanout queries = %v, want type-context probe %q present", got, want)
		}
	}
	if containsString(got, "different types") {
		t.Fatalf("fanout queries = %v, do not want low-signal type phrase", got)
	}
}

func TestBuildVectorFanoutQueries_AddsTypeContextProbes(t *testing.T) {
	t.Parallel()

	got := buildVectorFanoutQueries("How many different types of citrus fruits have I used in my cocktail recipes?")

	for _, want := range []string{
		"How many different types of citrus fruits have I used in my cocktail recipes?",
		"cocktail citrus",
		"citrus cocktail",
	} {
		if !containsString(got, want) {
			t.Fatalf("vector fanout queries = %v, want %q present", got, want)
		}
	}
	if len(got) > maxVectorFanoutQueries {
		t.Fatalf("vector fanout queries = %v, want at most %d", got, maxVectorFanoutQueries)
	}
}

func TestBuildBM25FanoutQueries_DoesNotDisplaceComparisonPredicate(t *testing.T) {
	t.Parallel()

	got := buildBM25FanoutQueries(
		"Which group did I join first, 'Page Turners' or 'Marketing Professionals'?",
		"",
	)

	if len(got) == 0 || !strings.Contains(got[0], "join") {
		t.Fatalf("fanout queries = %v, want comparison predicate retained", got)
	}
}

func TestBuildBM25FanoutQueries_AddsCoverageFacetsForCompositeEntities(t *testing.T) {
	t.Parallel()

	got := buildBM25FanoutQueries(
		"What is the total number of comments on my recent Facebook Live session and my most popular YouTube video?",
		"",
	)

	for _, want := range []string{
		"facebook live session",
		"youtube video",
	} {
		if !containsString(got, want) {
			t.Fatalf("fanout queries = %v, want coverage facet %q present", got, want)
		}
	}
}

func TestBuildBM25FanoutQueries_AddsCoverageFacetsForTemporalEventSlots(t *testing.T) {
	t.Parallel()

	got := buildBM25FanoutQueries(
		"How many days passed between the day I received feedback about my car's suspension and the day I tested my new suspension setup?",
		"",
	)

	for _, want := range []string{
		"received feedback car suspension",
		"tested new suspension setup",
	} {
		if !containsString(got, want) {
			t.Fatalf("fanout queries = %v, want temporal facet %q present", got, want)
		}
	}
}

func TestBuildBM25FanoutQueries_KeepsCapitalisedEntityGlueWords(t *testing.T) {
	t.Parallel()

	got := buildBM25FanoutQueries(
		"How many days passed between my Museum of Modern Art visit and the Metropolitan Museum of Art exhibit?",
		"",
	)

	for _, want := range []string{
		"museum modern art visit",
		"metropolitan museum art exhibit",
	} {
		if !containsString(got, want) {
			t.Fatalf("fanout queries = %v, want capitalised entity facet %q present", got, want)
		}
	}
}

func TestBuildBM25FanoutQueries_AddsCountTotalListProbes(t *testing.T) {
	t.Parallel()

	got := buildBM25FanoutQueries(
		"How many total pieces of writing have I completed since I started writing again three weeks ago, including short stories, poems, and pieces for the writing challenge?",
		"",
	)

	for _, want := range []string{
		"short stories",
		"poems",
		"writing challenge",
	} {
		if !containsString(got, want) {
			t.Fatalf("fanout queries = %v, want count-total query %q present", got, want)
		}
	}
}

func TestBuildBM25FanoutQueries_AddsDateArithmeticEventProbes(t *testing.T) {
	t.Parallel()

	got := buildBM25FanoutQueries(
		"How many days ago did I launch my website when I signed a contract with my first client?",
		"",
	)

	for _, want := range []string{
		"launch website",
		"launched website",
		"signed contract first client",
	} {
		if !containsString(got, want) {
			t.Fatalf("fanout queries = %v, want date arithmetic query %q present", got, want)
		}
	}
}

func TestBuildBM25FanoutQueries_AddsRecallTitleProbeAfterPredicate(t *testing.T) {
	t.Parallel()

	got := buildBM25FanoutQueries(
		"I was going through our previous chat about Tanqueray's Spiritual Life treatise. Which chapter discusses vocal prayer and meditation?",
		"",
	)

	if !containsString(got, "tanqueray spiritual life") {
		t.Fatalf("fanout queries = %v, want constrained title phrase", got)
	}
	if len(got) == 0 || !strings.Contains(got[0], "vocal prayer") {
		t.Fatalf("fanout queries = %v, want predicate phrase retained first", got)
	}
}

func TestBuildBM25FanoutQueries_AddsActionDateProbeForWhenDidISubmit(t *testing.T) {
	t.Parallel()

	got := buildBM25FanoutQueries(
		"When did I submit my research paper on sentiment analysis?",
		"",
	)

	if len(got) < 2 {
		t.Fatalf("fanout queries = %v, want submission-date probes", got)
	}
	for _, want := range []string{
		"sentiment analysis submission date",
		"research paper submission date",
		"research paper",
		"sentiment analysis",
	} {
		if !containsString(got, want) {
			t.Fatalf("fanout queries = %v, want query %q present", got, want)
		}
	}
}

func TestBuildBM25FanoutQueries_AddsInspirationSourceProbe(t *testing.T) {
	t.Parallel()

	got := buildBM25FanoutQueries(
		"How can I find new inspiration for my paintings?",
		"",
	)

	if len(got) < 2 {
		t.Fatalf("fanout queries = %v, want inspiration-source probe", got)
	}
	if !containsString(got, "paintings inspiration") {
		t.Fatalf("fanout queries = %v, want paintings inspiration present", got)
	}
	if containsString(got, "paintings social media tutorials") {
		t.Fatalf("fanout queries = %v, should not include domain-specific social media tutorials probe", got)
	}
}

func TestRetrieve_BM25Fanout_UsesPhraseProbesForCompoundTotals(t *testing.T) {
	t.Parallel()

	src := newFakeSource(newTestCorpus())
	src.bm25Override = func(expr string) ([]BM25Hit, bool) {
		switch {
		case strings.Contains(expr, "designer") && strings.Contains(expr, "handbag") && !strings.Contains(expr, "skincare"):
			return []BM25Hit{{
				ID:      "bag",
				Path:    "raw/lme/designer-handbag.md",
				Title:   "Designer handbag",
				Summary: "High-value purchase",
				Content: "I spent 1800 on the designer handbag.",
			}}, true
		case strings.Contains(expr, "skincare") && strings.Contains(expr, "products"):
			return []BM25Hit{{
				ID:      "skincare",
				Path:    "raw/lme/skincare-products.md",
				Title:   "Skincare products",
				Summary: "Beauty purchase",
				Content: "I spent 320 on the high-end skincare products.",
			}}, true
		case strings.Contains(expr, "handbag") && strings.Contains(expr, "skincare"):
			return nil, true
		default:
			return nil, false
		}
	}

	r, err := New(Config{Source: src})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	resp, err := r.Retrieve(context.Background(), Request{
		Query: "What is the total amount I spent on the designer handbag and high-end skincare products?",
		TopK:  5,
		Mode:  ModeBM25,
	})
	if err != nil {
		t.Fatalf("Retrieve: %v", err)
	}
	if len(resp.Chunks) < 2 {
		t.Fatalf("expected phrase fanout to merge two hits, got %d", len(resp.Chunks))
	}
	got := []string{resp.Chunks[0].Path, resp.Chunks[1].Path}
	if !containsPath(got, "raw/lme/designer-handbag.md") || !containsPath(got, "raw/lme/skincare-products.md") {
		t.Fatalf("fanout merged paths = %v, want both compound item hits", got)
	}
	if len(resp.Attempts) == 0 {
		t.Fatalf("expected initial attempt trace, got none")
	}
	if !strings.Contains(resp.Attempts[0].Query, "designer") || !strings.Contains(resp.Attempts[0].Query, "skincare") {
		t.Fatalf("initial attempt should record phrase probes, got %+v", resp.Attempts)
	}
}

func TestRetrieve_TrigramFallback_RespectsExactPathFilters(t *testing.T) {
	t.Parallel()

	src := newFakeSource(nil)
	src.bm25Override = func(expr string) ([]BM25Hit, bool) {
		return nil, true
	}

	r, err := New(Config{
		Source: src,
		TrigramChunks: []trigramChunk{
			{
				ID:      "blocked",
				Path:    "notes/photosynthasis.md",
				Title:   "Blocked note",
				Summary: "blocked summary",
				Content: "blocked content",
			},
			{
				ID:      "allowed",
				Path:    "notes/photosynthasis-log.md",
				Title:   "Allowed note",
				Summary: "allowed summary",
				Content: "allowed content",
			},
		},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	resp, err := r.Retrieve(context.Background(), Request{
		Query: "photosynthasis",
		TopK:  5,
		Mode:  ModeBM25,
		Filters: Filters{
			Paths: []string{"notes/photosynthasis-log.md"},
		},
	})
	if err != nil {
		t.Fatalf("Retrieve: %v", err)
	}
	if len(resp.Chunks) == 0 {
		t.Fatal("expected filtered trigram fallback hit")
	}
	for _, chunk := range resp.Chunks {
		if chunk.Path != "notes/photosynthasis-log.md" {
			t.Fatalf("path leak: %s", chunk.Path)
		}
	}
	found := false
	for _, attempt := range resp.Attempts {
		if attempt.Reason == "trigram_fuzzy" {
			found = true
			break
		}
	}
	if !found {
		t.Fatalf("expected trigram_fuzzy attempt, got %+v", resp.Attempts)
	}
}

func TestRetrieve_TrigramFallback_RespectsMetadataFilters(t *testing.T) {
	t.Parallel()

	src := newFakeSource(nil)
	src.bm25Override = func(expr string) ([]BM25Hit, bool) {
		return nil, true
	}

	r, err := New(Config{
		Source: src,
		TrigramChunks: []trigramChunk{
			{
				ID:      "wrong-scope",
				Path:    "raw/lme/photosynthasis.md",
				Title:   "Wrong scope",
				Tags:    "science",
				Scope:   "raw_lme",
				Project: "eval-lme",
				Session: "2024-03-10",
			},
			{
				ID:      "wrong-project",
				Path:    "memory/projects/other/photosynthasis.md",
				Title:   "Wrong project",
				Tags:    "science",
				Scope:   "project_memory",
				Project: "other",
				Session: "2024-03-10",
			},
			{
				ID:      "wrong-tag",
				Path:    "memory/projects/eval-lme/photosynthasis-cooking.md",
				Title:   "Wrong tag",
				Tags:    "cooking",
				Scope:   "project_memory",
				Project: "eval-lme",
				Session: "2024-03-10",
			},
			{
				ID:      "wrong-date",
				Path:    "memory/projects/eval-lme/photosynthasis-future.md",
				Title:   "Wrong date",
				Tags:    "science",
				Scope:   "project_memory",
				Project: "eval-lme",
				Session: "2024-04-10",
			},
			{
				ID:      "allowed",
				Path:    "memory/projects/eval-lme/photosynthasis-log.md",
				Title:   "Allowed note",
				Summary: "allowed summary",
				Content: "allowed content",
				Tags:    "science botany",
				Scope:   "project_memory",
				Project: "eval-lme",
				Session: "2024-03-10",
			},
		},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	resp, err := r.Retrieve(context.Background(), Request{
		Query: "photosynthasis",
		TopK:  5,
		Mode:  ModeBM25,
		Filters: Filters{
			Scope:   "project_memory",
			Project: "eval-lme",
			Tags:    []string{"science"},
			DateTo:  time.Date(2024, 3, 31, 23, 59, 59, 0, time.UTC),
		},
	})
	if err != nil {
		t.Fatalf("Retrieve: %v", err)
	}
	if got := chunkPaths(resp.Chunks); len(got) != 1 || got[0] != "memory/projects/eval-lme/photosynthasis-log.md" {
		t.Fatalf("filtered trigram hits = %v, want only allowed project note", got)
	}
}

func containsPath(paths []string, want string) bool {
	for _, path := range paths {
		if path == want {
			return true
		}
	}
	return false
}

func chunkPaths(chunks []RetrievedChunk) []string {
	paths := make([]string, 0, len(chunks))
	for _, chunk := range chunks {
		paths = append(paths, chunk.Path)
	}
	return paths
}

func containsString(values []string, want string) bool {
	for _, value := range values {
		if value == want {
			return true
		}
	}
	return false
}

// rerankerFn is a function-as-Reranker adapter used by tests.
type rerankerFn func(ctx context.Context, query string, chunks []RetrievedChunk) ([]RetrievedChunk, error)

func (f rerankerFn) Rerank(ctx context.Context, query string, chunks []RetrievedChunk) ([]RetrievedChunk, error) {
	return f(ctx, query, chunks)
}

func (f rerankerFn) Name() string { return "test-reranker" }
