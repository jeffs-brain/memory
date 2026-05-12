// SPDX-License-Identifier: Apache-2.0

package memory

import (
	"context"
	"errors"
	"fmt"
	"testing"
	"time"

	"github.com/jeffs-brain/memory/go/brain"
	"github.com/jeffs-brain/memory/go/store/mem"
)

func newTestEpisodeStore(t *testing.T) *BrainEpisodeStore {
	t.Helper()
	s := mem.New()
	t.Cleanup(func() { _ = s.Close() })
	return NewBrainEpisodeStore(s)
}

func makeEpisode(sessionID string) EpisodeRecord {
	return EpisodeRecord{
		SessionID:           sessionID,
		ActorID:             "test-actor",
		Scope:               EpisodeScopeProject,
		Name:                "Episode " + sessionID,
		Summary:             "Fixed a critical auth bug in the login flow",
		Outcome:             EpisodeOutcomeSuccess,
		RetryFeedback:       "",
		ShouldRecordEpisode: true,
		OpenQuestions:        []string{},
		Heuristics: []EpisodeHeuristic{
			{
				Rule:       "Check token expiry before refreshing",
				Context:    "auth module",
				Confidence: "medium",
				Category:   "auth",
				Scope:      "project",
			},
		},
		Tags: []string{"auth", "bugfix"},
		Signals: EpisodeSignals{
			MessageCount:            12,
			SubstantiveMessageCount: 10,
			UserMessageCount:        4,
			AssistantMessageCount:   4,
			ToolMessageCount:        4,
			ToolCallCount:           6,
			WriteSignal:             true,
			EditSignal:              true,
			ToolSignal:              true,
		},
	}
}

func TestBrainEpisodeStore_CRUD(t *testing.T) {
	tests := []struct {
		name string
		fn   func(t *testing.T, store *BrainEpisodeStore)
	}{
		{"CreateEpisode stores and persists", testCreateEpisode},
		{"CreateEpisode rejects duplicate", testCreateEpisodeDuplicate},
		{"CreateEpisode sets defaults", testCreateEpisodeDefaults},
		{"GetEpisode returns correct episode", testGetEpisode},
		{"GetEpisode non-existent returns error", testGetEpisodeNotFound},
		{"ListEpisodes returns all", testListEpisodesAll},
		{"ListEpisodes returns empty slice not nil", testListEpisodesEmpty},
		{"ListEpisodes filters by actor", testListEpisodesFilterActor},
		{"ListEpisodes filters by scope", testListEpisodesFilterScope},
		{"ListEpisodes filters by outcome", testListEpisodesFilterOutcome},
		{"ListEpisodes filters by session", testListEpisodesFilterSession},
		{"ListEpisodes filters by tags", testListEpisodesFilterTags},
		{"ListEpisodes filters by date range", testListEpisodesFilterDateRange},
		{"ListEpisodes overlapping date ranges", testListEpisodesOverlappingDateRanges},
		{"ListEpisodes respects limit", testListEpisodesLimit},
		{"UpdateEpisode persists changes", testUpdateEpisode},
		{"UpdateEpisode non-existent returns error", testUpdateEpisodeNotFound},
		{"DeleteEpisode removes episode", testDeleteEpisode},
		{"DeleteEpisode non-existent returns error", testDeleteEpisodeNotFound},
		{"QueryEpisodes by text", testQueryEpisodesByText},
		{"QueryEpisodes by participant", testQueryEpisodesByParticipant},
		{"QueryEpisodes empty query returns empty", testQueryEpisodesEmpty},
		{"QueryEpisodes respects limit", testQueryEpisodesLimit},
		{"Round-trip preserves all fields", testRoundTripPreservesFields},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			store := newTestEpisodeStore(t)
			tc.fn(t, store)
		})
	}
}

func testCreateEpisode(t *testing.T, store *BrainEpisodeStore) {
	ctx := context.Background()
	ep := makeEpisode("session-001")

	err := store.CreateEpisode(ctx, ep)
	if err != nil {
		t.Fatalf("CreateEpisode: %v", err)
	}

	got, err := store.GetEpisode(ctx, "session-001")
	if err != nil {
		t.Fatalf("GetEpisode: %v", err)
	}
	if got.SessionID != "session-001" {
		t.Errorf("SessionID = %q, want %q", got.SessionID, "session-001")
	}
	if got.Summary != ep.Summary {
		t.Errorf("Summary = %q, want %q", got.Summary, ep.Summary)
	}
}

func testCreateEpisodeDuplicate(t *testing.T, store *BrainEpisodeStore) {
	ctx := context.Background()
	ep := makeEpisode("session-dup")

	if err := store.CreateEpisode(ctx, ep); err != nil {
		t.Fatalf("first CreateEpisode: %v", err)
	}
	err := store.CreateEpisode(ctx, ep)
	if err == nil {
		t.Fatal("second CreateEpisode should have failed")
	}
}

func testCreateEpisodeDefaults(t *testing.T, store *BrainEpisodeStore) {
	ctx := context.Background()
	ep := EpisodeRecord{
		SessionID: "session-defaults",
		Summary:   "test defaults",
		Outcome:   EpisodeOutcomeSuccess,
	}

	if err := store.CreateEpisode(ctx, ep); err != nil {
		t.Fatalf("CreateEpisode: %v", err)
	}

	got, err := store.GetEpisode(ctx, "session-defaults")
	if err != nil {
		t.Fatalf("GetEpisode: %v", err)
	}
	if got.Created == "" {
		t.Error("Created should be auto-set")
	}
	if got.Modified == "" {
		t.Error("Modified should be auto-set")
	}
	if got.Name == "" {
		t.Error("Name should be auto-set")
	}
	if got.OpenQuestions == nil {
		t.Error("OpenQuestions should not be nil")
	}
	if got.Heuristics == nil {
		t.Error("Heuristics should not be nil")
	}
	if got.Tags == nil {
		t.Error("Tags should not be nil")
	}
}

func testGetEpisode(t *testing.T, store *BrainEpisodeStore) {
	ctx := context.Background()
	ep := makeEpisode("session-get")

	if err := store.CreateEpisode(ctx, ep); err != nil {
		t.Fatalf("CreateEpisode: %v", err)
	}

	got, err := store.GetEpisode(ctx, "session-get")
	if err != nil {
		t.Fatalf("GetEpisode: %v", err)
	}
	if got.SessionID != "session-get" {
		t.Errorf("SessionID = %q, want %q", got.SessionID, "session-get")
	}
	if got.Outcome != EpisodeOutcomeSuccess {
		t.Errorf("Outcome = %q, want %q", got.Outcome, EpisodeOutcomeSuccess)
	}
	if got.ActorID != "test-actor" {
		t.Errorf("ActorID = %q, want %q", got.ActorID, "test-actor")
	}
}

func testGetEpisodeNotFound(t *testing.T, store *BrainEpisodeStore) {
	ctx := context.Background()
	_, err := store.GetEpisode(ctx, "nonexistent-session")
	if err == nil {
		t.Fatal("GetEpisode should return error for non-existent session")
	}
	if !errors.Is(err, brain.ErrNotFound) {
		t.Logf("error = %v (wraps ErrNotFound: %v)", err, errors.Is(err, brain.ErrNotFound))
	}
}

func testListEpisodesAll(t *testing.T, store *BrainEpisodeStore) {
	ctx := context.Background()
	for i := 0; i < 3; i++ {
		ep := makeEpisode(fmt.Sprintf("session-list-%d", i))
		mustCreateEp(t, store, ctx, ep)
	}

	episodes, err := store.ListEpisodes(ctx, EpisodeListOptions{})
	if err != nil {
		t.Fatalf("ListEpisodes: %v", err)
	}
	if len(episodes) != 3 {
		t.Errorf("got %d episodes, want 3", len(episodes))
	}
}

func testListEpisodesEmpty(t *testing.T, store *BrainEpisodeStore) {
	ctx := context.Background()
	episodes, err := store.ListEpisodes(ctx, EpisodeListOptions{})
	if err != nil {
		t.Fatalf("ListEpisodes: %v", err)
	}
	if episodes == nil {
		t.Fatal("ListEpisodes should return empty slice, not nil")
	}
	if len(episodes) != 0 {
		t.Errorf("got %d episodes, want 0", len(episodes))
	}
}

func testListEpisodesFilterActor(t *testing.T, store *BrainEpisodeStore) {
	ctx := context.Background()

	ep1 := makeEpisode("session-actor-1")
	ep1.ActorID = "actor-alpha"
	ep2 := makeEpisode("session-actor-2")
	ep2.ActorID = "actor-beta"

	mustCreateEp(t, store, ctx, ep1)
	mustCreateEp(t, store, ctx, ep2)

	episodes, err := store.ListEpisodes(ctx, EpisodeListOptions{ActorID: "actor-alpha"})
	if err != nil {
		t.Fatalf("ListEpisodes: %v", err)
	}
	if len(episodes) != 1 {
		t.Fatalf("got %d episodes, want 1", len(episodes))
	}
	if episodes[0].ActorID != "actor-alpha" {
		t.Errorf("ActorID = %q, want %q", episodes[0].ActorID, "actor-alpha")
	}
}

func testListEpisodesFilterScope(t *testing.T, store *BrainEpisodeStore) {
	ctx := context.Background()

	ep1 := makeEpisode("session-scope-1")
	ep1.Scope = EpisodeScopeGlobal
	ep2 := makeEpisode("session-scope-2")
	ep2.Scope = EpisodeScopeProject

	mustCreateEp(t, store, ctx, ep1)
	mustCreateEp(t, store, ctx, ep2)

	episodes, err := store.ListEpisodes(ctx, EpisodeListOptions{Scope: EpisodeScopeGlobal})
	if err != nil {
		t.Fatalf("ListEpisodes: %v", err)
	}
	if len(episodes) != 1 {
		t.Fatalf("got %d episodes, want 1", len(episodes))
	}
	if episodes[0].Scope != EpisodeScopeGlobal {
		t.Errorf("Scope = %q, want %q", episodes[0].Scope, EpisodeScopeGlobal)
	}
}

func testListEpisodesFilterOutcome(t *testing.T, store *BrainEpisodeStore) {
	ctx := context.Background()

	ep1 := makeEpisode("session-outcome-1")
	ep1.Outcome = EpisodeOutcomeSuccess
	ep2 := makeEpisode("session-outcome-2")
	ep2.Outcome = EpisodeOutcomeFailure

	mustCreateEp(t, store, ctx, ep1)
	mustCreateEp(t, store, ctx, ep2)

	episodes, err := store.ListEpisodes(ctx, EpisodeListOptions{Outcome: EpisodeOutcomeFailure})
	if err != nil {
		t.Fatalf("ListEpisodes: %v", err)
	}
	if len(episodes) != 1 {
		t.Fatalf("got %d episodes, want 1", len(episodes))
	}
	if episodes[0].Outcome != EpisodeOutcomeFailure {
		t.Errorf("Outcome = %q, want %q", episodes[0].Outcome, EpisodeOutcomeFailure)
	}
}

func testListEpisodesFilterSession(t *testing.T, store *BrainEpisodeStore) {
	ctx := context.Background()

	mustCreateEp(t, store, ctx, makeEpisode("session-filter-a"))
	mustCreateEp(t, store, ctx, makeEpisode("session-filter-b"))

	episodes, err := store.ListEpisodes(ctx, EpisodeListOptions{SessionID: "session-filter-a"})
	if err != nil {
		t.Fatalf("ListEpisodes: %v", err)
	}
	if len(episodes) != 1 {
		t.Fatalf("got %d episodes, want 1", len(episodes))
	}
	if episodes[0].SessionID != "session-filter-a" {
		t.Errorf("SessionID = %q, want %q", episodes[0].SessionID, "session-filter-a")
	}
}

func testListEpisodesFilterTags(t *testing.T, store *BrainEpisodeStore) {
	ctx := context.Background()

	ep1 := makeEpisode("session-tags-1")
	ep1.Tags = []string{"auth", "bugfix"}
	ep2 := makeEpisode("session-tags-2")
	ep2.Tags = []string{"feature", "api"}
	ep3 := makeEpisode("session-tags-3")
	ep3.Tags = []string{"auth", "feature"}

	mustCreateEp(t, store, ctx, ep1)
	mustCreateEp(t, store, ctx, ep2)
	mustCreateEp(t, store, ctx, ep3)

	episodes, err := store.ListEpisodes(ctx, EpisodeListOptions{Tags: []string{"auth"}})
	if err != nil {
		t.Fatalf("ListEpisodes: %v", err)
	}
	if len(episodes) != 2 {
		t.Errorf("got %d episodes matching tag 'auth', want 2", len(episodes))
	}
}

func testListEpisodesFilterDateRange(t *testing.T, store *BrainEpisodeStore) {
	ctx := context.Background()

	baseTime := time.Date(2025, 3, 15, 10, 0, 0, 0, time.UTC)

	ep1 := makeEpisode("session-date-1")
	ep1.EndedAt = baseTime.Add(-48 * time.Hour).Format(time.RFC3339)
	ep2 := makeEpisode("session-date-2")
	ep2.EndedAt = baseTime.Format(time.RFC3339)
	ep3 := makeEpisode("session-date-3")
	ep3.EndedAt = baseTime.Add(48 * time.Hour).Format(time.RFC3339)

	mustCreateEp(t, store, ctx, ep1)
	mustCreateEp(t, store, ctx, ep2)
	mustCreateEp(t, store, ctx, ep3)

	from := baseTime.Add(-1 * time.Hour)
	to := baseTime.Add(1 * time.Hour)
	episodes, err := store.ListEpisodes(ctx, EpisodeListOptions{
		From: &from,
		To:   &to,
	})
	if err != nil {
		t.Fatalf("ListEpisodes: %v", err)
	}
	if len(episodes) != 1 {
		t.Fatalf("got %d episodes in date range, want 1", len(episodes))
	}
	if episodes[0].SessionID != "session-date-2" {
		t.Errorf("SessionID = %q, want %q", episodes[0].SessionID, "session-date-2")
	}
}

func testListEpisodesOverlappingDateRanges(t *testing.T, store *BrainEpisodeStore) {
	ctx := context.Background()

	t1 := time.Date(2025, 6, 1, 10, 0, 0, 0, time.UTC)
	t2 := time.Date(2025, 6, 2, 10, 0, 0, 0, time.UTC)
	t3 := time.Date(2025, 6, 3, 10, 0, 0, 0, time.UTC)
	t4 := time.Date(2025, 6, 4, 10, 0, 0, 0, time.UTC)

	ep1 := makeEpisode("session-overlap-1")
	ep1.StartedAt = t1.Format(time.RFC3339)
	ep1.EndedAt = t2.Format(time.RFC3339)
	ep2 := makeEpisode("session-overlap-2")
	ep2.StartedAt = t2.Format(time.RFC3339)
	ep2.EndedAt = t3.Format(time.RFC3339)
	ep3 := makeEpisode("session-overlap-3")
	ep3.StartedAt = t3.Format(time.RFC3339)
	ep3.EndedAt = t4.Format(time.RFC3339)

	mustCreateEp(t, store, ctx, ep1)
	mustCreateEp(t, store, ctx, ep2)
	mustCreateEp(t, store, ctx, ep3)

	// Query for range that covers ep1.endedAt=t2 and ep2.endedAt=t3
	from := t1.Add(12 * time.Hour)
	to := t3.Add(1 * time.Hour)
	episodes, err := store.ListEpisodes(ctx, EpisodeListOptions{
		From: &from,
		To:   &to,
	})
	if err != nil {
		t.Fatalf("ListEpisodes: %v", err)
	}
	if len(episodes) != 2 {
		t.Errorf("got %d episodes in overlapping range, want 2", len(episodes))
	}
}

func testListEpisodesLimit(t *testing.T, store *BrainEpisodeStore) {
	ctx := context.Background()

	for i := 0; i < 5; i++ {
		ep := makeEpisode(fmt.Sprintf("session-limit-%d", i))
		mustCreateEp(t, store, ctx, ep)
	}

	episodes, err := store.ListEpisodes(ctx, EpisodeListOptions{Limit: 2})
	if err != nil {
		t.Fatalf("ListEpisodes: %v", err)
	}
	if len(episodes) != 2 {
		t.Errorf("got %d episodes with limit 2, want 2", len(episodes))
	}
}

func testUpdateEpisode(t *testing.T, store *BrainEpisodeStore) {
	ctx := context.Background()
	ep := makeEpisode("session-update")
	mustCreateEp(t, store, ctx, ep)

	ep.Summary = "Updated summary after code review"
	ep.Outcome = EpisodeOutcomePartial
	ep.Tags = []string{"updated", "review"}

	if err := store.UpdateEpisode(ctx, ep); err != nil {
		t.Fatalf("UpdateEpisode: %v", err)
	}

	got, err := store.GetEpisode(ctx, "session-update")
	if err != nil {
		t.Fatalf("GetEpisode: %v", err)
	}
	if got.Summary != "Updated summary after code review" {
		t.Errorf("Summary = %q, want %q", got.Summary, "Updated summary after code review")
	}
	if got.Outcome != EpisodeOutcomePartial {
		t.Errorf("Outcome = %q, want %q", got.Outcome, EpisodeOutcomePartial)
	}
}

func testUpdateEpisodeNotFound(t *testing.T, store *BrainEpisodeStore) {
	ctx := context.Background()
	ep := makeEpisode("session-nonexistent")

	err := store.UpdateEpisode(ctx, ep)
	if err == nil {
		t.Fatal("UpdateEpisode should return error for non-existent session")
	}
	if !errors.Is(err, brain.ErrNotFound) {
		t.Logf("error = %v (wraps ErrNotFound: %v)", err, errors.Is(err, brain.ErrNotFound))
	}
}

func testDeleteEpisode(t *testing.T, store *BrainEpisodeStore) {
	ctx := context.Background()
	ep := makeEpisode("session-delete")
	mustCreateEp(t, store, ctx, ep)

	if err := store.DeleteEpisode(ctx, "session-delete"); err != nil {
		t.Fatalf("DeleteEpisode: %v", err)
	}

	_, err := store.GetEpisode(ctx, "session-delete")
	if err == nil {
		t.Fatal("GetEpisode should return error after deletion")
	}

	episodes, err := store.ListEpisodes(ctx, EpisodeListOptions{})
	if err != nil {
		t.Fatalf("ListEpisodes: %v", err)
	}
	if len(episodes) != 0 {
		t.Errorf("got %d episodes after deletion, want 0", len(episodes))
	}
}

func testDeleteEpisodeNotFound(t *testing.T, store *BrainEpisodeStore) {
	ctx := context.Background()
	err := store.DeleteEpisode(ctx, "nonexistent-delete")
	if err == nil {
		t.Fatal("DeleteEpisode should return error for non-existent session")
	}
	if !errors.Is(err, brain.ErrNotFound) {
		t.Logf("error = %v (wraps ErrNotFound: %v)", err, errors.Is(err, brain.ErrNotFound))
	}
}

func testQueryEpisodesByText(t *testing.T, store *BrainEpisodeStore) {
	ctx := context.Background()

	ep1 := makeEpisode("session-query-1")
	ep1.Summary = "Fixed authentication bug in login flow"
	ep1.Tags = []string{"auth", "bugfix"}
	ep2 := makeEpisode("session-query-2")
	ep2.Summary = "Added new REST API endpoints for users"
	ep2.Tags = []string{"api", "feature"}
	ep3 := makeEpisode("session-query-3")
	ep3.Summary = "Refactored database connection pooling"
	ep3.Tags = []string{"database", "refactor"}

	mustCreateEp(t, store, ctx, ep1)
	mustCreateEp(t, store, ctx, ep2)
	mustCreateEp(t, store, ctx, ep3)

	hits, err := store.QueryEpisodes(ctx, EpisodeQueryOptions{
		Query: "auth",
	})
	if err != nil {
		t.Fatalf("QueryEpisodes: %v", err)
	}
	if len(hits) == 0 {
		t.Fatal("QueryEpisodes returned no results for 'auth'")
	}
	if hits[0].SessionID != "session-query-1" {
		t.Errorf("top hit SessionID = %q, want %q", hits[0].SessionID, "session-query-1")
	}
	if hits[0].Score <= 0 {
		t.Errorf("top hit Score = %d, want > 0", hits[0].Score)
	}
}

func testQueryEpisodesByParticipant(t *testing.T, store *BrainEpisodeStore) {
	ctx := context.Background()

	ep1 := makeEpisode("session-participant-1")
	ep1.ActorID = "alice"
	ep2 := makeEpisode("session-participant-2")
	ep2.ActorID = "bob"

	mustCreateEp(t, store, ctx, ep1)
	mustCreateEp(t, store, ctx, ep2)

	hits, err := store.QueryEpisodes(ctx, EpisodeQueryOptions{
		EpisodeListOptions: EpisodeListOptions{ActorID: "alice"},
		Query:              "auth",
	})
	if err != nil {
		t.Fatalf("QueryEpisodes: %v", err)
	}
	for _, hit := range hits {
		if hit.ActorID != "alice" {
			t.Errorf("hit ActorID = %q, want %q", hit.ActorID, "alice")
		}
	}
}

func testQueryEpisodesEmpty(t *testing.T, store *BrainEpisodeStore) {
	ctx := context.Background()

	hits, err := store.QueryEpisodes(ctx, EpisodeQueryOptions{Query: ""})
	if err != nil {
		t.Fatalf("QueryEpisodes: %v", err)
	}
	if hits == nil {
		t.Fatal("QueryEpisodes should return empty slice, not nil")
	}
	if len(hits) != 0 {
		t.Errorf("got %d hits for empty query, want 0", len(hits))
	}
}

func testQueryEpisodesLimit(t *testing.T, store *BrainEpisodeStore) {
	ctx := context.Background()

	for i := 0; i < 5; i++ {
		ep := makeEpisode(fmt.Sprintf("session-qlimit-%d", i))
		ep.Summary = "authentication token refresh handling"
		mustCreateEp(t, store, ctx, ep)
	}

	hits, err := store.QueryEpisodes(ctx, EpisodeQueryOptions{
		EpisodeListOptions: EpisodeListOptions{Limit: 2},
		Query:              "auth",
	})
	if err != nil {
		t.Fatalf("QueryEpisodes: %v", err)
	}
	if len(hits) > 2 {
		t.Errorf("got %d hits with limit 2, want <= 2", len(hits))
	}
}

func testRoundTripPreservesFields(t *testing.T, store *BrainEpisodeStore) {
	ctx := context.Background()
	now := time.Now().UTC().Format(time.RFC3339)

	ep := EpisodeRecord{
		SessionID:           "session-roundtrip",
		ActorID:             "actor-roundtrip",
		Scope:               EpisodeScopeGlobal,
		Name:                "Roundtrip Test Episode",
		Summary:             "Testing that all fields survive persistence",
		Outcome:             EpisodeOutcomePartial,
		RetryFeedback:       "Try using a different approach next time",
		ShouldRecordEpisode: true,
		OpenQuestions:        []string{"Should we cache the result?", "Is the timeout too short?"},
		Heuristics: []EpisodeHeuristic{
			{
				Rule:        "Always validate input before processing",
				Context:     "API endpoints",
				Confidence:  "high",
				Category:    "validation",
				Scope:       "global",
				AntiPattern: false,
			},
			{
				Rule:        "Avoid using global mutable state",
				Context:     "concurrency",
				Confidence:  "medium",
				Category:    "architecture",
				Scope:       "project",
				AntiPattern: true,
			},
		},
		Tags:      []string{"roundtrip", "test", "validation"},
		Created:   now,
		Modified:  now,
		StartedAt: now,
		EndedAt:   now,
		Signals: EpisodeSignals{
			MessageCount:            20,
			SubstantiveMessageCount: 18,
			UserMessageCount:        6,
			AssistantMessageCount:   6,
			ToolMessageCount:        8,
			ToolCallCount:           10,
			WriteSignal:             true,
			EditSignal:              false,
			ToolSignal:              true,
		},
	}

	if err := store.CreateEpisode(ctx, ep); err != nil {
		t.Fatalf("CreateEpisode: %v", err)
	}

	got, err := store.GetEpisode(ctx, "session-roundtrip")
	if err != nil {
		t.Fatalf("GetEpisode: %v", err)
	}

	if got.SessionID != ep.SessionID {
		t.Errorf("SessionID = %q, want %q", got.SessionID, ep.SessionID)
	}
	if got.ActorID != ep.ActorID {
		t.Errorf("ActorID = %q, want %q", got.ActorID, ep.ActorID)
	}
	if got.Scope != ep.Scope {
		t.Errorf("Scope = %q, want %q", got.Scope, ep.Scope)
	}
	if got.Summary != ep.Summary {
		t.Errorf("Summary = %q, want %q", got.Summary, ep.Summary)
	}
	if got.Outcome != ep.Outcome {
		t.Errorf("Outcome = %q, want %q", got.Outcome, ep.Outcome)
	}
	if got.RetryFeedback != ep.RetryFeedback {
		t.Errorf("RetryFeedback = %q, want %q", got.RetryFeedback, ep.RetryFeedback)
	}
	if len(got.OpenQuestions) != len(ep.OpenQuestions) {
		t.Errorf("OpenQuestions len = %d, want %d", len(got.OpenQuestions), len(ep.OpenQuestions))
	}
	if len(got.Heuristics) != len(ep.Heuristics) {
		t.Fatalf("Heuristics len = %d, want %d", len(got.Heuristics), len(ep.Heuristics))
	}
	if got.Heuristics[0].Rule != ep.Heuristics[0].Rule {
		t.Errorf("Heuristics[0].Rule = %q, want %q", got.Heuristics[0].Rule, ep.Heuristics[0].Rule)
	}
	if got.Heuristics[1].AntiPattern != true {
		t.Error("Heuristics[1].AntiPattern should be true")
	}
	if len(got.Tags) != len(ep.Tags) {
		t.Errorf("Tags len = %d, want %d", len(got.Tags), len(ep.Tags))
	}
	if got.Signals.MessageCount != ep.Signals.MessageCount {
		t.Errorf("Signals.MessageCount = %d, want %d", got.Signals.MessageCount, ep.Signals.MessageCount)
	}
	if got.Signals.WriteSignal != ep.Signals.WriteSignal {
		t.Errorf("Signals.WriteSignal = %v, want %v", got.Signals.WriteSignal, ep.Signals.WriteSignal)
	}
	if got.Signals.EditSignal != ep.Signals.EditSignal {
		t.Errorf("Signals.EditSignal = %v, want %v", got.Signals.EditSignal, ep.Signals.EditSignal)
	}
}

func mustCreateEp(t *testing.T, store *BrainEpisodeStore, ctx context.Context, ep EpisodeRecord) {
	t.Helper()
	if err := store.CreateEpisode(ctx, ep); err != nil {
		t.Fatalf("CreateEpisode(%s): %v", ep.SessionID, err)
	}
}
