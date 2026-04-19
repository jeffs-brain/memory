// SPDX-License-Identifier: Apache-2.0

package lme

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/jeffs-brain/memory/go/brain"
	"github.com/jeffs-brain/memory/go/llm"
	"github.com/jeffs-brain/memory/go/memory"
	"github.com/jeffs-brain/memory/go/store/mem"
	"github.com/jeffs-brain/memory/go/store/pt"
)

// replayFakeProvider is a test double that returns canned extraction
// responses as raw JSON strings. Tracks the model field of each
// incoming request so tests can assert the extraction model wiring.
type replayFakeProvider struct {
	mu        sync.Mutex
	responses []string
	calls     int
	models    []string
}

func (f *replayFakeProvider) Complete(_ context.Context, req llm.CompleteRequest) (llm.CompleteResponse, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.models = append(f.models, req.Model)
	if f.calls >= len(f.responses) {
		return llm.CompleteResponse{Text: `{"memories": []}`}, nil
	}
	resp := f.responses[f.calls]
	f.calls++
	return llm.CompleteResponse{Text: resp}, nil
}

func (f *replayFakeProvider) CompleteStream(_ context.Context, _ llm.CompleteRequest) (<-chan llm.StreamChunk, error) {
	return nil, fmt.Errorf("not implemented")
}

func (f *replayFakeProvider) Close() error { return nil }

func (f *replayFakeProvider) callCount() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.calls
}

func (f *replayFakeProvider) modelsSeen() []string {
	f.mu.Lock()
	defer f.mu.Unlock()
	out := make([]string, len(f.models))
	copy(out, f.models)
	return out
}

func TestIngestReplay_BasicExtraction(t *testing.T) {
	store := mem.New()
	ds := &Dataset{
		Questions: []Question{
			{
				ID:            "q1",
				Category:      "single-session",
				Question:      "What colour was the car?",
				Answer:        "red",
				SessionIDs:    []string{"sess-001"},
				HaystackDates: []string{"2023/04/10 (Mon) 17:50"},
				HaystackSessions: [][]SessionMessage{
					{
						{Role: "user", Content: "I bought a new red car today."},
						{Role: "assistant", Content: "That sounds exciting! Red is a bold choice."},
					},
				},
			},
		},
	}

	provider := &replayFakeProvider{
		responses: []string{
			`{"memories": [{"action": "create", "filename": "car_purchase.md", "name": "Car Purchase", "description": "User bought a red car", "type": "project", "content": "User purchased a new red car.", "index_entry": "- [Car Purchase](car_purchase.md)", "scope": "project"}]}`,
		},
	}

	result, err := IngestReplay(context.Background(), store, ds, provider, ReplayOpts{})
	if err != nil {
		t.Fatalf("IngestReplay: %v", err)
	}

	if result.SessionsProcessed != 1 {
		t.Errorf("SessionsProcessed = %d, want 1", result.SessionsProcessed)
	}
	if result.FactsExtracted != 1 {
		t.Errorf("FactsExtracted = %d, want 1", result.FactsExtracted)
	}
	if result.FactsWritten != 1 {
		t.Errorf("FactsWritten = %d, want 1", result.FactsWritten)
	}
	if len(result.Warnings) != 0 {
		t.Errorf("unexpected warnings: %v", result.Warnings)
	}
}

func TestIngestReplay_TemporalMetadata(t *testing.T) {
	store := mem.New()
	ds := &Dataset{
		Questions: []Question{
			{
				ID:            "q1",
				Category:      "temporal",
				Question:      "When did we discuss the project?",
				Answer:        "Monday",
				SessionIDs:    []string{"sess-001"},
				HaystackDates: []string{"2023/04/10 (Mon) 17:50"},
				HaystackSessions: [][]SessionMessage{
					{
						{Role: "user", Content: "Let us discuss the project timeline."},
						{Role: "assistant", Content: "Sure, the project kicks off next week."},
					},
				},
			},
		},
	}

	provider := &replayFakeProvider{
		responses: []string{
			`{"memories": [{"action": "create", "filename": "project_timeline.md", "name": "Project Timeline", "description": "Project starts next week", "type": "project", "content": "The project kicks off next week.", "index_entry": "- [Project Timeline](project_timeline.md)", "scope": "project"}]}`,
		},
	}

	result, err := IngestReplay(context.Background(), store, ds, provider, ReplayOpts{})
	if err != nil {
		t.Fatalf("IngestReplay: %v", err)
	}
	if result.FactsWritten != 1 {
		t.Fatalf("FactsWritten = %d, want 1", result.FactsWritten)
	}

	slug := memory.ProjectSlug("/eval/lme")
	files, err := store.List(context.Background(), brain.MemoryProjectPrefix(slug), brain.ListOpts{
		Recursive:        true,
		IncludeGenerated: true,
	})
	if err != nil {
		t.Fatalf("list memory files: %v", err)
	}

	found := false
	for _, f := range files {
		if f.IsDir || strings.HasSuffix(string(f.Path), "MEMORY.md") {
			continue
		}
		data, err := store.Read(context.Background(), f.Path)
		if err != nil {
			continue
		}
		if strings.Contains(string(data), "[Observed on 2023/04/10 (Mon) 17:50]") {
			found = true
			break
		}
	}
	if !found {
		t.Error("expected temporal metadata '[Observed on ...]' in extracted fact content")
	}
}

func TestIngestReplay_DeduplicatesSessions(t *testing.T) {
	store := mem.New()
	ds := &Dataset{
		Questions: []Question{
			{
				ID: "q1", Category: "single-session", Question: "Q1?", Answer: "A1",
				SessionIDs: []string{"sess-001"},
				HaystackSessions: [][]SessionMessage{{
					{Role: "user", Content: "Session content here."},
					{Role: "assistant", Content: "I understand."},
				}},
			},
			{
				ID: "q2", Category: "single-session", Question: "Q2?", Answer: "A2",
				SessionIDs: []string{"sess-001"},
				HaystackSessions: [][]SessionMessage{{
					{Role: "user", Content: "Session content here."},
					{Role: "assistant", Content: "I understand."},
				}},
			},
		},
	}

	provider := &replayFakeProvider{
		responses: []string{
			`{"memories": [{"action": "create", "filename": "fact.md", "name": "Fact", "description": "A fact", "type": "project", "content": "Some fact.", "index_entry": "- [Fact](fact.md)", "scope": "project"}]}`,
		},
	}

	result, err := IngestReplay(context.Background(), store, ds, provider, ReplayOpts{})
	if err != nil {
		t.Fatalf("IngestReplay: %v", err)
	}

	if result.SessionsProcessed != 1 {
		t.Errorf("SessionsProcessed = %d, want 1 (should deduplicate)", result.SessionsProcessed)
	}
	if provider.callCount() != 1 {
		t.Errorf("provider.calls = %d, want 1 (should only extract once per unique session)", provider.callCount())
	}
}

func TestIngestReplay_NoExtractions(t *testing.T) {
	store := mem.New()
	ds := &Dataset{
		Questions: []Question{
			{
				ID: "q1", Category: "single-session", Question: "Q1?", Answer: "A1",
				SessionIDs: []string{"sess-001"},
				HaystackSessions: [][]SessionMessage{{
					{Role: "user", Content: "Hello."},
					{Role: "assistant", Content: "Hi."},
				}},
			},
		},
	}

	provider := &replayFakeProvider{
		responses: []string{`{"memories": []}`},
	}

	result, err := IngestReplay(context.Background(), store, ds, provider, ReplayOpts{})
	if err != nil {
		t.Fatalf("IngestReplay: %v", err)
	}

	if result.SessionsProcessed != 1 {
		t.Errorf("SessionsProcessed = %d, want 1", result.SessionsProcessed)
	}
	if result.FactsExtracted != 0 {
		t.Errorf("FactsExtracted = %d, want 0", result.FactsExtracted)
	}
	if result.FactsWritten != 0 {
		t.Errorf("FactsWritten = %d, want 0", result.FactsWritten)
	}
}

func TestIngestReplay_NilProvider(t *testing.T) {
	store := mem.New()
	ds := &Dataset{Questions: []Question{{
		ID: "q1", Category: "test", Question: "Q?", Answer: "A",
		SessionIDs:       []string{"s1"},
		HaystackSessions: [][]SessionMessage{{{Role: "user", Content: "hi"}}},
	}}}

	_, err := IngestReplay(context.Background(), store, ds, nil, ReplayOpts{})
	if err == nil {
		t.Fatal("expected error for nil provider")
	}
}

func TestIngestReplay_EmptyDataset(t *testing.T) {
	store := mem.New()
	ds := &Dataset{Questions: nil}

	provider := &replayFakeProvider{}
	result, err := IngestReplay(context.Background(), store, ds, provider, ReplayOpts{})
	if err != nil {
		t.Fatalf("IngestReplay: %v", err)
	}
	if result.SessionsProcessed != 0 {
		t.Errorf("SessionsProcessed = %d, want 0", result.SessionsProcessed)
	}
}

func TestIngestReplay_DefaultModel(t *testing.T) {
	store := mem.New()
	ds := &Dataset{Questions: []Question{{
		ID: "q1", Category: "single-session", Question: "q?", Answer: "a",
		SessionIDs: []string{"sess-001"},
		HaystackSessions: [][]SessionMessage{{
			{Role: "user", Content: "Hello."},
			{Role: "assistant", Content: "Hi."},
		}},
	}}}
	provider := &replayFakeProvider{responses: []string{`{"memories": []}`}}

	_, err := IngestReplay(context.Background(), store, ds, provider, ReplayOpts{})
	if err != nil {
		t.Fatalf("IngestReplay: %v", err)
	}

	models := provider.modelsSeen()
	if len(models) == 0 {
		t.Fatal("expected at least one Complete call")
	}
	for i, m := range models {
		if m != DefaultReplayExtractModel {
			t.Errorf("model[%d] = %q, want default %q", i, m, DefaultReplayExtractModel)
		}
	}
}

func TestIngestReplay_OverrideModel(t *testing.T) {
	store := mem.New()
	ds := &Dataset{Questions: []Question{{
		ID: "q1", Category: "single-session", Question: "q?", Answer: "a",
		SessionIDs: []string{"sess-001"},
		HaystackSessions: [][]SessionMessage{{
			{Role: "user", Content: "Hello."},
			{Role: "assistant", Content: "Hi."},
		}},
	}}}
	provider := &replayFakeProvider{responses: []string{`{"memories": []}`}}

	_, err := IngestReplay(context.Background(), store, ds, provider, ReplayOpts{
		ExtractModel: "gpt-4o-mini",
	})
	if err != nil {
		t.Fatalf("IngestReplay: %v", err)
	}

	models := provider.modelsSeen()
	if len(models) == 0 {
		t.Fatal("expected at least one Complete call")
	}
	for i, m := range models {
		if m != "gpt-4o-mini" {
			t.Errorf("model[%d] = %q, want gpt-4o-mini", i, m)
		}
	}
}

func TestIngestReplay_WritesSessionDateISO(t *testing.T) {
	store := mem.New()
	ds := &Dataset{Questions: []Question{{
		ID: "q1", Category: "single-session",
		Question:      "What did they discuss?",
		Answer:        "the project",
		SessionIDs:    []string{"sess-001"},
		HaystackDates: []string{"2024/03/25 (Mon) 10:00"},
		HaystackSessions: [][]SessionMessage{{
			{Role: "user", Content: "Let us discuss the project timeline."},
			{Role: "assistant", Content: "Sure."},
		}},
	}}}

	provider := &replayFakeProvider{responses: []string{
		`{"memories": [{"action": "create", "filename": "project.md", "name": "Project", "description": "d", "type": "project", "content": "They discussed the project.", "index_entry": "- x", "scope": "project"}]}`,
	}}

	result, err := IngestReplay(context.Background(), store, ds, provider, ReplayOpts{})
	if err != nil {
		t.Fatalf("IngestReplay: %v", err)
	}
	if result.FactsWritten != 1 {
		t.Fatalf("FactsWritten = %d, want 1", result.FactsWritten)
	}

	slug := memory.ProjectSlug("/eval/lme")
	files, err := store.List(context.Background(), brain.MemoryProjectPrefix(slug),
		brain.ListOpts{Recursive: true, IncludeGenerated: true})
	if err != nil {
		t.Fatalf("list: %v", err)
	}

	found := false
	for _, f := range files {
		if f.IsDir || strings.HasSuffix(string(f.Path), "MEMORY.md") {
			continue
		}
		data, err := store.Read(context.Background(), f.Path)
		if err != nil {
			continue
		}
		if strings.Contains(string(data), "session_date: 2024-03-25") {
			found = true
			break
		}
	}
	if !found {
		t.Error("expected session_date: 2024-03-25 in extracted fact frontmatter")
	}
}

func TestIngestReplay_OneExtractionPerSession(t *testing.T) {
	store := mem.New()
	ds := &Dataset{Questions: []Question{
		{
			ID: "q1", Category: "multi-session",
			Question:      "Q?",
			Answer:        "A",
			SessionIDs:    []string{"sess-A", "sess-B", "sess-C"},
			HaystackDates: []string{"2024/01/01 (Mon) 10:00", "2024/02/02 (Fri) 11:00", "2024/03/03 (Sun) 12:00"},
			HaystackSessions: [][]SessionMessage{
				{{Role: "user", Content: "Alpha content."}, {Role: "assistant", Content: "ok"}},
				{{Role: "user", Content: "Bravo content."}, {Role: "assistant", Content: "ok"}},
				{{Role: "user", Content: "Charlie content."}, {Role: "assistant", Content: "ok"}},
			},
		},
	}}

	provider := &replayFakeProvider{responses: []string{
		`{"memories": [{"action": "create", "filename": "a.md", "name": "A", "description": "d", "type": "project", "content": "Alpha fact.", "index_entry": "- a", "scope": "project"}]}`,
		`{"memories": [{"action": "create", "filename": "b.md", "name": "B", "description": "d", "type": "project", "content": "Bravo fact.", "index_entry": "- b", "scope": "project"}]}`,
		`{"memories": [{"action": "create", "filename": "c.md", "name": "C", "description": "d", "type": "project", "content": "Charlie fact.", "index_entry": "- c", "scope": "project"}]}`,
	}}

	_, err := IngestReplay(context.Background(), store, ds, provider, ReplayOpts{Concurrency: 1})
	if err != nil {
		t.Fatalf("IngestReplay: %v", err)
	}
	if provider.callCount() != 3 {
		t.Errorf("provider.calls = %d, want 3 (one per session)", provider.callCount())
	}

	slug := memory.ProjectSlug("/eval/lme")
	files, err := store.List(context.Background(), brain.MemoryProjectPrefix(slug),
		brain.ListOpts{Recursive: true, IncludeGenerated: true})
	if err != nil {
		t.Fatalf("list: %v", err)
	}
	wantDates := map[string]string{
		"Alpha fact.":   "2024-01-01",
		"Bravo fact.":   "2024-02-02",
		"Charlie fact.": "2024-03-03",
	}
	seen := make(map[string]bool)
	for _, f := range files {
		if f.IsDir || strings.HasSuffix(string(f.Path), "MEMORY.md") {
			continue
		}
		data, err := store.Read(context.Background(), f.Path)
		if err != nil {
			continue
		}
		body := string(data)
		for factBody, wantDate := range wantDates {
			if strings.Contains(body, factBody) {
				if !strings.Contains(body, "session_date: "+wantDate) {
					t.Errorf("fact %q should carry session_date %q, got:\n%s", factBody, wantDate, body)
				}
				seen[factBody] = true
			}
		}
	}
	for fact := range wantDates {
		if !seen[fact] {
			t.Errorf("fact %q not found in store", fact)
		}
	}
}

func TestIngestReplay_EmptySessionIsHandled(t *testing.T) {
	store := mem.New()
	ds := &Dataset{Questions: []Question{{
		ID: "q1", Category: "single-session", Question: "Q?", Answer: "A",
		SessionIDs:       []string{"sess-001"},
		HaystackSessions: [][]SessionMessage{{{Role: "user", Content: ""}}},
	}}}

	provider := &replayFakeProvider{responses: []string{`{"memories": []}`}}
	_, err := IngestReplay(context.Background(), store, ds, provider, ReplayOpts{})
	if err != nil {
		t.Fatalf("IngestReplay panic-free: %v", err)
	}
}

func TestIngestReplay_SessionDatePresentEvenWithoutObservedOn(t *testing.T) {
	store := mem.New()
	ds := &Dataset{Questions: []Question{{
		ID: "q1", Category: "single-session",
		Question:      "Q?",
		Answer:        "A",
		SessionIDs:    []string{"sess-001"},
		HaystackDates: []string{"2024-07-04"},
		HaystackSessions: [][]SessionMessage{{
			{Role: "user", Content: "Independence day event."},
			{Role: "assistant", Content: "Noted."},
		}},
	}}}

	provider := &replayFakeProvider{responses: []string{
		`{"memories": [{"action": "create", "filename": "event.md", "name": "Event", "description": "d", "type": "project", "content": "An independence day event.", "index_entry": "- e", "scope": "project"}]}`,
	}}

	_, err := IngestReplay(context.Background(), store, ds, provider, ReplayOpts{})
	if err != nil {
		t.Fatalf("IngestReplay: %v", err)
	}

	slug := memory.ProjectSlug("/eval/lme")
	files, _ := store.List(context.Background(), brain.MemoryProjectPrefix(slug),
		brain.ListOpts{Recursive: true, IncludeGenerated: true})
	foundSessionDate := false
	for _, f := range files {
		if f.IsDir || strings.HasSuffix(string(f.Path), "MEMORY.md") {
			continue
		}
		data, _ := store.Read(context.Background(), f.Path)
		if strings.Contains(string(data), "session_date: 2024-07-04") {
			foundSessionDate = true
			break
		}
	}
	if !foundSessionDate {
		t.Error("expected session_date: 2024-07-04 even with YYYY-MM-DD only HaystackDate")
	}
}

func TestIngestReplay_EmittedMarkdownCarriesTags(t *testing.T) {
	store := mem.New()
	ds := &Dataset{Questions: []Question{{
		ID: "q1", Category: "single-session", Question: "Q?", Answer: "A",
		SessionIDs:    []string{"sess-001"},
		HaystackDates: []string{"2024/03/25 (Mon) 10:00"},
		HaystackSessions: [][]SessionMessage{{
			{Role: "user", Content: "Appointment on 2024-03-25, cost was $185."},
			{Role: "assistant", Content: "Noted."},
		}},
	}}}

	provider := &replayFakeProvider{responses: []string{
		`{"memories": [{"action": "create", "filename": "appt.md", "name": "Appointment", "description": "d", "type": "project", "content": "Appointment on Monday 2024-03-25 cost $185 and lasted 45 minutes.", "index_entry": "- a", "scope": "project"}]}`,
	}}

	_, err := IngestReplay(context.Background(), store, ds, provider, ReplayOpts{})
	if err != nil {
		t.Fatalf("IngestReplay: %v", err)
	}

	slug := memory.ProjectSlug("/eval/lme")
	files, _ := store.List(context.Background(), brain.MemoryProjectPrefix(slug),
		brain.ListOpts{Recursive: true, IncludeGenerated: true})

	var factBody string
	for _, f := range files {
		if f.IsDir || strings.HasSuffix(string(f.Path), "MEMORY.md") {
			continue
		}
		data, _ := store.Read(context.Background(), f.Path)
		if strings.Contains(string(data), "Appointment on Monday") {
			factBody = string(data)
			break
		}
	}
	if factBody == "" {
		t.Fatal("appointment fact not found in store")
	}

	frontmatter := factBody
	if idx := strings.Index(factBody, "\n---\n\n"); idx > 0 {
		frontmatter = factBody[:idx]
	}

	mustContain := []string{
		"tags: [",
		"Monday",
		"$185",
		"45 minutes",
		"2024-03-25",
	}
	for _, want := range mustContain {
		if !strings.Contains(frontmatter, want) {
			t.Errorf("frontmatter missing %q in:\n%s", want, frontmatter)
		}
	}
}

// TestIngestReplay_PassthroughLayoutPersistsGlobalMemory is the
// tri-SDK contract test: when a replay ingest runs over a pt.Store
// rooted at a brain cache, the resulting memory/global/*.md files
// must land at the same literal on-disk path. The TS/Py SDKs and the
// Go HTTP daemon all open the cache with a passthrough store and walk
// `memory/global/` verbatim; the fs.Store remap from `memory/global/`
// to `memory/` would break every downstream daemon at search time.
func TestIngestReplay_PassthroughLayoutPersistsGlobalMemory(t *testing.T) {
	dir := t.TempDir()
	store, err := pt.New(dir)
	if err != nil {
		t.Fatalf("pt.New: %v", err)
	}
	t.Cleanup(func() { _ = store.Close() })

	ds := &Dataset{Questions: []Question{{
		ID: "q1", Category: "single-session",
		Question:      "Q?",
		Answer:        "A",
		SessionIDs:    []string{"sess-001"},
		HaystackDates: []string{"2024/03/25 (Mon) 10:00"},
		HaystackSessions: [][]SessionMessage{{
			{Role: "user", Content: "Globally important fact."},
			{Role: "assistant", Content: "Understood."},
		}},
	}}}

	// Note the `scope: "global"` on the extraction — triggers the
	// global-memory write path inside ApplyExtractions.
	provider := &replayFakeProvider{responses: []string{
		`{"memories": [{"action": "create", "filename": "globally_important.md", "name": "GloballyImportant", "description": "d", "type": "global", "content": "A globally important fact.", "index_entry": "- g", "scope": "global"}]}`,
	}}

	result, err := IngestReplay(context.Background(), store, ds, provider, ReplayOpts{})
	if err != nil {
		t.Fatalf("IngestReplay: %v", err)
	}
	if result.FactsWritten != 1 {
		t.Fatalf("FactsWritten = %d, want 1", result.FactsWritten)
	}

	literal := filepath.Join(dir, "memory", "global", "globally_important.md")
	body, err := os.ReadFile(literal)
	if err != nil {
		t.Fatalf("expected file at %s (passthrough layout): %v", literal, err)
	}
	if !strings.Contains(string(body), "globally important fact") {
		t.Fatalf("fact content missing from %s: %q", literal, body)
	}

	// Defence in depth: the fs.Store remap would create the file at
	// memory/globally_important.md under the root. Confirm that is
	// absent so a future regression that reintroduces fs.New here
	// fails this test.
	remapped := filepath.Join(dir, "memory", "globally_important.md")
	if _, err := os.Stat(remapped); err == nil {
		t.Fatalf("found file at remapped path %s; passthrough store must not remap", remapped)
	}
}

func TestAutoFactTags_ExtractsUnitQuantitiesAndMoney(t *testing.T) {
	body := "The appointment lasted 45 minutes and cost $185 on 2024-03-25."
	tags := autoFactTags(body)
	got := strings.Join(tags, "|")

	if !strings.Contains(got, "45 minutes") {
		t.Errorf("expected '45 minutes' in tags, got %v", tags)
	}
	if !strings.Contains(got, "$185") {
		t.Errorf("expected '$185' in tags, got %v", tags)
	}
	if !strings.Contains(got, "2024-03-25") {
		t.Errorf("expected ISO date in tags, got %v", tags)
	}
	if !strings.Contains(got, "Monday") {
		t.Errorf("expected weekday 'Monday' from 2024-03-25 date, got %v", tags)
	}
}

func TestAutoFactTags_IgnoresCommonStopNouns(t *testing.T) {
	body := "The user said that this is good. When they arrived on Monday, it worked."
	tags := autoFactTags(body)
	for _, t0 := range tags {
		low := strings.ToLower(t0)
		if autoTagStopNoun[low] {
			t.Errorf("tag %q should have been filtered as a stop noun; full set: %v", t0, tags)
		}
	}
	found := false
	for _, t0 := range tags {
		if t0 == "Monday" {
			found = true
			break
		}
	}
	if !found {
		t.Errorf("expected Monday in tags, got %v", tags)
	}
}

func TestParseSessionDateRFC3339(t *testing.T) {
	cases := []struct {
		in   string
		want string
	}{
		{"", ""},
		{"2023/04/10 (Mon) 23:07", "2023-04-10T23:07:00Z"},
		{"2024/04/15", "2024-04-15T00:00:00Z"},
		{"2024-04-15", "2024-04-15T00:00:00Z"},
		{"not a date", ""},
	}
	for _, c := range cases {
		got := parseSessionDateRFC3339(c.in)
		if got != c.want {
			t.Errorf("parseSessionDateRFC3339(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}

func TestSessionToMessages_WithRoleMarkers(t *testing.T) {
	sess := sessionData{
		id:   "test-001",
		text: "[user]: Hello there.\n\n[assistant]: Hi! How can I help?\n\n[user]: Tell me about Go.\n\n[assistant]: Go is a great language.",
		date: "2023/04/10 (Mon) 17:50",
	}

	messages := sessionToMessages(sess)

	if len(messages) < 3 {
		t.Fatalf("expected at least 3 messages, got %d", len(messages))
	}

	if messages[0].Role != memory.RoleSystem {
		t.Errorf("messages[0].Role = %q, want %q", messages[0].Role, memory.RoleSystem)
	}
	if !strings.Contains(messages[0].Content, "2023/04/10") {
		t.Errorf("system message should contain date, got: %s", messages[0].Content)
	}

	hasUser := false
	hasAssistant := false
	for _, m := range messages[1:] {
		if m.Role == memory.RoleUser {
			hasUser = true
		}
		if m.Role == memory.RoleAssistant {
			hasAssistant = true
		}
	}
	if !hasUser {
		t.Error("expected at least one user message")
	}
	if !hasAssistant {
		t.Error("expected at least one assistant message")
	}
}

func TestSessionToMessages_NoRoleMarkers(t *testing.T) {
	sess := sessionData{
		id:   "test-002",
		text: "Just some raw text without role markers.",
	}

	messages := sessionToMessages(sess)

	if len(messages) < 1 {
		t.Fatal("expected at least 1 message")
	}

	lastMsg := messages[len(messages)-1]
	if lastMsg.Role != memory.RoleUser {
		t.Errorf("fallback message role = %q, want %q", lastMsg.Role, memory.RoleUser)
	}
	if !strings.Contains(lastMsg.Content, "raw text") {
		t.Error("fallback message should contain the original text")
	}
}

func TestSessionToMessages_NoDate(t *testing.T) {
	sess := sessionData{
		id:   "test-003",
		text: "[user]: Hello.\n\n[assistant]: Hi.",
	}

	messages := sessionToMessages(sess)

	if len(messages) > 0 && messages[0].Role == memory.RoleSystem {
		t.Error("should not have system message when no date")
	}
}

func TestDeduplicateSessions_AcrossQuestions(t *testing.T) {
	questions := []Question{
		{
			ID:         "q1",
			SessionIDs: []string{"s1", "s2"},
			HaystackSessions: [][]SessionMessage{
				{{Role: "user", Content: "alpha"}},
				{{Role: "user", Content: "bravo"}},
			},
		},
		{
			ID:         "q2",
			SessionIDs: []string{"s2", "s3"},
			HaystackSessions: [][]SessionMessage{
				{{Role: "user", Content: "bravo"}},
				{{Role: "user", Content: "charlie"}},
			},
		},
	}

	sessions := deduplicateSessions(questions)
	if len(sessions) != 3 {
		t.Fatalf("len(sessions) = %d, want 3", len(sessions))
	}
	seen := map[string]bool{}
	for _, s := range sessions {
		seen[s.id] = true
	}
	for _, want := range []string{"s1", "s2", "s3"} {
		if !seen[want] {
			t.Errorf("missing session %q", want)
		}
	}
}
