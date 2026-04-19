// SPDX-License-Identifier: Apache-2.0

package memory

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/jeffs-brain/memory/go/brain"
	"github.com/jeffs-brain/memory/go/llm"
)

func TestParseExtractionResult_ValidJSON(t *testing.T) {
	input := `{"memories": [{"action": "create", "filename": "test.md", "name": "Test", "description": "A test", "type": "project", "content": "hello", "index_entry": "- [Test](test.md)"}]}`

	result := parseExtractionResult(input)
	if len(result.Memories) != 1 {
		t.Fatalf("expected 1 memory, got %d", len(result.Memories))
	}
	if result.Memories[0].Filename != "test.md" {
		t.Errorf("filename = %q, want %q", result.Memories[0].Filename, "test.md")
	}
	if result.Memories[0].Content != "hello" {
		t.Errorf("content = %q, want %q", result.Memories[0].Content, "hello")
	}
}

func TestParseExtractionResult_EmptyMemories(t *testing.T) {
	input := `{"memories": []}`
	result := parseExtractionResult(input)
	if len(result.Memories) != 0 {
		t.Errorf("expected 0 memories, got %d", len(result.Memories))
	}
}

func TestParseExtractionResult_InvalidJSON(t *testing.T) {
	result := parseExtractionResult("not json")
	if len(result.Memories) != 0 {
		t.Errorf("expected 0 memories for invalid JSON, got %d", len(result.Memories))
	}
}

func TestParseExtractionResult_WrappedInMarkdown(t *testing.T) {
	input := "```json\n{\"memories\": [{\"action\": \"create\", \"filename\": \"x.md\", \"content\": \"hi\"}]}\n```"
	result := parseExtractionResult(input)
	if len(result.Memories) != 1 {
		t.Fatalf("expected 1 memory from markdown-wrapped JSON, got %d", len(result.Memories))
	}
}

func TestHasMemoryWrites_NoWrites(t *testing.T) {
	msgs := []Message{
		{Role: RoleUser, Content: "hello"},
		{Role: RoleAssistant, Content: "hi back"},
	}
	if hasMemoryWrites(msgs, "/home/user/.config/jeff/projects/test/memory", "/home/user/.config/jeff/memory") {
		t.Error("expected no memory writes")
	}
}

func TestHasMemoryWrites_WithWriteToMemory(t *testing.T) {
	msgs := []Message{
		{Role: RoleAssistant, Content: "", ToolCalls: []ToolCall{
			{Name: "write", Arguments: []byte(`{"file_path": "/home/user/.config/jeff/projects/test/memory/feedback.md"}`)},
		}},
	}
	if !hasMemoryWrites(msgs, "/home/user/.config/jeff/projects/test/memory", "/home/user/.config/jeff/memory") {
		t.Error("expected memory write detected")
	}
}

func TestHasMemoryWrites_WithWriteToGlobalMemory(t *testing.T) {
	msgs := []Message{
		{Role: RoleAssistant, Content: "", ToolCalls: []ToolCall{
			{Name: "write", Arguments: []byte(`{"file_path": "/home/user/.config/jeff/memory/user_prefs.md"}`)},
		}},
	}
	if !hasMemoryWrites(msgs, "/home/user/.config/jeff/projects/test/memory", "/home/user/.config/jeff/memory") {
		t.Error("expected global memory write detected")
	}
}

func TestHasMemoryWrites_WithWriteElsewhere(t *testing.T) {
	msgs := []Message{
		{Role: RoleAssistant, Content: "", ToolCalls: []ToolCall{
			{Name: "write", Arguments: []byte(`{"file_path": "/tmp/other.md"}`)},
		}},
	}
	if hasMemoryWrites(msgs, "/home/user/.config/jeff/projects/test/memory", "/home/user/.config/jeff/memory") {
		t.Error("write to /tmp should not count as memory write")
	}
}

func TestApplyExtractions_CreatesFiles(t *testing.T) {
	mem, store := newTestMemory(t)
	projectPath := "/example/project"
	slug := ProjectSlug(projectPath)

	memories := []ExtractedMemory{
		{
			Action:      "create",
			Filename:    "feedback_style.md",
			Name:        "Code Style",
			Description: "Prefer short functions",
			Type:        "feedback",
			Content:     "Always use early returns.",
			IndexEntry:  "- [Code Style](feedback_style.md) — prefer short functions",
			Scope:       "project",
		},
	}

	if err := mem.ApplyExtractions(context.Background(), slug, memories); err != nil {
		t.Fatalf("ApplyExtractions: %v", err)
	}

	data, err := store.Read(context.Background(), brain.MemoryProjectTopic(slug, "feedback_style"))
	if err != nil {
		t.Fatalf("file not created: %v", err)
	}
	content := string(data)
	if !strings.Contains(content, "---") {
		t.Error("expected frontmatter delimiters")
	}
	if !strings.Contains(content, "name: Code Style") {
		t.Error("expected name in frontmatter")
	}
	if !strings.Contains(content, "type: feedback") {
		t.Error("expected type in frontmatter")
	}
	if !strings.Contains(content, "Always use early returns.") {
		t.Error("expected content body")
	}
}

func TestApplyExtractions_SkipsEmptyFilename(t *testing.T) {
	mem, store := newTestMemory(t)
	projectPath := "/example/project"
	slug := ProjectSlug(projectPath)

	memories := []ExtractedMemory{
		{Action: "create", Filename: "", Content: "should be skipped"},
	}

	if err := mem.ApplyExtractions(context.Background(), slug, memories); err != nil {
		t.Fatalf("ApplyExtractions: %v", err)
	}

	entries, _ := store.List(context.Background(), brain.MemoryProjectPrefix(slug), brain.ListOpts{IncludeGenerated: true})
	if len(entries) != 0 {
		t.Errorf("expected no files created, got %d", len(entries))
	}
}

func TestApplyExtractions_AddsMdExtension(t *testing.T) {
	mem, store := newTestMemory(t)
	projectPath := "/example/project"
	slug := ProjectSlug(projectPath)

	memories := []ExtractedMemory{
		{Action: "create", Filename: "no_extension", Content: "test"},
	}

	if err := mem.ApplyExtractions(context.Background(), slug, memories); err != nil {
		t.Fatalf("ApplyExtractions: %v", err)
	}

	if _, err := store.Read(context.Background(), brain.MemoryProjectTopic(slug, "no_extension")); err != nil {
		t.Errorf("expected file with .md extension: %v", err)
	}
}

func TestApplyExtractions_UpdatesIndex(t *testing.T) {
	mem, store := newTestMemory(t)
	projectPath := "/example/project"
	slug := ProjectSlug(projectPath)

	memories := []ExtractedMemory{
		{
			Action:     "create",
			Filename:   "test.md",
			Name:       "Test",
			Content:    "body",
			IndexEntry: "- [Test](test.md) — a test entry",
			Scope:      "project",
		},
	}

	if err := mem.ApplyExtractions(context.Background(), slug, memories); err != nil {
		t.Fatalf("ApplyExtractions: %v", err)
	}

	if _, err := store.Read(context.Background(), brain.MemoryProjectTopic(slug, "test")); err != nil {
		t.Fatalf("topic file not created: %v", err)
	}

	indexData, err := store.Read(context.Background(), brain.MemoryProjectIndex(slug))
	if err != nil {
		t.Fatalf("index file not created: %v", err)
	}
	if !strings.Contains(string(indexData), "a test entry") {
		t.Errorf("expected entry in index, got: %s", indexData)
	}
}

func TestApplyExtractions_NoDuplicateIndexEntries(t *testing.T) {
	mem, store := newTestMemory(t)
	projectPath := "/example/project"
	slug := ProjectSlug(projectPath)

	memories := []ExtractedMemory{
		{
			Action:     "create",
			Filename:   "existing.md",
			Name:       "Existing",
			Content:    "first body",
			IndexEntry: "- [Existing](existing.md) — already here",
			Scope:      "project",
		},
	}
	if err := mem.ApplyExtractions(context.Background(), slug, memories); err != nil {
		t.Fatalf("first ApplyExtractions: %v", err)
	}

	more := []ExtractedMemory{
		{
			Action:     "create",
			Filename:   "existing.md",
			Name:       "Existing",
			Content:    "second body",
			IndexEntry: "- [Existing](existing.md) — already here",
			Scope:      "project",
		},
		{
			Action:     "create",
			Filename:   "new.md",
			Name:       "New",
			Content:    "fresh body",
			IndexEntry: "- [New](new.md) — fresh entry",
			Scope:      "project",
		},
	}
	if err := mem.ApplyExtractions(context.Background(), slug, more); err != nil {
		t.Fatalf("second ApplyExtractions: %v", err)
	}

	indexData, err := store.Read(context.Background(), brain.MemoryProjectIndex(slug))
	if err != nil {
		t.Fatalf("index file missing: %v", err)
	}
	content := string(indexData)
	if strings.Count(content, "already here") != 1 {
		t.Errorf("expected 1 occurrence of existing entry, got %d; content:\n%s",
			strings.Count(content, "already here"), content)
	}
	if !strings.Contains(content, "fresh entry") {
		t.Error("expected new entry to be added")
	}
}

func TestExtractUserPrompt_IncludesManifest(t *testing.T) {
	msgs := []Message{
		{Role: RoleUser, Content: "hello"},
		{Role: RoleAssistant, Content: "hi"},
	}

	result := extractUserPrompt(msgs, "- [project] auth.md: auth notes", "/tmp/mem")
	if !strings.Contains(result, "Existing memory files") {
		t.Error("expected existing manifest header")
	}
	if !strings.Contains(result, "auth.md") {
		t.Error("expected manifest content")
	}
}

func TestExtractUserPrompt_NoManifest(t *testing.T) {
	msgs := []Message{
		{Role: RoleUser, Content: "hello"},
	}

	result := extractUserPrompt(msgs, "", "/tmp/mem")
	if strings.Contains(result, "Existing memory files") {
		t.Error("should not include manifest header when empty")
	}
}

func TestExtractUserPrompt_TruncatesLongMessages(t *testing.T) {
	long := strings.Repeat("x", 5000)
	msgs := []Message{
		{Role: RoleUser, Content: long},
	}

	result := extractUserPrompt(msgs, "", "/tmp/mem")
	if !strings.Contains(result, "[...truncated]") {
		t.Error("expected truncation of long message")
	}
}

func TestNewExtractor(t *testing.T) {
	mem, _ := newTestMemory(t)
	e := NewExtractor(mem)
	if e == nil {
		t.Fatal("expected non-nil extractor")
	}
	if e.inProgress {
		t.Error("should not be in progress initially")
	}
	if e.lastCursor != 0 {
		t.Errorf("expected cursor 0, got %d", e.lastCursor)
	}
}

func TestParseExtractionResult_WithScope(t *testing.T) {
	input := `{"memories": [{"action": "create", "filename": "user_prefs.md", "name": "User Prefs", "description": "User preferences", "type": "user", "content": "Prefers British English", "index_entry": "- [User Prefs](user_prefs.md)", "scope": "global"}]}`

	result := parseExtractionResult(input)
	if len(result.Memories) != 1 {
		t.Fatalf("expected 1 memory, got %d", len(result.Memories))
	}
	if result.Memories[0].Scope != "global" {
		t.Errorf("scope = %q, want %q", result.Memories[0].Scope, "global")
	}
	if result.Memories[0].Type != "user" {
		t.Errorf("type = %q, want %q", result.Memories[0].Type, "user")
	}
}

func TestApplyExtractions_RoutesToGlobalDir(t *testing.T) {
	mem, store := newTestMemory(t)
	projectPath := "/example/project"
	slug := ProjectSlug(projectPath)

	memories := []ExtractedMemory{
		{
			Action:      "create",
			Filename:    "user_prefs.md",
			Name:        "User Prefs",
			Description: "User preferences",
			Type:        "user",
			Content:     "Prefers British English",
			IndexEntry:  "- [User Prefs](user_prefs.md) — user preferences",
			Scope:       "global",
		},
	}

	if err := mem.ApplyExtractions(context.Background(), slug, memories); err != nil {
		t.Fatalf("ApplyExtractions: %v", err)
	}

	data, err := store.Read(context.Background(), brain.MemoryGlobalTopic("user_prefs"))
	if err != nil {
		t.Fatalf("global file not created: %v", err)
	}
	if !strings.Contains(string(data), "Prefers British English") {
		t.Error("expected content in global memory file")
	}

	if _, err := store.Read(context.Background(), brain.MemoryProjectTopic(slug, "user_prefs")); err == nil {
		t.Error("file should not exist in project memory directory")
	}

	indexData, err := store.Read(context.Background(), brain.MemoryGlobalIndex())
	if err != nil {
		t.Fatalf("global index not created: %v", err)
	}
	if !strings.Contains(string(indexData), "User Prefs") {
		t.Error("expected entry in global MEMORY.md")
	}
}

func TestApplyExtractions_DefaultsToProject(t *testing.T) {
	mem, store := newTestMemory(t)
	projectPath := "/example/project"
	slug := ProjectSlug(projectPath)

	memories := []ExtractedMemory{
		{
			Action:   "create",
			Filename: "project_notes.md",
			Content:  "Some project notes",
			Scope:    "",
		},
	}

	if err := mem.ApplyExtractions(context.Background(), slug, memories); err != nil {
		t.Fatalf("ApplyExtractions: %v", err)
	}

	if _, err := store.Read(context.Background(), brain.MemoryProjectTopic(slug, "project_notes")); err != nil {
		t.Errorf("expected file in project memory dir: %v", err)
	}

	if _, err := store.Read(context.Background(), brain.MemoryGlobalTopic("project_notes")); err == nil {
		t.Error("file with empty scope should not be in global directory")
	}
}

func TestApplyExtractions_MixedScopes(t *testing.T) {
	mem, store := newTestMemory(t)
	projectPath := "/example/project"
	slug := ProjectSlug(projectPath)

	memories := []ExtractedMemory{
		{
			Action:     "create",
			Filename:   "arch_decisions.md",
			Content:    "Use microservices",
			IndexEntry: "- [Arch](arch_decisions.md)",
			Scope:      "project",
		},
		{
			Action:     "create",
			Filename:   "user_style.md",
			Content:    "Prefers early returns",
			IndexEntry: "- [Style](user_style.md)",
			Scope:      "global",
		},
	}

	if err := mem.ApplyExtractions(context.Background(), slug, memories); err != nil {
		t.Fatalf("ApplyExtractions: %v", err)
	}

	if _, err := store.Read(context.Background(), brain.MemoryProjectTopic(slug, "arch_decisions")); err != nil {
		t.Error("expected project-scoped file in project dir")
	}
	if _, err := store.Read(context.Background(), brain.MemoryGlobalTopic("user_style")); err != nil {
		t.Error("expected global-scoped file in global dir")
	}

	projectIndex, _ := store.Read(context.Background(), brain.MemoryProjectIndex(slug))
	if !strings.Contains(string(projectIndex), "Arch") {
		t.Error("expected project index entry")
	}
	globalIndex, _ := store.Read(context.Background(), brain.MemoryGlobalIndex())
	if !strings.Contains(string(globalIndex), "Style") {
		t.Error("expected global index entry")
	}
}

// TestExtractionPrompt_CoversAssistantTurns guards against drift that
// would silently drop assistant-turn facts from extraction.
func TestExtractionPrompt_CoversAssistantTurns(t *testing.T) {
	if !strings.Contains(strings.ToLower(extractionPrompt), "assistant") {
		t.Fatal("extractionPrompt must mention 'assistant'")
	}
	for _, kw := range []string{"recommend", "enumerat"} {
		if !strings.Contains(strings.ToLower(extractionPrompt), kw) {
			t.Errorf("extractionPrompt should cover assistant-turn keyword %q", kw)
		}
	}
}

// TestToolCallArguments_DecodeSmoke confirms the json.RawMessage-backed
// Arguments field round-trips without allocation problems when used in
// tests.
func TestToolCallArguments_DecodeSmoke(t *testing.T) {
	args := json.RawMessage(`{"file_path": "/tmp/x.md"}`)
	tc := ToolCall{Name: "write", Arguments: args}
	if !strings.Contains(string(tc.Arguments), "/tmp/x.md") {
		t.Fatalf("arguments lost: %q", tc.Arguments)
	}
}

type extractStubProvider struct {
	reply string
}

func (p *extractStubProvider) Complete(_ context.Context, _ llm.CompleteRequest) (llm.CompleteResponse, error) {
	return llm.CompleteResponse{Text: p.reply}, nil
}

func (p *extractStubProvider) CompleteStream(_ context.Context, _ llm.CompleteRequest) (<-chan llm.StreamChunk, error) {
	return nil, nil
}

func (p *extractStubProvider) Close() error { return nil }

func TestExtractFromMessages_AddsHeuristicUserFactForQuantifiedUpdate(t *testing.T) {
	mem, _ := newTestMemory(t)
	provider := &extractStubProvider{reply: `{"memories":[]}`}

	out, err := ExtractFromMessages(context.Background(), provider, "test-model", mem, "/project", []Message{
		{Role: RoleSystem, Content: "This conversation took place on 2023/07/15 (Sat) 22:42."},
		{Role: RoleUser, Content: "I've been reading about the Amazon rainforest and its indigenous communities in National Geographic, and I just finished my fifth issue."},
		{Role: RoleAssistant, Content: "That sounds fascinating."},
	})
	if err != nil {
		t.Fatalf("ExtractFromMessages: %v", err)
	}

	var found *ExtractedMemory
	for i := range out {
		if strings.Contains(out[i].Content, "fifth issue") {
			found = &out[i]
			break
		}
	}
	if found == nil {
		t.Fatalf("expected heuristic fact for quantified update, got %#v", out)
	}
	if found.Scope != "global" || found.Type != "user" {
		t.Fatalf("heuristic fact scope/type = %q/%q, want global/user", found.Scope, found.Type)
	}
	if !strings.Contains(found.Filename, "user-fact-2023-07-15") {
		t.Fatalf("heuristic filename = %q, want dated user-fact prefix", found.Filename)
	}
}

func TestExtractFromMessages_AddsHeuristicMilestoneFact(t *testing.T) {
	mem, _ := newTestMemory(t)
	provider := &extractStubProvider{reply: `{"memories":[]}`}

	out, err := ExtractFromMessages(context.Background(), provider, "test-model", mem, "/project", []Message{
		{Role: RoleSystem, Content: "This conversation took place on 2022/11/17 (Thu) 15:34."},
		{Role: RoleUser, Content: "I'm planning to start learning about deep learning. By the way, I just completed my undergraduate degree in computer science."},
		{Role: RoleAssistant, Content: "That foundation will help."},
	})
	if err != nil {
		t.Fatalf("ExtractFromMessages: %v", err)
	}

	for _, memory := range out {
		if !strings.Contains(memory.Content, "completed my undergraduate degree") {
			continue
		}
		if !strings.Contains(memory.Filename, "milestone") {
			t.Fatalf("milestone heuristic filename = %q, want milestone prefix", memory.Filename)
		}
		return
	}
	t.Fatalf("expected milestone heuristic in %#v", out)
}

func TestExtractFromMessages_AddsMonthNameDateFact(t *testing.T) {
	mem, _ := newTestMemory(t)
	provider := &extractStubProvider{reply: `{"memories":[]}`}

	out, err := ExtractFromMessages(context.Background(), provider, "test-model", mem, "/project", []Message{
		{Role: RoleSystem, Content: "This conversation took place on 2023/07/07 (Fri) 04:44."},
		{Role: RoleUser, Content: "My close friend Rachel got engaged on May 15th, and we're already planning her bachelorette party."},
		{Role: RoleAssistant, Content: "That sounds exciting."},
	})
	if err != nil {
		t.Fatalf("ExtractFromMessages: %v", err)
	}

	for _, memory := range out {
		if !strings.Contains(memory.Content, "May 15th") {
			continue
		}
		if !strings.HasPrefix(memory.Content, "[Date: 2023-05-15") {
			t.Fatalf("month-name heuristic content = %q, want prefixed actual date", memory.Content)
		}
		if memory.ObservedOn != "2023-05-15T00:00:00Z" {
			t.Fatalf("month-name heuristic observed_on = %q, want 2023-05-15T00:00:00Z", memory.ObservedOn)
		}
		return
	}

	t.Fatalf("expected month-name heuristic in %#v", out)
}

func TestExtractFromMessages_AddsGroupJoinMilestoneWithResolvedDate(t *testing.T) {
	mem, _ := newTestMemory(t)
	provider := &extractStubProvider{reply: `{"memories":[]}`}

	out, err := ExtractFromMessages(context.Background(), provider, "test-model", mem, "/project", []Message{
		{Role: RoleSystem, Content: "This conversation took place on 2023/05/25 (Thu) 01:50."},
		{Role: RoleUser, Content: `I just joined a new book club group called "Page Turners" last week, where we discuss our favourite novels and share recommendations.`},
		{Role: RoleAssistant, Content: "That sounds like a great group."},
	})
	if err != nil {
		t.Fatalf("ExtractFromMessages: %v", err)
	}

	for _, memory := range out {
		if !strings.Contains(memory.Content, "Page Turners") {
			continue
		}
		if !strings.HasPrefix(memory.Content, "[Date: 2023-05-18") {
			t.Fatalf("group-join heuristic content = %q, want prefixed actual date", memory.Content)
		}
		if memory.ObservedOn != "2023-05-18T00:00:00Z" {
			t.Fatalf("group-join heuristic observed_on = %q, want 2023-05-18T00:00:00Z", memory.ObservedOn)
		}
		return
	}

	t.Fatalf("expected group-join heuristic in %#v", out)
}

func TestExtractFromMessages_AddsHeuristicUserFactForRelativeTimeNumberWords(t *testing.T) {
	mem, _ := newTestMemory(t)
	provider := &extractStubProvider{reply: `{"memories":[]}`}

	out, err := ExtractFromMessages(context.Background(), provider, "test-model", mem, "/project", []Message{
		{Role: RoleSystem, Content: "This conversation took place on 2023/05/26 (Fri) 18:20."},
		{Role: RoleUser, Content: "I'm actually planning to buy a new phone charger, since I lost my old one at the gym about two weeks ago."},
		{Role: RoleAssistant, Content: "Buying a new charger makes sense."},
	})
	if err != nil {
		t.Fatalf("ExtractFromMessages: %v", err)
	}

	for _, memory := range out {
		if !strings.Contains(memory.Content, "lost my old one at the gym about two weeks ago") {
			continue
		}
		if memory.ObservedOn != "2023-05-12T00:00:00Z" {
			t.Fatalf("relative-time heuristic observed_on = %q, want 2023-05-12T00:00:00Z", memory.ObservedOn)
		}
		return
	}

	t.Fatalf("expected relative-time heuristic in %#v", out)
}

func TestExtractFromMessages_AddsHeuristicUserFactForAirbnbBookingLeadTime(t *testing.T) {
	mem, _ := newTestMemory(t)
	provider := &extractStubProvider{reply: `{"memories":[]}`}

	out, err := ExtractFromMessages(context.Background(), provider, "test-model", mem, "/project", []Message{
		{Role: RoleSystem, Content: "This conversation took place on 2023/05/27 (Sat) 03:04."},
		{Role: RoleUser, Content: "I've had a great experience with Airbnb in the past, like when I stayed in Haight-Ashbury for my best friend's wedding and had to book three months in advance."},
		{Role: RoleAssistant, Content: "That sounds like a memorable trip."},
	})
	if err != nil {
		t.Fatalf("ExtractFromMessages: %v", err)
	}

	for _, memory := range out {
		if strings.Contains(memory.Content, "Airbnb") &&
			strings.Contains(memory.Content, "book three months in advance") {
			return
		}
	}

	t.Fatalf("expected Airbnb booking heuristic in %#v", out)
}

func TestExtractFromMessages_AddsPendingTaskFactWithoutCreatingAppointmentEvent(t *testing.T) {
	mem, _ := newTestMemory(t)
	provider := &extractStubProvider{reply: `{"memories":[]}`}

	out, err := ExtractFromMessages(context.Background(), provider, "test-model", mem, "/project", []Message{
		{Role: RoleSystem, Content: "This conversation took place on 2023/05/27 (Sat) 09:00."},
		{Role: RoleUser, Content: "I still need to schedule a dentist appointment and pick up my prescription."},
		{Role: RoleAssistant, Content: "I can help you remember both tasks."},
	})
	if err != nil {
		t.Fatalf("ExtractFromMessages: %v", err)
	}

	foundTask := false
	for _, memory := range out {
		if strings.Contains(memory.Content, "The user has a dentist appointment") {
			t.Fatalf("pending appointment should not become an event: %#v", out)
		}
		if strings.Contains(memory.Content, "The user still needs to schedule a dentist appointment.") {
			foundTask = true
		}
	}
	if !foundTask {
		t.Fatalf("expected pending-task heuristic in %#v", out)
	}
}

func TestExtractFromMessages_AddsHeuristicPreferenceFact(t *testing.T) {
	mem, _ := newTestMemory(t)
	provider := &extractStubProvider{reply: `{"memories":[]}`}

	out, err := ExtractFromMessages(context.Background(), provider, "test-model", mem, "/project", []Message{
		{Role: RoleSystem, Content: "This conversation took place on 2023/07/14 (Fri) 20:05."},
		{Role: RoleUser, Content: "Can you recommend a family-friendly, light-hearted film for tonight, ideally under 100 minutes and without gore?"},
		{Role: RoleAssistant, Content: "I will keep the suggestions gentle and short."},
	})
	if err != nil {
		t.Fatalf("ExtractFromMessages: %v", err)
	}

	for _, memory := range out {
		if !strings.Contains(memory.Filename, "user-preference-2023-07-14") {
			continue
		}
		if !strings.Contains(memory.Content, "family-friendly") || !strings.Contains(memory.Content, "without gore") {
			t.Fatalf("preference heuristic content = %q, want captured constraints", memory.Content)
		}
		return
	}
	t.Fatalf("expected preference heuristic in %#v", out)
}

func TestExtractFromMessages_AddsReligiousServiceEventFact(t *testing.T) {
	mem, _ := newTestMemory(t)
	provider := &extractStubProvider{reply: `{"memories":[]}`}

	out, err := ExtractFromMessages(context.Background(), provider, "test-model", mem, "/project", []Message{
		{Role: RoleSystem, Content: "This conversation took place on 2023/04/06 (Thu) 05:36."},
		{Role: RoleUser, Content: "I'm glad I got to attend the Maundy Thursday service at the Episcopal Church, it was a beautiful and moving experience."},
		{Role: RoleAssistant, Content: "That sounds meaningful."},
	})
	if err != nil {
		t.Fatalf("ExtractFromMessages: %v", err)
	}

	for _, memory := range out {
		if !strings.Contains(memory.Content, "Maundy Thursday service at the Episcopal Church") {
			continue
		}
		if memory.ObservedOn != "2023-04-06T05:36:00Z" {
			t.Fatalf("service heuristic observed_on = %q, want 2023-04-06T05:36:00Z", memory.ObservedOn)
		}
		return
	}

	t.Fatalf("expected religious-service heuristic in %#v", out)
}
