// SPDX-License-Identifier: Apache-2.0

package memory

import (
	"context"
	"encoding/json"
	"errors"
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
	if !result.Parsed {
		t.Fatalf("expected empty memories response to parse, got err %v", result.Err)
	}
	if len(result.Memories) != 0 {
		t.Errorf("expected 0 memories, got %d", len(result.Memories))
	}
}

func TestParseExtractionResult_InvalidJSON(t *testing.T) {
	result := parseExtractionResult("not json")
	if result.Parsed {
		t.Fatal("expected invalid JSON to be unparsed")
	}
	if !errors.Is(result.Err, ErrExtractionParse) {
		t.Fatalf("parse err = %v, want ErrExtractionParse", result.Err)
	}
	if len(result.Memories) != 0 {
		t.Errorf("expected 0 memories for invalid JSON, got %d", len(result.Memories))
	}
}

func TestExtractFromMessages_ReturnsParseErrorForMalformedJSON(t *testing.T) {
	mem, _ := newTestMemory(t)
	provider := &extractStubProvider{reply: `{"memories":[`}

	_, err := ExtractFromMessages(context.Background(), provider, "test-model", mem, "/project", []Message{
		{Role: RoleUser, Content: "Remember that the car is red."},
		{Role: RoleAssistant, Content: "Understood."},
	})
	if !errors.Is(err, ErrExtractionParse) {
		t.Fatalf("ExtractFromMessages err = %v, want ErrExtractionParse", err)
	}
}

func TestParseExtractionResult_WrappedInMarkdown(t *testing.T) {
	input := "```json\n{\"memories\": [{\"action\": \"create\", \"filename\": \"x.md\", \"content\": \"hi\"}]}\n```"
	result := parseExtractionResult(input)
	if len(result.Memories) != 1 {
		t.Fatalf("expected 1 memory from markdown-wrapped JSON, got %d", len(result.Memories))
	}
}

func TestParseExtractionResult_RepairsTrailingComma(t *testing.T) {
	input := `{"memories":[{"filename":"x.md","content":"hi",}]}`
	result := parseExtractionResult(input)
	if len(result.Memories) != 1 {
		t.Fatalf("expected 1 memory from repaired JSON, got %d", len(result.Memories))
	}
}

func TestParseExtractionResult_AcceptsBareArray(t *testing.T) {
	input := `[{"filename":"x.md","content":"hi"}]`
	result := parseExtractionResult(input)
	if len(result.Memories) != 1 {
		t.Fatalf("expected 1 memory from bare array, got %d", len(result.Memories))
	}
}

func TestParseExtractionResult_NormalisesFields(t *testing.T) {
	input := `{"memories":[{"action":"unexpected","filename":"x.md","type":"unexpected","scope":"unexpected","content":"hello","indexEntry":"entry","observedOn":"2023-07-15T22:42:00Z","sessionDate":"2023-07-15","sourceRole":"assistant","eventDate":"2023-07-14","contextPrefix":"session context","modifiedOverride":"2023-07-15T22:42:00Z"}]}`

	result := parseExtractionResult(input)
	if len(result.Memories) != 1 {
		t.Fatalf("expected 1 memory, got %d", len(result.Memories))
	}

	got := result.Memories[0]
	if got.Action != "create" {
		t.Fatalf("action = %q, want create", got.Action)
	}
	if got.Type != "project" {
		t.Fatalf("type = %q, want project", got.Type)
	}
	if got.Scope != "project" {
		t.Fatalf("scope = %q, want project", got.Scope)
	}
	if got.IndexEntry != "entry" {
		t.Fatalf("indexEntry = %q, want entry", got.IndexEntry)
	}
	if got.ObservedOn != "2023-07-15T22:42:00Z" {
		t.Fatalf("observedOn = %q, want propagated value", got.ObservedOn)
	}
	if got.SessionDate != "2023-07-15" {
		t.Fatalf("sessionDate = %q, want propagated value", got.SessionDate)
	}
	if got.SourceRole != "assistant" {
		t.Fatalf("sourceRole = %q, want assistant", got.SourceRole)
	}
	if got.EventDate != "2023-07-14" {
		t.Fatalf("eventDate = %q, want propagated value", got.EventDate)
	}
	if got.ContextPrefix != "session context" {
		t.Fatalf("contextPrefix = %q, want propagated value", got.ContextPrefix)
	}
	if got.ModifiedOverride != "2023-07-15T22:42:00Z" {
		t.Fatalf("modifiedOverride = %q, want propagated value", got.ModifiedOverride)
	}
}

func TestParseExtractionResult_RewritesHeuristicFilenameForSessionID(t *testing.T) {
	input := `{"memories":[{"action":"create","filename":"user-fact-2023-07-15-reading-progress.md","type":"user","scope":"global","content":"hello","sessionId":"session-a"}]}`

	result := parseExtractionResult(input)
	if len(result.Memories) != 1 {
		t.Fatalf("expected 1 memory, got %d", len(result.Memories))
	}
	if got := result.Memories[0].Filename; got != "user-fact-2023-07-15-session-a-reading-progress.md" {
		t.Fatalf("filename = %q, want session-aware rewrite", got)
	}
	if got := result.Memories[0].SessionID; got != "session-a" {
		t.Fatalf("sessionID = %q, want session-a", got)
	}
}

func TestRewriteHeuristicFilenameForSession(t *testing.T) {
	cases := []struct {
		name      string
		filename  string
		sessionID string
		want      string
	}{
		{
			name:      "dated user fact",
			filename:  "user-fact-2024-03-25-commute-takes-minutes-each.md",
			sessionID: "sess-001",
			want:      "user-fact-2024-03-25-sess-001-commute-takes-minutes-each.md",
		},
		{
			name:      "dated preference sanitises session id",
			filename:  "user-preference-2024-03-25-films-family-friendly.md",
			sessionID: "sess 001",
			want:      "user-preference-2024-03-25-sess-001-films-family-friendly.md",
		},
		{
			name:      "already rewritten stays stable",
			filename:  "user-fact-2024-03-25-sess-001-commute-takes-minutes-each.md",
			sessionID: "sess-001",
			want:      "user-fact-2024-03-25-sess-001-commute-takes-minutes-each.md",
		},
		{
			name:      "non heuristic file unchanged",
			filename:  "project-note.md",
			sessionID: "sess-001",
			want:      "project-note.md",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := RewriteHeuristicFilenameForSession(tc.filename, tc.sessionID)
			if got != tc.want {
				t.Fatalf("RewriteHeuristicFilenameForSession(%q, %q) = %q, want %q", tc.filename, tc.sessionID, got, tc.want)
			}
		})
	}
}

func TestRewriteFilenameForSessionContent(t *testing.T) {
	a := RewriteFilenameForSessionContent("shared.md", "sess 001", "First durable fact.")
	b := RewriteFilenameForSessionContent("shared.md", "sess 001", "Second durable fact.")

	if a == b {
		t.Fatalf("filenames should differ for different content, got %q", a)
	}
	for _, got := range []string{a, b} {
		if !strings.HasPrefix(got, "shared-sess-001-") {
			t.Fatalf("filename = %q, want readable slug and sanitised session", got)
		}
		if !strings.HasSuffix(got, ".md") {
			t.Fatalf("filename = %q, want .md suffix", got)
		}
	}
}

func TestExtractFromMessages_RewritesHeuristicFilenameUsingExistingSessionID(t *testing.T) {
	mem, _ := newTestMemory(t)
	provider := &extractStubProvider{reply: `{"memories":[{"action":"create","filename":"project-note.md","name":"Project note","type":"project","scope":"project","content":"Keep the plan handy.","sessionId":"session-a"}]}`}

	out, err := ExtractFromMessages(context.Background(), provider, "test-model", mem, "/project", []Message{
		{Role: RoleSystem, Content: "This conversation took place on 2023/07/15 (Sat) 22:42."},
		{Role: RoleUser, Content: "I've been reading about the Amazon rainforest and I just finished my fifth issue."},
		{Role: RoleAssistant, Content: "That sounds fascinating."},
	})
	if err != nil {
		t.Fatalf("ExtractFromMessages: %v", err)
	}

	for _, memory := range out {
		if !strings.Contains(memory.Content, "fifth issue") {
			continue
		}
		if !strings.Contains(memory.Filename, "session-a") {
			t.Fatalf("heuristic filename = %q, want session-aware segment", memory.Filename)
		}
		if memory.SessionID != "session-a" {
			t.Fatalf("heuristic sessionID = %q, want session-a", memory.SessionID)
		}
		return
	}

	t.Fatalf("expected heuristic fact for quantified update, got %#v", out)
}

func TestExtractFromMessages_RewritesHeuristicFilenameUsingSystemSessionID(t *testing.T) {
	mem, _ := newTestMemory(t)
	provider := &extractStubProvider{reply: `{"memories":[]}`}

	out, err := ExtractFromMessages(context.Background(), provider, "test-model", mem, "/project", []Message{
		{Role: RoleSystem, Content: "session_id: session-b\nThis conversation took place on 2023/07/15 (Sat) 22:42."},
		{Role: RoleUser, Content: "I've been reading about the Amazon rainforest and I just finished my fifth issue."},
		{Role: RoleAssistant, Content: "That sounds fascinating."},
	})
	if err != nil {
		t.Fatalf("ExtractFromMessages: %v", err)
	}

	for _, memory := range out {
		if !strings.Contains(memory.Content, "fifth issue") {
			continue
		}
		if !strings.Contains(memory.Filename, "session-b") {
			t.Fatalf("heuristic filename = %q, want system-derived session-aware segment", memory.Filename)
		}
		if memory.SessionID != "session-b" {
			t.Fatalf("heuristic sessionID = %q, want session-b", memory.SessionID)
		}
		return
	}

	t.Fatalf("expected heuristic fact for quantified update, got %#v", out)
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
			SourceRole:  "assistant",
			EventDate:   "2024-03-13",
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
	if !strings.Contains(content, "source_role: assistant") {
		t.Error("expected source role in frontmatter")
	}
	if !strings.Contains(content, "event_date: 2024-03-13") {
		t.Error("expected event date in frontmatter")
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

func TestExtractUserPrompt_IncludesExistingMemorySummaries(t *testing.T) {
	msgs := []Message{
		{Role: RoleUser, Content: "hello"},
		{Role: RoleAssistant, Content: "hi"},
	}

	result := extractUserPrompt(msgs, []existingMemorySummary{
		{
			Path:        "memory/project/example/project-auth.md",
			Scope:       "project",
			Name:        "Project auth",
			Description: "Use OIDC for auth",
			Type:        "project",
			Modified:    "2026-04-18T11:00:00Z",
			Content:     "The project uses OIDC.",
		},
	})
	if !strings.Contains(result, "## Existing memories") {
		t.Error("expected existing memories header")
	}
	for _, want := range []string{
		"### [project] project-auth.md",
		"name: Project auth",
		"description: Use OIDC for auth",
		"type: project",
		"modified: 2026-04-18T11:00:00Z",
		"content: The project uses OIDC.",
	} {
		if !strings.Contains(result, want) {
			t.Errorf("expected prompt to contain %q", want)
		}
	}
}

func TestExtractUserPrompt_NoExistingMemories(t *testing.T) {
	msgs := []Message{
		{Role: RoleUser, Content: "hello"},
	}

	result := extractUserPrompt(msgs, nil)
	if strings.Contains(result, "## Existing memories") {
		t.Error("should not include existing memories header when empty")
	}
}

func TestExtractUserPrompt_TruncatesLongMessages(t *testing.T) {
	long := "start-" + strings.Repeat("x", 5000) + "-end"
	msgs := []Message{
		{Role: RoleUser, Content: long},
	}

	result := extractUserPrompt(msgs, nil)
	if !strings.Contains(result, "[...middle truncated...]") {
		t.Error("expected truncation of long message")
	}
	if !strings.Contains(result, "start-") {
		t.Error("expected head of long message to be preserved")
	}
	if !strings.Contains(result, "-end") {
		t.Error("expected tail of long message to be preserved")
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

func TestApplyExtractions_SupersedesAcrossScopes(t *testing.T) {
	mem, store := newTestMemory(t)
	projectPath := "/example/project"
	slug := ProjectSlug(projectPath)

	writeTopic(t, store, brain.MemoryProjectTopic(slug, "old_commute_fact"), strings.TrimSpace(`
---
name: Old commute fact
type: project
---

The assistant previously guessed the commute took an hour each way.
`))

	memories := []ExtractedMemory{
		{
			Action:      "create",
			Filename:    "new_commute_fact.md",
			Name:        "New commute fact",
			Description: "Corrected commute duration",
			Type:        "user",
			Scope:       "global",
			Content:     "My commute actually takes 45 minutes each way.",
			Supersedes:  "old_commute_fact.md",
		},
	}

	if err := mem.ApplyExtractions(context.Background(), slug, memories); err != nil {
		t.Fatalf("ApplyExtractions: %v", err)
	}

	data, err := store.Read(context.Background(), brain.MemoryProjectTopic(slug, "old_commute_fact"))
	if err != nil {
		t.Fatalf("project file not readable after supersession: %v", err)
	}
	if !strings.Contains(string(data), "superseded_by: new_commute_fact.md") {
		t.Fatalf("project fact missing superseded_by marker: %s", string(data))
	}
}

// TestExtractionPrompt_CoversAssistantTurns guards against drift that
// would silently drop assistant-turn facts from extraction.
func TestExtractionPrompt_CoversAssistantTurns(t *testing.T) {
	if !strings.Contains(strings.ToLower(extractionPrompt), "assistant") {
		t.Fatal("extractionPrompt must mention 'assistant'")
	}
	for _, kw := range []string{"recommend", "enumerat", "preserve concrete historical facts exactly", "relative time phrases"} {
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
	req   llm.CompleteRequest
}

func (p *extractStubProvider) Complete(_ context.Context, req llm.CompleteRequest) (llm.CompleteResponse, error) {
	p.req = req
	return llm.CompleteResponse{Text: p.reply}, nil
}

func (p *extractStubProvider) CompleteStream(_ context.Context, _ llm.CompleteRequest) (<-chan llm.StreamChunk, error) {
	return nil, nil
}

func (p *extractStubProvider) Close() error { return nil }

func TestExtractFromMessages_UsesExistingMemoryPromptContext(t *testing.T) {
	mem, store := newTestMemory(t)
	projectPath := "/project"
	slug := ProjectSlug(projectPath)

	writeTopic(t, store, brain.MemoryGlobalTopic("feedback-testing"), strings.TrimSpace(`
---
name: Testing feedback
description: Prefer integration tests
type: feedback
modified: 2026-04-18T10:00:00Z
---

Prefer integration tests over snapshots.
`))
	writeTopic(t, store, brain.MemoryProjectTopic(slug, "project-auth"), strings.TrimSpace(`
---
name: Auth choice
description: Use OIDC for auth
type: project
modified: 2026-04-18T11:00:00Z
---

The project uses OIDC.
`))

	provider := &extractStubProvider{reply: `{"memories":[]}`}
	if _, err := ExtractFromMessages(context.Background(), provider, "test-model", mem, projectPath, []Message{
		{Role: RoleUser, Content: "msg 0"},
		{Role: RoleAssistant, Content: "msg 1"},
	}); err != nil {
		t.Fatalf("ExtractFromMessages: %v", err)
	}

	if provider.req.Temperature != 0 {
		t.Fatalf("temperature = %v, want 0", provider.req.Temperature)
	}
	if provider.req.MaxTokens != extractMaxTokens {
		t.Fatalf("max tokens = %d, want %d", provider.req.MaxTokens, extractMaxTokens)
	}
	if !provider.req.ResponseFormatJSON {
		t.Fatal("expected extraction request to enable JSON response format")
	}

	if len(provider.req.Messages) != 2 {
		t.Fatalf("expected system and user prompt messages, got %d", len(provider.req.Messages))
	}
	prompt := provider.req.Messages[1].Content
	for _, want := range []string{
		"## Existing memories",
		"### [project] project-auth.md",
		"### [global] feedback-testing.md",
		"content: The project uses OIDC.",
		"content: Prefer integration tests over snapshots.",
	} {
		if !strings.Contains(prompt, want) {
			t.Errorf("expected prompt to contain %q", want)
		}
	}
	if strings.Contains(prompt, "Memory directory:") {
		t.Error("prompt should not include a memory-directory footer")
	}
	if strings.Index(prompt, "### [project] project-auth.md") > strings.Index(prompt, "### [global] feedback-testing.md") {
		t.Error("expected project memory with newer modified timestamp to appear first")
	}
}

func TestExtractorMaybeExtract_DefaultsToTwoMessageTurn(t *testing.T) {
	mem, store := newTestMemory(t)
	extractor := NewExtractor(mem)
	provider := &extractStubProvider{reply: `{"memories":[{"action":"create","filename":"facts.md","name":"Facts","description":"facts","type":"project","scope":"project","content":"A fact.","index_entry":"- facts"}]}`}

	extractor.MaybeExtract(context.Background(), provider, "test-model", "/project", []Message{
		{Role: RoleUser, Content: "tell me"},
		{Role: RoleAssistant, Content: "here is info"},
	})

	slug := ProjectSlug("/project")
	data, err := store.Read(context.Background(), brain.MemoryProjectTopic(slug, "facts"))
	if err != nil {
		t.Fatalf("expected extracted fact to be written: %v", err)
	}
	if !strings.Contains(string(data), "A fact.") {
		t.Fatalf("expected extracted content in stored file, got:\n%s", data)
	}
}

func TestExtractFromMessages_RefinesSummaryAndTags(t *testing.T) {
	mem, _ := newTestMemory(t)
	provider := &extractStubProvider{reply: `{"memories":[{"action":"create","filename":"user-appointment.md","name":"Appointment","description":"Upcoming appointment","type":"user","scope":"global","content":"The user has a dermatologist appointment with Dr Patel next Tuesday at 3 pm.","index_entry":"- user-appointment.md: appointment"}]}`}

	out, err := ExtractFromMessages(context.Background(), provider, "test-model", mem, "/project", []Message{
		{Role: RoleSystem, Content: "This conversation took place on 2023/06/06 (Tue) 09:00."},
		{Role: RoleUser, Content: "I've got a dermatologist appointment with Dr Patel next Tuesday at 3 pm."},
		{Role: RoleAssistant, Content: "That is useful to keep in mind."},
	})
	if err != nil {
		t.Fatalf("ExtractFromMessages: %v", err)
	}

	var got *ExtractedMemory
	for i := range out {
		if out[i].Filename == "user-appointment.md" {
			got = &out[i]
			break
		}
	}
	if got == nil {
		t.Fatalf("expected provider memory in %#v", out)
	}
	if !strings.Contains(got.Description, "dermatologist appointment") || !strings.Contains(got.Description, "Dr Patel") {
		t.Fatalf("description = %q, want specific appointment summary", got.Description)
	}
	if !strings.Contains(got.IndexEntry, "dermatologist appointment") || !strings.Contains(got.IndexEntry, "Dr Patel") {
		t.Fatalf("index entry = %q, want specific appointment summary", got.IndexEntry)
	}
	if !strings.HasPrefix(got.Content, "[Date: 2023-06-06 Tuesday June 2023]\n\n[Observed on 2023/06/06 (Tue) 09:00]\n\n") {
		t.Fatalf("content = %q, want session date prefixes", got.Content)
	}
	if got.ObservedOn != "2023-06-06T09:00:00Z" {
		t.Fatalf("observedOn = %q, want session timestamp", got.ObservedOn)
	}
	if got.ModifiedOverride != "2023-06-06T09:00:00Z" {
		t.Fatalf("modifiedOverride = %q, want session timestamp", got.ModifiedOverride)
	}
	if got.SessionDate != "2023-06-06" {
		t.Fatalf("sessionDate = %q, want short ISO date", got.SessionDate)
	}
	for _, tag := range []string{"appointment", "medical", "dermatologist", "next tuesday", "3 pm"} {
		found := false
		for _, gotTag := range got.Tags {
			if gotTag == tag {
				found = true
				break
			}
		}
		if !found {
			t.Fatalf("expected tag %q in %#v", tag, got.Tags)
		}
	}
}

func TestExtractFromMessages_DoesNotInferPendingTaskFromAdviceQuestion(t *testing.T) {
	mem, _ := newTestMemory(t)
	provider := &extractStubProvider{reply: `{"memories":[]}`}

	out, err := ExtractFromMessages(context.Background(), provider, "test-model", mem, "/project", []Message{
		{Role: RoleSystem, Content: "This conversation took place on 2022/03/02 (Wed) 04:59."},
		{Role: RoleUser, Content: "Can you tell me what are some things I should consider before making an offer?"},
		{Role: RoleAssistant, Content: "I can walk you through the main factors to check."},
	})
	if err != nil {
		t.Fatalf("ExtractFromMessages: %v", err)
	}

	for _, memory := range out {
		if strings.Contains(memory.Content, "still needs to consider before making an offer") {
			t.Fatalf("unexpected pending-task heuristic in %#v", out)
		}
	}
}

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

func TestExtractFromMessages_AddsHeuristicUserFactForTookMeDuration(t *testing.T) {
	mem, _ := newTestMemory(t)
	provider := &extractStubProvider{reply: `{"memories":[]}`}

	out, err := ExtractFromMessages(context.Background(), provider, "test-model", mem, "/project", []Message{
		{Role: RoleSystem, Content: "This conversation took place on 2023/05/27 (Sat) 12:34."},
		{Role: RoleUser, Content: "Can you recommend games similar to Celeste, which took me 10 hours to complete?"},
		{Role: RoleAssistant, Content: "Celeste is an excellent game."},
	})
	if err != nil {
		t.Fatalf("ExtractFromMessages: %v", err)
	}

	for _, memory := range out {
		if strings.Contains(memory.Content, "took me 10 hours to complete") {
			if memory.Scope != "global" || memory.Type != "user" {
				t.Fatalf("heuristic fact scope/type = %q/%q, want global/user", memory.Scope, memory.Type)
			}
			return
		}
	}
	t.Fatalf("expected took-me duration heuristic fact, got %#v", out)
}

func TestExtractFromMessages_AddsHeuristicCadenceFact(t *testing.T) {
	mem, _ := newTestMemory(t)
	provider := &extractStubProvider{reply: `{"memories":[]}`}

	out, err := ExtractFromMessages(context.Background(), provider, "test-model", mem, "/project", []Message{
		{Role: RoleSystem, Content: "This conversation took place on 2023/11/03 (Fri) 18:00."},
		{Role: RoleUser, Content: "I see Dr. Smith every week, and she's been helping me work on this stuff."},
		{Role: RoleAssistant, Content: "That sounds useful."},
	})
	if err != nil {
		t.Fatalf("ExtractFromMessages: %v", err)
	}

	for _, memory := range out {
		if !strings.Contains(memory.Content, "every week") {
			continue
		}
		if memory.Scope != "global" || memory.Type != "user" {
			t.Fatalf("cadence heuristic scope/type = %q/%q, want global/user", memory.Scope, memory.Type)
		}
		return
	}

	t.Fatalf("expected cadence heuristic in %#v", out)
}

func TestExtractFromMessagesWithSession_HeuristicUsesExplicitSessionMetadata(t *testing.T) {
	mem, _ := newTestMemory(t)
	provider := &extractStubProvider{reply: `{"memories":[]}`}

	out, err := ExtractFromMessagesWithSession(
		context.Background(),
		provider,
		"test-model",
		mem,
		"/project",
		[]Message{
			{Role: RoleUser, Content: "My commute actually takes 45 minutes each way."},
			{Role: RoleAssistant, Content: "That is worth remembering."},
		},
		"session-commute",
		"2024/03/25 (Mon) 09:15",
	)
	if err != nil {
		t.Fatalf("ExtractFromMessagesWithSession: %v", err)
	}

	for _, memory := range out {
		if !strings.Contains(memory.Content, "45 minutes each way") {
			continue
		}
		if memory.SessionID != "session-commute" {
			t.Fatalf("sessionID = %q, want session-commute", memory.SessionID)
		}
		if memory.SessionDate != "2024-03-25" {
			t.Fatalf("sessionDate = %q, want 2024-03-25", memory.SessionDate)
		}
		if !strings.Contains(memory.Filename, "user-fact-2024-03-25-session-commute") {
			t.Fatalf("filename = %q, want explicit session-aware filename", memory.Filename)
		}
		return
	}

	t.Fatalf("expected explicit-session heuristic fact in %#v", out)
}

func TestExtractFromMessages_AddsHeuristicBandwidthFact(t *testing.T) {
	mem, _ := newTestMemory(t)
	provider := &extractStubProvider{reply: `{"memories":[]}`}

	out, err := ExtractFromMessages(context.Background(), provider, "test-model", mem, "/project", []Message{
		{Role: RoleSystem, Content: "This conversation took place on 2023/08/14 (Mon) 09:30."},
		{Role: RoleUser, Content: "I've upgraded my line to 500 Mbps and the backup sync now runs at 1.5 GB/s."},
		{Role: RoleAssistant, Content: "That is a substantial improvement."},
	})
	if err != nil {
		t.Fatalf("ExtractFromMessages: %v", err)
	}

	for _, memory := range out {
		if !strings.Contains(memory.Content, "500 Mbps") {
			continue
		}
		if !strings.Contains(memory.Content, "1.5 GB/s") {
			t.Fatalf("bandwidth heuristic content = %q, want both bandwidth facts", memory.Content)
		}
		return
	}

	t.Fatalf("expected bandwidth heuristic in %#v", out)
}

func TestExtractFromMessages_AddsAssistantTableRowFact(t *testing.T) {
	mem, _ := newTestMemory(t)
	provider := &extractStubProvider{reply: `{"memories":[]}`}

	out, err := ExtractFromMessagesWithSession(
		context.Background(),
		provider,
		"test-model",
		mem,
		"/project",
		[]Message{
			{Role: RoleUser, Content: "Can you put the rota into a table?"},
			{
				Role: RoleAssistant,
				Content: strings.Join([]string{
					"| Day | Day Shift | Evening Shift |",
					"| --- | --- | --- |",
					"| Sunday | Admon, 8 am - 4 pm | Bea, 4 pm - 12 am |",
				}, "\n"),
			},
		},
		"session-shift",
		"2024/03/25 (Mon) 09:15",
	)
	if err != nil {
		t.Fatalf("ExtractFromMessagesWithSession: %v", err)
	}

	for _, memory := range out {
		if !strings.Contains(memory.Content, "Admon") {
			continue
		}
		if memory.Scope != "project" || memory.Type != "project" {
			t.Fatalf("assistant table scope/type = %q/%q, want project/project", memory.Scope, memory.Type)
		}
		if !strings.HasPrefix(memory.Filename, "assistant-table-2024-03-25-session-shift") {
			t.Fatalf("assistant table filename = %q, want assistant-table-2024-03-25-session-shift...", memory.Filename)
		}
		if !strings.Contains(memory.Content, "Sunday roster: Admon, 8 am - 4 pm (Day Shift); Bea, 4 pm - 12 am (Evening Shift).") {
			t.Fatalf("assistant table content = %q, want preserved shift roster", memory.Content)
		}
		return
	}

	t.Fatalf("expected assistant-table heuristic in %#v", out)
}

func TestExtractFromMessages_AddsHeuristicStorageLocationFact(t *testing.T) {
	mem, _ := newTestMemory(t)
	provider := &extractStubProvider{reply: `{"memories":[]}`}

	out, err := ExtractFromMessages(context.Background(), provider, "test-model", mem, "/project", []Message{
		{Role: RoleSystem, Content: "This conversation took place on 2023/08/12 (Sat) 09:15."},
		{Role: RoleUser, Content: "I've been keeping my old sneakers under my bed for storage."},
		{Role: RoleAssistant, Content: "That should keep them out of the way."},
	})
	if err != nil {
		t.Fatalf("ExtractFromMessages: %v", err)
	}

	for _, memory := range out {
		if !strings.Contains(memory.Content, "under my bed") {
			continue
		}
		if memory.Scope != "global" || memory.Type != "user" {
			t.Fatalf("location heuristic scope/type = %q/%q, want global/user", memory.Scope, memory.Type)
		}
		return
	}

	t.Fatalf("expected storage-location heuristic in %#v", out)
}

func TestExtractFromMessages_AddsHeuristicStorageLocationForPlannedRack(t *testing.T) {
	mem, _ := newTestMemory(t)
	provider := &extractStubProvider{reply: `{"memories":[]}`}

	out, err := ExtractFromMessages(context.Background(), provider, "test-model", mem, "/project", []Message{
		{Role: RoleSystem, Content: "This conversation took place on 2023/05/29 (Mon) 15:01."},
		{Role: RoleUser, Content: "I'm thinking of buying a new pair of sandals. By the way, I need to organize my closet this weekend, and I'm looking forward to storing my old sneakers in a shoe rack, they're currently taking up space."},
		{Role: RoleAssistant, Content: "A shoe rack should help free up closet space."},
	})
	if err != nil {
		t.Fatalf("ExtractFromMessages: %v", err)
	}

	for _, memory := range out {
		if !strings.Contains(memory.Content, "old sneakers in a shoe rack") {
			continue
		}
		if memory.Scope != "global" || memory.Type != "user" {
			t.Fatalf("planned storage heuristic scope/type = %q/%q, want global/user", memory.Scope, memory.Type)
		}
		return
	}

	t.Fatalf("expected planned storage-location heuristic in %#v", out)
}

func TestExtractFromMessages_AddsHeuristicPersonalSetupFacts(t *testing.T) {
	mem, _ := newTestMemory(t)
	provider := &extractStubProvider{reply: `{"memories":[]}`}

	out, err := ExtractFromMessages(context.Background(), provider, "test-model", mem, "/project", []Message{
		{Role: RoleSystem, Content: "This conversation took place on 2023/05/22 (Mon) 05:54."},
		{Role: RoleUser, Content: "I recently installed a slim wall shelf to keep the hallway clutter-free. I noticed a crack on my oak desk near the lamp."},
		{Role: RoleAssistant, Content: "Those are useful household details to remember."},
	})
	if err != nil {
		t.Fatalf("ExtractFromMessages: %v", err)
	}

	want := []string{
		"wall shelf to keep the hallway clutter-free",
		"crack on my oak desk near the lamp",
	}
	for _, phrase := range want {
		found := false
		for _, memory := range out {
			if !strings.Contains(memory.Content, phrase) {
				continue
			}
			if memory.Scope != "global" || memory.Type != "user" {
				t.Fatalf("personal setup fact scope/type = %q/%q, want global/user", memory.Scope, memory.Type)
			}
			found = true
			break
		}
		if !found {
			t.Fatalf("expected personal setup fact containing %q in %#v", phrase, out)
		}
	}
}

func TestExtractFromMessages_SkipsKitchenRecommendationRequestBeforePersonalSetupFacts(t *testing.T) {
	mem, _ := newTestMemory(t)
	provider := &extractStubProvider{reply: `{"memories":[]}`}

	out, err := ExtractFromMessages(context.Background(), provider, "test-model", mem, "/project", []Message{
		{Role: RoleSystem, Content: "This conversation took place on 2023/05/22 (Mon) 05:54."},
		{Role: RoleUser, Content: "Can you give me tips for creating a boot tray that fits inside the narrow cupboard? I recently bought a slim wall shelf to keep my hallway clutter-free. I noticed a crack on my oak desk near the lamp."},
		{Role: RoleAssistant, Content: "A fitted storage tray and surface repair approach would both help."},
	})
	if err != nil {
		t.Fatalf("ExtractFromMessages: %v", err)
	}

	for _, memory := range out {
		if strings.Contains(memory.Content, "give me tips for creating a boot tray") {
			t.Fatalf("recommendation request should not become a heuristic user fact: %#v", out)
		}
	}

	want := []string{
		"wall shelf to keep my hallway clutter-free",
		"crack on my oak desk near the lamp",
	}
	for _, phrase := range want {
		found := false
		for _, memory := range out {
			if !strings.Contains(memory.Content, phrase) {
				continue
			}
			if memory.Scope != "global" || memory.Type != "user" {
				t.Fatalf("personal setup fact scope/type = %q/%q, want global/user", memory.Scope, memory.Type)
			}
			found = true
			break
		}
		if !found {
			t.Fatalf("expected personal setup fact containing %q in %#v", phrase, out)
		}
	}
}

func TestExtractFromMessages_AddsApplicationApprovalDurationFact(t *testing.T) {
	mem, _ := newTestMemory(t)
	provider := &extractStubProvider{reply: `{"memories":[]}`}

	out, err := ExtractFromMessages(context.Background(), provider, "test-model", mem, "/project", []Message{
		{Role: RoleSystem, Content: "This conversation took place on 2023/05/22 (Mon) 05:54."},
		{Role: RoleUser, Content: "I've been keeping my documents in the hallway cabinet. I go to the community centre every week. It's crazy how long it took for my asylum application to get approved. Over a year of uncertainty was really tough."},
		{Role: RoleAssistant, Content: "That was a long period to deal with."},
	})
	if err != nil {
		t.Fatalf("ExtractFromMessages: %v", err)
	}

	for _, memory := range out {
		if !strings.Contains(memory.Content, "asylum application to get approved") {
			continue
		}
		if !strings.Contains(memory.Content, "Over a year of uncertainty") {
			t.Fatalf("application timeline content = %q, want approval and duration", memory.Content)
		}
		if memory.Scope != "global" || memory.Type != "user" {
			t.Fatalf("application timeline scope/type = %q/%q, want global/user", memory.Scope, memory.Type)
		}
		return
	}

	t.Fatalf("expected application approval duration fact in %#v", out)
}

func TestExtractFromMessages_DoesNotExtractAssistantApplicationTimelineAdvice(t *testing.T) {
	mem, _ := newTestMemory(t)
	provider := &extractStubProvider{reply: `{"memories":[]}`}

	out, err := ExtractFromMessages(context.Background(), provider, "test-model", mem, "/project", []Message{
		{Role: RoleSystem, Content: "This conversation took place on 2023/05/22 (Mon) 05:54."},
		{Role: RoleUser, Content: "How long can visa applications take?"},
		{Role: RoleAssistant, Content: "A visa application can take over a year to get approved, depending on the case."},
	})
	if err != nil {
		t.Fatalf("ExtractFromMessages: %v", err)
	}

	for _, memory := range out {
		if strings.Contains(memory.Content, "visa application can take over a year") {
			t.Fatalf("assistant advice should not become a user fact: %#v", out)
		}
	}
}

func TestExtractFromMessages_AddsCreativeProgressCountFact(t *testing.T) {
	mem, _ := newTestMemory(t)
	provider := &extractStubProvider{reply: `{"memories":[]}`}

	out, err := ExtractFromMessages(context.Background(), provider, "test-model", mem, "/project", []Message{
		{Role: RoleSystem, Content: "This conversation took place on 2023/05/22 (Mon) 05:54."},
		{Role: RoleUser, Content: "I've written five short stories so far."},
		{Role: RoleAssistant, Content: "That is good progress."},
	})
	if err != nil {
		t.Fatalf("ExtractFromMessages: %v", err)
	}

	for _, memory := range out {
		if !strings.Contains(memory.Content, "written five short stories") {
			continue
		}
		if memory.Scope != "global" || memory.Type != "user" {
			t.Fatalf("creative progress scope/type = %q/%q, want global/user", memory.Scope, memory.Type)
		}
		return
	}

	t.Fatalf("expected creative progress count fact in %#v", out)
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

func TestExtractFromMessages_AddsParticipationEventFact(t *testing.T) {
	mem, _ := newTestMemory(t)
	provider := &extractStubProvider{reply: `{"memories":[]}`}

	out, err := ExtractFromMessages(context.Background(), provider, "test-model", mem, "/project", []Message{
		{Role: RoleSystem, Content: "This conversation took place on 2023/05/29 (Mon) 15:01."},
		{Role: RoleUser, Content: "I will participate in the company's annual charity soccer tournament today."},
		{Role: RoleAssistant, Content: "That sounds worthwhile."},
	})
	if err != nil {
		t.Fatalf("ExtractFromMessages: %v", err)
	}

	for _, memory := range out {
		if !strings.Contains(memory.Content, "company's annual charity soccer tournament") {
			continue
		}
		if !strings.Contains(memory.Content, "The user will participate in the company's annual charity soccer tournament") {
			t.Fatalf("participation event content = %q, want future participation summary", memory.Content)
		}
		if memory.Scope != "global" || memory.Type != "user" {
			t.Fatalf("participation event scope/type = %q/%q, want global/user", memory.Scope, memory.Type)
		}
		return
	}

	t.Fatalf("expected participation event heuristic in %#v", out)
}

func TestExtractFromMessages_DoesNotAddParticipationAspirationEventFact(t *testing.T) {
	mem, _ := newTestMemory(t)
	provider := &extractStubProvider{reply: `{"memories":[]}`}

	out, err := ExtractFromMessages(context.Background(), provider, "test-model", mem, "/project", []Message{
		{Role: RoleSystem, Content: "This conversation took place on 2023/05/29 (Mon) 15:01."},
		{Role: RoleUser, Content: "I hope to participate in the company's annual charity soccer tournament someday."},
		{Role: RoleAssistant, Content: "That would be a good goal."},
	})
	if err != nil {
		t.Fatalf("ExtractFromMessages: %v", err)
	}

	for _, memory := range out {
		if strings.Contains(memory.Content, "company's annual charity soccer tournament") {
			t.Fatalf("aspiration should not become an event fact: %#v", out)
		}
	}
}
