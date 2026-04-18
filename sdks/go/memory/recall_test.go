// SPDX-License-Identifier: Apache-2.0

package memory

import (
	"context"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/jeffs-brain/memory/go/brain"
)

func TestBuildManifest_Empty(t *testing.T) {
	result := buildManifest(nil)
	if result != "" {
		t.Errorf("expected empty manifest, got %q", result)
	}
}

func TestBuildManifest_WithTopics(t *testing.T) {
	topics := []TopicFile{
		{Name: "Auth", Description: "Auth migration notes", Type: "project", Path: "memory/project/x/project_auth.md"},
		{Name: "Style", Description: "", Type: "feedback", Path: "memory/project/x/feedback_style.md"},
		{Name: "NoType", Description: "Something", Type: "", Path: "memory/project/x/notype.md"},
	}

	result := buildManifest(topics)

	if !strings.Contains(result, "[project] project_auth.md: Auth migration notes") {
		t.Errorf("expected project entry, got:\n%s", result)
	}
	if !strings.Contains(result, "[feedback] feedback_style.md") {
		t.Errorf("expected feedback entry, got:\n%s", result)
	}
	if strings.Contains(result, "[feedback] feedback_style.md:") {
		t.Errorf("entry with no description should not have colon, got:\n%s", result)
	}
	if strings.Contains(result, "[]") {
		t.Errorf("entry with no type should omit brackets, got:\n%s", result)
	}
}

func TestParseSelectedMemories_ValidJSON(t *testing.T) {
	input := `{"selected": ["auth.md", "style.md", "prefs.md"]}`
	result := parseSelectedMemories(input)

	if len(result) != 3 {
		t.Fatalf("expected 3 selections, got %d", len(result))
	}
	if result[0] != "auth.md" {
		t.Errorf("result[0] = %q, want %q", result[0], "auth.md")
	}
}

func TestParseSelectedMemories_WrappedInMarkdown(t *testing.T) {
	input := "```json\n{\"selected\": [\"auth.md\"]}\n```"
	result := parseSelectedMemories(input)

	if len(result) != 1 {
		t.Fatalf("expected 1 selection, got %d", len(result))
	}
}

func TestParseSelectedMemories_EmptyArray(t *testing.T) {
	input := `{"selected": []}`
	result := parseSelectedMemories(input)

	if len(result) != 0 {
		t.Errorf("expected 0 selections, got %d", len(result))
	}
}

func TestParseSelectedMemories_InvalidJSON(t *testing.T) {
	result := parseSelectedMemories("not json at all")
	if len(result) != 0 {
		t.Errorf("expected 0 selections for invalid JSON, got %d", len(result))
	}
}

func TestParseSelectedMemories_CapsAtMax(t *testing.T) {
	input := `{"selected": ["a.md", "b.md", "c.md", "d.md", "e.md", "f.md", "g.md"]}`
	result := parseSelectedMemories(input)

	if len(result) != maxRecallTopics {
		t.Errorf("expected max %d selections, got %d", maxRecallTopics, len(result))
	}
}

func TestReadCappedTopic_SmallFile(t *testing.T) {
	mem, store := newTestMemory(t)
	p := brain.MemoryGlobalTopic("test")
	writeTopic(t, store, p, "short content")

	content, err := mem.readCappedTopic(context.Background(), p)
	if err != nil {
		t.Fatal(err)
	}
	if content != "short content" {
		t.Errorf("expected 'short content', got %q", content)
	}
}

func TestReadCappedTopic_LargeFile(t *testing.T) {
	mem, store := newTestMemory(t)
	data := make([]byte, maxMemoryBytes+1000)
	for i := range data {
		data[i] = 'x'
	}
	p := brain.MemoryGlobalTopic("big")
	writeTopic(t, store, p, string(data))

	content, err := mem.readCappedTopic(context.Background(), p)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(content, "[...truncated]") {
		t.Error("expected truncation marker")
	}
	if len(content) > maxMemoryBytes+50 {
		t.Errorf("content too long: %d bytes", len(content))
	}
}

func TestReadCappedTopic_ManyLines(t *testing.T) {
	mem, store := newTestMemory(t)
	var lines []string
	for i := 0; i < maxMemoryLines+50; i++ {
		lines = append(lines, "line")
	}
	p := brain.MemoryGlobalTopic("lines")
	writeTopic(t, store, p, strings.Join(lines, "\n"))

	content, err := mem.readCappedTopic(context.Background(), p)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(content, "[...truncated]") {
		t.Error("expected truncation marker for many lines")
	}
}

func TestTopicAge_Empty(t *testing.T) {
	if got := topicAge("", time.Now()); got != "unknown time ago" {
		t.Errorf("expected 'unknown time ago', got %q", got)
	}
}

func TestTopicAge_Unparseable(t *testing.T) {
	if got := topicAge("not a date", time.Now()); got != "unknown time ago" {
		t.Errorf("expected 'unknown time ago', got %q", got)
	}
}

func TestFormatRecalledMemories_Empty(t *testing.T) {
	result := FormatRecalledMemories(nil)
	if result != "" {
		t.Errorf("expected empty string, got %q", result)
	}
}

func TestFormatRecalledMemories_WithMemories(t *testing.T) {
	memories := []SurfacedMemory{
		{Path: brain.MemoryGlobalTopic("test"), Content: "memory content", Topic: TopicFile{Name: "Test"}},
	}

	result := FormatRecalledMemories(memories)
	if !strings.Contains(result, "<system-reminder>") {
		t.Error("expected system-reminder tags")
	}
	if !strings.Contains(result, "memory content") {
		t.Error("expected memory content in output")
	}
	if !strings.Contains(result, "Memory (saved") {
		t.Error("expected age header")
	}
}

func TestBuildManifest_WithGlobalTopics(t *testing.T) {
	topics := []TopicFile{
		{Name: "Prefs", Description: "Coding style preferences", Type: "user", Path: "memory/global/user_preferences.md", Scope: "global"},
		{Name: "Patterns", Description: "Common patterns", Type: "reference", Path: "memory/global/patterns.md", Scope: "global"},
	}

	result := buildManifest(topics)

	if !strings.Contains(result, "[global:user] user_preferences.md: Coding style preferences") {
		t.Errorf("expected global:user entry, got:\n%s", result)
	}
	if !strings.Contains(result, "[global:reference] patterns.md: Common patterns") {
		t.Errorf("expected global:reference entry, got:\n%s", result)
	}
}

func TestBuildManifest_MixedScopes(t *testing.T) {
	topics := []TopicFile{
		{Name: "Auth", Description: "Auth migration notes", Type: "project", Path: "memory/project/x/project_auth.md", Scope: "project"},
		{Name: "Prefs", Description: "Coding style", Type: "user", Path: "memory/global/user_preferences.md", Scope: "global"},
		{Name: "NoType", Description: "Something", Type: "", Path: "memory/project/x/notype.md", Scope: "project"},
	}

	result := buildManifest(topics)

	if !strings.Contains(result, "[project] project_auth.md: Auth migration notes") {
		t.Errorf("expected project entry with [type], got:\n%s", result)
	}
	if !strings.Contains(result, "[global:user] user_preferences.md: Coding style") {
		t.Errorf("expected global entry with [global:type], got:\n%s", result)
	}
	if strings.Contains(result, "[]") {
		t.Errorf("entry with empty type should not produce empty brackets, got:\n%s", result)
	}
	if !strings.Contains(result, "- notype.md: Something") {
		t.Errorf("expected plain entry for no-type topic, got:\n%s", result)
	}
}

func TestFormatRecalledMemories_GlobalScope(t *testing.T) {
	memories := []SurfacedMemory{
		{
			Path:    brain.MemoryGlobalTopic("prefs"),
			Content: "global memory content",
			Topic:   TopicFile{Name: "Prefs", Scope: "global"},
		},
	}

	result := FormatRecalledMemories(memories)

	if !strings.Contains(result, "Global memory (saved") {
		t.Errorf("expected 'Global memory' label, got:\n%s", result)
	}
	if !strings.Contains(result, "global memory content") {
		t.Error("expected memory content in output")
	}
}

func TestFormatRecalledMemories_LinkedMemory(t *testing.T) {
	memories := []SurfacedMemory{
		{
			Path:       brain.MemoryGlobalTopic("related"),
			Content:    "linked memory content",
			Topic:      TopicFile{Name: "Related", Scope: "project"},
			LinkedFrom: "some-topic",
		},
	}

	result := FormatRecalledMemories(memories)

	if !strings.Contains(result, "Linked memory (via [[some-topic]])") {
		t.Errorf("expected 'Linked memory (via [[some-topic]])' label, got:\n%s", result)
	}
}

func TestFormatRecalledMemories_MixedLabels(t *testing.T) {
	memories := []SurfacedMemory{
		{Path: brain.MemoryGlobalTopic("proj"), Content: "proj content", Topic: TopicFile{Name: "Proj", Scope: "project"}},
		{Path: brain.MemoryGlobalTopic("global"), Content: "global content", Topic: TopicFile{Name: "Glob", Scope: "global"}},
		{Path: brain.MemoryGlobalTopic("linked"), Content: "linked content", Topic: TopicFile{Name: "Link", Scope: "project"}, LinkedFrom: "arch"},
	}

	result := FormatRecalledMemories(memories)

	if !strings.Contains(result, "Memory (saved") {
		t.Error("expected 'Memory' label for project memory")
	}
	if !strings.Contains(result, "Global memory (saved") {
		t.Error("expected 'Global memory' label for global memory")
	}
	if !strings.Contains(result, "Linked memory (via [[arch]])") {
		t.Error("expected 'Linked memory' label for linked memory")
	}
}

func TestMemoryLabel_ProjectDefault(t *testing.T) {
	m := SurfacedMemory{Topic: TopicFile{Scope: "project"}}
	if label := memoryLabel(m); label != "Memory" {
		t.Errorf("expected 'Memory', got %q", label)
	}
}

func TestMemoryLabel_Global(t *testing.T) {
	m := SurfacedMemory{Topic: TopicFile{Scope: "global"}}
	if label := memoryLabel(m); label != "Global memory" {
		t.Errorf("expected 'Global memory', got %q", label)
	}
}

func TestMemoryLabel_LinkedTakesPrecedence(t *testing.T) {
	m := SurfacedMemory{Topic: TopicFile{Scope: "global"}, LinkedFrom: "topic"}
	label := memoryLabel(m)
	if !strings.Contains(label, "Linked memory") {
		t.Errorf("expected 'Linked memory' label, got %q", label)
	}
	if !strings.Contains(label, "[[topic]]") {
		t.Errorf("expected [[topic]] in label, got %q", label)
	}
}

func TestFollowWikilinks_ResolvesLinks(t *testing.T) {
	mem, store := newTestMemory(t)
	projectPath := "/example/project"
	slug := ProjectSlug(projectPath)

	writeTopic(t, store, brain.MemoryProjectTopic(slug, "linked-topic"), `---
name: Linked Topic
description: A linked topic
type: reference
---

Linked content here.`)

	memories := []SurfacedMemory{
		{
			Path:    brain.MemoryProjectTopic(slug, "main"),
			Content: "See also [[linked-topic]]",
			Topic:   TopicFile{Scope: "project"},
		},
	}
	surfaced := map[brain.Path]bool{}

	linked := mem.followWikilinks(context.Background(), memories, surfaced, projectPath)

	if len(linked) != 1 {
		t.Fatalf("expected 1 linked memory, got %d", len(linked))
	}
	if linked[0].LinkedFrom != "linked-topic" {
		t.Errorf("LinkedFrom = %q, want %q", linked[0].LinkedFrom, "linked-topic")
	}
	if !strings.Contains(linked[0].Content, "Linked content here.") {
		t.Error("expected linked content")
	}
}

func TestFollowWikilinks_CapsAtMax(t *testing.T) {
	mem, store := newTestMemory(t)
	projectPath := "/example/project"
	slug := ProjectSlug(projectPath)

	for i := 0; i < 5; i++ {
		name := fmt.Sprintf("topic-%d", i)
		writeTopic(t, store, brain.MemoryProjectTopic(slug, name),
			fmt.Sprintf("---\nname: Topic %d\n---\nContent %d", i, i))
	}

	mainContent := "See [[topic-0]] and [[topic-1]] and [[topic-2]] and [[topic-3]] and [[topic-4]]"
	memories := []SurfacedMemory{
		{Path: brain.MemoryProjectTopic(slug, "main"), Content: mainContent, Topic: TopicFile{Scope: "project"}},
	}

	linked := mem.followWikilinks(context.Background(), memories, map[brain.Path]bool{}, projectPath)

	if len(linked) > maxLinkedMemories {
		t.Errorf("expected at most %d linked memories, got %d", maxLinkedMemories, len(linked))
	}
}

func TestFollowWikilinks_SkipsSurfacedAndLoaded(t *testing.T) {
	mem, store := newTestMemory(t)
	projectPath := "/example/project"
	slug := ProjectSlug(projectPath)

	writeTopic(t, store, brain.MemoryProjectTopic(slug, "already-loaded"),
		"---\nname: Already\n---\nContent")

	mainContent := "See [[already-loaded]]"
	memories := []SurfacedMemory{
		{Path: brain.MemoryProjectTopic(slug, "main"), Content: mainContent, Topic: TopicFile{Scope: "project"}},
		{Path: brain.MemoryProjectTopic(slug, "already-loaded"), Content: "already loaded", Topic: TopicFile{Scope: "project"}},
	}

	linked := mem.followWikilinks(context.Background(), memories, map[brain.Path]bool{}, projectPath)

	if len(linked) != 0 {
		t.Errorf("expected 0 linked memories (already loaded), got %d", len(linked))
	}
}

func TestBuildManifest_HeuristicTopics(t *testing.T) {
	topics := []TopicFile{
		{
			Name:        "Testing: Always run Go",
			Description: "Always run Go tests before committing",
			Type:        "feedback",
			Path:        "memory/project/x/heuristic-testing-always-run.md",
			Scope:       "project",
			Tags:        []string{"heuristic", "testing"},
			Confidence:  "high",
		},
		{
			Name:        "Auth notes",
			Description: "Auth migration notes",
			Type:        "project",
			Path:        "memory/project/x/project_auth.md",
			Scope:       "project",
		},
		{
			Name:        "Anti SQL concat",
			Description: "Avoid SQL string concatenation",
			Type:        "feedback",
			Path:        "memory/global/heuristic-anti-debugging-sql.md",
			Scope:       "global",
			Tags:        []string{"heuristic", "debugging", "anti-pattern"},
			Confidence:  "medium",
		},
	}

	result := buildManifest(topics)

	if !strings.Contains(result, "[heuristic:high]") {
		t.Errorf("expected [heuristic:high] for high-confidence heuristic, got:\n%s", result)
	}
	if !strings.Contains(result, "[heuristic:medium]") {
		t.Errorf("expected [heuristic:medium] for medium-confidence heuristic, got:\n%s", result)
	}
	if strings.Contains(result, "project_auth.md") && strings.Contains(result, "[heuristic") {
		for _, line := range strings.Split(result, "\n") {
			if strings.Contains(line, "project_auth.md") && strings.Contains(line, "[heuristic") {
				t.Errorf("non-heuristic topic should not have [heuristic] label, got:\n%s", line)
			}
		}
	}
}

func TestBuildManifest_HeuristicNoConfidence(t *testing.T) {
	topics := []TopicFile{
		{
			Name:        "Some heuristic",
			Description: "A heuristic with no confidence set",
			Type:        "feedback",
			Path:        "memory/project/x/heuristic-general-some.md",
			Scope:       "project",
			Tags:        []string{"heuristic"},
			Confidence:  "",
		},
	}

	result := buildManifest(topics)

	if !strings.Contains(result, "[heuristic:low]") {
		t.Errorf("expected [heuristic:low] for empty confidence, got:\n%s", result)
	}
}

func TestFormatRecalledMemories_HeuristicLabel(t *testing.T) {
	memories := []SurfacedMemory{
		{
			Path:    brain.MemoryProjectTopic("x", "heuristic-testing-always-run"),
			Content: "Always run Go tests before committing",
			Topic: TopicFile{
				Name:       "Testing: Always run Go",
				Scope:      "project",
				Tags:       []string{"heuristic", "testing"},
				Confidence: "high",
			},
		},
	}

	result := FormatRecalledMemories(memories)

	if !strings.Contains(result, "Learned heuristic (high confidence)") {
		t.Errorf("expected 'Learned heuristic (high confidence)' label, got:\n%s", result)
	}
	if !strings.Contains(result, "Always run Go tests") {
		t.Error("expected heuristic content in output")
	}
}

func TestFormatRecalledMemories_HeuristicDefaultConfidence(t *testing.T) {
	memories := []SurfacedMemory{
		{
			Path:    brain.MemoryProjectTopic("x", "heuristic-general-some"),
			Content: "some heuristic rule",
			Topic: TopicFile{
				Name:       "General heuristic",
				Scope:      "project",
				Tags:       []string{"heuristic"},
				Confidence: "",
			},
		},
	}

	result := FormatRecalledMemories(memories)

	if !strings.Contains(result, "Learned heuristic (low confidence)") {
		t.Errorf("expected 'Learned heuristic (low confidence)' for empty confidence, got:\n%s", result)
	}
}

func TestMemoryLabel_Heuristic(t *testing.T) {
	m := SurfacedMemory{
		Topic: TopicFile{
			Scope:      "project",
			Tags:       []string{"heuristic", "testing"},
			Confidence: "high",
		},
	}
	label := memoryLabel(m)
	if label != "Learned heuristic (high confidence)" {
		t.Errorf("expected 'Learned heuristic (high confidence)', got %q", label)
	}
}

func TestMemoryLabel_HeuristicTakesPrecedenceOverGlobal(t *testing.T) {
	m := SurfacedMemory{
		Topic: TopicFile{
			Scope:      "global",
			Tags:       []string{"heuristic"},
			Confidence: "medium",
		},
	}
	label := memoryLabel(m)
	if label != "Learned heuristic (medium confidence)" {
		t.Errorf("expected heuristic label to take precedence over global, got %q", label)
	}
}

func TestMemoryLabel_LinkedStillTakesPrecedenceOverHeuristic(t *testing.T) {
	m := SurfacedMemory{
		Topic: TopicFile{
			Scope:      "project",
			Tags:       []string{"heuristic"},
			Confidence: "high",
		},
		LinkedFrom: "some-topic",
	}
	label := memoryLabel(m)
	if !strings.Contains(label, "Linked memory") {
		t.Errorf("expected linked label to take precedence, got %q", label)
	}
}
