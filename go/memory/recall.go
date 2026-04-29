// SPDX-License-Identifier: Apache-2.0

package memory

import (
	"context"
	"encoding/json"
	"fmt"
	"sort"
	"strings"
	"time"

	"github.com/jeffs-brain/memory/go/brain"
	"github.com/jeffs-brain/memory/go/llm"
)

// Recall thresholds.
const (
	maxRecallTopics   = 5
	maxScanFiles      = 200
	maxMemoryLines    = 200
	maxMemoryBytes    = 4096
	maxLinkedMemories = 2
	recallMaxTokens   = 256
	recallTemperature = 0.0
)

// SurfacedMemory holds a single memory file selected for injection.
type SurfacedMemory struct {
	Path       brain.Path
	Content    string
	Topic      TopicFile
	LinkedFrom string
}

// RecallWeights biases memory recall toward global or project
// memories.
type RecallWeights struct {
	Global  float64
	Project float64
}

// globalPlentifulThreshold is the number of global candidates required
// before assistant-mode recall drops project memories entirely.
const globalPlentifulThreshold = 5

// assistantProjectCap limits project candidates when assistant mode
// still needs to fall back to project memories.
const assistantProjectCap = 10

// recallSelectorPrompt is the base system prompt for the memory
// selector. Ported verbatim from jeff.
const recallSelectorPrompt = `You are selecting memories that will be useful to an AI assistant as it processes a user's query. You will be given the user's query and a list of available memory files with their filenames and descriptions.

Return a JSON object with a "selected" array of filenames for the memories that will clearly be useful (up to 5). Only include memories you are certain will be helpful based on their name and description.

- If unsure whether a memory is relevant, do not include it. Be selective.
- If no memories are relevant, return an empty array.

Memories may be project-scoped (specific to this codebase) or global (cross-project knowledge about the user, their preferences and history).
Both can be useful — prefer project memories when the query is about this specific codebase, and global memories when the query is about general patterns, personal context, or user preferences.

Memories tagged [heuristic] are learned patterns from past sessions. Prefer high-confidence heuristics when they match the task.

Respond with ONLY valid JSON, no other text. Example: {"selected": ["feedback_testing.md", "project_auth.md"]}`

const recallHintAssistant = "\n\nThe user is currently in assistant mode — a conversational personal assistant session. Prefer global/personal memories over project-specific ones unless the query is clearly about a specific codebase."

const recallHintCoding = "\n\nThe user is currently in coding mode — an AI coding harness session. Prefer project-specific memories over global ones unless the query is clearly about general user preferences."

// Recall runs the memory recall side-query. It gathers candidates from
// project and global scopes, asks the provider to pick the most
// relevant, and returns the surfaced memories plus any linked via
// wikilinks.
func (m *Memory) Recall(
	ctx context.Context,
	provider llm.Provider,
	model string,
	projectPath string,
	userQuery string,
	surfaced map[brain.Path]bool,
	weights RecallWeights,
) ([]SurfacedMemory, error) {
	projectTopics, err := m.ListProjectTopics(ctx, projectPath)
	if err != nil {
		return nil, err
	}
	globalTopics, _ := m.ListGlobalTopics(ctx)

	projectTopics = filterSurfaced(projectTopics, surfaced)
	globalTopics = filterSurfaced(globalTopics, surfaced)

	var candidates []TopicFile
	switch {
	case weights.Global > weights.Project:
		candidates = append(candidates, globalTopics...)
		if len(globalTopics) < globalPlentifulThreshold {
			projCap := assistantProjectCap
			if len(projectTopics) < projCap {
				projCap = len(projectTopics)
			}
			candidates = append(candidates, projectTopics[:projCap]...)
		}
	case weights.Project > weights.Global:
		candidates = append(candidates, projectTopics...)
		candidates = append(candidates, globalTopics...)
	default:
		candidates = append(candidates, projectTopics...)
		candidates = append(candidates, globalTopics...)
	}

	if len(candidates) == 0 {
		return nil, nil
	}

	if len(candidates) > maxScanFiles {
		candidates = candidates[:maxScanFiles]
	}

	manifest := buildManifest(candidates)
	userPrompt := fmt.Sprintf("Query: %s\n\nAvailable memories:\n%s", userQuery, manifest)

	systemPrompt := recallSelectorPrompt
	switch {
	case weights.Global > weights.Project:
		systemPrompt += recallHintAssistant
	case weights.Project > weights.Global:
		systemPrompt += recallHintCoding
	}

	resp, err := provider.Complete(ctx, llm.CompleteRequest{
		Model: model,
		Messages: []llm.Message{
			{Role: RoleSystem, Content: systemPrompt},
			{Role: RoleUser, Content: userPrompt},
		},
		MaxTokens:   recallMaxTokens,
		Temperature: recallTemperature,
	})
	if err != nil {
		return nil, nil
	}

	selected := parseSelectedMemories(resp.Text)
	if len(selected) == 0 {
		return nil, nil
	}

	topicByFile := make(map[string]TopicFile, len(candidates))
	for _, t := range candidates {
		topicByFile[baseName(string(t.Path))] = t
	}

	var memories []SurfacedMemory
	for _, filename := range selected {
		topic, ok := topicByFile[filename]
		if !ok {
			continue
		}

		content, err := m.readCappedTopic(ctx, topic.Path)
		if err != nil {
			continue
		}

		memories = append(memories, SurfacedMemory{
			Path:    topic.Path,
			Content: content,
			Topic:   topic,
		})

		if len(memories) >= maxRecallTopics {
			break
		}
	}

	linked := m.followWikilinks(ctx, memories, surfaced, projectPath)
	memories = append(memories, linked...)

	return memories, nil
}

// FormatRecalledMemories formats surfaced memories as system-reminder
// content suitable for injection as a user message.
func FormatRecalledMemories(memories []SurfacedMemory) string {
	return FormatRecalledMemoriesWithContext(memories, time.Now())
}

// FormatRecalledMemoriesWithContext formats surfaced memories with an
// explicit "now" anchor for relative-time annotations.
func FormatRecalledMemoriesWithContext(memories []SurfacedMemory, now time.Time) string {
	if len(memories) == 0 {
		return ""
	}

	var b strings.Builder
	for _, m := range memories {
		age := topicAge(m.Topic.Modified, now)
		b.WriteString("<system-reminder>\n")

		label := memoryLabel(m)
		b.WriteString(fmt.Sprintf("%s (saved %s): %s\n", label, age, baseName(string(m.Path))))

		if header := dateHeader(m.Topic.Modified, now); header != "" {
			b.WriteString(header)
			b.WriteByte('\n')
		}
		b.WriteByte('\n')

		b.WriteString(m.Content)
		b.WriteString("\n</system-reminder>\n")
	}

	return strings.TrimSpace(b.String())
}

// SortMemoriesChronologically returns memories sorted oldest-first by
// their frontmatter "modified" timestamp.
func SortMemoriesChronologically(memories []SurfacedMemory) []SurfacedMemory {
	if len(memories) == 0 {
		return nil
	}

	type indexed struct {
		mem     SurfacedMemory
		origIdx int
		ts      time.Time
		hasTS   bool
	}

	dated := make([]indexed, 0, len(memories))
	undated := make([]indexed, 0, len(memories))
	for i, m := range memories {
		t, ok := parseTopicTime(m.Topic.Modified)
		entry := indexed{mem: m, origIdx: i, ts: t, hasTS: ok}
		if ok {
			dated = append(dated, entry)
		} else {
			undated = append(undated, entry)
		}
	}

	sort.SliceStable(dated, func(i, j int) bool {
		return dated[i].ts.Before(dated[j].ts)
	})

	out := make([]SurfacedMemory, 0, len(memories))
	for _, d := range dated {
		out = append(out, d.mem)
	}
	for _, u := range undated {
		out = append(out, u.mem)
	}
	return out
}

// parseTopicTime tries to parse a topic timestamp as RFC 3339.
func parseTopicTime(modified string) (time.Time, bool) {
	if modified == "" {
		return time.Time{}, false
	}
	t, err := time.Parse(time.RFC3339, modified)
	if err != nil || t.IsZero() {
		return time.Time{}, false
	}
	return t, true
}

// dateHeader returns a date banner for a memory.
func dateHeader(modified string, now time.Time) string {
	t, ok := parseTopicTime(modified)
	if !ok {
		return ""
	}
	iso := t.UTC().Format("2006-01-02")
	rel := relativeTimeString(t, now)
	if rel == "" {
		return fmt.Sprintf("=== %s ===", iso)
	}
	return fmt.Sprintf("=== %s (%s) ===", iso, rel)
}

// topicAge returns a human-readable age string.
func topicAge(modified string, now time.Time) string {
	t, ok := parseTopicTime(modified)
	if !ok {
		return "unknown time ago"
	}
	days := int(now.Sub(t).Hours() / 24)
	switch {
	case days <= 0:
		return "today"
	case days == 1:
		return "yesterday"
	default:
		return fmt.Sprintf("%d days ago", days)
	}
}

// relativeTimeString describes "then" relative to "now" in natural
// English.
func relativeTimeString(then, now time.Time) string {
	diff := now.Sub(then)
	if diff < 0 {
		return ""
	}

	days := int(diff.Hours() / 24)
	switch {
	case days == 0:
		return "today"
	case days == 1:
		return "yesterday"
	case days <= 6:
		return fmt.Sprintf("%d days ago", days)
	case days <= 27:
		weeks := (days + 3) / 7
		if weeks == 1 {
			return "1 week ago"
		}
		return fmt.Sprintf("%d weeks ago", weeks)
	case days <= 364:
		months := (days + 15) / 30
		if months < 1 {
			months = 1
		}
		if months == 1 {
			return "1 month ago"
		}
		return fmt.Sprintf("%d months ago", months)
	default:
		years := days / 365
		if years == 1 {
			return "1 year ago"
		}
		return fmt.Sprintf("%d years ago", years)
	}
}

// memoryLabel returns the appropriate label for a surfaced memory.
func memoryLabel(m SurfacedMemory) string {
	if m.LinkedFrom != "" {
		return fmt.Sprintf("Linked memory (via [[%s]])", m.LinkedFrom)
	}
	if hasTag(m.Topic.Tags, "heuristic") {
		conf := m.Topic.Confidence
		if conf == "" {
			conf = "low"
		}
		return fmt.Sprintf("Learned heuristic (%s confidence)", conf)
	}
	if m.Topic.Scope == "global" {
		return "Global memory"
	}
	return "Memory"
}

// buildManifest creates a one-line-per-topic manifest for the
// selector.
func buildManifest(topics []TopicFile) string {
	var b strings.Builder
	for _, t := range topics {
		line := "- "
		if t.Scope == "global" {
			line += "[global:" + t.Type + "] "
		} else if t.Type != "" {
			line += "[" + t.Type + "] "
		}
		if hasTag(t.Tags, "heuristic") {
			conf := t.Confidence
			if conf == "" {
				conf = "low"
			}
			line += "[heuristic:" + conf + "] "
		}
		line += baseName(string(t.Path))
		if t.Description != "" {
			line += ": " + t.Description
		}
		b.WriteString(line + "\n")
	}
	return strings.TrimSpace(b.String())
}

// followWikilinks extracts wikilinks from loaded memories and resolves
// them.
func (m *Memory) followWikilinks(ctx context.Context, memories []SurfacedMemory, surfaced map[brain.Path]bool, projectPath string) []SurfacedMemory {
	loaded := make(map[brain.Path]bool, len(memories))
	for _, mem := range memories {
		loaded[mem.Path] = true
	}

	var linked []SurfacedMemory
	for _, mem := range memories {
		if len(linked) >= maxLinkedMemories {
			break
		}

		links := ExtractWikilinks(mem.Content)
		for _, link := range links {
			if len(linked) >= maxLinkedMemories {
				break
			}

			resolved := m.ResolveWikilink(ctx, link, projectPath)
			if resolved == "" || loaded[resolved] || surfaced[resolved] {
				continue
			}

			content, err := m.readCappedTopic(ctx, resolved)
			if err != nil {
				continue
			}

			linkTarget := link
			if idx := strings.Index(linkTarget, "|"); idx >= 0 {
				linkTarget = linkTarget[:idx]
			}

			linked = append(linked, SurfacedMemory{
				Path:       resolved,
				Content:    content,
				Topic:      m.topicFromPath(ctx, resolved),
				LinkedFrom: strings.TrimSpace(linkTarget),
			})
			loaded[resolved] = true
		}
	}

	return linked
}

// topicFromPath creates a minimal TopicFile from a resolved path.
func (m *Memory) topicFromPath(ctx context.Context, p brain.Path) TopicFile {
	content, err := m.ReadTopic(ctx, p)
	if err != nil {
		return TopicFile{Path: p, Scope: m.inferScope(p)}
	}

	fm, _ := ParseFrontmatter(content)
	name := fm.Name
	if name == "" {
		name = strings.TrimSuffix(baseName(string(p)), ".md")
	}

	return TopicFile{
		Name:        name,
		Description: fm.Description,
		Type:        fm.Type,
		Path:        p,
		Created:     fm.Created,
		Modified:    fm.Modified,
		Tags:        fm.Tags,
		Confidence:  fm.Confidence,
		Source:      fm.Source,
		Scope:       m.inferScope(p),
	}
}

// inferScope returns "global" if the path is inside the global memory
// tree, otherwise "project".
func (m *Memory) inferScope(p brain.Path) string {
	s := string(p)
	if strings.HasPrefix(s, string(brain.MemoryGlobalPrefix())+"/") || s == string(brain.MemoryGlobalPrefix()) {
		return "global"
	}
	return "project"
}

// parseSelectedMemories extracts filenames from the selector's JSON
// response.
func parseSelectedMemories(content string) []string {
	content = strings.TrimSpace(content)

	if idx := strings.Index(content, "{"); idx >= 0 {
		if end := strings.LastIndex(content, "}"); end > idx {
			content = content[idx : end+1]
		}
	}

	var result struct {
		Selected []string `json:"selected"`
	}
	if err := json.Unmarshal([]byte(content), &result); err != nil {
		return nil
	}

	if len(result.Selected) > maxRecallTopics {
		result.Selected = result.Selected[:maxRecallTopics]
	}

	return result.Selected
}

// readCappedTopic reads a topic file, capping content at
// maxMemoryLines and maxMemoryBytes.
func (m *Memory) readCappedTopic(ctx context.Context, p brain.Path) (string, error) {
	content, err := m.ReadTopic(ctx, p)
	if err != nil {
		return "", err
	}

	if len(content) > maxMemoryBytes {
		content = content[:maxMemoryBytes] + "\n[...truncated]"
	}

	lines := strings.SplitN(content, "\n", maxMemoryLines+1)
	if len(lines) > maxMemoryLines {
		lines = lines[:maxMemoryLines]
		lines = append(lines, "[...truncated]")
		content = strings.Join(lines, "\n")
	}

	return content, nil
}

// filterSurfaced returns topics whose paths are not already in the
// surfaced set.
func filterSurfaced(topics []TopicFile, surfaced map[brain.Path]bool) []TopicFile {
	if len(topics) == 0 {
		return nil
	}
	out := make([]TopicFile, 0, len(topics))
	for _, t := range topics {
		if !surfaced[t.Path] {
			out = append(out, t)
		}
	}
	return out
}
