// SPDX-License-Identifier: Apache-2.0

package memory

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"strings"
	"sync"
	"time"

	"github.com/jeffs-brain/memory/go/brain"
	"github.com/jeffs-brain/memory/go/llm"
)

// Extraction thresholds.
const (
	extractMaxTokens   = 4096
	extractTemperature = 0.2
	extractMinMessages = 6
	extractMaxRecent   = 40
)

// Extractor manages background memory extraction. It runs after each
// turn to distil durable knowledge from the conversation into memory
// files.
type Extractor struct {
	mem *Memory

	mu         sync.Mutex
	lastCursor int
	inProgress bool

	ctx *Contextualiser
}

// NewExtractor creates a new Extractor bound to the supplied Memory.
func NewExtractor(mem *Memory) *Extractor {
	return &Extractor{mem: mem}
}

// SetContextualiser wires an optional [Contextualiser] into the
// extractor.
func (e *Extractor) SetContextualiser(c *Contextualiser) {
	if e == nil {
		return
	}
	e.ctx = c
}

// extractionPrompt is the system prompt for the extraction agent. Ported
// verbatim from jeff; prompt content is tuned.
const extractionPrompt = `You are a memory extraction agent. Analyse the recent conversation messages below and determine what durable knowledge should be saved to the persistent memory system.

You MUST respond with ONLY a JSON object. Do NOT call tools, do NOT write prose. Just output the JSON.

Both speakers contribute durable knowledge. Treat user turns and assistant turns as equally valid sources of facts. Capture everything the user stated AND everything the assistant provided: recommendations (restaurants, hotels, shops, books), specific named suggestions, recipes, itineraries, enumerated lists or rankings the assistant gave, answers the assistant produced, corrections the assistant issued, plans the assistant proposed, colours or attributes the assistant described, and any quantities or dates the assistant cited. If the assistant enumerated items (a list of jobs, options, steps, or candidates), save the full enumeration verbatim including positions where relevant. When in doubt, extract both sides.

Memory types:
- user: User's role, preferences, knowledge level, working style
- feedback: Corrections or confirmations about approach (what to avoid or keep doing)
- project: Non-obvious context about ongoing work, goals, decisions, deadlines (includes assistant recommendations and enumerations worth recalling later)
- reference: Pointers to external systems, URLs, project names, named entities the assistant surfaced (restaurants, hotels, businesses, books, product names)

Memory scopes:
- global (~/.config/jeff/memory/): Cross-project knowledge. Types: user, feedback
- project (project memory directory): Project-specific knowledge. Types: project, reference

When deciding scope:
- user preferences, working style, general corrections → global
- project architecture, project-specific decisions, external system pointers, assistant recommendations and enumerations → project
- default to "project" if unsure

Examples of assistant-turn facts that MUST be captured:
- "I recommend Roscioli for romantic Italian in Rome." → create a reference memory naming the restaurant, cuisine, city.
- "Here are seven work-from-home jobs for seniors: 1. Virtual Assistant, 2. ..., 7. Transcriptionist." → save the full numbered list so later recall can reconstruct any position.
- "The Plesiosaur in the children's book had a blue scaly body." → save the attribute with its subject.

Do NOT save:
- Code patterns, architecture, or file paths derivable from the codebase
- Git history or recent changes (use git log for those)
- Debugging solutions (the fix is in the code)
- Ephemeral task details or in-progress work
- Anything already in the existing memories listed below

For each memory worth saving, output:
- action: "create" (new file) or "update" (modify existing)
- filename: e.g. "feedback_testing.md" (kebab-case, descriptive)
- name: human-readable name
- description: one-line description (used for future recall)
- type: user | feedback | project | reference
- scope: "global" or "project" (default to "project" if unsure)
- content: the memory content (structured with Why: and How to apply: lines for feedback/project types)
- index_entry: one-line entry for MEMORY.md (under 150 chars)
- supersedes (optional): when the user has corrected, updated, or contradicted an earlier stated fact for the same topic, set this to the filename of the earlier memory so it is retired. Only fill when you are confident the new fact replaces a specific older one; prefer leaving empty when unsure.

If nothing is worth saving, return: {"memories": []}

Respond with ONLY valid JSON: {"memories": [...]}`

// extractUserPrompt builds the user message for the extraction agent.
func extractUserPrompt(messages []Message, existingManifest, memDirDisplay string) string {
	var b strings.Builder

	if existingManifest != "" {
		b.WriteString("## Existing memory files\n\n")
		b.WriteString(existingManifest)
		b.WriteString("\n\nCheck this list before writing — update an existing file rather than creating a duplicate.\n\n")
	}

	b.WriteString("## Recent conversation\n\n")
	for _, m := range messages {
		role := string(m.Role)
		content := m.Content
		if len(content) > 2000 {
			content = content[:2000] + "\n[...truncated]"
		}
		if m.Role == RoleTool {
			if len(content) > 300 {
				content = content[:300] + "..."
			}
			b.WriteString(fmt.Sprintf("[%s (%s)]: %s\n\n", role, m.Name, content))
			continue
		}
		b.WriteString(fmt.Sprintf("[%s]: %s\n\n", role, content))
	}

	b.WriteString(fmt.Sprintf("\nMemory directory: %s\n", memDirDisplay))

	return b.String()
}

// extractionResult represents a parsed extraction response.
type extractionResult struct {
	Memories []ExtractedMemory `json:"memories"`
}

// ExtractedMemory represents a single memory extracted from a
// conversation by the extraction LLM.
type ExtractedMemory struct {
	Action      string `json:"action"`
	Filename    string `json:"filename"`
	Name        string `json:"name"`
	Description string `json:"description"`
	Type        string `json:"type"`
	Content     string `json:"content"`
	IndexEntry  string `json:"index_entry"`
	Scope       string `json:"scope"`
	Supersedes  string `json:"supersedes,omitempty"`
	Tags        []string `json:"tags,omitempty"`
	// SessionID, when set, is written into the fact's frontmatter so
	// multi-session queries can filter or aggregate by origin session.
	SessionID string `json:"-"`
	// ObservedOn mirrors the session date for the fact. Populated by
	// replay ingests.
	ObservedOn string `json:"-"`
	// SessionDate is the short ISO YYYY-MM-DD form of the parent
	// session's date, written into frontmatter as session_date.
	SessionDate string `json:"-"`
	// ContextPrefix, when non-empty, is a short LLM-generated situating
	// prefix prepended to the fact body before writing.
	ContextPrefix string `json:"-"`
	// ModifiedOverride, when non-empty, replaces the default "now"
	// timestamp written into the memory's frontmatter modified field.
	ModifiedOverride string `json:"-"`
}

// MaybeExtract checks if extraction should run and, if so, distils
// durable knowledge from recent conversation messages into memory
// files.
//
// This is designed to be called from a background goroutine after the
// main agentic loop completes. It is safe for concurrent use.
func (e *Extractor) MaybeExtract(
	ctx context.Context,
	provider llm.Provider,
	model string,
	projectPath string,
	messages []Message,
) {
	e.mu.Lock()
	if e.inProgress {
		e.mu.Unlock()
		return
	}
	e.inProgress = true
	cursor := e.lastCursor
	e.mu.Unlock()

	defer func() {
		e.mu.Lock()
		e.inProgress = false
		e.mu.Unlock()
	}()

	if len(messages)-cursor < extractMinMessages {
		return
	}

	slug := ProjectSlug(projectPath)

	var physicalHints []string
	if p, ok := e.mem.store.LocalPath(brain.MemoryGlobalPrefix()); ok {
		physicalHints = append(physicalHints, p)
	}
	if p, ok := e.mem.store.LocalPath(brain.MemoryProjectPrefix(slug)); ok {
		physicalHints = append(physicalHints, p)
	}
	if hasMemoryWrites(messages[cursor:], physicalHints...) {
		e.mu.Lock()
		e.lastCursor = len(messages)
		e.mu.Unlock()
		return
	}

	recent := messages[cursor:]
	if len(recent) > extractMaxRecent {
		recent = recent[len(recent)-extractMaxRecent:]
	}

	projectTopics, _ := e.mem.ListProjectTopics(ctx, projectPath)
	globalTopics, _ := e.mem.ListGlobalTopics(ctx)

	manifest := buildManifests(projectTopics, globalTopics)

	memDirDisplay := string(brain.MemoryProjectPrefix(slug))
	if len(physicalHints) > 0 {
		memDirDisplay = physicalHints[len(physicalHints)-1]
	}
	userPrompt := extractUserPrompt(recent, manifest, memDirDisplay)

	resp, err := provider.Complete(ctx, llm.CompleteRequest{
		Model: model,
		Messages: []llm.Message{
			{Role: RoleSystem, Content: extractionPrompt},
			{Role: RoleUser, Content: userPrompt},
		},
		MaxTokens:   extractMaxTokens,
		Temperature: extractTemperature,
	})
	if err != nil {
		return
	}

	result := parseExtractionResult(resp.Text)
	if len(result.Memories) == 0 {
		e.mu.Lock()
		e.lastCursor = len(messages)
		e.mu.Unlock()
		return
	}

	if e.ctx.Enabled() {
		summary := extractSessionSummary(recent)
		for i := range result.Memories {
			prefix := e.ctx.BuildPrefix(ctx, "", summary, result.Memories[i].Content)
			if prefix != "" {
				result.Memories[i].ContextPrefix = prefix
			}
		}
	}

	if err := e.mem.ApplyExtractions(ctx, slug, result.Memories); err != nil {
		slog.Warn("memory: apply extractions failed", "err", err)
	}

	e.mu.Lock()
	e.lastCursor = len(messages)
	e.mu.Unlock()
}

// ResetCursor resets the extraction cursor so the next MaybeExtract
// call processes all messages from scratch.
func (e *Extractor) ResetCursor() {
	e.mu.Lock()
	e.lastCursor = 0
	e.mu.Unlock()
}

// ExtractFromMessages runs the extraction LLM call and returns
// structured results without applying them. Useful for replay-style
// ingests that want to post-process before writing.
func ExtractFromMessages(
	ctx context.Context,
	provider llm.Provider,
	model string,
	mem *Memory,
	projectPath string,
	messages []Message,
) ([]ExtractedMemory, error) {
	if len(messages) < 2 {
		return nil, nil
	}

	slug := ProjectSlug(projectPath)
	recent := messages
	if len(recent) > extractMaxRecent {
		recent = recent[len(recent)-extractMaxRecent:]
	}

	projectTopics, _ := mem.ListProjectTopics(ctx, projectPath)
	globalTopics, _ := mem.ListGlobalTopics(ctx)

	manifest := buildManifests(projectTopics, globalTopics)

	memDirDisplay := string(brain.MemoryProjectPrefix(slug))
	userPrompt := extractUserPrompt(recent, manifest, memDirDisplay)

	resp, err := provider.Complete(ctx, llm.CompleteRequest{
		Model: model,
		Messages: []llm.Message{
			{Role: RoleSystem, Content: extractionPrompt},
			{Role: RoleUser, Content: userPrompt},
		},
		MaxTokens:   extractMaxTokens,
		Temperature: extractTemperature,
	})
	if err != nil {
		return nil, fmt.Errorf("extraction LLM call: %w", err)
	}

	result := parseExtractionResult(resp.Text)
	return result.Memories, nil
}

// buildManifests glues the project and global manifests into a single
// labelled block for the extraction prompt.
func buildManifests(projectTopics, globalTopics []TopicFile) string {
	var b strings.Builder
	if pm := buildManifest(projectTopics); pm != "" {
		b.WriteString("## Project memory files\n\n")
		b.WriteString(pm)
	}
	if gm := buildManifest(globalTopics); gm != "" {
		if b.Len() > 0 {
			b.WriteString("\n\n")
		}
		b.WriteString("## Global memory files\n\n")
		b.WriteString(gm)
	}
	return b.String()
}

// extractSessionSummary derives a short one-line session header from a
// conversation slice.
func extractSessionSummary(messages []Message) string {
	for _, m := range messages {
		if m.Role == RoleSystem && strings.TrimSpace(m.Content) != "" {
			return truncateOneLine(m.Content, 240)
		}
	}
	for _, m := range messages {
		if m.Role == RoleUser && strings.TrimSpace(m.Content) != "" {
			return truncateOneLine(m.Content, 240)
		}
	}
	return ""
}

// truncateOneLine collapses newlines to spaces and caps at n runes.
func truncateOneLine(s string, n int) string {
	s = strings.ReplaceAll(s, "\r\n", " ")
	s = strings.ReplaceAll(s, "\n", " ")
	s = strings.Join(strings.Fields(s), " ")
	if n > 0 && len(s) > n {
		s = s[:n] + "..."
	}
	return s
}

// hasMemoryWrites checks if any assistant message in the slice contains
// tool calls that wrote to either the project or global memory
// directory.
func hasMemoryWrites(messages []Message, memDirs ...string) bool {
	for _, m := range messages {
		if m.Role != RoleAssistant {
			continue
		}
		for _, tc := range m.ToolCalls {
			if tc.Name == "write" || tc.Name == "edit" {
				args := string(tc.Arguments)
				for _, dir := range memDirs {
					if dir != "" && strings.Contains(args, dir) {
						return true
					}
				}
				if strings.Contains(args, "memory/") {
					return true
				}
			}
		}
	}
	return false
}

// parseExtractionResult extracts memories from the model's JSON
// response.
func parseExtractionResult(content string) extractionResult {
	content = strings.TrimSpace(content)

	if idx := strings.Index(content, "{"); idx >= 0 {
		if end := strings.LastIndex(content, "}"); end > idx {
			content = content[idx : end+1]
		}
	}

	var result extractionResult
	if err := json.Unmarshal([]byte(content), &result); err != nil {
		return extractionResult{}
	}

	return result
}

// ApplyExtractions writes extracted memories through the brain store
// and updates MEMORY.md indices. All writes happen in a single batch.
func (m *Memory) ApplyExtractions(ctx context.Context, projectSlug string, memories []ExtractedMemory) error {
	var projectEntries, globalEntries []string
	type pendingTopic struct {
		path    brain.Path
		content []byte
	}
	var pending []pendingTopic

	for _, em := range memories {
		if em.Filename == "" || em.Content == "" {
			continue
		}

		filename := sanitiseFilename(em.Filename)
		if !strings.HasSuffix(filename, ".md") {
			filename += ".md"
		}
		slug := strings.TrimSuffix(filename, ".md")

		var p brain.Path
		if em.Scope == "global" {
			p = brain.MemoryGlobalTopic(slug)
		} else {
			p = brain.MemoryProjectTopic(projectSlug, slug)
		}

		content := buildTopicFileContent(em)
		pending = append(pending, pendingTopic{path: p, content: content})

		if em.IndexEntry != "" {
			if em.Scope == "global" {
				globalEntries = append(globalEntries, em.IndexEntry)
			} else {
				projectEntries = append(projectEntries, em.IndexEntry)
			}
		}
	}

	if len(pending) == 0 {
		return nil
	}

	return m.store.Batch(ctx, brain.BatchOptions{Reason: "extract"}, func(b brain.Batch) error {
		for _, p := range pending {
			if err := b.Write(ctx, p.path, p.content); err != nil {
				return err
			}
		}
		for _, em := range memories {
			if em.Supersedes == "" {
				continue
			}
			oldFile := sanitiseFilename(em.Supersedes)
			if !strings.HasSuffix(oldFile, ".md") {
				oldFile += ".md"
			}
			oldSlug := strings.TrimSuffix(oldFile, ".md")
			newFile := sanitiseFilename(em.Filename)
			if !strings.HasSuffix(newFile, ".md") {
				newFile += ".md"
			}
			var oldPath brain.Path
			if em.Scope == "global" {
				oldPath = brain.MemoryGlobalTopic(oldSlug)
			} else {
				oldPath = brain.MemoryProjectTopic(projectSlug, oldSlug)
			}
			if err := stampSupersededBy(ctx, b, oldPath, newFile); err != nil {
				return err
			}
		}
		if len(projectEntries) > 0 {
			if err := m.appendIndexEntries(ctx, b, brain.MemoryProjectIndex(projectSlug), projectEntries); err != nil {
				return err
			}
		}
		if len(globalEntries) > 0 {
			if err := m.appendIndexEntries(ctx, b, brain.MemoryGlobalIndex(), globalEntries); err != nil {
				return err
			}
		}
		return nil
	})
}

// stampSupersededBy rewrites an existing memory file's frontmatter with
// a superseded_by pointer to the new file.
func stampSupersededBy(ctx context.Context, b brain.Batch, oldPath brain.Path, newFile string) error {
	raw, err := b.Read(ctx, oldPath)
	if err != nil {
		return nil
	}
	content := string(raw)
	lines := strings.Split(content, "\n")
	if len(lines) < 2 || strings.TrimSpace(lines[0]) != "---" {
		return nil
	}
	closeIdx := -1
	for i := 1; i < len(lines); i++ {
		if strings.TrimSpace(lines[i]) == "---" {
			closeIdx = i
			break
		}
	}
	if closeIdx < 0 {
		return nil
	}

	replaced := false
	for i := 1; i < closeIdx; i++ {
		if strings.HasPrefix(strings.TrimSpace(lines[i]), "superseded_by:") {
			lines[i] = fmt.Sprintf("superseded_by: %s", newFile)
			replaced = true
			break
		}
	}
	if !replaced {
		inserted := make([]string, 0, len(lines)+1)
		inserted = append(inserted, lines[:closeIdx]...)
		inserted = append(inserted, fmt.Sprintf("superseded_by: %s", newFile))
		inserted = append(inserted, lines[closeIdx:]...)
		lines = inserted
	}
	return b.Write(ctx, oldPath, []byte(strings.Join(lines, "\n")))
}

// sanitiseFilename strips path traversal from an LLM-supplied filename.
func sanitiseFilename(name string) string {
	if idx := strings.LastIndexAny(name, "/\\"); idx >= 0 {
		name = name[idx+1:]
	}
	return name
}

// buildTopicFileContent builds the full markdown file contents for an
// extracted memory, including YAML frontmatter.
func buildTopicFileContent(em ExtractedMemory) []byte {
	now := time.Now().UTC().Format(time.RFC3339)
	modified := now
	created := now
	if em.ModifiedOverride != "" {
		modified = em.ModifiedOverride
		created = em.ModifiedOverride
	}
	var b strings.Builder
	b.WriteString("---\n")
	if em.Name != "" {
		b.WriteString(fmt.Sprintf("name: %s\n", em.Name))
	}
	if em.Description != "" {
		b.WriteString(fmt.Sprintf("description: %s\n", em.Description))
	}
	if em.Type != "" {
		b.WriteString(fmt.Sprintf("type: %s\n", em.Type))
	}
	if em.Action == "create" {
		b.WriteString(fmt.Sprintf("created: %s\n", created))
	}
	b.WriteString(fmt.Sprintf("modified: %s\n", modified))
	b.WriteString("source: session\n")
	if em.Supersedes != "" {
		b.WriteString(fmt.Sprintf("supersedes: %s\n", em.Supersedes))
	}
	if em.SessionID != "" {
		b.WriteString(fmt.Sprintf("session_id: %s\n", em.SessionID))
	}
	if em.ObservedOn != "" {
		b.WriteString(fmt.Sprintf("observed_on: %s\n", em.ObservedOn))
	}
	if em.SessionDate != "" {
		b.WriteString(fmt.Sprintf("session_date: %s\n", em.SessionDate))
	}
	if len(em.Tags) > 0 {
		b.WriteString(fmt.Sprintf("tags: [%s]\n", strings.Join(em.Tags, ", ")))
	}
	b.WriteString("---\n\n")
	b.WriteString(ApplyContextualPrefix(em.ContextPrefix, em.Content))
	b.WriteString("\n")
	return []byte(b.String())
}

// appendIndexEntries reads the current index, appends any new entries
// that are not already present, and writes it back via the batch.
func (m *Memory) appendIndexEntries(ctx context.Context, b brain.Batch, indexPath brain.Path, entries []string) error {
	var content string
	existing, err := b.Read(ctx, indexPath)
	if err == nil {
		content = strings.TrimSpace(string(existing))
	}
	for _, entry := range entries {
		entry = strings.TrimSpace(entry)
		if entry == "" {
			continue
		}
		if strings.Contains(content, entry) {
			continue
		}
		if content != "" {
			content += "\n"
		}
		content += entry
	}
	return b.Write(ctx, indexPath, []byte(content+"\n"))
}
