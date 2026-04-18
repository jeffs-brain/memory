// SPDX-License-Identifier: Apache-2.0

package memory

import (
	"context"
	"fmt"
	"regexp"
	"strings"
	"time"

	"github.com/jeffs-brain/memory/go/brain"
)

// HeuristicSummary is a lightweight view of a heuristic memory file,
// used by callers that list or display heuristics.
type HeuristicSummary struct {
	Name       string
	Path       brain.Path
	Confidence string
	Tags       []string
	Scope      string
	IsAnti     bool
}

// stopWords are filtered out when extracting significant words from a rule.
var stopWords = map[string]bool{
	"the": true, "a": true, "an": true, "in": true, "on": true,
	"for": true, "to": true, "of": true, "with": true, "when": true,
	"and": true, "or": true, "but": true, "is": true, "are": true,
	"was": true, "were": true, "be": true, "not": true, "do": true,
}

// nonAlphaNum matches any character that is not a letter, digit, or
// space.
var nonAlphaNum = regexp.MustCompile(`[^a-z0-9 ]`)

// ApplyHeuristics writes or updates heuristic memory files based on
// reflection output. All writes happen in a single batch so heuristics
// land as one coherent commit.
func (m *Memory) ApplyHeuristics(ctx context.Context, projectSlug string, heuristics []Heuristic) error {
	var projectEntries, globalEntries []string

	type pending struct {
		path    brain.Path
		content []byte
	}
	var writes []pending

	for _, h := range heuristics {
		if h.Rule == "" {
			continue
		}

		var prefix brain.Path
		if h.Scope == "global" {
			prefix = brain.MemoryGlobalPrefix()
		} else {
			prefix = brain.MemoryProjectPrefix(projectSlug)
		}

		existingPath, existingContent, found := m.findExistingHeuristic(ctx, h, prefix)

		var path brain.Path
		var content string

		if found {
			content = mergeHeuristic(existingContent, h)
			path = existingPath
		} else {
			content = buildHeuristicContent(h)
			if h.Scope == "global" {
				path = brain.MemoryGlobalTopic(strings.TrimSuffix(heuristicFilename(h), ".md"))
			} else {
				path = brain.MemoryProjectTopic(projectSlug, strings.TrimSuffix(heuristicFilename(h), ".md"))
			}
		}

		writes = append(writes, pending{path: path, content: []byte(content)})

		entry := fmt.Sprintf("- [heuristic] %s: %s", baseName(string(path)), truncate(h.Rule, 100))
		if h.Scope == "global" {
			globalEntries = append(globalEntries, entry)
		} else {
			projectEntries = append(projectEntries, entry)
		}
	}

	if len(writes) == 0 {
		return nil
	}

	return m.store.Batch(ctx, brain.BatchOptions{Reason: "reflect"}, func(b brain.Batch) error {
		for _, w := range writes {
			if err := b.Write(ctx, w.path, w.content); err != nil {
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

// heuristicFilename generates a filename from the heuristic's category
// and the first two significant words of the rule.
func heuristicFilename(h Heuristic) string {
	words := significantWords(h.Rule)
	slug := h.Category
	if slug == "" {
		slug = "general"
	}

	prefix := "heuristic"
	if h.AntiPattern {
		prefix = "heuristic-anti"
	}

	limit := 2
	if len(words) < limit {
		limit = len(words)
	}

	parts := []string{prefix, slug}
	parts = append(parts, words[:limit]...)

	return strings.Join(parts, "-") + ".md"
}

// significantWords extracts lowercase words from text, filtering out
// stop words and non-alphanumeric characters.
func significantWords(text string) []string {
	cleaned := nonAlphaNum.ReplaceAllString(strings.ToLower(text), " ")
	raw := strings.Fields(cleaned)

	var result []string
	for _, w := range raw {
		if !stopWords[w] && len(w) > 1 {
			result = append(result, w)
		}
	}
	return result
}

// findExistingHeuristic searches under the given prefix for an existing
// heuristic file matching the given heuristic.
func (m *Memory) findExistingHeuristic(ctx context.Context, h Heuristic, prefix brain.Path) (path brain.Path, content string, found bool) {
	entries, err := m.store.List(ctx, prefix, brain.ListOpts{IncludeGenerated: true})
	if err != nil {
		return "", "", false
	}

	candidateWords := significantWords(heuristicFilename(h))

	for _, entry := range entries {
		if entry.IsDir {
			continue
		}
		name := baseName(string(entry.Path))
		if !strings.HasSuffix(name, ".md") {
			continue
		}
		if strings.EqualFold(name, "MEMORY.md") {
			continue
		}

		data, err := m.store.Read(ctx, entry.Path)
		if err != nil {
			continue
		}

		fm, _ := ParseFrontmatter(string(data))

		if !hasTag(fm.Tags, "heuristic") {
			continue
		}
		if !hasTag(fm.Tags, h.Category) {
			continue
		}

		existingWords := significantWords(name)
		if jaccardSimilarity(candidateWords, existingWords) > 0.5 {
			return entry.Path, string(data), true
		}
	}

	return "", "", false
}

// buildHeuristicContent produces a complete heuristic memory file with
// YAML frontmatter and structured body.
func buildHeuristicContent(h Heuristic) string {
	now := time.Now().UTC().Format(time.RFC3339)
	heading := firstNWords(h.Rule, 5)

	var b strings.Builder

	b.WriteString("---\n")
	b.WriteString(fmt.Sprintf("name: \"%s: %s\"\n", capitalise(h.Category), heading))
	b.WriteString(fmt.Sprintf("description: \"%s\"\n", truncate(h.Rule, 100)))
	b.WriteString("type: feedback\n")
	b.WriteString(fmt.Sprintf("created: %s\n", now))
	b.WriteString(fmt.Sprintf("modified: %s\n", now))
	b.WriteString(fmt.Sprintf("confidence: %s\n", h.Confidence))
	b.WriteString("source: reflection\n")
	b.WriteString("tags:\n")
	b.WriteString("  - heuristic\n")
	b.WriteString(fmt.Sprintf("  - %s\n", h.Category))
	if h.AntiPattern {
		b.WriteString("  - anti-pattern\n")
	}
	b.WriteString("---\n\n")

	if h.AntiPattern {
		b.WriteString(fmt.Sprintf("## Anti-pattern: %s\n\n", heading))
		b.WriteString(fmt.Sprintf("**Don't:** %s\n\n", h.Rule))
		if alt := extractAlternative(h.Rule); alt != "" {
			b.WriteString(fmt.Sprintf("**Instead:** %s\n\n", alt))
		}
		b.WriteString("**Why:** Observed during reflection\n\n")
		b.WriteString(fmt.Sprintf("**Confidence:** %s (1 observation)\n", h.Confidence))
	} else {
		b.WriteString(fmt.Sprintf("## %s\n\n", heading))
		b.WriteString(fmt.Sprintf("%s\n\n", h.Rule))
		if h.Context != "" {
			b.WriteString(fmt.Sprintf("**Context:** %s\n\n", h.Context))
		}
		b.WriteString("**Why:** Observed during reflection\n\n")
		b.WriteString(fmt.Sprintf("**Confidence:** %s (1 observation)\n", h.Confidence))
	}

	return b.String()
}

// mergeHeuristic adds a new observation section to an existing heuristic
// file.
func mergeHeuristic(existingContent string, h Heuristic) string {
	fm, body := ParseFrontmatter(existingContent)
	count := countSections(body)
	newConfidence := confidenceFromObservations(count + 1)

	now := time.Now().UTC().Format(time.RFC3339)
	fm.Modified = now
	fm.Confidence = newConfidence

	heading := firstNWords(h.Rule, 5)
	var section strings.Builder

	if h.AntiPattern {
		section.WriteString(fmt.Sprintf("\n\n## Anti-pattern: %s\n\n", heading))
		section.WriteString(fmt.Sprintf("**Don't:** %s\n\n", h.Rule))
		section.WriteString("**Why:** Observed during reflection\n\n")
		section.WriteString(fmt.Sprintf("**Confidence:** %s (%d observations)\n", newConfidence, count+1))
	} else {
		section.WriteString(fmt.Sprintf("\n\n## %s\n\n", heading))
		section.WriteString(fmt.Sprintf("%s\n\n", h.Rule))
		if h.Context != "" {
			section.WriteString(fmt.Sprintf("**Context:** %s\n\n", h.Context))
		}
		section.WriteString("**Why:** Observed during reflection\n\n")
		section.WriteString(fmt.Sprintf("**Confidence:** %s (%d observations)\n", newConfidence, count+1))
	}

	var b strings.Builder
	b.WriteString("---\n")
	if fm.Name != "" {
		b.WriteString(fmt.Sprintf("name: \"%s\"\n", fm.Name))
	}
	if fm.Description != "" {
		b.WriteString(fmt.Sprintf("description: \"%s\"\n", fm.Description))
	}
	if fm.Type != "" {
		b.WriteString(fmt.Sprintf("type: %s\n", fm.Type))
	}
	if fm.Created != "" {
		b.WriteString(fmt.Sprintf("created: %s\n", fm.Created))
	}
	b.WriteString(fmt.Sprintf("modified: %s\n", now))
	b.WriteString(fmt.Sprintf("confidence: %s\n", newConfidence))
	if fm.Source != "" {
		b.WriteString(fmt.Sprintf("source: %s\n", fm.Source))
	}
	if len(fm.Tags) > 0 {
		b.WriteString("tags:\n")
		for _, tag := range fm.Tags {
			b.WriteString(fmt.Sprintf("  - %s\n", tag))
		}
	}
	b.WriteString("---\n\n")

	b.WriteString(strings.TrimSpace(body))
	b.WriteString(section.String())

	return b.String()
}

// confidenceFromObservations returns a confidence level based on the
// number of times a heuristic has been observed.
func confidenceFromObservations(count int) string {
	switch {
	case count >= 4:
		return "high"
	case count >= 2:
		return "medium"
	default:
		return "low"
	}
}

// ListHeuristicsIn returns all heuristic memory files from both project
// and global memory for the given project path.
func (m *Memory) ListHeuristicsIn(ctx context.Context, projectPath string) []HeuristicSummary {
	var summaries []HeuristicSummary

	projectTopics, _ := m.ListProjectTopics(ctx, projectPath)
	for _, t := range projectTopics {
		if hasTag(t.Tags, "heuristic") {
			summaries = append(summaries, HeuristicSummary{
				Name:       t.Name,
				Path:       t.Path,
				Confidence: t.Confidence,
				Tags:       t.Tags,
				Scope:      "project",
				IsAnti:     hasTag(t.Tags, "anti-pattern"),
			})
		}
	}

	globalTopics, _ := m.ListGlobalTopics(ctx)
	for _, t := range globalTopics {
		if hasTag(t.Tags, "heuristic") {
			summaries = append(summaries, HeuristicSummary{
				Name:       t.Name,
				Path:       t.Path,
				Confidence: t.Confidence,
				Tags:       t.Tags,
				Scope:      "global",
				IsAnti:     hasTag(t.Tags, "anti-pattern"),
			})
		}
	}

	return summaries
}

// hasTag checks whether a tag slice contains the given tag
// (case-insensitive).
func hasTag(tags []string, tag string) bool {
	target := strings.ToLower(tag)
	for _, t := range tags {
		if strings.ToLower(t) == target {
			return true
		}
	}
	return false
}

// jaccardSimilarity computes the Jaccard index between two word slices.
func jaccardSimilarity(a, b []string) float64 {
	if len(a) == 0 && len(b) == 0 {
		return 1.0
	}

	setA := make(map[string]bool, len(a))
	for _, w := range a {
		setA[w] = true
	}

	setB := make(map[string]bool, len(b))
	for _, w := range b {
		setB[w] = true
	}

	intersection := 0
	for w := range setA {
		if setB[w] {
			intersection++
		}
	}

	union := len(setA)
	for w := range setB {
		if !setA[w] {
			union++
		}
	}

	if union == 0 {
		return 0
	}
	return float64(intersection) / float64(union)
}

// countSections counts the number of ## headings in a markdown body.
func countSections(body string) int {
	count := 0
	for _, line := range strings.Split(body, "\n") {
		if strings.HasPrefix(strings.TrimSpace(line), "## ") {
			count++
		}
	}
	return count
}

// firstNWords returns the first n words of text, joined by spaces.
func firstNWords(text string, n int) string {
	words := strings.Fields(text)
	if len(words) > n {
		words = words[:n]
	}
	return strings.Join(words, " ")
}

// truncate shortens text to maxLen characters, appending "..." if
// truncated.
func truncate(text string, maxLen int) string {
	if len(text) <= maxLen {
		return text
	}
	if maxLen < 4 {
		return text[:maxLen]
	}
	return text[:maxLen-3] + "..."
}

// capitalise returns the string with the first letter upper-cased.
func capitalise(s string) string {
	if s == "" {
		return s
	}
	return strings.ToUpper(s[:1]) + s[1:]
}

// extractAlternative attempts to find an alternative suggestion in a
// rule string.
func extractAlternative(rule string) string {
	lower := strings.ToLower(rule)
	for _, marker := range []string{"instead ", "use ", "prefer "} {
		if idx := strings.Index(lower, marker); idx >= 0 {
			return strings.TrimSpace(rule[idx:])
		}
	}
	return ""
}
