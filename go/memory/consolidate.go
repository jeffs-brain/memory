// SPDX-License-Identifier: Apache-2.0

package memory

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/jeffs-brain/memory/go/brain"
	"github.com/jeffs-brain/memory/go/llm"
)

// Consolidation thresholds.
const (
	stalenessThresholdDays     = 90
	deduplicationJaccardCutoff = 0.3
	deduplicationMaxTokens     = 512
	deduplicationTemperature   = 0.0
)

// Consolidator maintains memory health by regenerating indexes,
// detecting stale files, deduplicating overlapping memories, and
// reinforcing heuristics. All I/O is routed through the injected
// [*Memory]; each public run wraps its mutations in a single batch
// per scope.
//
// TODO(integration): jeff wires an injected KnowledgeBase for wiki
// promotion. That subsystem has not been ported, so the promotion
// step is omitted. The consolidator still handles index rebuilds,
// dedup, staleness, and heuristic reinforcement.
type Consolidator struct {
	mem      *Memory
	provider llm.Provider
	model    string

	mu         sync.Mutex
	inProgress bool
}

// ConsolidationReport summarises the work performed by a consolidation
// run.
type ConsolidationReport struct {
	EpisodesReviewed     int
	MemoriesMerged       int
	HeuristicsUpdated    int
	IndexesRebuilt       int
	StaleMemoriesFlagged int
	InsightsPromoted     int
	Duration             time.Duration
	Errors               []string
}

// NewConsolidator creates a Consolidator bound to the supplied Memory.
// The provider and model are used for LLM-powered dedup; pass a nil
// provider to skip LLM steps.
func NewConsolidator(provider llm.Provider, model string, mem *Memory) *Consolidator {
	return &Consolidator{provider: provider, model: model, mem: mem}
}

// RunFull performs all consolidation tasks including LLM-powered
// analysis.
func (c *Consolidator) RunFull(ctx context.Context) (*ConsolidationReport, error) {
	c.mu.Lock()
	if c.inProgress {
		c.mu.Unlock()
		return nil, fmt.Errorf("consolidation already in progress")
	}
	c.inProgress = true
	c.mu.Unlock()

	defer func() {
		c.mu.Lock()
		c.inProgress = false
		c.mu.Unlock()
	}()

	start := time.Now()
	report := &ConsolidationReport{}

	for _, prefix := range c.scopePrefixes(ctx) {
		c.runScopeFull(ctx, prefix, report)
	}

	report.Duration = time.Since(start)
	return report, nil
}

// RunQuick performs only cheap tasks (no LLM calls).
func (c *Consolidator) RunQuick(ctx context.Context) (*ConsolidationReport, error) {
	c.mu.Lock()
	if c.inProgress {
		c.mu.Unlock()
		return nil, fmt.Errorf("consolidation already in progress")
	}
	c.inProgress = true
	c.mu.Unlock()

	defer func() {
		c.mu.Lock()
		c.inProgress = false
		c.mu.Unlock()
	}()

	start := time.Now()
	report := &ConsolidationReport{}

	for _, prefix := range c.scopePrefixes(ctx) {
		c.runScopeQuick(ctx, prefix, report)
	}

	report.Duration = time.Since(start)
	return report, nil
}

func (c *Consolidator) runScopeQuick(ctx context.Context, prefix brain.Path, report *ConsolidationReport) {
	c.detectStalenessIn(ctx, prefix, report)

	err := c.mem.store.Batch(ctx, brain.BatchOptions{Reason: "consolidate"}, func(b brain.Batch) error {
		if rebuildErr := c.rebuildIndexInBatch(ctx, b, prefix); rebuildErr != nil {
			report.Errors = append(report.Errors, fmt.Sprintf("rebuilding index %s: %s", prefix, rebuildErr))
		} else {
			report.IndexesRebuilt++
		}
		updated, errs := c.reinforceHeuristicsInBatch(ctx, b, prefix)
		report.HeuristicsUpdated += updated
		report.Errors = append(report.Errors, errs...)
		return nil
	})
	if err != nil {
		report.Errors = append(report.Errors, fmt.Sprintf("consolidate batch %s: %s", prefix, err))
	}
}

func (c *Consolidator) runScopeFull(ctx context.Context, prefix brain.Path, report *ConsolidationReport) {
	c.detectStalenessIn(ctx, prefix, report)

	err := c.mem.store.Batch(ctx, brain.BatchOptions{Reason: "consolidate"}, func(b brain.Batch) error {
		if rebuildErr := c.rebuildIndexInBatch(ctx, b, prefix); rebuildErr != nil {
			report.Errors = append(report.Errors, fmt.Sprintf("rebuilding index %s: %s", prefix, rebuildErr))
		} else {
			report.IndexesRebuilt++
		}

		merged, dedupErrs := c.deduplicateScopeInBatch(ctx, b, prefix)
		report.MemoriesMerged += merged
		report.Errors = append(report.Errors, dedupErrs...)
		if merged > 0 {
			if rebuildErr := c.rebuildIndexInBatch(ctx, b, prefix); rebuildErr != nil {
				report.Errors = append(report.Errors, fmt.Sprintf("rebuilding index post-merge %s: %s", prefix, rebuildErr))
			}
		}

		updated, heurErrs := c.reinforceHeuristicsInBatch(ctx, b, prefix)
		report.HeuristicsUpdated += updated
		report.Errors = append(report.Errors, heurErrs...)
		return nil
	})
	if err != nil {
		report.Errors = append(report.Errors, fmt.Sprintf("consolidate batch %s: %s", prefix, err))
	}
}

// scopePrefixes returns the set of logical prefixes consolidation
// should iterate over.
func (c *Consolidator) scopePrefixes(ctx context.Context) []brain.Path {
	var prefixes []brain.Path
	prefixes = append(prefixes, brain.MemoryGlobalPrefix())

	entries, err := c.mem.store.List(ctx, brain.MemoryProjectsPrefix(), brain.ListOpts{IncludeGenerated: true})
	if err != nil && !errors.Is(err, brain.ErrNotFound) {
		return prefixes
	}
	for _, e := range entries {
		if !e.IsDir {
			continue
		}
		prefixes = append(prefixes, e.Path)
	}
	return prefixes
}

// rebuildIndexInBatch regenerates MEMORY.md under the given prefix.
func (c *Consolidator) rebuildIndexInBatch(ctx context.Context, b brain.Batch, prefix brain.Path) error {
	entries, err := b.List(ctx, prefix, brain.ListOpts{IncludeGenerated: true})
	if err != nil {
		if errors.Is(err, brain.ErrNotFound) {
			return nil
		}
		return err
	}

	var lines []string
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

		content, readErr := b.Read(ctx, entry.Path)
		if readErr != nil {
			continue
		}

		fm, _ := ParseFrontmatter(string(content))
		displayName := fm.Name
		if displayName == "" {
			displayName = strings.TrimSuffix(name, ".md")
		}

		desc := fm.Description
		if desc == "" {
			desc = fm.Type
		}
		if desc == "" {
			desc = "no description"
		}

		lines = append(lines, fmt.Sprintf("- [%s](%s) — %s", displayName, name, desc))
	}

	indexPath := brain.Path(string(prefix) + "/MEMORY.md")
	content := strings.Join(lines, "\n")
	if content != "" {
		content += "\n"
	}
	return b.Write(ctx, indexPath, []byte(content))
}

// detectStalenessIn counts topic files older than the staleness
// threshold.
func (c *Consolidator) detectStalenessIn(ctx context.Context, prefix brain.Path, report *ConsolidationReport) {
	threshold := time.Now().AddDate(0, 0, -stalenessThresholdDays)

	entries, err := c.mem.store.List(ctx, prefix, brain.ListOpts{IncludeGenerated: true})
	if err != nil {
		if errors.Is(err, brain.ErrNotFound) {
			return
		}
		report.Errors = append(report.Errors, fmt.Sprintf("reading prefix for staleness: %s: %s", prefix, err))
		return
	}

	for _, entry := range entries {
		if entry.IsDir {
			continue
		}
		name := baseName(string(entry.Path))
		if !strings.HasSuffix(name, ".md") || strings.EqualFold(name, "MEMORY.md") {
			continue
		}

		modTime := c.modifiedTime(ctx, entry.Path)
		if modTime.Before(threshold) {
			report.StaleMemoriesFlagged++
		}
	}
}

// modifiedTime returns the modified date from frontmatter if
// available, otherwise falls back to the store's Stat mtime.
func (c *Consolidator) modifiedTime(ctx context.Context, p brain.Path) time.Time {
	data, err := c.mem.store.Read(ctx, p)
	if err == nil {
		fm, _ := ParseFrontmatter(string(data))
		if fm.Modified != "" {
			if t, parseErr := time.Parse(time.RFC3339, fm.Modified); parseErr == nil {
				return t
			}
		}
	}
	info, err := c.mem.store.Stat(ctx, p)
	if err != nil {
		return time.Time{}
	}
	return info.ModTime
}

// deduplicationSystemPrompt describes how the LLM should resolve two
// overlapping memory files. Ported verbatim from jeff.
const deduplicationSystemPrompt = `You are analysing two memory files for overlap. Determine whether they cover the same topic or are distinct.

Respond with ONLY a JSON object:
{
  "verdict": "keep_first" | "keep_second" | "merge" | "distinct",
  "reason": "brief explanation"
}

- "distinct": files cover different topics, keep both
- "keep_first": files overlap, the first is more complete — delete the second
- "keep_second": files overlap, the second is more complete — delete the first
- "merge": files have complementary information — combine into one

Respond with ONLY valid JSON, no other text.`

type deduplicationResult struct {
	Verdict string `json:"verdict"`
	Reason  string `json:"reason"`
}

func (c *Consolidator) deduplicateScopeInBatch(ctx context.Context, b brain.Batch, prefix brain.Path) (int, []string) {
	if c.provider == nil {
		return 0, []string{"deduplication skipped: no LLM provider"}
	}
	scope := "global"
	if strings.HasPrefix(string(prefix), "memory/project/") {
		scope = "project"
	}
	topics, err := listTopicsInBatch(ctx, b, prefix, scope)
	if err != nil || len(topics) < 2 {
		return 0, nil
	}

	var merged int
	var errs []string

	for i := 0; i < len(topics); i++ {
		for j := i + 1; j < len(topics); j++ {
			wordsA := significantWords(baseName(string(topics[i].Path)))
			wordsB := significantWords(baseName(string(topics[j].Path)))

			if jaccardSimilarity(wordsA, wordsB) < deduplicationJaccardCutoff {
				continue
			}

			contentA, errA := b.Read(ctx, topics[i].Path)
			contentB, errB := b.Read(ctx, topics[j].Path)
			if errA != nil || errB != nil {
				continue
			}

			prompt := fmt.Sprintf("## File 1: %s\n\n%s\n\n---\n\n## File 2: %s\n\n%s",
				baseName(string(topics[i].Path)), string(contentA),
				baseName(string(topics[j].Path)), string(contentB))

			resp, llmErr := c.provider.Complete(ctx, llm.CompleteRequest{
				Model: c.model,
				Messages: []llm.Message{
					{Role: RoleSystem, Content: deduplicationSystemPrompt},
					{Role: RoleUser, Content: prompt},
				},
				MaxTokens:   deduplicationMaxTokens,
				Temperature: deduplicationTemperature,
			})
			if llmErr != nil {
				errs = append(errs, fmt.Sprintf("dedup LLM call: %s", llmErr))
				continue
			}

			verdict := parseDeduplicationResult(resp.Text)

			switch verdict.Verdict {
			case "keep_first":
				if err := b.Delete(ctx, topics[j].Path); err != nil {
					errs = append(errs, fmt.Sprintf("removing %s: %s", topics[j].Path, err))
				} else {
					merged++
				}
			case "keep_second":
				if err := b.Delete(ctx, topics[i].Path); err != nil {
					errs = append(errs, fmt.Sprintf("removing %s: %s", topics[i].Path, err))
				} else {
					merged++
				}
			case "merge":
				if mergeErr := c.mergeTopicsInBatch(ctx, b, topics[i], topics[j]); mergeErr != nil {
					errs = append(errs, fmt.Sprintf("merging %s + %s: %s",
						baseName(string(topics[i].Path)), baseName(string(topics[j].Path)), mergeErr))
				} else {
					merged++
				}
			}
		}
	}

	return merged, errs
}

// mergeTopicsInBatch combines two topics into the more recently
// modified one.
func (c *Consolidator) mergeTopicsInBatch(ctx context.Context, b brain.Batch, a, topicB TopicFile) error {
	modA := c.modifiedTime(ctx, a.Path)
	modBt := c.modifiedTime(ctx, topicB.Path)

	keeper, donor := a, topicB
	if modBt.After(modA) {
		keeper, donor = topicB, a
	}

	keeperData, err := b.Read(ctx, keeper.Path)
	if err != nil {
		return err
	}
	donorData, err := b.Read(ctx, donor.Path)
	if err != nil {
		return err
	}

	_, donorBody := ParseFrontmatter(string(donorData))

	combined := strings.TrimSpace(string(keeperData)) + "\n\n---\n\n" +
		fmt.Sprintf("*Merged from %s:*\n\n", baseName(string(donor.Path))) +
		strings.TrimSpace(donorBody) + "\n"

	if err := b.Write(ctx, keeper.Path, []byte(combined)); err != nil {
		return err
	}
	return b.Delete(ctx, donor.Path)
}

func parseDeduplicationResult(content string) deduplicationResult {
	content = strings.TrimSpace(content)

	if idx := strings.Index(content, "{"); idx >= 0 {
		if end := strings.LastIndex(content, "}"); end > idx {
			content = content[idx : end+1]
		}
	}

	var result deduplicationResult
	if err := json.Unmarshal([]byte(content), &result); err != nil {
		return deduplicationResult{Verdict: "distinct"}
	}

	return result
}

// reinforceHeuristicsInBatch recalculates confidence levels for
// heuristic files under the given logical prefix.
func (c *Consolidator) reinforceHeuristicsInBatch(ctx context.Context, b brain.Batch, prefix brain.Path) (int, []string) {
	entries, err := b.List(ctx, prefix, brain.ListOpts{IncludeGenerated: true})
	if err != nil {
		if errors.Is(err, brain.ErrNotFound) {
			return 0, nil
		}
		return 0, []string{fmt.Sprintf("reading prefix for heuristics: %s: %s", prefix, err)}
	}

	var updated int
	var errs []string

	for _, entry := range entries {
		if entry.IsDir {
			continue
		}
		name := baseName(string(entry.Path))
		if !strings.HasSuffix(name, ".md") || strings.EqualFold(name, "MEMORY.md") {
			continue
		}

		data, readErr := b.Read(ctx, entry.Path)
		if readErr != nil {
			continue
		}

		fm, body := ParseFrontmatter(string(data))
		if !hasTag(fm.Tags, "heuristic") {
			continue
		}

		count := countSections(body)
		newConfidence := confidenceFromObservations(count)
		if newConfidence == fm.Confidence {
			continue
		}

		rebuiltContent := rebuildWithUpdatedConfidence(fm, body, newConfidence)
		if writeErr := b.Write(ctx, entry.Path, []byte(rebuiltContent)); writeErr != nil {
			errs = append(errs, fmt.Sprintf("updating heuristic %s: %s", name, writeErr))
			continue
		}
		updated++
	}

	return updated, errs
}

// rebuildWithUpdatedConfidence reconstructs a memory file with new
// confidence.
func rebuildWithUpdatedConfidence(fm Frontmatter, body, newConfidence string) string {
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
	if fm.Modified != "" {
		b.WriteString(fmt.Sprintf("modified: %s\n", fm.Modified))
	}
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
	b.WriteString("\n")

	return b.String()
}

// listTopicsInBatch is the batch-aware analogue of listTopicsUnder.
func listTopicsInBatch(ctx context.Context, b brain.Batch, prefix brain.Path, scope string) ([]TopicFile, error) {
	entries, err := b.List(ctx, prefix, brain.ListOpts{IncludeGenerated: true})
	if err != nil {
		if errors.Is(err, brain.ErrNotFound) {
			return nil, nil
		}
		return nil, err
	}

	var topics []TopicFile
	for _, entry := range entries {
		if entry.IsDir {
			continue
		}
		name := baseName(string(entry.Path))
		if !strings.HasSuffix(name, ".md") || strings.EqualFold(name, "MEMORY.md") {
			continue
		}

		data, readErr := b.Read(ctx, entry.Path)
		if readErr != nil {
			continue
		}

		fm, _ := ParseFrontmatter(string(data))
		topicName := fm.Name
		if topicName == "" {
			topicName = strings.TrimSuffix(name, ".md")
		}

		topics = append(topics, TopicFile{
			Name:        topicName,
			Description: fm.Description,
			Type:        fm.Type,
			Path:        entry.Path,
			Created:     fm.Created,
			Modified:    fm.Modified,
			Tags:        fm.Tags,
			Confidence:  fm.Confidence,
			Source:      fm.Source,
			Scope:       scope,
		})
	}

	return topics, nil
}
