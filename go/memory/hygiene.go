// SPDX-License-Identifier: Apache-2.0

package memory

import (
	"context"
	"errors"
	"fmt"
	"path"
	"sort"
	"strings"
	"time"

	"github.com/jeffs-brain/memory/go/brain"
)

// DefaultRetiredAgeDays is the default age before a superseded memory
// file is stamped with retired: true by the hygiene pass.
const DefaultRetiredAgeDays = 30

// HygieneOptions tunes a single hygiene pass.
type HygieneOptions struct {
	// RetiredAgeDays controls when superseded entries are soft-retired.
	// Zero or negative means use DefaultRetiredAgeDays.
	RetiredAgeDays int

	// Apply controls whether the pass writes changes. When false the pass
	// populates the report but never mutates the store.
	Apply bool

	// Now pins time for tests and reproducible reports. Zero means time.Now.
	Now time.Time
}

// HygieneReport summarises what the contradiction and aging pass found.
type HygieneReport struct {
	Contradictions []ContradictionGroup
	AgingRetired   []AgingRetirement
	Errors         []string
}

// ContradictionGroup describes live memory files that cover the same
// topic. When Apply is set, every entry except Canonical is stamped with
// superseded_by pointing at Canonical.
type ContradictionGroup struct {
	Key       string
	KeyReason string
	Scope     string
	Project   string
	Members   []TopicFile
	Canonical brain.Path
}

// AgingRetirement describes a superseded file that should be soft-retired.
type AgingRetirement struct {
	Path brain.Path
	Age  time.Duration
}

// RunHygiene performs a contradiction and aging pass across every memory
// scope. Dry-run mode reports intended changes. Apply mode writes one
// batch per scope.
func (c *Consolidator) RunHygiene(ctx context.Context, opts HygieneOptions) (*HygieneReport, error) {
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

	report := &HygieneReport{}

	maxAgeDays := opts.RetiredAgeDays
	if maxAgeDays <= 0 {
		maxAgeDays = DefaultRetiredAgeDays
	}
	maxAge := time.Duration(maxAgeDays) * 24 * time.Hour
	now := opts.Now
	if now.IsZero() {
		now = time.Now()
	}

	prefixes := c.scopePrefixes(ctx)
	for _, prefix := range prefixes {
		c.detectContradictionsForPrefix(ctx, prefix, report)
		c.ageRetireSupersededForPrefix(ctx, prefix, now, maxAge, report)
	}

	if !opts.Apply {
		for i := range report.Contradictions {
			report.Contradictions[i].Canonical = ""
		}
		return report, nil
	}

	for _, prefix := range prefixes {
		c.applyHygieneForPrefix(ctx, prefix, report, now)
	}

	return report, nil
}

func (c *Consolidator) detectContradictionsForPrefix(ctx context.Context, prefix brain.Path, report *HygieneReport) {
	scope, project := hygieneScopeForPrefix(prefix)

	entries, err := c.mem.store.List(ctx, prefix, brain.ListOpts{IncludeGenerated: true})
	if err != nil {
		if !errors.Is(err, brain.ErrNotFound) {
			report.Errors = append(report.Errors, fmt.Sprintf("listing %s: %s", prefix, err))
		}
		return
	}

	groupsByClaim := map[string][]TopicFile{}
	groupsByName := map[string][]TopicFile{}

	for _, e := range entries {
		if e.IsDir {
			continue
		}
		base := path.Base(string(e.Path))
		if !strings.HasSuffix(base, ".md") || strings.EqualFold(base, "MEMORY.md") {
			continue
		}
		data, err := c.mem.store.Read(ctx, e.Path)
		if err != nil {
			continue
		}
		fm, _ := ParseFrontmatter(string(data))
		if fm.SupersededBy != "" || fm.Retired {
			continue
		}
		topic := TopicFile{
			Name:        fm.Name,
			Description: fm.Description,
			Type:        fm.Type,
			Path:        e.Path,
			Created:     fm.Created,
			Modified:    fm.Modified,
			Tags:        fm.Tags,
			Confidence:  fm.Confidence,
			Source:      fm.Source,
			Scope:       scope,
		}

		if fm.ClaimKey != "" {
			groupsByClaim[fm.ClaimKey] = append(groupsByClaim[fm.ClaimKey], topic)
		}
		if fm.Name != "" {
			key := strings.ToLower(strings.TrimSpace(fm.Name))
			groupsByName[key] = append(groupsByName[key], topic)
		}
	}

	emitContradictionGroups(report, groupsByClaim, "claim_key", scope, project)
	emitContradictionGroups(report, groupsByName, "name", scope, project)
}

func emitContradictionGroups(report *HygieneReport, groupsByKey map[string][]TopicFile, reason, scope, project string) {
	keys := make([]string, 0, len(groupsByKey))
	for k := range groupsByKey {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		members := groupsByKey[k]
		if len(members) < 2 {
			continue
		}
		seen := map[brain.Path]bool{}
		unique := make([]TopicFile, 0, len(members))
		for _, m := range members {
			if seen[m.Path] {
				continue
			}
			seen[m.Path] = true
			unique = append(unique, m)
		}
		if len(unique) < 2 {
			continue
		}
		report.Contradictions = append(report.Contradictions, ContradictionGroup{
			Key:       k,
			KeyReason: reason,
			Scope:     scope,
			Project:   project,
			Members:   unique,
		})
	}
}

func pickCanonical(members []TopicFile) TopicFile {
	if len(members) == 0 {
		return TopicFile{}
	}
	confidenceRank := func(s string) int {
		switch strings.ToLower(strings.TrimSpace(s)) {
		case "high":
			return 3
		case "medium":
			return 2
		case "low":
			return 1
		}
		return 0
	}
	sorted := make([]TopicFile, len(members))
	copy(sorted, members)
	sort.SliceStable(sorted, func(i, j int) bool {
		ri, rj := confidenceRank(sorted[i].Confidence), confidenceRank(sorted[j].Confidence)
		if ri != rj {
			return ri > rj
		}
		ti := parseHygieneModified(sorted[i].Modified)
		tj := parseHygieneModified(sorted[j].Modified)
		if !ti.Equal(tj) {
			return ti.After(tj)
		}
		return string(sorted[i].Path) < string(sorted[j].Path)
	})
	return sorted[0]
}

func parseHygieneModified(s string) time.Time {
	if s == "" {
		return time.Time{}
	}
	if t, err := time.Parse(time.RFC3339, s); err == nil {
		return t
	}
	return time.Time{}
}

func (c *Consolidator) ageRetireSupersededForPrefix(ctx context.Context, prefix brain.Path, now time.Time, maxAge time.Duration, report *HygieneReport) {
	threshold := now.Add(-maxAge)

	entries, err := c.mem.store.List(ctx, prefix, brain.ListOpts{IncludeGenerated: true})
	if err != nil {
		if !errors.Is(err, brain.ErrNotFound) {
			report.Errors = append(report.Errors, fmt.Sprintf("listing %s: %s", prefix, err))
		}
		return
	}
	for _, e := range entries {
		if e.IsDir {
			continue
		}
		base := path.Base(string(e.Path))
		if !strings.HasSuffix(base, ".md") || strings.EqualFold(base, "MEMORY.md") {
			continue
		}
		data, err := c.mem.store.Read(ctx, e.Path)
		if err != nil {
			continue
		}
		fm, _ := ParseFrontmatter(string(data))
		if fm.SupersededBy == "" || fm.Retired {
			continue
		}
		modified := parseHygieneModified(fm.Modified)
		if modified.IsZero() || !modified.Before(threshold) {
			continue
		}
		report.AgingRetired = append(report.AgingRetired, AgingRetirement{
			Path: e.Path,
			Age:  now.Sub(modified),
		})
	}
}

func (c *Consolidator) applyHygieneForPrefix(ctx context.Context, prefix brain.Path, report *HygieneReport, now time.Time) {
	scope, project := hygieneScopeForPrefix(prefix)

	var groupsHere []*ContradictionGroup
	for i := range report.Contradictions {
		g := &report.Contradictions[i]
		if g.Scope == scope && g.Project == project {
			groupsHere = append(groupsHere, g)
		}
	}
	var retireHere []*AgingRetirement
	for i := range report.AgingRetired {
		a := &report.AgingRetired[i]
		if strings.HasPrefix(string(a.Path), string(prefix)+"/") {
			retireHere = append(retireHere, a)
		}
	}
	if len(groupsHere) == 0 && len(retireHere) == 0 {
		return
	}

	err := c.mem.store.Batch(ctx, brain.BatchOptions{Reason: "memory:hygiene " + scope + hygieneProjectSuffix(project)}, func(b brain.Batch) error {
		for _, g := range groupsHere {
			canonical := pickCanonical(g.Members)
			g.Canonical = canonical.Path
			canonicalFile := path.Base(string(canonical.Path))
			for _, m := range g.Members {
				if m.Path == canonical.Path {
					continue
				}
				if _, err := stampSupersededBy(ctx, b, m.Path, canonicalFile); err != nil {
					report.Errors = append(report.Errors, fmt.Sprintf("stamp %s: %s", m.Path, err))
				}
			}
		}
		for _, a := range retireHere {
			existing, err := b.Read(ctx, a.Path)
			if err != nil {
				report.Errors = append(report.Errors, fmt.Sprintf("read %s: %s", a.Path, err))
				continue
			}
			reason := fmt.Sprintf("auto-retired by hygiene: superseded for %s", a.Age.Round(24*time.Hour))
			stamped := retireFrontmatterInPlace(existing, now, reason)
			if err := b.Write(ctx, a.Path, stamped); err != nil {
				report.Errors = append(report.Errors, fmt.Sprintf("retire %s: %s", a.Path, err))
			}
		}
		return nil
	})
	if err != nil {
		report.Errors = append(report.Errors, fmt.Sprintf("hygiene batch %s: %s", prefix, err))
	}
}

func retireFrontmatterInPlace(existing []byte, now time.Time, reason string) []byte {
	content := string(existing)
	lines := strings.Split(content, "\n")
	if len(lines) < 2 || strings.TrimSpace(lines[0]) != "---" {
		return existing
	}
	closeIdx := -1
	for i := 1; i < len(lines); i++ {
		if strings.TrimSpace(lines[i]) == "---" {
			closeIdx = i
			break
		}
	}
	if closeIdx < 0 {
		return existing
	}
	for i := 1; i < closeIdx; i++ {
		if strings.HasPrefix(strings.TrimSpace(lines[i]), "retired:") {
			return existing
		}
	}

	insert := []string{
		"retired: true",
		"retired_on: " + now.UTC().Format("2006-01-02"),
	}
	if reason != "" {
		insert = append(insert, "retired_reason: "+reason)
	}
	merged := make([]string, 0, len(lines)+len(insert))
	merged = append(merged, lines[:closeIdx]...)
	merged = append(merged, insert...)
	merged = append(merged, lines[closeIdx:]...)
	return []byte(strings.Join(merged, "\n"))
}

func hygieneScopeForPrefix(prefix brain.Path) (scope, project string) {
	s := string(prefix)
	if s == "memory/global" {
		return "global", ""
	}
	if strings.HasPrefix(s, "memory/project/") {
		return "project", strings.TrimPrefix(s, "memory/project/")
	}
	return "", ""
}

func hygieneProjectSuffix(project string) string {
	if project == "" {
		return ""
	}
	return "/" + project
}
