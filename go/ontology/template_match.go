// SPDX-License-Identifier: Apache-2.0
package ontology

import (
	"context"
	"fmt"
	"sort"

	"github.com/jeffs-brain/memory/go/llm"
)

// Template matching constants ported from the intelligence service.
const (
	// TemplateMatchSemanticThreshold is the minimum cosine similarity for
	// a semantic embedding pair to count as a match during bipartite
	// greedy matching.
	TemplateMatchSemanticThreshold = 0.8

	// TemplateMatchCombinedMinimum is the minimum combined score
	// (overlap + coverage) / 2 required for a template to qualify as a
	// suggestion.
	TemplateMatchCombinedMinimum = 0.3
)

// TemplateSuggestion is the result of template matching.
type TemplateSuggestion struct {
	TemplateKey         string      `json:"templateKey"`
	TemplateLabel       string      `json:"templateLabel"`
	OverlapScore        float64     `json:"overlapScore"`
	CoverageScore       float64     `json:"coverageScore"`
	AdditionalTypes     []TypeEntry `json:"additionalTypes"`
	MissingFromTemplate []TypeEntry `json:"missingFromTemplate"`
}

// TemplateMatcherConfig configures the TemplateMatcher.
type TemplateMatcherConfig struct {
	// Embedder provides semantic embeddings. May be nil for exact-only matching.
	Embedder llm.Embedder

	// SemanticThreshold overrides TemplateMatchSemanticThreshold.
	// Zero uses the default (0.8).
	SemanticThreshold float64

	// CombinedMinimum overrides TemplateMatchCombinedMinimum.
	// Zero uses the default (0.3).
	CombinedMinimum float64
}

// TemplateMatcher suggests industry templates based on extracted types.
type TemplateMatcher struct {
	embedder          llm.Embedder
	semanticThreshold float64
	combinedMinimum   float64
	embeddingCache    map[string][][]float32
}

// NewTemplateMatcher creates a matcher. Embedder may be nil for exact-only matching.
func NewTemplateMatcher(embedder llm.Embedder) *TemplateMatcher {
	return NewTemplateMatcherWithConfig(TemplateMatcherConfig{Embedder: embedder})
}

// NewTemplateMatcherWithConfig creates a TemplateMatcher with configurable options.
func NewTemplateMatcherWithConfig(cfg TemplateMatcherConfig) *TemplateMatcher {
	semThresh := cfg.SemanticThreshold
	if semThresh == 0 {
		semThresh = TemplateMatchSemanticThreshold
	}
	combMin := cfg.CombinedMinimum
	if combMin == 0 {
		combMin = TemplateMatchCombinedMinimum
	}
	return &TemplateMatcher{
		embedder:          cfg.Embedder,
		semanticThreshold: semThresh,
		combinedMinimum:   combMin,
		embeddingCache:    make(map[string][][]float32),
	}
}

// Match finds the best-matching industry template for the given extraction.
// Uses semantic matching when an embedder is available, falling back to exact.
// Returns nil if no template scores above the combined threshold.
func (m *TemplateMatcher) Match(ctx context.Context, extracted ExtractionResult) (*TemplateSuggestion, error) {
	if len(extracted.NodeTypes) == 0 && len(extracted.EdgeTypes) == 0 {
		return nil, nil
	}

	if m.embedder == nil {
		return m.MatchExact(extracted), nil
	}

	return m.matchSemantic(ctx, extracted)
}

// MatchExact performs set-intersection-only matching (no embeddings).
// Returns nil if no template scores above the combined threshold.
func (m *TemplateMatcher) MatchExact(extracted ExtractionResult) *TemplateSuggestion {
	if len(extracted.NodeTypes) == 0 && len(extracted.EdgeTypes) == 0 {
		return nil
	}

	extractedNodeSet := make(map[string]TypeEntry, len(extracted.NodeTypes))
	for _, nt := range extracted.NodeTypes {
		extractedNodeSet[nt.Type] = nt
	}
	extractedEdgeSet := make(map[string]TypeEntry, len(extracted.EdgeTypes))
	for _, et := range extracted.EdgeTypes {
		extractedEdgeSet[et.Type] = et
	}

	extractedCount := len(extracted.NodeTypes) + len(extracted.EdgeTypes)

	var best *TemplateSuggestion
	var bestCombined float64

	keys := ListTemplates()
	for _, key := range keys {
		tmpl, ok := GetTemplate(key)
		if !ok {
			continue
		}

		templateNodeSet := make(map[string]TypeEntry, len(tmpl.NodeTypes))
		for _, nt := range tmpl.NodeTypes {
			templateNodeSet[nt.Type] = nt
		}
		templateEdgeSet := make(map[string]TypeEntry, len(tmpl.EdgeTypes))
		for _, et := range tmpl.EdgeTypes {
			templateEdgeSet[et.Type] = et
		}

		templateCount := len(tmpl.NodeTypes) + len(tmpl.EdgeTypes)
		if templateCount == 0 {
			continue
		}

		intersectionCount := 0
		for typeID := range extractedNodeSet {
			if _, ok := templateNodeSet[typeID]; ok {
				intersectionCount++
			}
		}
		for typeID := range extractedEdgeSet {
			if _, ok := templateEdgeSet[typeID]; ok {
				intersectionCount++
			}
		}

		overlapScore := 0.0
		if extractedCount > 0 {
			overlapScore = float64(intersectionCount) / float64(extractedCount)
		}
		coverageScore := float64(intersectionCount) / float64(templateCount)
		combined := (overlapScore + coverageScore) / 2

		if combined >= m.combinedMinimum && combined > bestCombined {
			additional := computeAdditionalTypes(extracted, templateNodeSet, templateEdgeSet)
			missing := computeMissingTypes(tmpl, extractedNodeSet, extractedEdgeSet)

			bestCombined = combined
			best = &TemplateSuggestion{
				TemplateKey:         key,
				TemplateLabel:       tmpl.Label,
				OverlapScore:        overlapScore,
				CoverageScore:       coverageScore,
				AdditionalTypes:     additional,
				MissingFromTemplate: missing,
			}
		}
	}

	return best
}

func (m *TemplateMatcher) matchSemantic(ctx context.Context, extracted ExtractionResult) (*TemplateSuggestion, error) {
	extractedTypes := combineTypes(extracted.NodeTypes, extracted.EdgeTypes)
	if len(extractedTypes) == 0 {
		return nil, nil
	}

	extractedTexts := make([]string, len(extractedTypes))
	for i, t := range extractedTypes {
		extractedTexts[i] = t.Label + ": " + t.Description
	}

	extractedEmbeddings, err := m.embedder.Embed(ctx, extractedTexts)
	if err != nil {
		return nil, fmt.Errorf("ontology: embed extracted types: %w", err)
	}

	var best *TemplateSuggestion
	var bestCombined float64

	keys := ListTemplates()
	for _, key := range keys {
		tmpl, ok := GetTemplate(key)
		if !ok {
			continue
		}

		templateTypes := combineTypes(tmpl.NodeTypes, tmpl.EdgeTypes)
		if len(templateTypes) == 0 {
			continue
		}

		var templateEmbeddings [][]float32
		if cached, ok := m.embeddingCache[key]; ok {
			templateEmbeddings = cached
		} else {
			templateTexts := make([]string, len(templateTypes))
			for i, t := range templateTypes {
				templateTexts[i] = t.Label + ": " + t.Description
			}

			var embedErr error
			templateEmbeddings, embedErr = m.embedder.Embed(ctx, templateTexts)
			if embedErr != nil {
				return nil, fmt.Errorf("ontology: embed template %q types: %w", key, embedErr)
			}
			m.embeddingCache[key] = templateEmbeddings
		}

		matchCount := greedyBipartiteMatch(extractedEmbeddings, templateEmbeddings, m.semanticThreshold)

		extractedCount := len(extractedTypes)
		templateCount := len(templateTypes)

		overlapScore := 0.0
		if extractedCount > 0 {
			overlapScore = float64(matchCount) / float64(extractedCount)
		}
		coverageScore := 0.0
		if templateCount > 0 {
			coverageScore = float64(matchCount) / float64(templateCount)
		}
		combined := (overlapScore + coverageScore) / 2

		if combined >= m.combinedMinimum && combined > bestCombined {
			extractedNodeSet := make(map[string]TypeEntry, len(extracted.NodeTypes))
			for _, nt := range extracted.NodeTypes {
				extractedNodeSet[nt.Type] = nt
			}
			extractedEdgeSet := make(map[string]TypeEntry, len(extracted.EdgeTypes))
			for _, et := range extracted.EdgeTypes {
				extractedEdgeSet[et.Type] = et
			}
			templateNodeSet := make(map[string]TypeEntry, len(tmpl.NodeTypes))
			for _, nt := range tmpl.NodeTypes {
				templateNodeSet[nt.Type] = nt
			}
			templateEdgeSet := make(map[string]TypeEntry, len(tmpl.EdgeTypes))
			for _, et := range tmpl.EdgeTypes {
				templateEdgeSet[et.Type] = et
			}

			additional := computeAdditionalTypes(extracted, templateNodeSet, templateEdgeSet)
			missing := computeMissingTypes(tmpl, extractedNodeSet, extractedEdgeSet)

			bestCombined = combined
			best = &TemplateSuggestion{
				TemplateKey:         key,
				TemplateLabel:       tmpl.Label,
				OverlapScore:        overlapScore,
				CoverageScore:       coverageScore,
				AdditionalTypes:     additional,
				MissingFromTemplate: missing,
			}
		}
	}

	return best, nil
}

// greedyBipartiteMatch counts the number of matched pairs between
// extracted and template embeddings using greedy best-first matching.
// Each extracted embedding is matched to the most-similar template
// embedding with cosine similarity >= threshold. Once matched, a
// template embedding is removed from the candidate pool.
func greedyBipartiteMatch(extractedVecs, templateVecs [][]float32, threshold float64) int {
	if len(extractedVecs) == 0 || len(templateVecs) == 0 {
		return 0
	}

	type scoredPair struct {
		extractedIdx int
		templateIdx  int
		similarity   float64
	}

	var pairs []scoredPair
	for i, ev := range extractedVecs {
		for j, tv := range templateVecs {
			sim, err := CosineSimilarity(ev, tv)
			if err != nil || sim < threshold {
				continue
			}
			pairs = append(pairs, scoredPair{i, j, sim})
		}
	}

	sort.Slice(pairs, func(a, b int) bool {
		return pairs[a].similarity > pairs[b].similarity
	})

	usedExtracted := make(map[int]bool)
	usedTemplate := make(map[int]bool)
	matchCount := 0

	for _, p := range pairs {
		if usedExtracted[p.extractedIdx] || usedTemplate[p.templateIdx] {
			continue
		}
		usedExtracted[p.extractedIdx] = true
		usedTemplate[p.templateIdx] = true
		matchCount++
	}

	return matchCount
}

// computeAdditionalTypes returns types in extracted that are not in the template.
func computeAdditionalTypes(extracted ExtractionResult, templateNodeSet, templateEdgeSet map[string]TypeEntry) []TypeEntry {
	var additional []TypeEntry
	for _, nt := range extracted.NodeTypes {
		if _, ok := templateNodeSet[nt.Type]; !ok {
			additional = append(additional, nt)
		}
	}
	for _, et := range extracted.EdgeTypes {
		if _, ok := templateEdgeSet[et.Type]; !ok {
			additional = append(additional, et)
		}
	}
	if additional == nil {
		return []TypeEntry{}
	}
	return additional
}

// computeMissingTypes returns types in the template that are not in extracted.
func computeMissingTypes(tmpl IndustryTemplate, extractedNodeSet, extractedEdgeSet map[string]TypeEntry) []TypeEntry {
	var missing []TypeEntry
	for _, nt := range tmpl.NodeTypes {
		if _, ok := extractedNodeSet[nt.Type]; !ok {
			missing = append(missing, nt)
		}
	}
	for _, et := range tmpl.EdgeTypes {
		if _, ok := extractedEdgeSet[et.Type]; !ok {
			missing = append(missing, et)
		}
	}
	if missing == nil {
		return []TypeEntry{}
	}
	return missing
}

// combineTypes merges node and edge type entries into a single slice.
func combineTypes(nodeTypes, edgeTypes []TypeEntry) []TypeEntry {
	combined := make([]TypeEntry, 0, len(nodeTypes)+len(edgeTypes))
	combined = append(combined, nodeTypes...)
	combined = append(combined, edgeTypes...)
	return combined
}
