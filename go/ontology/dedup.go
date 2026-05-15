// SPDX-License-Identifier: Apache-2.0
package ontology

import (
	"context"
	"fmt"
	"strings"

	"github.com/jeffs-brain/memory/go/llm"
)

// Default thresholds matching the intelligence service configuration.
const (
	// FuzzyLabelThreshold is the default minimum Jaro-Winkler similarity
	// for fuzzy label matching within the same type prefix group.
	FuzzyLabelThreshold = 0.85

	// EmbeddingAutoMerge is the default minimum cosine similarity for
	// automatic semantic merging without human review.
	EmbeddingAutoMerge = 0.9

	// EmbeddingReviewThreshold is the default minimum cosine similarity
	// that flags a type pair for human review.
	EmbeddingReviewThreshold = 0.75
)

// SimilarityFunc is a pluggable string similarity function.
// It accepts two strings and returns a similarity value in [0, 1].
type SimilarityFunc func(a, b string) float64

// DedupMethod describes which deduplication tier matched.
type DedupMethod string

const (
	DedupMethodExact      DedupMethod = "exact"
	DedupMethodFuzzyLabel DedupMethod = "fuzzy_label"
	DedupMethodEmbedding  DedupMethod = "embedding"
)

// DedupResultKind is the discriminant for single-type dedup results.
type DedupResultKind string

const (
	DedupResultExact          DedupResultKind = "exact"
	DedupResultFuzzyMatch     DedupResultKind = "fuzzy_match"
	DedupResultSemanticMatch  DedupResultKind = "semantic_match"
	DedupResultSemanticReview DedupResultKind = "semantic_review"
	DedupResultNew            DedupResultKind = "new"
)

// DedupResult is the outcome of deduplicating a single candidate type
// against an existing registry.
type DedupResult struct {
	Kind         DedupResultKind
	ExistingType *TypeDefinition
	Similarity   float64
}

// MergedPair records a deduplication match between an extracted type
// and an existing type.
type MergedPair struct {
	Extracted     TypeDefinition `json:"extracted"`
	ExistingMatch TypeDefinition `json:"existingMatch"`
	Similarity    float64        `json:"similarity"`
	Method        DedupMethod    `json:"method"`
}

// DeduplicationResult holds the outcome of type deduplication across
// three tiers.
type DeduplicationResult struct {
	Unique           []TypeDefinition `json:"unique"`
	AutoMerged       []MergedPair     `json:"autoMerged"`
	ReviewCandidates []MergedPair     `json:"reviewCandidates"`
}

// DeduplicatorConfig configures the Deduplicator's thresholds and
// similarity algorithm.
type DeduplicatorConfig struct {
	// Embedder provides semantic embeddings. May be nil to skip the
	// semantic tier (exact + fuzzy only).
	Embedder llm.Embedder

	// FuzzyThreshold is the minimum string similarity for fuzzy label
	// matching within the same type prefix group. Default: 0.85.
	FuzzyThreshold float64

	// AutoMergeThreshold is the minimum cosine similarity for automatic
	// semantic merging. Default: 0.9.
	AutoMergeThreshold float64

	// ReviewThreshold is the minimum cosine similarity that flags a pair
	// for human review. Default: 0.75.
	ReviewThreshold float64

	// Similarity is the string similarity function used for fuzzy label
	// matching. Default: JaroWinklerDistance.
	Similarity SimilarityFunc
}

// Deduplicator performs three-tier type deduplication: exact match on
// type ID, fuzzy label match via configurable similarity function
// within the same prefix group, and semantic embedding match via
// cosine similarity.
type Deduplicator struct {
	embedder       llm.Embedder
	fuzzyThreshold float64
	autoMerge      float64
	reviewThreshold float64
	similarity     SimilarityFunc
}

// NewDeduplicator creates a deduplicator. Pass nil for embedder to
// skip the semantic tier and run exact + fuzzy only.
// Deprecated: Use NewDeduplicatorWithConfig for configurable thresholds.
func NewDeduplicator(embedder llm.Embedder) *Deduplicator {
	return NewDeduplicatorWithConfig(DeduplicatorConfig{Embedder: embedder})
}

// NewDeduplicatorWithConfig creates a Deduplicator with configurable
// thresholds and similarity algorithm. Zero-value fields use defaults.
func NewDeduplicatorWithConfig(cfg DeduplicatorConfig) *Deduplicator {
	fuzzy := cfg.FuzzyThreshold
	if fuzzy == 0 {
		fuzzy = FuzzyLabelThreshold
	}
	autoMerge := cfg.AutoMergeThreshold
	if autoMerge == 0 {
		autoMerge = EmbeddingAutoMerge
	}
	review := cfg.ReviewThreshold
	if review == 0 {
		review = EmbeddingReviewThreshold
	}
	sim := cfg.Similarity
	if sim == nil {
		sim = JaroWinklerDistance
	}
	return &Deduplicator{
		embedder:        cfg.Embedder,
		fuzzyThreshold:  fuzzy,
		autoMerge:       autoMerge,
		reviewThreshold: review,
		similarity:      sim,
	}
}

// Deduplicate compares extracted types against existing types using
// three tiers: exact ID match, fuzzy label match, and semantic
// embedding match.
//
// Time: O(E*X) where E = len(extracted), X = len(existing) for
// exact+fuzzy tiers; semantic tier adds embedding cost.
func (d *Deduplicator) Deduplicate(ctx context.Context, extracted, existing []TypeDefinition) (DeduplicationResult, error) {
	result := DeduplicationResult{
		Unique:           make([]TypeDefinition, 0),
		AutoMerged:       make([]MergedPair, 0),
		ReviewCandidates: make([]MergedPair, 0),
	}

	if len(extracted) == 0 {
		return result, nil
	}
	if len(existing) == 0 {
		result.Unique = append(result.Unique, extracted...)
		return result, nil
	}

	existingByID := buildTypeIDIndex(existing)
	existingByPrefix := groupByPrefix(existing)

	afterExact := d.deduplicateExact(extracted, existingByID, &result)
	afterFuzzy := d.deduplicateFuzzy(afterExact, existingByPrefix, &result)

	if d.embedder == nil || len(afterFuzzy) == 0 {
		result.Unique = append(result.Unique, afterFuzzy...)
		return result, nil
	}

	if err := d.deduplicateSemantic(ctx, afterFuzzy, existing, &result); err != nil {
		return DeduplicationResult{}, fmt.Errorf("ontology: semantic dedup: %w", err)
	}
	return result, nil
}

func (d *Deduplicator) deduplicateExact(
	extracted []TypeDefinition,
	existingByID map[string]TypeDefinition,
	result *DeduplicationResult,
) []TypeDefinition {
	remaining := make([]TypeDefinition, 0, len(extracted))
	for _, entry := range extracted {
		if match, found := existingByID[entry.Type]; found {
			result.AutoMerged = append(result.AutoMerged, MergedPair{
				Extracted:     entry,
				ExistingMatch: match,
				Similarity:    1.0,
				Method:        DedupMethodExact,
			})
			continue
		}
		remaining = append(remaining, entry)
	}
	return remaining
}

func (d *Deduplicator) deduplicateFuzzy(
	candidates []TypeDefinition,
	existingByPrefix map[string][]TypeDefinition,
	result *DeduplicationResult,
) []TypeDefinition {
	remaining := make([]TypeDefinition, 0, len(candidates))
	for _, entry := range candidates {
		prefix := typePrefix(entry.Type)
		samePrefix := existingByPrefix[prefix]
		if len(samePrefix) == 0 {
			remaining = append(remaining, entry)
			continue
		}

		matched := false
		for _, existing := range samePrefix {
			sim := d.similarity(entry.Label, existing.Label)
			if sim >= d.fuzzyThreshold {
				result.AutoMerged = append(result.AutoMerged, MergedPair{
					Extracted:     entry,
					ExistingMatch: existing,
					Similarity:    sim,
					Method:        DedupMethodFuzzyLabel,
				})
				matched = true
				break
			}
		}

		if !matched {
			remaining = append(remaining, entry)
		}
	}
	return remaining
}

func (d *Deduplicator) deduplicateSemantic(
	ctx context.Context,
	candidates []TypeDefinition,
	existing []TypeDefinition,
	result *DeduplicationResult,
) error {
	candidateTexts := make([]string, len(candidates))
	for i, c := range candidates {
		candidateTexts[i] = c.Label + ": " + c.Description
	}

	existingTexts := make([]string, len(existing))
	for i, e := range existing {
		existingTexts[i] = e.Label + ": " + e.Description
	}

	candidateEmbeddings, err := d.embedder.Embed(ctx, candidateTexts)
	if err != nil {
		return fmt.Errorf("embedding candidates: %w", err)
	}
	existingEmbeddings, err := d.embedder.Embed(ctx, existingTexts)
	if err != nil {
		return fmt.Errorf("embedding existing types: %w", err)
	}

	for i, candidate := range candidates {
		if i >= len(candidateEmbeddings) {
			result.Unique = append(result.Unique, candidate)
			continue
		}
		candidateVec := candidateEmbeddings[i]

		var bestSimilarity float64
		var bestMatch TypeDefinition
		hasBest := false

		for j, existingEntry := range existing {
			if j >= len(existingEmbeddings) {
				break
			}
			similarity, err := CosineSimilarity(candidateVec, existingEmbeddings[j])
			if err != nil {
				return fmt.Errorf("ontology: cosine similarity: %w", err)
			}
			if similarity > bestSimilarity {
				bestSimilarity = similarity
				bestMatch = existingEntry
				hasBest = true
			}
		}

		switch {
		case hasBest && bestSimilarity >= d.autoMerge:
			result.AutoMerged = append(result.AutoMerged, MergedPair{
				Extracted:     candidate,
				ExistingMatch: bestMatch,
				Similarity:    bestSimilarity,
				Method:        DedupMethodEmbedding,
			})
		case hasBest && bestSimilarity >= d.reviewThreshold:
			result.ReviewCandidates = append(result.ReviewCandidates, MergedPair{
				Extracted:     candidate,
				ExistingMatch: bestMatch,
				Similarity:    bestSimilarity,
				Method:        DedupMethodEmbedding,
			})
		default:
			result.Unique = append(result.Unique, candidate)
		}
	}
	return nil
}

// DeduplicateType evaluates a single candidate against a list of
// existing types using three tiers: exact ID match, fuzzy label match,
// and semantic embedding. Pass nil for embedder to skip the semantic tier.
func DeduplicateType(ctx context.Context, candidate TypeDefinition, existing []TypeDefinition, embedder llm.Embedder) (DedupResult, error) {
	if len(existing) == 0 {
		return DedupResult{Kind: DedupResultNew}, nil
	}

	for i := range existing {
		if candidate.Type == existing[i].Type {
			return DedupResult{Kind: DedupResultExact, ExistingType: &existing[i], Similarity: 1.0}, nil
		}
	}

	candidatePrefix := typePrefix(candidate.Type)
	for i := range existing {
		if typePrefix(existing[i].Type) != candidatePrefix {
			continue
		}
		similarity := JaroWinklerDistance(candidate.Label, existing[i].Label)
		if similarity >= FuzzyLabelThreshold {
			return DedupResult{Kind: DedupResultFuzzyMatch, ExistingType: &existing[i], Similarity: similarity}, nil
		}
	}

	if embedder == nil {
		return DedupResult{Kind: DedupResultNew}, nil
	}

	candidateText := candidate.Label + ": " + candidate.Description
	existingTexts := make([]string, len(existing))
	for i, e := range existing {
		existingTexts[i] = e.Label + ": " + e.Description
	}

	candidateEmbeddings, err := embedder.Embed(ctx, []string{candidateText})
	if err != nil {
		return DedupResult{}, fmt.Errorf("ontology: embed candidate: %w", err)
	}
	existingEmbeddings, err := embedder.Embed(ctx, existingTexts)
	if err != nil {
		return DedupResult{}, fmt.Errorf("ontology: embed existing: %w", err)
	}
	if len(candidateEmbeddings) == 0 {
		return DedupResult{Kind: DedupResultNew}, nil
	}

	candidateVec := candidateEmbeddings[0]
	var bestSimilarity float64
	var bestIdx int

	for j := range existingEmbeddings {
		sim, err := CosineSimilarity(candidateVec, existingEmbeddings[j])
		if err != nil {
			return DedupResult{}, fmt.Errorf("ontology: cosine similarity: %w", err)
		}
		if sim > bestSimilarity {
			bestSimilarity = sim
			bestIdx = j
		}
	}

	switch {
	case bestSimilarity >= EmbeddingAutoMerge:
		return DedupResult{Kind: DedupResultSemanticMatch, ExistingType: &existing[bestIdx], Similarity: bestSimilarity}, nil
	case bestSimilarity >= EmbeddingReviewThreshold:
		return DedupResult{Kind: DedupResultSemanticReview, ExistingType: &existing[bestIdx], Similarity: bestSimilarity}, nil
	default:
		return DedupResult{Kind: DedupResultNew}, nil
	}
}

func buildTypeIDIndex(types []TypeDefinition) map[string]TypeDefinition {
	index := make(map[string]TypeDefinition, len(types))
	for _, t := range types {
		index[t.Type] = t
	}
	return index
}

func groupByPrefix(types []TypeDefinition) map[string][]TypeDefinition {
	groups := make(map[string][]TypeDefinition)
	for _, t := range types {
		prefix := typePrefix(t.Type)
		groups[prefix] = append(groups[prefix], t)
	}
	return groups
}

func typePrefix(typeID string) string {
	idx := strings.IndexByte(typeID, '.')
	if idx < 0 {
		return typeID
	}
	return typeID[:idx]
}
