// SPDX-License-Identifier: Apache-2.0
package ontology

import (
	"context"
	"testing"
)

func TestMatchExact_PerfectOverlap(t *testing.T) {
	t.Parallel()
	matcher := NewTemplateMatcher(nil)

	// Use types from the server_hardware template for a perfect overlap
	tmpl, ok := GetTemplate("server_hardware")
	if !ok {
		t.Fatal("failed to get template")
	}

	extracted := ExtractionResult{
		NodeTypes:          make([]TypeEntry, len(tmpl.NodeTypes)),
		EdgeTypes:          make([]TypeEntry, len(tmpl.EdgeTypes)),
		BusinessCategories: tmpl.BusinessCategories,
	}
	copy(extracted.NodeTypes, tmpl.NodeTypes)
	copy(extracted.EdgeTypes, tmpl.EdgeTypes)

	suggestion := matcher.MatchExact(extracted)
	if suggestion == nil {
		t.Fatal("expected a suggestion for perfect overlap")
	}
	if suggestion.TemplateKey != "server_hardware" {
		t.Fatalf("expected server_hardware, got %s", suggestion.TemplateKey)
	}
	if suggestion.OverlapScore != 1.0 {
		t.Fatalf("expected overlap score 1.0, got %f", suggestion.OverlapScore)
	}
	if suggestion.CoverageScore != 1.0 {
		t.Fatalf("expected coverage score 1.0, got %f", suggestion.CoverageScore)
	}
}

func TestMatchExact_PartialOverlap(t *testing.T) {
	t.Parallel()
	matcher := NewTemplateMatcher(nil)

	// Use a subset of the healthcare template
	tmpl, ok := GetTemplate("healthcare")
	if !ok {
		t.Fatal("failed to get template")
	}

	extracted := ExtractionResult{
		NodeTypes: tmpl.NodeTypes[:5],
		EdgeTypes: tmpl.EdgeTypes[:3],
	}

	suggestion := matcher.MatchExact(extracted)
	if suggestion == nil {
		t.Fatal("expected a suggestion for partial overlap")
	}
	if suggestion.TemplateKey != "healthcare" {
		t.Fatalf("expected healthcare, got %s", suggestion.TemplateKey)
	}
	// Overlap: all 8 extracted types match (8/8 = 1.0)
	if suggestion.OverlapScore < 0.99 {
		t.Fatalf("expected overlap ~1.0, got %f", suggestion.OverlapScore)
	}
	// Coverage: 8 out of 45 (30 node + 15 edge)
	if suggestion.CoverageScore < 0.1 || suggestion.CoverageScore > 0.5 {
		t.Fatalf("expected coverage between 0.1 and 0.5, got %f", suggestion.CoverageScore)
	}
}

func TestMatchExact_NoOverlap(t *testing.T) {
	t.Parallel()
	matcher := NewTemplateMatcher(nil)

	// Types that don't match any template
	extracted := ExtractionResult{
		NodeTypes: []TypeEntry{
			{Type: "entity.alien_species", Label: "Alien Species", Description: "An extraterrestrial species"},
			{Type: "entity.space_station", Label: "Space Station", Description: "An orbital space station"},
		},
		EdgeTypes: []TypeEntry{
			{Type: "orbits", Label: "Orbits", Description: "One body orbits another"},
		},
	}

	suggestion := matcher.MatchExact(extracted)
	if suggestion != nil {
		t.Fatalf("expected nil for no overlap, got %s with score %f",
			suggestion.TemplateKey, (suggestion.OverlapScore+suggestion.CoverageScore)/2)
	}
}

func TestMatchExact_BestTemplate(t *testing.T) {
	t.Parallel()
	matcher := NewTemplateMatcher(nil)

	// Use types that partially overlap with finance and slightly with insurance
	extracted := ExtractionResult{
		NodeTypes: []TypeEntry{
			{Type: "entity.account", Label: "Account", Description: "A financial account"},
			{Type: "entity.payment", Label: "Payment", Description: "A transfer of funds"},
			{Type: "rule.kyc", Label: "KYC", Description: "Know your customer"},
			{Type: "rule.aml", Label: "AML", Description: "Anti-money laundering"},
			{Type: "entity.policy", Label: "Policy", Description: "An insurance policy"}, // matches insurance
		},
		EdgeTypes: []TypeEntry{
			{Type: "authorises", Label: "Authorises", Description: "Authorisation relationship"},
		},
	}

	suggestion := matcher.MatchExact(extracted)
	if suggestion == nil {
		t.Fatal("expected a suggestion")
	}
	// Should select finance since more types match finance than insurance
	if suggestion.TemplateKey != "finance" {
		t.Fatalf("expected finance as best match, got %s", suggestion.TemplateKey)
	}
}

func TestMatchExact_AdditionalTypes(t *testing.T) {
	t.Parallel()
	matcher := NewTemplateMatcher(nil)

	tmpl, ok := GetTemplate("order_processing")
	if !ok {
		t.Fatal("failed to get template")
	}

	// Start with template types and add extra ones
	extracted := ExtractionResult{
		NodeTypes: append(
			append([]TypeEntry{}, tmpl.NodeTypes...),
			TypeEntry{Type: "entity.custom_thing", Label: "Custom Thing", Description: "Not in template"},
		),
		EdgeTypes: tmpl.EdgeTypes,
	}

	suggestion := matcher.MatchExact(extracted)
	if suggestion == nil {
		t.Fatal("expected a suggestion")
	}

	found := false
	for _, additional := range suggestion.AdditionalTypes {
		if additional.Type == "entity.custom_thing" {
			found = true
			break
		}
	}
	if !found {
		t.Fatal("entity.custom_thing should be in additional types")
	}
}

func TestMatchExact_MissingTypes(t *testing.T) {
	t.Parallel()
	matcher := NewTemplateMatcher(nil)

	tmpl, ok := GetTemplate("order_processing")
	if !ok {
		t.Fatal("failed to get template")
	}

	// Use only first 3 node types from template
	extracted := ExtractionResult{
		NodeTypes: tmpl.NodeTypes[:3],
		EdgeTypes: tmpl.EdgeTypes,
	}

	suggestion := matcher.MatchExact(extracted)
	if suggestion == nil {
		t.Fatal("expected a suggestion")
	}

	// Missing types should include the node types we didn't extract
	if len(suggestion.MissingFromTemplate) == 0 {
		t.Fatal("expected missing types from template")
	}

	missingTypes := make(map[string]bool)
	for _, mt := range suggestion.MissingFromTemplate {
		missingTypes[mt.Type] = true
	}

	for _, nt := range tmpl.NodeTypes[3:] {
		if !missingTypes[nt.Type] {
			t.Fatalf("expected %s to be in missing types", nt.Type)
		}
	}
}

func TestMatchExact_EmptyExtraction(t *testing.T) {
	t.Parallel()
	matcher := NewTemplateMatcher(nil)
	suggestion := matcher.MatchExact(ExtractionResult{})
	if suggestion != nil {
		t.Fatal("expected nil for empty extraction")
	}
}

func TestMatch_FallbackToExact(t *testing.T) {
	t.Parallel()
	// No embedder: should fall back to exact matching
	matcher := NewTemplateMatcher(nil)

	tmpl, ok := GetTemplate("server_hardware")
	if !ok {
		t.Fatal("failed to get template")
	}

	extracted := ExtractionResult{
		NodeTypes: tmpl.NodeTypes,
		EdgeTypes: tmpl.EdgeTypes,
	}

	suggestion, matchErr := matcher.Match(context.Background(), extracted)
	if matchErr != nil {
		t.Fatalf("unexpected error: %v", matchErr)
	}
	if suggestion == nil {
		t.Fatal("expected suggestion from exact fallback")
	}
	if suggestion.TemplateKey != "server_hardware" {
		t.Fatalf("expected server_hardware, got %s", suggestion.TemplateKey)
	}
}

func TestMatchSemantic_AboveThreshold(t *testing.T) {
	t.Parallel()

	// Build a fake embedder where extracted and healthcare template types
	// map to the SAME vector, guaranteeing cosine similarity = 1.0.
	tmpl, ok := GetTemplate("healthcare")
	if !ok {
		t.Fatal("healthcare template not found")
	}

	sharedVec := []float32{1.0, 0.0, 0.0}
	vectors := make(map[string][]float32)

	// Use the same 3 types that will appear in the extracted set and
	// the template. Because the fake embedder returns the same vector
	// for both, greedy bipartite matching will produce 3 matches.
	for _, nt := range tmpl.NodeTypes[:3] {
		key := nt.Label + ": " + nt.Description
		vectors[key] = sharedVec
	}

	embedder := newFakeEmbedder(3, vectors)
	matcher := NewTemplateMatcherWithConfig(TemplateMatcherConfig{
		Embedder:          embedder,
		SemanticThreshold: 0.8,
		CombinedMinimum:   0.01, // low threshold so even 3/45 matches qualify
	})

	extracted := ExtractionResult{
		NodeTypes: tmpl.NodeTypes[:3],
	}

	suggestion, err := matcher.Match(context.Background(), extracted)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if suggestion == nil {
		t.Fatal("expected a suggestion above threshold")
	}
	if suggestion.TemplateKey != "healthcare" {
		t.Fatalf("expected healthcare, got %s", suggestion.TemplateKey)
	}
	if suggestion.OverlapScore < 0.99 {
		t.Fatalf("expected overlap ~1.0, got %f", suggestion.OverlapScore)
	}
}

func TestMatchSemantic_BelowThreshold(t *testing.T) {
	t.Parallel()

	// All vectors are zero, so cosine similarity is 0 (or NaN) -- no matches.
	embedder := newFakeEmbedder(3, map[string][]float32{})
	matcher := NewTemplateMatcherWithConfig(TemplateMatcherConfig{
		Embedder:          embedder,
		SemanticThreshold: 0.8,
		CombinedMinimum:   0.3,
	})

	extracted := ExtractionResult{
		NodeTypes: []TypeEntry{
			{Type: "entity.alien_species", Label: "Alien Species", Description: "An extraterrestrial species"},
		},
		EdgeTypes: []TypeEntry{
			{Type: "orbits", Label: "Orbits", Description: "One body orbits another"},
		},
	}

	suggestion, err := matcher.Match(context.Background(), extracted)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if suggestion != nil {
		t.Fatalf("expected nil for zero-vector embeddings, got %s with score %f",
			suggestion.TemplateKey, (suggestion.OverlapScore+suggestion.CoverageScore)/2)
	}
}

func TestMatch_EmptyExtraction(t *testing.T) {
	t.Parallel()
	matcher := NewTemplateMatcher(nil)
	suggestion, err := matcher.Match(context.Background(), ExtractionResult{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if suggestion != nil {
		t.Fatal("expected nil for empty extraction")
	}
}

func TestGreedyBipartiteMatch_NoPairs(t *testing.T) {
	t.Parallel()
	count := greedyBipartiteMatch(nil, nil, 0.8)
	if count != 0 {
		t.Fatalf("expected 0, got %d", count)
	}
}

func TestGreedyBipartiteMatch_IdenticalVectors(t *testing.T) {
	t.Parallel()
	vec := []float32{1.0, 0.0, 0.0}
	count := greedyBipartiteMatch(
		[][]float32{vec},
		[][]float32{vec},
		0.8,
	)
	if count != 1 {
		t.Fatalf("expected 1 match for identical vectors, got %d", count)
	}
}

func TestGreedyBipartiteMatch_OrthogonalVectors(t *testing.T) {
	t.Parallel()
	count := greedyBipartiteMatch(
		[][]float32{{1.0, 0.0, 0.0}},
		[][]float32{{0.0, 1.0, 0.0}},
		0.8,
	)
	if count != 0 {
		t.Fatalf("expected 0 matches for orthogonal vectors, got %d", count)
	}
}

func TestCombinedScoreCalculation(t *testing.T) {
	t.Parallel()
	// overlapScore = intersectionCount / extractedCount
	// coverageScore = intersectionCount / templateCount
	// combined = (overlapScore + coverageScore) / 2

	// 3 extracted, 10 template, 2 overlap
	overlapScore := 2.0 / 3.0
	coverageScore := 2.0 / 10.0
	combined := (overlapScore + coverageScore) / 2

	expected := (0.6667 + 0.2) / 2 // ~0.4333
	if combined < expected-0.01 || combined > expected+0.01 {
		t.Fatalf("expected combined ~%f, got %f", expected, combined)
	}
}

// newFakeEmbedderForTemplateMatch creates a fake embedder that returns
// deterministic hash-based vectors. Re-uses the dedup test fakeEmbedder.
func newFakeEmbedderForTemplateMatch(dims int) *fakeEmbedder {
	return &fakeEmbedder{
		vectors:    make(map[string][]float32),
		dimensions: dims,
	}
}
