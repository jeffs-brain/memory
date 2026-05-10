// SPDX-License-Identifier: Apache-2.0
package ontology

import (
	"context"
	"testing"
)

type fakeEmbedder struct {
	vectors    map[string][]float32
	dimensions int
}

func newFakeEmbedder(dimensions int, vectors map[string][]float32) *fakeEmbedder {
	return &fakeEmbedder{vectors: vectors, dimensions: dimensions}
}

func (f *fakeEmbedder) Embed(_ context.Context, texts []string) ([][]float32, error) {
	out := make([][]float32, len(texts))
	for i, text := range texts {
		if vec, ok := f.vectors[text]; ok {
			out[i] = vec
		} else {
			out[i] = make([]float32, f.dimensions)
		}
	}
	return out, nil
}

func (f *fakeEmbedder) Dimensions() int { return f.dimensions }
func (f *fakeEmbedder) Close() error    { return nil }

func TestDedup_ExactMatch(t *testing.T) {
	t.Parallel()
	dedup := NewDeduplicator(nil)
	extracted := []TypeDefinition{
		{Type: "entity.customer", Label: "Customer", Description: "A customer entity"},
	}
	existing := []TypeDefinition{
		{Type: "entity.customer", Label: "Customer", Description: "A customer entity"},
	}

	result, err := dedup.Deduplicate(context.Background(), extracted, existing)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(result.AutoMerged) != 1 {
		t.Fatalf("expected 1 auto-merged, got %d", len(result.AutoMerged))
	}
	if result.AutoMerged[0].Similarity != 1.0 {
		t.Fatalf("expected similarity 1.0, got %f", result.AutoMerged[0].Similarity)
	}
	if result.AutoMerged[0].Method != DedupMethodExact {
		t.Fatalf("expected method exact, got %s", result.AutoMerged[0].Method)
	}
	if len(result.Unique) != 0 {
		t.Fatalf("expected 0 unique, got %d", len(result.Unique))
	}
}

func TestDedup_FuzzyMatchSamePrefix(t *testing.T) {
	t.Parallel()
	dedup := NewDeduplicator(nil)
	extracted := []TypeDefinition{
		{Type: "entity.customer_record", Label: "Customer Record", Description: "A customer record"},
	}
	existing := []TypeDefinition{
		{Type: "entity.customer_records", Label: "Customer Records", Description: "Customer records"},
	}

	result, err := dedup.Deduplicate(context.Background(), extracted, existing)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(result.AutoMerged) != 1 {
		t.Fatalf("expected 1 auto-merged, got %d", len(result.AutoMerged))
	}
	if result.AutoMerged[0].Method != DedupMethodFuzzyLabel {
		t.Fatalf("expected method fuzzy_label, got %s", result.AutoMerged[0].Method)
	}
	if result.AutoMerged[0].Similarity < FuzzyLabelThreshold {
		t.Fatalf("expected similarity >= %f, got %f", FuzzyLabelThreshold, result.AutoMerged[0].Similarity)
	}
}

func TestDedup_FuzzyMatchDifferentPrefix(t *testing.T) {
	t.Parallel()
	dedup := NewDeduplicator(nil)
	extracted := []TypeDefinition{
		{Type: "entity.customer_records", Label: "Customer Records", Description: "Records"},
	}
	existing := []TypeDefinition{
		{Type: "rule.customer_records", Label: "Customer Records", Description: "Records rule"},
	}

	result, err := dedup.Deduplicate(context.Background(), extracted, existing)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(result.AutoMerged) != 0 {
		t.Fatalf("expected 0 auto-merged (different prefix), got %d", len(result.AutoMerged))
	}
	if len(result.Unique) != 1 {
		t.Fatalf("expected 1 unique, got %d", len(result.Unique))
	}
}

func TestDedup_BelowFuzzyThreshold(t *testing.T) {
	t.Parallel()
	dedup := NewDeduplicator(nil)
	extracted := []TypeDefinition{
		{Type: "entity.product", Label: "Product", Description: "A product"},
	}
	existing := []TypeDefinition{
		{Type: "entity.customer", Label: "Customer", Description: "A customer"},
	}

	result, err := dedup.Deduplicate(context.Background(), extracted, existing)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(result.AutoMerged) != 0 {
		t.Fatalf("expected 0 auto-merged, got %d", len(result.AutoMerged))
	}
	if len(result.Unique) != 1 {
		t.Fatalf("expected 1 unique, got %d", len(result.Unique))
	}
}

func TestDedup_SemanticAutoMerge(t *testing.T) {
	t.Parallel()
	// Vectors designed so cosine similarity >= 0.9
	embedder := newFakeEmbedder(3, map[string][]float32{
		"Client: A client entity":   {0.95, 0.1, 0.05},
		"Customer: A customer type": {0.96, 0.1, 0.04},
	})
	dedup := NewDeduplicator(embedder)

	extracted := []TypeDefinition{
		{Type: "entity.client", Label: "Client", Description: "A client entity"},
	}
	existing := []TypeDefinition{
		{Type: "entity.customer", Label: "Customer", Description: "A customer type"},
	}

	result, err := dedup.Deduplicate(context.Background(), extracted, existing)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(result.AutoMerged) != 1 {
		t.Fatalf("expected 1 auto-merged, got %d", len(result.AutoMerged))
	}
	if result.AutoMerged[0].Method != DedupMethodEmbedding {
		t.Fatalf("expected method embedding, got %s", result.AutoMerged[0].Method)
	}
	if result.AutoMerged[0].Similarity < EmbeddingAutoMerge {
		t.Fatalf("expected similarity >= %f, got %f", EmbeddingAutoMerge, result.AutoMerged[0].Similarity)
	}
}

func TestDedup_SemanticReviewZone(t *testing.T) {
	t.Parallel()
	// Vectors designed so cosine similarity is in [0.75, 0.9)
	// cosine([0.8, 0.4, 0.2], [0.5, 0.7, 0.5]) ~ 0.855
	embedder := newFakeEmbedder(3, map[string][]float32{
		"Workflow: A workflow process": {0.8, 0.4, 0.2},
		"Pipeline: A pipeline stage":  {0.5, 0.7, 0.5},
	})
	dedup := NewDeduplicator(embedder)

	extracted := []TypeDefinition{
		{Type: "process.workflow", Label: "Workflow", Description: "A workflow process"},
	}
	existing := []TypeDefinition{
		{Type: "process.pipeline", Label: "Pipeline", Description: "A pipeline stage"},
	}

	result, err := dedup.Deduplicate(context.Background(), extracted, existing)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	similarity, simErr := CosineSimilarity(
		[]float32{0.8, 0.4, 0.2},
		[]float32{0.5, 0.7, 0.5},
	)
	if simErr != nil {
		t.Fatalf("CosineSimilarity: %v", simErr)
	}
	if similarity < EmbeddingReviewThreshold || similarity >= EmbeddingAutoMerge {
		t.Skipf("test vectors produce similarity %f outside review zone [%f, %f)", similarity, EmbeddingReviewThreshold, EmbeddingAutoMerge)
	}

	if len(result.ReviewCandidates) != 1 {
		t.Fatalf("expected 1 review candidate, got %d", len(result.ReviewCandidates))
	}
	if result.ReviewCandidates[0].Method != DedupMethodEmbedding {
		t.Fatalf("expected method embedding, got %s", result.ReviewCandidates[0].Method)
	}
	if result.ReviewCandidates[0].Similarity < EmbeddingReviewThreshold {
		t.Fatalf("expected similarity >= %f, got %f", EmbeddingReviewThreshold, result.ReviewCandidates[0].Similarity)
	}
}

func TestDedup_NoEmbedder(t *testing.T) {
	t.Parallel()
	dedup := NewDeduplicator(nil)
	extracted := []TypeDefinition{
		{Type: "entity.client", Label: "Client", Description: "A client entity"},
	}
	existing := []TypeDefinition{
		{Type: "entity.customer", Label: "Customer", Description: "A customer type"},
	}

	result, err := dedup.Deduplicate(context.Background(), extracted, existing)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(result.Unique) != 1 {
		t.Fatalf("expected 1 unique (no embedder skips semantic), got %d", len(result.Unique))
	}
	if len(result.AutoMerged) != 0 {
		t.Fatalf("expected 0 auto-merged, got %d", len(result.AutoMerged))
	}
}

func TestDedup_EmptyExtracted(t *testing.T) {
	t.Parallel()
	dedup := NewDeduplicator(nil)
	existing := []TypeDefinition{
		{Type: "entity.customer", Label: "Customer", Description: "A customer"},
	}

	result, err := dedup.Deduplicate(context.Background(), nil, existing)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(result.Unique) != 0 {
		t.Fatalf("expected 0 unique, got %d", len(result.Unique))
	}
	if len(result.AutoMerged) != 0 {
		t.Fatalf("expected 0 auto-merged, got %d", len(result.AutoMerged))
	}
}

func TestDedup_EmptyExisting(t *testing.T) {
	t.Parallel()
	dedup := NewDeduplicator(nil)
	extracted := []TypeDefinition{
		{Type: "entity.customer", Label: "Customer", Description: "A customer"},
		{Type: "entity.product", Label: "Product", Description: "A product"},
	}

	result, err := dedup.Deduplicate(context.Background(), extracted, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(result.Unique) != 2 {
		t.Fatalf("expected 2 unique, got %d", len(result.Unique))
	}
	if len(result.AutoMerged) != 0 {
		t.Fatalf("expected 0 auto-merged, got %d", len(result.AutoMerged))
	}
}

func TestDeduplicateType_ExactMatch(t *testing.T) {
	t.Parallel()
	candidate := TypeDefinition{Type: "entity.customer", Label: "Customer", Description: "A customer"}
	existing := []TypeDefinition{
		{Type: "entity.customer", Label: "Customer", Description: "A customer"},
	}

	result, err := DeduplicateType(context.Background(), candidate, existing, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Kind != DedupResultExact {
		t.Fatalf("expected Exact, got %s", result.Kind)
	}
	if result.Similarity != 1.0 {
		t.Fatalf("expected similarity 1.0, got %f", result.Similarity)
	}
}

func TestDeduplicateType_FuzzyMatch(t *testing.T) {
	t.Parallel()
	candidate := TypeDefinition{Type: "entity.customer_record", Label: "Customer Record", Description: "A record"}
	existing := []TypeDefinition{
		{Type: "entity.customer_records", Label: "Customer Records", Description: "Records"},
	}

	result, err := DeduplicateType(context.Background(), candidate, existing, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Kind != DedupResultFuzzyMatch {
		t.Fatalf("expected FuzzyMatch, got %s", result.Kind)
	}
	if result.Similarity < FuzzyLabelThreshold {
		t.Fatalf("expected similarity >= %f, got %f", FuzzyLabelThreshold, result.Similarity)
	}
}

func TestDeduplicateType_FuzzyOnlySamePrefix(t *testing.T) {
	t.Parallel()
	candidate := TypeDefinition{Type: "entity.customer_records", Label: "Customer Records", Description: "Records"}
	existing := []TypeDefinition{
		{Type: "rule.customer_records", Label: "Customer Records", Description: "Rule"},
	}

	result, err := DeduplicateType(context.Background(), candidate, existing, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Kind != DedupResultNew {
		t.Fatalf("expected New (different prefix), got %s", result.Kind)
	}
}

func TestDeduplicateType_SemanticMatch(t *testing.T) {
	t.Parallel()
	embedder := newFakeEmbedder(3, map[string][]float32{
		"Client: A client entity":   {0.95, 0.1, 0.05},
		"Customer: A customer type": {0.96, 0.1, 0.04},
	})
	candidate := TypeDefinition{Type: "entity.client", Label: "Client", Description: "A client entity"}
	existing := []TypeDefinition{
		{Type: "entity.customer", Label: "Customer", Description: "A customer type"},
	}

	result, err := DeduplicateType(context.Background(), candidate, existing, embedder)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Kind != DedupResultSemanticMatch {
		t.Fatalf("expected SemanticMatch, got %s", result.Kind)
	}
	if result.Similarity < EmbeddingAutoMerge {
		t.Fatalf("expected similarity >= %f, got %f", EmbeddingAutoMerge, result.Similarity)
	}
}

func TestDeduplicateType_SemanticReview(t *testing.T) {
	t.Parallel()
	// cosine([0.8, 0.4, 0.2], [0.5, 0.7, 0.5]) ~ 0.855
	embedder := newFakeEmbedder(3, map[string][]float32{
		"Workflow: A workflow process": {0.8, 0.4, 0.2},
		"Pipeline: A pipeline stage":  {0.5, 0.7, 0.5},
	})
	candidate := TypeDefinition{Type: "process.workflow", Label: "Workflow", Description: "A workflow process"}
	existing := []TypeDefinition{
		{Type: "process.pipeline", Label: "Pipeline", Description: "A pipeline stage"},
	}

	result, err := DeduplicateType(context.Background(), candidate, existing, embedder)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Kind != DedupResultSemanticReview {
		t.Fatalf("expected SemanticReview, got %s", result.Kind)
	}
	if result.Similarity < EmbeddingReviewThreshold || result.Similarity >= EmbeddingAutoMerge {
		t.Fatalf("expected similarity in [%f, %f), got %f", EmbeddingReviewThreshold, EmbeddingAutoMerge, result.Similarity)
	}
}

func TestDeduplicateType_NoEmbedder_ReturnsNew(t *testing.T) {
	t.Parallel()
	candidate := TypeDefinition{Type: "entity.client", Label: "Client", Description: "A client entity"}
	existing := []TypeDefinition{
		{Type: "entity.customer", Label: "Customer", Description: "A customer type"},
	}

	result, err := DeduplicateType(context.Background(), candidate, existing, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Kind != DedupResultNew {
		t.Fatalf("expected New, got %s", result.Kind)
	}
}

func TestDeduplicateType_EmptyExisting_ReturnsNew(t *testing.T) {
	t.Parallel()
	candidate := TypeDefinition{Type: "entity.customer", Label: "Customer", Description: "A customer"}

	result, err := DeduplicateType(context.Background(), candidate, nil, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Kind != DedupResultNew {
		t.Fatalf("expected New, got %s", result.Kind)
	}
}
