// SPDX-License-Identifier: Apache-2.0
package ontology_test

import (
	"context"
	"testing"

	"github.com/jeffs-brain/memory/go/ontology"
)

func sampleResolvedOntology() *ontology.ResolvedOntology {
	return &ontology.ResolvedOntology{
		NodeTypes: []ontology.ResolvedType{
			{TypeDefinition: ontology.TypeDefinition{
				Type: "entity.customer", Label: "Customer (Entity)",
				Description: "A customer entity", CreatedAt: "2026-01-01T00:00:00Z", Status: "active",
			}, Scope: "built-in"},
			{TypeDefinition: ontology.TypeDefinition{
				Type: "entity.product", Label: "Product (Entity)",
				Description: "A product entity", CreatedAt: "2026-01-01T00:00:00Z", Status: "active",
			}, Scope: "built-in"},
			{TypeDefinition: ontology.TypeDefinition{
				Type: "process.workflow", Label: "Workflow (Process)",
				Description: "A workflow process", CreatedAt: "2026-01-01T00:00:00Z", Status: "active",
			}, Scope: "built-in"},
			{TypeDefinition: ontology.TypeDefinition{
				Type: "rule.validation", Label: "Validation (Rule)",
				Description: "A validation rule", CreatedAt: "2026-01-01T00:00:00Z", Status: "active",
			}, Scope: "built-in"},
		},
		EdgeTypes: []ontology.ResolvedType{
			{TypeDefinition: ontology.TypeDefinition{
				Type: "triggers", Label: "Triggers",
				Description: "Triggers relationship", CreatedAt: "2026-01-01T00:00:00Z", Status: "active",
			}, Scope: "built-in"},
		},
		BusinessCategories: []string{"customer", "order", "product", "general"},
	}
}

func TestTagChunk_WithOntology(t *testing.T) {
	t.Parallel()
	ont := sampleResolvedOntology()
	classification := ontology.ClassificationResult{
		Class:      ontology.DocumentClassUnstructured,
		Category:   "general",
		Confidence: 0.8,
	}

	content := "The customer submitted a product order through the workflow."
	tag := ontology.TagChunk(content, classification, ont)

	if tag.DocumentClass != "unstructured" {
		t.Fatalf("expected unstructured, got %q", tag.DocumentClass)
	}
	if tag.Confidence != 0.8 {
		t.Fatalf("expected confidence 0.8, got %f", tag.Confidence)
	}

	// Should find entity.customer and entity.product, process.workflow
	found := make(map[string]bool)
	for _, et := range tag.EntityTypes {
		found[et] = true
	}
	if !found["entity.customer"] {
		t.Error("expected entity.customer in entity types")
	}
	if !found["entity.product"] {
		t.Error("expected entity.product in entity types")
	}
	if !found["process.workflow"] {
		t.Error("expected process.workflow in entity types")
	}
}

func TestTagChunk_WithCategory(t *testing.T) {
	t.Parallel()
	ont := sampleResolvedOntology()
	classification := ontology.ClassificationResult{
		Class:      ontology.DocumentClassStructured,
		Category:   "customer",
		Confidence: 0.95,
	}

	content := "Customer account details for enterprise clients."
	tag := ontology.TagChunk(content, classification, ont)

	// Category should be preserved from classification (or overridden by ontology)
	if tag.BusinessCategory == "" {
		t.Fatal("expected non-empty business category")
	}
}

func TestTagChunk_MinimalClassification(t *testing.T) {
	t.Parallel()
	classification := ontology.ClassificationResult{
		Class:      ontology.DocumentClassUnstructured,
		Category:   "general",
		Confidence: 0.5,
	}

	content := "A simple text with no known ontology terms."
	tag := ontology.TagChunk(content, classification, nil)

	if tag.DocumentClass != "unstructured" {
		t.Fatalf("expected unstructured, got %q", tag.DocumentClass)
	}
	if tag.BusinessCategory != "general" {
		t.Fatalf("expected general category, got %q", tag.BusinessCategory)
	}
	if len(tag.EntityTypes) != 0 {
		t.Fatalf("expected no entity types without ontology, got %d", len(tag.EntityTypes))
	}
}

func TestTagChunks_BatchTagging(t *testing.T) {
	t.Parallel()
	ont := sampleResolvedOntology()
	classification := ontology.ClassificationResult{
		Class:      ontology.DocumentClassUnstructured,
		Category:   "general",
		Confidence: 0.7,
	}

	contents := []string{
		"The customer placed an order.",
		"Product validation rules apply.",
		"No relevant terms here.",
	}

	tags := ontology.TagChunks(contents, classification, ont)
	if len(tags) != 3 {
		t.Fatalf("expected 3 tags, got %d", len(tags))
	}

	// First chunk should find customer
	foundCustomer := false
	for _, et := range tags[0].EntityTypes {
		if et == "entity.customer" {
			foundCustomer = true
			break
		}
	}
	if !foundCustomer {
		t.Error("expected entity.customer in first chunk tags")
	}

	// Second chunk should find product and validation
	foundProduct := false
	foundValidation := false
	for _, et := range tags[1].EntityTypes {
		if et == "entity.product" {
			foundProduct = true
		}
		if et == "rule.validation" {
			foundValidation = true
		}
	}
	if !foundProduct {
		t.Error("expected entity.product in second chunk tags")
	}
	if !foundValidation {
		t.Error("expected rule.validation in second chunk tags")
	}
}

func TestTagChunk_NoEntityTypes_WhenNoMatch(t *testing.T) {
	t.Parallel()
	ont := sampleResolvedOntology()
	classification := ontology.ClassificationResult{
		Class:      ontology.DocumentClassUnstructured,
		Category:   "general",
		Confidence: 0.5,
	}

	content := "This text mentions nothing from the ontology at all."
	tag := ontology.TagChunk(content, classification, ont)

	if len(tag.EntityTypes) != 0 {
		t.Fatalf("expected no entity types for non-matching content, got %v", tag.EntityTypes)
	}
}

func TestClassifyAndTag_EndToEnd(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	c := ontology.NewClassifier(nil)
	ont := sampleResolvedOntology()

	content := `{"customer_id": 1, "customer_name": "Acme", "product": "Widget"}`
	result, err := c.Classify(ctx, content, "data.json")
	if err != nil {
		t.Fatalf("Classify: %v", err)
	}

	tag := ontology.TagChunk(content, result, ont)
	if tag.DocumentClass != "structured" {
		t.Fatalf("expected structured, got %q", tag.DocumentClass)
	}
	if tag.Confidence < 0.8 {
		t.Fatalf("expected high confidence for JSON, got %f", tag.Confidence)
	}

	// Should find customer and product entity types
	foundCustomer := false
	foundProduct := false
	for _, et := range tag.EntityTypes {
		if et == "entity.customer" {
			foundCustomer = true
		}
		if et == "entity.product" {
			foundProduct = true
		}
	}
	if !foundCustomer {
		t.Error("expected entity.customer in end-to-end tag")
	}
	if !foundProduct {
		t.Error("expected entity.product in end-to-end tag")
	}
}
