// SPDX-License-Identifier: Apache-2.0
package ontology_test

import (
	"context"
	"testing"

	"github.com/jeffs-brain/memory/go/llm"
	"github.com/jeffs-brain/memory/go/ontology"
)

// fakeProvider is a mock LLM provider for testing classification.
type fakeProvider struct {
	response string
	err      error
}

func (f *fakeProvider) Complete(_ context.Context, _ llm.CompleteRequest) (llm.CompleteResponse, error) {
	if f.err != nil {
		return llm.CompleteResponse{}, f.err
	}
	return llm.CompleteResponse{Text: f.response}, nil
}

func (f *fakeProvider) CompleteStream(_ context.Context, _ llm.CompleteRequest) (<-chan llm.StreamChunk, error) {
	return nil, nil
}

func (f *fakeProvider) Close() error { return nil }

func TestIsJsonDocument_ValidObject(t *testing.T) {
	t.Parallel()
	content := `{"rules": [{"name": "discount"}, {"name": "tax"}]}`
	if !ontology.IsJsonDocument(content) {
		t.Fatal("expected true for valid JSON object")
	}
}

func TestIsJsonDocument_ValidArray(t *testing.T) {
	t.Parallel()
	content := `[{"name": "Alice"}, {"name": "Bob"}]`
	if !ontology.IsJsonDocument(content) {
		t.Fatal("expected true for valid JSON array")
	}
}

func TestIsJsonDocument_InvalidJson(t *testing.T) {
	t.Parallel()
	content := "This is just some regular text content."
	if ontology.IsJsonDocument(content) {
		t.Fatal("expected false for prose text")
	}
}

func TestIsJsonDocument_TrivialJson(t *testing.T) {
	t.Parallel()
	// Bare primitives are not business-relevant JSON
	if ontology.IsJsonDocument("42") {
		t.Fatal("expected false for bare number")
	}
	if ontology.IsJsonDocument(`"hello"`) {
		t.Fatal("expected false for bare string")
	}
}

func TestIsTabularDocument_Csv(t *testing.T) {
	t.Parallel()
	content := "name,age,city\nAlice,30,NYC\nBob,25,LA\nCharlie,35,SF\n"
	if !ontology.IsTabularDocument(content) {
		t.Fatal("expected true for CSV content")
	}
}

func TestIsTabularDocument_Pipe(t *testing.T) {
	t.Parallel()
	content := "name|age|city\nAlice|30|NYC\nBob|25|LA\nCharlie|35|SF\n"
	if !ontology.IsTabularDocument(content) {
		t.Fatal("expected true for pipe-delimited content")
	}
}

func TestIsTabularDocument_Prose(t *testing.T) {
	t.Parallel()
	content := "This is a paragraph of text.\nAnother paragraph follows.\nNo delimiters here.\n"
	if ontology.IsTabularDocument(content) {
		t.Fatal("expected false for prose text")
	}
}

func TestClassify_JsonDetected(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	c := ontology.NewClassifier(nil)

	result, err := c.Classify(ctx, `{"customers": [{"id": 1}]}`, "data.json")
	if err != nil {
		t.Fatalf("Classify: %v", err)
	}
	if result.Class != ontology.DocumentClassStructured {
		t.Fatalf("expected structured, got %q", result.Class)
	}
	if !result.IsStructured {
		t.Fatal("expected IsStructured to be true")
	}
	if result.Confidence < 0.9 {
		t.Fatalf("expected high confidence, got %f", result.Confidence)
	}
}

func TestClassify_TabularDetected(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	c := ontology.NewClassifier(nil)

	content := "name,age,city\nAlice,30,NYC\nBob,25,LA\nCharlie,35,SF\n"
	result, err := c.Classify(ctx, content, "contacts.csv")
	if err != nil {
		t.Fatalf("Classify: %v", err)
	}
	if result.Class != ontology.DocumentClassTabular {
		t.Fatalf("expected tabular, got %q", result.Class)
	}
	if !result.IsStructured {
		t.Fatal("expected IsStructured to be true for tabular")
	}
}

func TestClassify_FallbackToLLM(t *testing.T) {
	t.Parallel()
	ctx := context.Background()

	provider := &fakeProvider{
		response: `{"category": "entity", "confidence": 0.85, "reasoning": "contains customer data"}`,
	}
	c := ontology.NewClassifier(provider)

	content := "The customer approval process requires manager sign-off for orders above $10,000."
	result, err := c.Classify(ctx, content, "approval-rules.md")
	if err != nil {
		t.Fatalf("Classify: %v", err)
	}
	if result.Class != ontology.DocumentClassUnstructured {
		t.Fatalf("expected unstructured, got %q", result.Class)
	}
	if result.Confidence <= 0 {
		t.Fatal("expected non-zero confidence from LLM")
	}
}

func TestClassify_NoProvider(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	c := ontology.NewClassifier(nil)

	content := "The customer approval process requires manager sign-off."
	result, err := c.Classify(ctx, content, "doc.md")
	if err != nil {
		t.Fatalf("Classify: %v", err)
	}
	if result.Class != ontology.DocumentClassUnstructured {
		t.Fatalf("expected unstructured, got %q", result.Class)
	}
	if result.Category != "general" {
		t.Fatalf("expected general category without provider, got %q", result.Category)
	}
}

func TestClassify_JsonWithCustomerKeywords(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	c := ontology.NewClassifier(nil)

	content := `{"customer_id": 123, "customer_name": "Acme Corp", "account_type": "enterprise"}`
	result, err := c.Classify(ctx, content, "data.json")
	if err != nil {
		t.Fatalf("Classify: %v", err)
	}
	if result.Category != "customer" {
		t.Fatalf("expected customer category, got %q", result.Category)
	}
}

func TestDetermineCategory_WithOntology(t *testing.T) {
	t.Parallel()
	ontologyData := &ontology.ResolvedOntology{
		NodeTypes: []ontology.ResolvedType{
			{TypeDefinition: ontology.TypeDefinition{
				Type: "entity.customer", Label: "Customer",
				Description: "Customer entity for customer management",
				CreatedAt: "2026-01-01T00:00:00Z", Status: "active",
			}},
		},
		BusinessCategories: []string{"customer", "order", "general"},
	}

	content := "The customer submitted a new order for processing."
	category := ontology.DetermineCategory(content, ontologyData)
	// Should find "customer" in the content matching entity.customer
	if category == "" {
		t.Fatal("expected non-empty category")
	}
}

func TestDetermineCategory_NilOntology(t *testing.T) {
	t.Parallel()
	category := ontology.DetermineCategory("some content", nil)
	if category != "general" {
		t.Fatalf("expected general for nil ontology, got %q", category)
	}
}

func TestCategoryWinnerThreshold(t *testing.T) {
	t.Parallel()
	// Test that the threshold constant matches the intelligence service
	if ontology.CategoryWinnerThreshold != 0.4 {
		t.Fatalf("expected 0.4, got %f", ontology.CategoryWinnerThreshold)
	}
}
