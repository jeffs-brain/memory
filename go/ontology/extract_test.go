// SPDX-License-Identifier: Apache-2.0
package ontology

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/jeffs-brain/memory/go/llm"
)

func TestExtract_NilProvider(t *testing.T) {
	t.Parallel()
	ext := NewExtractor(nil)
	result, err := ext.Extract(context.Background(), ExtractionParams{
		Content:  "Some document content about servers and hardware.",
		FileName: "servers.md",
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(result.NodeTypes) != 0 {
		t.Fatalf("expected 0 node types, got %d", len(result.NodeTypes))
	}
	if len(result.EdgeTypes) != 0 {
		t.Fatalf("expected 0 edge types, got %d", len(result.EdgeTypes))
	}
	if len(result.BusinessCategories) != 0 {
		t.Fatalf("expected 0 business categories, got %d", len(result.BusinessCategories))
	}
}

func TestExtract_EmptyContent(t *testing.T) {
	t.Parallel()
	ext := NewExtractor(llm.NewFake([]string{`{}`}))
	result, err := ext.Extract(context.Background(), ExtractionParams{
		Content: "",
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(result.NodeTypes) != 0 {
		t.Fatalf("expected 0 node types, got %d", len(result.NodeTypes))
	}
}

func makeLLMResponse(domain string, confidence float64, nodeTypes []TypeEntry, edgeTypes []TypeEntry, categories []string) string {
	raw := rawExtractionResult{
		Domain:             domain,
		Confidence:         confidence,
		NodeTypes:          nodeTypes,
		EdgeTypes:          edgeTypes,
		BusinessCategories: categories,
	}
	data, _ := json.Marshal(raw)
	return string(data)
}

func TestExtract_SingleSection(t *testing.T) {
	t.Parallel()
	response := makeLLMResponse("healthcare", 0.85, []TypeEntry{
		{Type: "entity.patient", Label: "Patient", Description: "A person receiving care"},
		{Type: "rule.clinical_protocol", Label: "Clinical Protocol", Description: "A clinical treatment rule"},
	}, []TypeEntry{
		{Type: "treats", Label: "Treats", Description: "Treatment relationship"},
	}, []string{"patient_care"})

	ext := NewExtractor(llm.NewFake([]string{response}))
	result, err := ext.Extract(context.Background(), ExtractionParams{
		Content:  "Short healthcare document about patients and treatments.",
		FileName: "healthcare.md",
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Domain != "healthcare" {
		t.Fatalf("expected domain healthcare, got %s", result.Domain)
	}
	if result.Confidence != 0.85 {
		t.Fatalf("expected confidence 0.85, got %f", result.Confidence)
	}
	if len(result.NodeTypes) != 2 {
		t.Fatalf("expected 2 node types, got %d", len(result.NodeTypes))
	}
	if len(result.EdgeTypes) != 1 {
		t.Fatalf("expected 1 edge type, got %d", len(result.EdgeTypes))
	}
	if len(result.BusinessCategories) != 1 {
		t.Fatalf("expected 1 business category, got %d", len(result.BusinessCategories))
	}
}

func TestExtract_MultiSection(t *testing.T) {
	t.Parallel()
	// Create content larger than SingleSectionThreshold
	longContent := generateLongContent(SingleSectionThreshold + 1000)

	response1 := makeLLMResponse("finance", 0.7, []TypeEntry{
		{Type: "entity.account", Label: "Account", Description: "A financial account"},
	}, []TypeEntry{
		{Type: "holds", Label: "Holds", Description: "An account holds assets"},
	}, []string{"retail_banking"})

	response2 := makeLLMResponse("finance", 0.8, []TypeEntry{
		{Type: "entity.payment", Label: "Payment", Description: "A transfer of funds between accounts"},
	}, []TypeEntry{
		{Type: "settles", Label: "Settles", Description: "A transaction settles"},
	}, []string{"payments"})

	ext := NewExtractor(llm.NewFake([]string{response1, response2}))
	result, err := ext.Extract(context.Background(), ExtractionParams{
		Content:  longContent,
		FileName: "finance.md",
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Domain != "finance" {
		t.Fatalf("expected domain finance, got %s", result.Domain)
	}
	if len(result.NodeTypes) < 1 {
		t.Fatalf("expected at least 1 node type, got %d", len(result.NodeTypes))
	}
}

func TestExtract_ExistingTypesFiltered(t *testing.T) {
	t.Parallel()
	response := makeLLMResponse("healthcare", 0.85, []TypeEntry{
		{Type: "entity.patient", Label: "Patient", Description: "A person receiving care"},
		{Type: "entity.medication", Label: "Medication", Description: "A pharmaceutical product"},
	}, []TypeEntry{
		{Type: "treats", Label: "Treats", Description: "Treatment relationship"},
	}, []string{"patient_care"})

	ext := NewExtractor(llm.NewFake([]string{response}))

	existing := &ResolvedOntology{
		NodeTypes: []ResolvedType{
			{TypeDefinition: TypeDefinition{Type: "entity.patient", Label: "Patient", Description: "A person", Status: TypeStatusActive}, Scope: ScopeBuiltIn},
		},
		EdgeTypes:          []ResolvedType{},
		BusinessCategories: []string{},
	}

	result, err := ext.Extract(context.Background(), ExtractionParams{
		Content:       "Healthcare document about patients and medications.",
		FileName:      "healthcare.md",
		ExistingTypes: existing,
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// entity.patient should be filtered out
	for _, nt := range result.NodeTypes {
		if nt.Type == "entity.patient" {
			t.Fatal("entity.patient should have been filtered out")
		}
	}
}

func TestExtract_FuzzyMergeDuplicates(t *testing.T) {
	t.Parallel()
	// Two sections return near-identical types
	longContent := generateLongContent(SingleSectionThreshold + 500)

	response1 := makeLLMResponse("logistics", 0.7, []TypeEntry{
		{Type: "entity.shipment", Label: "Shipment", Description: "A collection of goods being transported"},
		{Type: "entity.shipment_order", Label: "Shipment Order", Description: "An order for shipping goods"},
	}, nil, nil)

	response2 := makeLLMResponse("logistics", 0.8, []TypeEntry{
		{Type: "entity.shipment", Label: "Shipment", Description: "A collection of goods being transported together"},
		{Type: "entity.shipment_orders", Label: "Shipment Orders", Description: "Orders for shipping"},
	}, nil, nil)

	ext := NewExtractor(llm.NewFake([]string{response1, response2}))
	result, err := ext.Extract(context.Background(), ExtractionParams{
		Content:  longContent,
		FileName: "logistics.md",
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// "Shipment Order" and "Shipment Orders" should be fuzzy-merged (>= 0.88)
	shipmentOrderCount := 0
	for _, nt := range result.NodeTypes {
		if nt.Type == "entity.shipment_order" || nt.Type == "entity.shipment_orders" {
			shipmentOrderCount++
		}
	}
	if shipmentOrderCount > 1 {
		t.Fatalf("expected shipment order variants to be fuzzy-merged, got %d", shipmentOrderCount)
	}
}

func TestExtract_MalformedJSON_RetriesSucceed(t *testing.T) {
	t.Parallel()
	goodResponse := makeLLMResponse("finance", 0.9, []TypeEntry{
		{Type: "entity.account", Label: "Account", Description: "A financial account"},
	}, nil, []string{"banking"})

	// First response is malformed, second is valid
	ext := NewExtractor(llm.NewFake([]string{
		"Here is the analysis:\n{invalid json...",
		goodResponse,
	}))
	result, err := ext.Extract(context.Background(), ExtractionParams{
		Content:  "Short finance document.",
		FileName: "finance.md",
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Domain != "finance" {
		t.Fatalf("expected domain finance, got %s", result.Domain)
	}
}

func TestExtract_MaxRetriesExhausted(t *testing.T) {
	t.Parallel()
	// All responses are malformed
	ext := NewExtractor(llm.NewFake([]string{
		"not json at all",
		"still not json",
		"nope",
	}))
	_, err := ext.Extract(context.Background(), ExtractionParams{
		Content:  "Some document.",
		FileName: "doc.md",
	})
	if err == nil {
		t.Fatal("expected error after exhausting retries")
	}
}

func TestExtract_WrappedJSON(t *testing.T) {
	t.Parallel()
	goodJSON := makeLLMResponse("tech", 0.8, []TypeEntry{
		{Type: "entity.server", Label: "Server", Description: "A server instance"},
	}, nil, []string{"infrastructure"})

	// LLM wraps JSON in prose
	wrappedResponse := "Here is my analysis:\n\n```json\n" + goodJSON + "\n```\n\nLet me know if you need more detail."

	ext := NewExtractor(llm.NewFake([]string{wrappedResponse}))
	result, err := ext.Extract(context.Background(), ExtractionParams{
		Content:  "Server documentation.",
		FileName: "servers.md",
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Domain != "tech" {
		t.Fatalf("expected domain tech, got %s", result.Domain)
	}
}

func TestNoisyOr_Empty(t *testing.T) {
	t.Parallel()
	result := NoisyOr(nil)
	if result != 0 {
		t.Fatalf("expected 0, got %f", result)
	}
}

func TestNoisyOr_SingleValue(t *testing.T) {
	t.Parallel()
	result := NoisyOr([]float64{0.7})
	if result != 0.7 {
		t.Fatalf("expected 0.7, got %f", result)
	}
}

func TestNoisyOr_SingleValueCapped(t *testing.T) {
	t.Parallel()
	result := NoisyOr([]float64{1.0})
	if result != ConfidenceCap {
		t.Fatalf("expected %f, got %f", ConfidenceCap, result)
	}
}

func TestNoisyOr_MultipleValues(t *testing.T) {
	t.Parallel()
	// NoisyOr([0.7, 0.8]) = 1 - (0.3 * 0.2) = 1 - 0.06 = 0.94
	result := NoisyOr([]float64{0.7, 0.8})
	expected := 0.94
	if result < expected-0.001 || result > expected+0.001 {
		t.Fatalf("expected ~%f, got %f", expected, result)
	}
}

func TestNoisyOr_SingleValueBelowFloor(t *testing.T) {
	t.Parallel()
	result := NoisyOr([]float64{0.2})
	if result != 0 {
		t.Fatalf("expected 0 for single value below floor, got %f", result)
	}
}

func TestNoisyOr_BelowFloor(t *testing.T) {
	t.Parallel()
	// All values below ConfidenceFloor should be filtered out
	result := NoisyOr([]float64{0.1, 0.2, 0.15})
	if result != 0 {
		t.Fatalf("expected 0 (all below floor), got %f", result)
	}
}

func TestNoisyOr_CappedAt099(t *testing.T) {
	t.Parallel()
	// Very high confidences: 1 - (0.05 * 0.05 * 0.05) = 0.999875 -> capped at 0.99
	result := NoisyOr([]float64{0.95, 0.95, 0.95})
	if result != ConfidenceCap {
		t.Fatalf("expected %f (capped), got %f", ConfidenceCap, result)
	}
}

func TestNoisyOr_MixedAboveAndBelowFloor(t *testing.T) {
	t.Parallel()
	// 0.1 is below floor, 0.7 and 0.8 are above
	result := NoisyOr([]float64{0.1, 0.7, 0.8})
	expected := 0.94 // 1 - (0.3 * 0.2)
	if result < expected-0.001 || result > expected+0.001 {
		t.Fatalf("expected ~%f, got %f", expected, result)
	}
}

func TestSplitContent_Short(t *testing.T) {
	t.Parallel()
	content := "Short document."
	sections := splitContent(content, SingleSectionThreshold)
	if len(sections) != 1 {
		t.Fatalf("expected 1 section, got %d", len(sections))
	}
	if sections[0] != content {
		t.Fatal("section content should match input")
	}
}

func TestSplitContent_Tabular(t *testing.T) {
	t.Parallel()
	content := "col1,col2,col3\nval1,val2,val3\nval4,val5,val6\nval7,val8,val9\nval10,val11,val12\nval13,val14,val15\nval16,val17,val18\nval19,val20,val21\nval22,val23,val24\nval25,val26,val27"
	// Make it exceed threshold by repeating
	for len(content) <= SingleSectionThreshold {
		content += "\nextra1,extra2,extra3"
	}
	sections := splitContent(content, SingleSectionThreshold)
	if len(sections) < 2 {
		t.Fatalf("expected at least 2 sections for tabular content, got %d", len(sections))
	}
	// Each section should start with the header
	for i, s := range sections {
		if !startsWith(s, "col1,col2,col3") {
			t.Fatalf("section %d should start with header, got: %s", i, s[:min(50, len(s))])
		}
	}
}

func TestSplitContent_Sections(t *testing.T) {
	t.Parallel()
	content := "# Section 1\n\nContent for section 1.\n\n# Section 2\n\nContent for section 2."
	for len(content) <= SingleSectionThreshold {
		content += "\n\nMore content to pad the document with additional text."
	}
	sections := splitContent(content, SingleSectionThreshold)
	if len(sections) < 2 {
		t.Fatalf("expected at least 2 sections for headed content, got %d", len(sections))
	}
}

func TestBuildPrompt_WithExistingTypes(t *testing.T) {
	t.Parallel()
	existing := &ResolvedOntology{
		NodeTypes: []ResolvedType{
			{TypeDefinition: TypeDefinition{Type: "entity.customer", Label: "Customer"}, Scope: ScopeBuiltIn},
		},
		EdgeTypes: []ResolvedType{
			{TypeDefinition: TypeDefinition{Type: "triggers", Label: "Triggers"}, Scope: ScopeBuiltIn},
		},
	}
	prompt := buildOntologyExtractionPrompt(existing)
	if !containsStr(prompt, "entity.customer") {
		t.Fatal("prompt should mention existing node types")
	}
	if !containsStr(prompt, "triggers") {
		t.Fatal("prompt should mention existing edge types")
	}
	if !containsStr(prompt, "Do NOT include these") {
		t.Fatal("prompt should instruct to exclude existing types")
	}
}

func TestBuildPrompt_WithoutExistingTypes(t *testing.T) {
	t.Parallel()
	prompt := buildOntologyExtractionPrompt(nil)
	if containsStr(prompt, "Do NOT include these") {
		t.Fatal("prompt should not mention filtering when no existing types")
	}
}

func TestIsTabularContent_CSV(t *testing.T) {
	t.Parallel()
	content := "col1,col2,col3\nval1,val2,val3\nval4,val5,val6\nval7,val8,val9"
	if !isTabularContent(content) {
		t.Fatal("CSV content should be detected as tabular")
	}
}

func TestIsTabularContent_Pipe(t *testing.T) {
	t.Parallel()
	content := "col1|col2|col3\nval1|val2|val3\nval4|val5|val6\nval7|val8|val9"
	if !isTabularContent(content) {
		t.Fatal("pipe-delimited content should be detected as tabular")
	}
}

func TestIsTabularContent_Prose(t *testing.T) {
	t.Parallel()
	content := "This is a normal paragraph.\nIt has multiple sentences.\nBut no delimiters.\nJust prose text.\nNothing tabular here."
	if isTabularContent(content) {
		t.Fatal("prose content should not be detected as tabular")
	}
}

func TestExtractJSONFromText_PlainJSON(t *testing.T) {
	t.Parallel()
	input := `{"domain": "test", "confidence": 0.5}`
	result, err := extractJSONFromText(input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	var parsed map[string]interface{}
	if json.Unmarshal([]byte(result), &parsed) != nil {
		t.Fatal("result should be valid JSON")
	}
}

func TestExtractJSONFromText_WrappedInProse(t *testing.T) {
	t.Parallel()
	input := `Here is the analysis:

{"domain": "test", "confidence": 0.5}

I hope this helps!`
	result, err := extractJSONFromText(input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	var parsed map[string]interface{}
	if json.Unmarshal([]byte(result), &parsed) != nil {
		t.Fatal("result should be valid JSON")
	}
}

func TestExtractJSONFromText_NoJSON(t *testing.T) {
	t.Parallel()
	_, err := extractJSONFromText("just plain text without any json")
	if err == nil {
		t.Fatal("expected error for text without JSON")
	}
}

func TestExtractJSONFromText_Empty(t *testing.T) {
	t.Parallel()
	_, err := extractJSONFromText("")
	if err == nil {
		t.Fatal("expected error for empty text")
	}
}

// --- helpers ---

func generateLongContent(minBytes int) string {
	var b []byte
	section := "# Section\n\nThis is a paragraph about server hardware and configuration management. " +
		"It discusses various components, compatibility rules, and operational procedures.\n\n"
	for len(b) < minBytes {
		b = append(b, section...)
	}
	return string(b)
}

func startsWith(s, prefix string) bool {
	return len(s) >= len(prefix) && s[:len(prefix)] == prefix
}

func containsStr(s, substr string) bool {
	return len(s) > 0 && len(substr) > 0 && contains(s, substr)
}

func contains(s, sub string) bool {
	for i := 0; i <= len(s)-len(sub); i++ {
		if s[i:i+len(sub)] == sub {
			return true
		}
	}
	return false
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
