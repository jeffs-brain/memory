// SPDX-License-Identifier: Apache-2.0
package ontology

import (
	"context"
	"encoding/json"
	"strings"

	"github.com/jeffs-brain/memory/go/llm"
)

// CategoryWinnerThreshold is the minimum proportion of entity type
// matches required for a business category to win classification.
// Matching the intelligence service constant.
const CategoryWinnerThreshold = 0.4

// DocumentClass categorises documents by structure.
type DocumentClass string

const (
	DocumentClassStructured   DocumentClass = "structured"
	DocumentClassTabular      DocumentClass = "tabular"
	DocumentClassUnstructured DocumentClass = "unstructured"
)

// ClassificationResult describes a classified document.
type ClassificationResult struct {
	Class        DocumentClass `json:"class"`
	Category     string        `json:"category"`
	Confidence   float64       `json:"confidence"`
	IsStructured bool          `json:"isStructured"`
}

// Classifier classifies documents and tags chunks with type metadata.
// When a Provider is supplied, unstructured documents use LLM-powered
// classification. Without a Provider, heuristic-only mode is used.
type Classifier struct {
	provider llm.Provider
}

// NewClassifier creates a classifier. Provider may be nil for
// heuristic-only mode.
func NewClassifier(provider llm.Provider) *Classifier {
	return &Classifier{provider: provider}
}

// Classify determines the document class and business category.
// Classification cascade: JSON detection -> tabular detection -> LLM.
func (c *Classifier) Classify(ctx context.Context, content string, fileName string) (ClassificationResult, error) {
	if IsJsonDocument(content) {
		return ClassificationResult{
			Class:        DocumentClassStructured,
			Category:     inferCategoryFromJSON(content),
			Confidence:   0.95,
			IsStructured: true,
		}, nil
	}

	if IsTabularDocument(content) {
		return ClassificationResult{
			Class:        DocumentClassTabular,
			Category:     inferCategoryFromFileName(fileName),
			Confidence:   0.85,
			IsStructured: true,
		}, nil
	}

	if c.provider != nil {
		return c.classifyWithLLM(ctx, content, fileName)
	}

	return ClassificationResult{
		Class:        DocumentClassUnstructured,
		Category:     "general",
		Confidence:   0.5,
		IsStructured: false,
	}, nil
}

// IsJsonDocument returns true if content parses as JSON with
// business-relevant structure: a non-empty object, or an array
// containing at least one object or nested array. Bare primitives,
// empty structures, and primitive-only arrays are excluded.
func IsJsonDocument(content string) bool {
	trimmed := strings.TrimSpace(content)
	if len(trimmed) == 0 {
		return false
	}

	first := trimmed[0]
	if first != '{' && first != '[' {
		return false
	}

	var parsed interface{}
	if err := json.Unmarshal([]byte(trimmed), &parsed); err != nil {
		return false
	}

	switch v := parsed.(type) {
	case map[string]interface{}:
		return len(v) > 0
	case []interface{}:
		if len(v) == 0 {
			return false
		}
		for _, item := range v {
			switch item.(type) {
			case map[string]interface{}, []interface{}:
				return true
			}
		}
		return false
	default:
		return false
	}
}

// IsTabularDocument returns true if content appears to be CSV/TSV data.
// Uses a heuristic: checks the first 5 lines for consistent comma or
// pipe delimiters (>= 2 delimiters per line on >= 3 of the first 5 lines).
func IsTabularDocument(content string) bool {
	lines := strings.SplitN(content, "\n", 6)
	if len(lines) < 3 {
		return false
	}

	checkLines := lines
	if len(checkLines) > 5 {
		checkLines = checkLines[:5]
	}

	commaCount := 0
	pipeCount := 0

	for _, line := range checkLines {
		trimmed := strings.TrimSpace(line)
		if trimmed == "" {
			continue
		}
		commas := strings.Count(trimmed, ",")
		pipes := strings.Count(trimmed, "|")

		if commas >= 2 {
			commaCount++
		}
		if pipes >= 2 {
			pipeCount++
		}
	}

	return commaCount >= 3 || pipeCount >= 3
}

// classifyWithLLM uses the LLM provider to classify unstructured
// content.
func (c *Classifier) classifyWithLLM(ctx context.Context, content string, fileName string) (ClassificationResult, error) {
	preview := content
	if len(preview) > 2000 {
		preview = preview[:2000]
	}

	prompt := buildClassificationPrompt(preview, fileName)

	resp, err := c.provider.Complete(ctx, llm.CompleteRequest{
		Messages: []llm.Message{
			{Role: llm.RoleSystem, Content: classificationSystemPrompt},
			{Role: llm.RoleUser, Content: prompt},
		},
		Temperature:        0.1,
		MaxTokens:          256,
		ResponseFormatJSON: true,
	})
	if err != nil {
		return ClassificationResult{
			Class:        DocumentClassUnstructured,
			Category:     "general",
			Confidence:   0.3,
			IsStructured: false,
		}, nil
	}

	return parseClassificationResponse(resp.Text)
}

const classificationSystemPrompt = `You are a document classifier. Analyse the provided document content and classify it into one of these categories:
- entity: Documents about customers, products, suppliers, or other business entities
- rule: Documents describing business rules, constraints, policies, or validation logic
- exception: Documents about workarounds, overrides, or special cases
- decision: Documents about decision trees, escalation paths, or branch logic
- process: Documents about workflows, procedures, approval chains, or integrations
- reference: General reference documentation that does not fit the above

Respond with a JSON object containing:
- "category": one of "entity", "rule", "exception", "decision", "process", "reference"
- "confidence": a number between 0 and 1 indicating your confidence
- "reasoning": a brief explanation of why you chose this category`

func buildClassificationPrompt(content string, fileName string) string {
	var b strings.Builder
	if fileName != "" {
		b.WriteString("File: ")
		b.WriteString(fileName)
		b.WriteString("\n\n")
	}
	b.WriteString("Document content:\n")
	b.WriteString(content)
	return b.String()
}

// llmClassification is the JSON shape returned by the LLM.
type llmClassification struct {
	Category   string  `json:"category"`
	Confidence float64 `json:"confidence"`
}

// extractJSONObject finds the first balanced JSON object in text using
// depth-tracking. Returns the extracted JSON string, or a non-empty
// error message on failure.
func extractJSONObject(text string) (string, string) {
	start := strings.IndexByte(text, '{')
	if start < 0 {
		return "", "no JSON object found"
	}

	depth := 0
	inString := false
	escaping := false

	for i := start; i < len(text); i++ {
		ch := text[i]
		if escaping {
			escaping = false
			continue
		}
		if inString {
			if ch == '\\' {
				escaping = true
			} else if ch == '"' {
				inString = false
			}
			continue
		}
		switch ch {
		case '"':
			inString = true
		case '{':
			depth++
		case '}':
			depth--
			if depth == 0 {
				candidate := text[start : i+1]
				var check json.RawMessage
				if json.Unmarshal([]byte(candidate), &check) == nil {
					return candidate, ""
				}
				return "", "extracted JSON is not valid"
			}
		}
	}

	return "", "unclosed JSON object"
}

func parseClassificationResponse(text string) (ClassificationResult, error) {
	trimmed := strings.TrimSpace(text)

	jsonStr, extractErr := extractJSONObject(trimmed)
	if extractErr != "" {
		return ClassificationResult{
			Class:        DocumentClassUnstructured,
			Category:     "general",
			Confidence:   0.3,
			IsStructured: false,
		}, nil
	}

	var result llmClassification
	if err := json.Unmarshal([]byte(jsonStr), &result); err != nil {
		return ClassificationResult{
			Class:        DocumentClassUnstructured,
			Category:     "general",
			Confidence:   0.3,
			IsStructured: false,
		}, nil
	}

	category := mapLLMCategory(result.Category)
	confidence := result.Confidence
	if confidence <= 0 || confidence > 1 {
		confidence = 0.7
	}

	return ClassificationResult{
		Class:        DocumentClassUnstructured,
		Category:     category,
		Confidence:   confidence,
		IsStructured: false,
	}, nil
}

// mapLLMCategory maps the LLM's category response to a business
// category string. The system prompt instructs the LLM to return one
// of: entity, rule, exception, decision, process, reference. These
// are preserved as-is since they are the ontology type prefixes the
// pipeline already understands. Additionally, common business
// categories like "customer" and "order" pass through directly.
// Falls back to "general" for unrecognised values.
func mapLLMCategory(raw string) string {
	categoryMap := map[string]string{
		"entity":    "entity",
		"rule":      "rule",
		"exception": "exception",
		"decision":  "decision",
		"process":   "process",
		"reference": "reference",
		"customer":  "customer",
		"order":     "order",
		"product":   "product",
		"address":   "address",
		"document":  "document",
	}
	lower := strings.ToLower(strings.TrimSpace(raw))
	if mapped, ok := categoryMap[lower]; ok {
		return mapped
	}
	if IsValidBusinessCategory(lower) {
		return lower
	}
	return "general"
}

// inferCategoryFromJSON attempts to determine a business category by
// inspecting JSON key names.
func inferCategoryFromJSON(content string) string {
	lower := strings.ToLower(content)
	categoryKeywords := map[string][]string{
		"customer":      {"customer", "client", "account"},
		"order":         {"order", "purchase", "transaction"},
		"product":       {"product", "item", "sku", "catalog"},
		"address":       {"address", "location", "postal", "zip"},
		"document":      {"document", "file", "attachment"},
		"authorization": {"authorization", "permission", "role", "access"},
		"integration":   {"integration", "api", "endpoint", "webhook"},
	}

	var bestCategory string
	bestCount := 0

	for category, keywords := range categoryKeywords {
		count := 0
		for _, kw := range keywords {
			count += strings.Count(lower, kw)
		}
		if count > bestCount {
			bestCount = count
			bestCategory = category
		}
	}

	if bestCategory != "" && bestCount >= 2 {
		return bestCategory
	}
	return "general"
}

// inferCategoryFromFileName attempts to determine a business category
// from the file name.
func inferCategoryFromFileName(fileName string) string {
	lower := strings.ToLower(fileName)
	keywords := map[string]string{
		"customer": "customer",
		"order":    "order",
		"product":  "product",
		"address":  "address",
		"invoice":  "order",
		"shipping": "order",
		"price":    "product",
		"auth":     "authorization",
		"api":      "integration",
	}
	for kw, category := range keywords {
		if strings.Contains(lower, kw) {
			return category
		}
	}
	return "general"
}

// buildCategoryCounts computes weighted category votes from ontology
// entity types found in content.
func buildCategoryCounts(content string, ontology *ResolvedOntology) map[string]int {
	counts := make(map[string]int)
	lower := strings.ToLower(content)

	for _, nt := range ontology.NodeTypes {
		parts := strings.SplitN(nt.Type, ".", 2)
		if len(parts) != 2 {
			continue
		}
		name := parts[1]
		if strings.Contains(lower, strings.ReplaceAll(name, "_", " ")) ||
			strings.Contains(lower, name) {
			for _, cat := range ontology.BusinessCategories {
				if strings.Contains(strings.ToLower(nt.Description), cat) {
					counts[cat]++
				}
			}
		}
	}

	return counts
}

// categoryWinner returns the business category with the highest vote
// count if it exceeds CategoryWinnerThreshold of total votes.
func categoryWinner(counts map[string]int) string {
	total := 0
	for _, c := range counts {
		total += c
	}
	if total == 0 {
		return "general"
	}

	var bestCategory string
	bestCount := 0
	for cat, count := range counts {
		if count > bestCount {
			bestCount = count
			bestCategory = cat
		}
	}

	if float64(bestCount)/float64(total) >= CategoryWinnerThreshold {
		return bestCategory
	}
	return "general"
}

// DetermineCategory uses ontology-aware weighted voting to determine
// the best business category for content. Falls back to "general" if
// no category exceeds the winner threshold.
func DetermineCategory(content string, ontology *ResolvedOntology) string {
	if ontology == nil {
		return "general"
	}
	counts := buildCategoryCounts(content, ontology)
	return categoryWinner(counts)
}

