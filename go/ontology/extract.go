// SPDX-License-Identifier: Apache-2.0
package ontology

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/jeffs-brain/memory/go/llm"
)

// Extraction constants ported from the intelligence service.
const (
	// SingleSectionThreshold is the byte count above which content is
	// split into multiple sections for extraction.
	SingleSectionThreshold = 8000

	// ExtractionTemperature is the LLM temperature used for extraction.
	ExtractionTemperature = 0.1

	// FuzzyLabelMerge is the Jaro-Winkler threshold for merging
	// similarly-labelled types during multi-section extraction.
	FuzzyLabelMerge = 0.88

	// ConfidenceFloor is the minimum confidence value considered in
	// noisy-OR aggregation.
	ConfidenceFloor = 0.3

	// ConfidenceCap is the maximum confidence value returned by
	// noisy-OR aggregation.
	ConfidenceCap = 0.99

	// extractionMaxRetries is the maximum number of retries when the
	// LLM returns malformed output.
	extractionMaxRetries = 2
)

// ExtractionResult holds extracted ontology types from a document.
type ExtractionResult struct {
	NodeTypes          []TypeEntry `json:"nodeTypes"`
	EdgeTypes          []TypeEntry `json:"edgeTypes"`
	BusinessCategories []string    `json:"businessCategories"`
	Domain             string      `json:"domain"`
	Confidence         float64     `json:"confidence"`
}

// ExtractionParams configures an extraction request.
type ExtractionParams struct {
	Content       string
	FileName      string
	ExistingTypes *ResolvedOntology
}

// ExtractorConfig configures an Extractor.
type ExtractorConfig struct {
	// Provider is the LLM provider for extraction. If nil, Extract
	// returns an empty result without error.
	Provider llm.Provider

	// Temperature overrides the default ExtractionTemperature.
	// Zero uses the default.
	Temperature float64

	// MaxRetries overrides the default retry count for malformed output.
	// Zero uses the default (2).
	MaxRetries int
}

// Extractor performs LLM-powered ontology extraction from documents.
type Extractor struct {
	provider    llm.Provider
	temperature float64
	maxRetries  int
}

// NewExtractor creates an extractor. If provider is nil, Extract
// returns empty results.
func NewExtractor(provider llm.Provider) *Extractor {
	return NewExtractorWithConfig(ExtractorConfig{Provider: provider})
}

// NewExtractorWithConfig creates an extractor with configurable options.
func NewExtractorWithConfig(cfg ExtractorConfig) *Extractor {
	temp := cfg.Temperature
	if temp == 0 {
		temp = ExtractionTemperature
	}
	retries := cfg.MaxRetries
	if retries == 0 {
		retries = extractionMaxRetries
	}
	return &Extractor{
		provider:    cfg.Provider,
		temperature: temp,
		maxRetries:  retries,
	}
}

// Extract analyses document content and returns discovered ontology types.
// Returns an empty result without error when the provider is nil.
func (e *Extractor) Extract(ctx context.Context, params ExtractionParams) (ExtractionResult, error) {
	if e.provider == nil {
		return emptyExtractionResult(), nil
	}

	content := params.Content
	if len(content) == 0 {
		return emptyExtractionResult(), nil
	}

	if len(content) <= SingleSectionThreshold {
		return e.extractSingleSection(ctx, content, params)
	}

	return e.extractMultiSection(ctx, content, params)
}

func (e *Extractor) extractSingleSection(ctx context.Context, content string, params ExtractionParams) (ExtractionResult, error) {
	prompt := buildOntologyExtractionPrompt(params.ExistingTypes)
	userMsg := buildUserMessage(content, params.FileName)

	result, err := e.callLLMWithRetry(ctx, prompt, userMsg)
	if err != nil {
		return emptyExtractionResult(), err
	}

	if params.ExistingTypes != nil {
		result = filterExistingTypes(result, params.ExistingTypes)
	}

	return result, nil
}

func (e *Extractor) extractMultiSection(ctx context.Context, content string, params ExtractionParams) (ExtractionResult, error) {
	sections := splitContent(content, SingleSectionThreshold)

	var allResults []ExtractionResult
	var discoveredTypes []TypeEntry

	for i, section := range sections {
		prompt := buildOntologyExtractionPrompt(params.ExistingTypes)
		if len(discoveredTypes) > 0 {
			prompt += buildContextPrefix(discoveredTypes)
		}

		userMsg := buildUserMessage(section, params.FileName)
		if len(sections) > 1 {
			userMsg = fmt.Sprintf("[Section %d of %d]\n\n%s", i+1, len(sections), userMsg)
		}

		result, err := e.callLLMWithRetry(ctx, prompt, userMsg)
		if err != nil {
			return emptyExtractionResult(), fmt.Errorf("ontology: extraction section %d: %w", i+1, err)
		}

		allResults = append(allResults, result)
		discoveredTypes = append(discoveredTypes, result.NodeTypes...)
		discoveredTypes = append(discoveredTypes, result.EdgeTypes...)
	}

	if len(allResults) == 0 {
		return emptyExtractionResult(), nil
	}

	merged := mergeOntologyExtractions(allResults)

	if params.ExistingTypes != nil {
		merged = filterExistingTypes(merged, params.ExistingTypes)
	}

	return merged, nil
}

func (e *Extractor) callLLMWithRetry(ctx context.Context, systemPrompt, userMsg string) (ExtractionResult, error) {
	messages := []llm.Message{
		{Role: llm.RoleSystem, Content: systemPrompt},
		{Role: llm.RoleUser, Content: userMsg},
	}

	for attempt := 0; attempt <= e.maxRetries; attempt++ {
		if err := ctx.Err(); err != nil {
			return emptyExtractionResult(), err
		}

		resp, err := e.provider.Complete(ctx, llm.CompleteRequest{
			Messages:           messages,
			Temperature:        e.temperature,
			MaxTokens:          4096,
			ResponseFormatJSON: true,
		})
		if err != nil {
			return emptyExtractionResult(), fmt.Errorf("ontology: llm complete: %w", err)
		}

		result, parseErr := parseExtractionResponse(resp.Text)
		if parseErr == nil {
			return result, nil
		}

		if attempt < e.maxRetries {
			messages = append(messages,
				llm.Message{Role: llm.RoleAssistant, Content: resp.Text},
				llm.Message{Role: llm.RoleUser, Content: "Your previous response was not valid JSON matching the expected schema. Return only valid JSON with the keys: domain, confidence, nodeTypes, edgeTypes, businessCategories."},
			)
			continue
		}

		return emptyExtractionResult(), fmt.Errorf("ontology: extraction failed after %d retries: %w", e.maxRetries, parseErr)
	}

	return emptyExtractionResult(), fmt.Errorf("ontology: extraction exhausted retries")
}

func parseExtractionResponse(text string) (ExtractionResult, error) {
	jsonStr, err := extractJSONFromText(text)
	if err != nil {
		return emptyExtractionResult(), err
	}

	var raw rawExtractionResult
	if err := json.Unmarshal([]byte(jsonStr), &raw); err != nil {
		return emptyExtractionResult(), fmt.Errorf("ontology: JSON unmarshal: %w", err)
	}

	if raw.Domain == "" {
		return emptyExtractionResult(), fmt.Errorf("ontology: missing required field 'domain'")
	}

	result := ExtractionResult{
		Domain:             raw.Domain,
		Confidence:         raw.Confidence,
		NodeTypes:          make([]TypeEntry, 0, len(raw.NodeTypes)),
		EdgeTypes:          make([]TypeEntry, 0, len(raw.EdgeTypes)),
		BusinessCategories: make([]string, 0, len(raw.BusinessCategories)),
	}

	for _, nt := range raw.NodeTypes {
		if nt.Type != "" && nt.Label != "" && nt.Description != "" {
			result.NodeTypes = append(result.NodeTypes, nt)
		}
	}
	for _, et := range raw.EdgeTypes {
		if et.Type != "" && et.Label != "" && et.Description != "" {
			result.EdgeTypes = append(result.EdgeTypes, et)
		}
	}
	for _, cat := range raw.BusinessCategories {
		if cat != "" {
			result.BusinessCategories = append(result.BusinessCategories, cat)
		}
	}

	return result, nil
}

// rawExtractionResult is the JSON shape returned by the LLM.
type rawExtractionResult struct {
	Domain             string      `json:"domain"`
	Confidence         float64     `json:"confidence"`
	NodeTypes          []TypeEntry `json:"nodeTypes"`
	EdgeTypes          []TypeEntry `json:"edgeTypes"`
	BusinessCategories []string    `json:"businessCategories"`
}

// extractJSONFromText finds the first JSON object in a text response.
func extractJSONFromText(text string) (string, error) {
	trimmed := strings.TrimSpace(text)
	if trimmed == "" {
		return "", fmt.Errorf("ontology: empty response from LLM")
	}

	start := strings.IndexByte(trimmed, '{')
	if start < 0 {
		return "", fmt.Errorf("ontology: no JSON object found in response")
	}

	depth := 0
	inString := false
	escaping := false

	for i := start; i < len(trimmed); i++ {
		ch := trimmed[i]
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
				candidate := trimmed[start : i+1]
				var check json.RawMessage
				if json.Unmarshal([]byte(candidate), &check) == nil {
					return candidate, nil
				}
				return "", fmt.Errorf("ontology: extracted JSON is not valid")
			}
		}
	}

	return "", fmt.Errorf("ontology: unclosed JSON object in response")
}

// NoisyOr computes the noisy-OR confidence aggregation.
// Filters values below ConfidenceFloor, caps at ConfidenceCap.
// Returns 0 for empty input.
func NoisyOr(confidences []float64) float64 {
	if len(confidences) == 0 {
		return 0
	}

	var filtered []float64
	for _, c := range confidences {
		if c >= ConfidenceFloor {
			filtered = append(filtered, c)
		}
	}

	if len(filtered) == 0 {
		return 0
	}

	if len(filtered) == 1 {
		if filtered[0] > ConfidenceCap {
			return ConfidenceCap
		}
		return filtered[0]
	}

	product := 1.0
	for _, c := range filtered {
		product *= (1.0 - c)
	}

	result := 1.0 - product
	if result > ConfidenceCap {
		return ConfidenceCap
	}
	return result
}

func emptyExtractionResult() ExtractionResult {
	return ExtractionResult{
		NodeTypes:          []TypeEntry{},
		EdgeTypes:          []TypeEntry{},
		BusinessCategories: []string{},
	}
}

// mergeOntologyExtractions merges results from multi-section extraction.
// Deduplicates by type key (keeps longest description), fuzzy-deduplicates
// labels within the same prefix (Jaro-Winkler >= 0.88), and aggregates
// confidence via noisy-OR.
func mergeOntologyExtractions(results []ExtractionResult) ExtractionResult {
	if len(results) == 0 {
		return emptyExtractionResult()
	}
	if len(results) == 1 {
		return results[0]
	}

	nodeMap := make(map[string]TypeEntry)
	edgeMap := make(map[string]TypeEntry)
	catSet := make(map[string]struct{})
	domains := make(map[string]int)
	var confidences []float64

	for _, r := range results {
		for _, nt := range r.NodeTypes {
			if existing, ok := nodeMap[nt.Type]; ok {
				if len(nt.Description) > len(existing.Description) {
					nodeMap[nt.Type] = nt
				}
			} else {
				nodeMap[nt.Type] = nt
			}
		}
		for _, et := range r.EdgeTypes {
			if existing, ok := edgeMap[et.Type]; ok {
				if len(et.Description) > len(existing.Description) {
					edgeMap[et.Type] = et
				}
			} else {
				edgeMap[et.Type] = et
			}
		}
		for _, cat := range r.BusinessCategories {
			catSet[cat] = struct{}{}
		}
		if r.Domain != "" {
			domains[r.Domain]++
		}
		if r.Confidence > 0 {
			confidences = append(confidences, r.Confidence)
		}
	}

	// Fuzzy-dedup labels within same prefix
	nodeMap = fuzzyDedupByPrefix(nodeMap)
	edgeMap = fuzzyDedupEdges(edgeMap)

	nodeTypes := make([]TypeEntry, 0, len(nodeMap))
	for _, nt := range nodeMap {
		nodeTypes = append(nodeTypes, nt)
	}

	edgeTypes := make([]TypeEntry, 0, len(edgeMap))
	for _, et := range edgeMap {
		edgeTypes = append(edgeTypes, et)
	}

	categories := make([]string, 0, len(catSet))
	for cat := range catSet {
		categories = append(categories, cat)
	}

	bestDomain := ""
	bestCount := 0
	for domain, count := range domains {
		if count > bestCount {
			bestCount = count
			bestDomain = domain
		}
	}

	return ExtractionResult{
		NodeTypes:          nodeTypes,
		EdgeTypes:          edgeTypes,
		BusinessCategories: categories,
		Domain:             bestDomain,
		Confidence:         NoisyOr(confidences),
	}
}

// fuzzyDedupByPrefix groups node types by prefix and merges entries
// whose labels are >= FuzzyLabelMerge similar, keeping the one with
// the longer description.
func fuzzyDedupByPrefix(nodeMap map[string]TypeEntry) map[string]TypeEntry {
	byPrefix := make(map[string][]string)
	for key := range nodeMap {
		prefix := typePrefix(key)
		byPrefix[prefix] = append(byPrefix[prefix], key)
	}

	for _, keys := range byPrefix {
		for i := 0; i < len(keys); i++ {
			for j := i + 1; j < len(keys); j++ {
				entryI := nodeMap[keys[i]]
				entryJ := nodeMap[keys[j]]
				sim := JaroWinklerDistance(entryI.Label, entryJ.Label)
				if sim >= FuzzyLabelMerge {
					// Keep the one with the longer description
					if len(entryJ.Description) > len(entryI.Description) {
						delete(nodeMap, keys[i])
					} else {
						delete(nodeMap, keys[j])
					}
				}
			}
		}
	}
	return nodeMap
}

// fuzzyDedupEdges merges edge types whose labels are >= FuzzyLabelMerge
// similar, keeping the one with the longer description.
func fuzzyDedupEdges(edgeMap map[string]TypeEntry) map[string]TypeEntry {
	keys := make([]string, 0, len(edgeMap))
	for k := range edgeMap {
		keys = append(keys, k)
	}

	for i := 0; i < len(keys); i++ {
		for j := i + 1; j < len(keys); j++ {
			entryI, okI := edgeMap[keys[i]]
			entryJ, okJ := edgeMap[keys[j]]
			if !okI || !okJ {
				continue
			}
			sim := JaroWinklerDistance(entryI.Label, entryJ.Label)
			if sim >= FuzzyLabelMerge {
				if len(entryJ.Description) > len(entryI.Description) {
					delete(edgeMap, keys[i])
				} else {
					delete(edgeMap, keys[j])
				}
			}
		}
	}
	return edgeMap
}

func filterExistingTypes(result ExtractionResult, existing *ResolvedOntology) ExtractionResult {
	existingNodeSet := make(map[string]struct{}, len(existing.NodeTypes))
	for _, nt := range existing.NodeTypes {
		existingNodeSet[nt.Type] = struct{}{}
	}

	existingEdgeSet := make(map[string]struct{}, len(existing.EdgeTypes))
	for _, et := range existing.EdgeTypes {
		existingEdgeSet[et.Type] = struct{}{}
	}

	existingCatSet := make(map[string]struct{}, len(existing.BusinessCategories))
	for _, cat := range existing.BusinessCategories {
		existingCatSet[cat] = struct{}{}
	}

	filtered := ExtractionResult{
		Domain:             result.Domain,
		Confidence:         result.Confidence,
		NodeTypes:          make([]TypeEntry, 0, len(result.NodeTypes)),
		EdgeTypes:          make([]TypeEntry, 0, len(result.EdgeTypes)),
		BusinessCategories: make([]string, 0, len(result.BusinessCategories)),
	}

	for _, nt := range result.NodeTypes {
		if _, exists := existingNodeSet[nt.Type]; !exists {
			filtered.NodeTypes = append(filtered.NodeTypes, nt)
		}
	}
	for _, et := range result.EdgeTypes {
		if _, exists := existingEdgeSet[et.Type]; !exists {
			filtered.EdgeTypes = append(filtered.EdgeTypes, et)
		}
	}
	for _, cat := range result.BusinessCategories {
		if _, exists := existingCatSet[cat]; !exists {
			filtered.BusinessCategories = append(filtered.BusinessCategories, cat)
		}
	}

	return filtered
}

// buildOntologyExtractionPrompt constructs the system prompt for ontology extraction.
func buildOntologyExtractionPrompt(existingTypes *ResolvedOntology) string {
	prompt := `You are a domain ontology analyst. Analyse this document and extract the SCHEMA -- the types of entities, rules, and relationships that exist in this domain.

Do NOT extract individual data items or instances. Instead, identify the CATEGORIES and TYPES of things described.

For each discovered type, provide:
- type: A dotted identifier following the pattern prefix.name where prefix is one of: entity, rule, exception, decision, process. The name should be snake_case.
- label: A human-readable name
- description: A one-sentence explanation of what this type represents

Also identify:
- Edge types: the kinds of relationships between entities (use snake_case identifiers like requires, compatible_with, belongs_to)
- Business categories: the high-level domain areas covered (use snake_case like server_hardware, customer_management)

Analyse the document structure, column headers, data patterns, and content to infer the domain ontology.

Examples of good ontology design patterns:

Pattern 1 - Entity-Rule binding:
entity.product is constrained by rule.pricing via the "constrains" edge.

Pattern 2 - Process-Decision-Exception:
process.approval_chain contains decision.escalation points that may trigger exception.override.

Pattern 3 - Classification hierarchy:
entity.category relates to entity.subcategory via "belongs_to" edge.

Return your analysis as a JSON object with the following structure:
{
  "domain": "string describing the domain",
  "confidence": 0.0 to 1.0,
  "nodeTypes": [{"type": "prefix.name", "label": "Human Label", "description": "One sentence"}],
  "edgeTypes": [{"type": "snake_case_name", "label": "Human Label", "description": "One sentence"}],
  "businessCategories": ["snake_case_category"]
}`

	if existingTypes != nil && (len(existingTypes.NodeTypes) > 0 || len(existingTypes.EdgeTypes) > 0) {
		prompt += "\n\nThe following types already exist in the ontology. Do NOT include these in your response -- only return NEW types that are not already present:\n"
		prompt += buildExistingTypesSection(existingTypes)
	}

	return prompt
}

func buildExistingTypesSection(existing *ResolvedOntology) string {
	var b strings.Builder
	if len(existing.NodeTypes) > 0 {
		b.WriteString("\nExisting node types:\n")
		for _, nt := range existing.NodeTypes {
			b.WriteString("- ")
			b.WriteString(nt.Type)
			b.WriteString(": ")
			b.WriteString(nt.Label)
			b.WriteByte('\n')
		}
	}
	if len(existing.EdgeTypes) > 0 {
		b.WriteString("\nExisting edge types:\n")
		for _, et := range existing.EdgeTypes {
			b.WriteString("- ")
			b.WriteString(et.Type)
			b.WriteString(": ")
			b.WriteString(et.Label)
			b.WriteByte('\n')
		}
	}
	return b.String()
}

func buildUserMessage(content, fileName string) string {
	if fileName != "" {
		return fmt.Sprintf("Document: %s\n\n%s", fileName, content)
	}
	return content
}

func buildContextPrefix(discoveredTypes []TypeEntry) string {
	var b strings.Builder
	b.WriteString("\n\nTypes discovered from earlier sections of this document (avoid duplicating these):\n")
	for _, t := range discoveredTypes {
		b.WriteString("- ")
		b.WriteString(t.Type)
		b.WriteString(": ")
		b.WriteString(t.Label)
		b.WriteByte('\n')
	}
	return b.String()
}

// splitContent splits content into sections of approximately maxBytes.
// If the content appears tabular (CSV/pipe-delimited), it samples the
// header plus 3 data rows per section. Otherwise, it splits on markdown
// headings or at byte boundaries.
func splitContent(content string, maxBytes int) []string {
	if len(content) <= maxBytes {
		return []string{content}
	}

	if isTabularContent(content) {
		return splitTabularContent(content, maxBytes)
	}

	return splitByHeadingsOrBytes(content, maxBytes)
}

// isTabularContent returns true if the content looks like CSV/TSV data.
// Heuristic: >= 2 delimiters per line on >= 3 of the first 5 lines.
func isTabularContent(content string) bool {
	lines := strings.SplitN(content, "\n", 6)
	if len(lines) < 3 {
		return false
	}

	limit := 5
	if len(lines) < limit {
		limit = len(lines)
	}

	tabularLines := 0
	for i := 0; i < limit; i++ {
		line := lines[i]
		commas := strings.Count(line, ",")
		pipes := strings.Count(line, "|")
		if commas >= 2 || pipes >= 2 {
			tabularLines++
		}
	}

	return tabularLines >= 3
}

func splitTabularContent(content string, maxBytes int) []string {
	lines := strings.Split(content, "\n")
	if len(lines) == 0 {
		return []string{content}
	}

	header := lines[0]
	dataLines := lines[1:]

	if len(dataLines) == 0 {
		return []string{content}
	}

	samplesPerSection := 3
	var sections []string

	for i := 0; i < len(dataLines); i += samplesPerSection {
		end := i + samplesPerSection
		if end > len(dataLines) {
			end = len(dataLines)
		}
		section := header + "\n" + strings.Join(dataLines[i:end], "\n")
		sections = append(sections, section)
	}

	return sections
}

func splitByHeadingsOrBytes(content string, maxBytes int) []string {
	lines := strings.Split(content, "\n")
	var sections []string
	var currentSection strings.Builder
	currentBytes := 0

	for _, line := range lines {
		lineBytes := len(line) + 1 // +1 for newline

		isHeading := strings.HasPrefix(line, "# ") ||
			strings.HasPrefix(line, "## ") ||
			strings.HasPrefix(line, "### ")

		if isHeading && currentBytes > 0 {
			sections = append(sections, currentSection.String())
			currentSection.Reset()
			currentBytes = 0
		}

		if currentBytes+lineBytes > maxBytes && currentBytes > 0 {
			sections = append(sections, currentSection.String())
			currentSection.Reset()
			currentBytes = 0
		}

		if currentBytes > 0 {
			currentSection.WriteByte('\n')
			currentBytes++
		}
		currentSection.WriteString(line)
		currentBytes += len(line)
	}

	if currentBytes > 0 {
		sections = append(sections, currentSection.String())
	}

	return sections
}
