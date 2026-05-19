// SPDX-License-Identifier: Apache-2.0
package ontology

import "strings"

// ChunkTag is type metadata attached to a chunk for retrieval
// enrichment. It records which ontology entity types are present,
// the dominant business category, classification confidence, and
// the document class.
type ChunkTag struct {
	EntityTypes      []string `json:"entityTypes,omitempty"`
	BusinessCategory string   `json:"businessCategory,omitempty"`
	Confidence       float64  `json:"confidence"`
	DocumentClass    string   `json:"documentClass"`
}

// TagChunk creates a ChunkTag for a given chunk based on the
// classification result and resolved ontology. It scans the chunk
// content for mentions of known entity types from the ontology and
// propagates the document classification's business category.
func TagChunk(content string, classification ClassificationResult, ontology *ResolvedOntology) ChunkTag {
	tag := ChunkTag{
		BusinessCategory: classification.Category,
		Confidence:       classification.Confidence,
		DocumentClass:    string(classification.Class),
	}

	if ontology == nil {
		return tag
	}

	lower := strings.ToLower(content)
	entityTypes := matchEntityTypes(lower, ontology)
	if len(entityTypes) > 0 {
		tag.EntityTypes = entityTypes
	}

	// Override category if ontology voting produces a stronger signal
	ontologyCategory := DetermineCategory(content, ontology)
	if ontologyCategory != "general" {
		tag.BusinessCategory = ontologyCategory
	}

	return tag
}

// matchEntityTypes scans lowercased content for mentions of known
// ontology node types and returns a deduplicated list of matched type
// identifiers.
func matchEntityTypes(lowerContent string, ontology *ResolvedOntology) []string {
	seen := make(map[string]struct{})
	matched := make([]string, 0)

	for _, nt := range ontology.NodeTypes {
		parts := strings.SplitN(nt.Type, ".", 2)
		if len(parts) != 2 {
			continue
		}
		name := parts[1]

		// Match against both snake_case and space-separated forms
		spaced := strings.ReplaceAll(name, "_", " ")
		if strings.Contains(lowerContent, name) || strings.Contains(lowerContent, spaced) {
			if _, ok := seen[nt.Type]; !ok {
				seen[nt.Type] = struct{}{}
				matched = append(matched, nt.Type)
			}
		}
	}

	return matched
}

// TagChunks creates ChunkTags for a batch of chunks. Each chunk is
// independently tagged against the classification and ontology.
func TagChunks(contents []string, classification ClassificationResult, ontology *ResolvedOntology) []ChunkTag {
	tags := make([]ChunkTag, len(contents))
	for i, content := range contents {
		tags[i] = TagChunk(content, classification, ontology)
	}
	return tags
}
