// SPDX-License-Identifier: Apache-2.0
package ontology

// ExtractionResult holds extracted ontology types from a document.
// Defined in a shared file so that both template matching (P6-6) and
// the proposal workflow (P6-7) can reference it without duplication.
// When P6-4 is merged, the canonical definition in extract.go will
// supersede this one.
type ExtractionResult struct {
	NodeTypes          []TypeEntry `json:"nodeTypes"`
	EdgeTypes          []TypeEntry `json:"edgeTypes"`
	BusinessCategories []string    `json:"businessCategories"`
	Domain             string      `json:"domain"`
	Confidence         float64     `json:"confidence"`
}
