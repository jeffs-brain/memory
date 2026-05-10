// SPDX-License-Identifier: Apache-2.0
package ontology

import (
	"fmt"
	"regexp"
	"sort"
)

// TypeEntry is a single type within an industry template.
type TypeEntry struct {
	Type        string `json:"type"`
	Label       string `json:"label"`
	Description string `json:"description"`
}

// IndustryTemplate defines a domain-specific set of types that seed an
// ontology for a particular industry vertical.
type IndustryTemplate struct {
	Label              string      `json:"label"`
	Description        string      `json:"description"`
	NodeTypes          []TypeEntry `json:"nodeTypes"`
	EdgeTypes          []TypeEntry `json:"edgeTypes"`
	BusinessCategories []string    `json:"businessCategories,omitempty"`
}

// nodeTypePattern validates node types: category.name where both parts are
// lowercase alphanumeric with underscores.
var nodeTypePattern = regexp.MustCompile(`^[a-z][a-z0-9]*\.[a-z][a-z0-9_]*$`)

// edgeTypePattern validates edge types: lowercase alphanumeric with underscores.
var edgeTypePattern = regexp.MustCompile(`^[a-z][a-z0-9_]*$`)

// businessCategoryPattern validates business categories: lowercase alphanumeric
// with underscores.
var businessCategoryPattern = regexp.MustCompile(`^[a-z][a-z0-9_]*$`)

// IsValidNodeType reports whether s matches the node type format (category.name).
func IsValidNodeType(s string) bool {
	return nodeTypePattern.MatchString(s)
}

// IsValidEdgeType reports whether s matches the edge type format (snake_case).
func IsValidEdgeType(s string) bool {
	return edgeTypePattern.MatchString(s)
}

// IsValidBusinessCategory reports whether s matches the business category
// format (snake_case).
func IsValidBusinessCategory(s string) bool {
	return businessCategoryPattern.MatchString(s)
}

// Templates is the read-only registry of built-in industry templates.
var Templates = map[string]IndustryTemplate{
	"server_hardware":  serverHardwareTemplate,
	"insurance":        insuranceTemplate,
	"logistics":        logisticsTemplate,
	"finance":          financeTemplate,
	"healthcare":       healthcareTemplate,
	"order_processing": orderProcessingTemplate,
}

// ListTemplates returns sorted template keys.
func ListTemplates() []string {
	keys := make([]string, 0, len(Templates))
	for k := range Templates {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

// GetTemplate returns a template by key, or an error if not found.
func GetTemplate(key string) (IndustryTemplate, error) {
	t, ok := Templates[key]
	if !ok {
		return IndustryTemplate{}, fmt.Errorf("ontology: template not found: %s", key)
	}
	return t, nil
}
