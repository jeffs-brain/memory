// SPDX-License-Identifier: Apache-2.0
package ontology

import (
	"fmt"
	"regexp"
	"strings"
)

// snakeCaseNameRe matches a valid snake_case name: starts with lowercase letter,
// followed by lowercase alphanumeric segments separated by underscores.
var snakeCaseNameRe = regexp.MustCompile(`^[a-z][a-z0-9]*(_[a-z0-9]+)*$`)

// validStatuses is the set of acceptable TypeStatus values for type definitions.
var validStatuses = map[TypeStatus]struct{}{
	TypeStatusActive:     {},
	TypeStatusProposed:   {},
	TypeStatusDeprecated: {},
}

// IsValidNodeType reports whether value is a valid node type identifier.
// A value is valid if it exactly matches a built-in node type, or if it
// starts with a valid prefix and has a valid snake_case name after the dot.
func IsValidNodeType(value string) bool {
	if IsBuiltInNodeType(value) {
		return true
	}
	prefixMu.RLock()
	prefixes := nodeTypePrefixes
	prefixMu.RUnlock()
	for _, prefix := range prefixes {
		if strings.HasPrefix(value, prefix) {
			name := value[len(prefix):]
			if len(name) == 0 {
				return false
			}
			return snakeCaseNameRe.MatchString(name)
		}
	}
	return false
}

// IsValidEdgeType reports whether value is a valid edge type identifier.
// A value is valid if it exactly matches a built-in edge type, or if it
// matches the snake_case pattern ^[a-z][a-z0-9]*(_[a-z0-9]+)*$.
func IsValidEdgeType(value string) bool {
	if IsBuiltInEdgeType(value) {
		return true
	}
	return snakeCaseNameRe.MatchString(value)
}

// IsValidBusinessCategory reports whether value is a valid business category.
// A value is valid if it matches the snake_case pattern.
func IsValidBusinessCategory(value string) bool {
	return snakeCaseNameRe.MatchString(value)
}

// HasPrefix reports whether the given node type starts with one of the
// known prefixes and returns the prefix. Returns empty string if no prefix matches.
func HasPrefix(nodeType string) string {
	prefixMu.RLock()
	prefixes := nodeTypePrefixes
	prefixMu.RUnlock()
	for _, prefix := range prefixes {
		if strings.HasPrefix(nodeType, prefix) {
			return prefix
		}
	}
	return ""
}

// ValidateNodeType returns an error if value is not a valid node type.
func ValidateNodeType(value string) error {
	if IsValidNodeType(value) {
		return nil
	}
	return fmt.Errorf("ontology: invalid node type %q: must start with a valid prefix (entity., rule., exception., decision., process.) followed by a snake_case name", value)
}

// ValidateEdgeType returns an error if value is not a valid edge type.
func ValidateEdgeType(value string) error {
	if IsValidEdgeType(value) {
		return nil
	}
	return fmt.Errorf("ontology: invalid edge type %q: must be lowercase snake_case starting with a letter", value)
}

// ValidateBusinessCategory returns an error if value is not a valid business category.
func ValidateBusinessCategory(value string) error {
	if IsValidBusinessCategory(value) {
		return nil
	}
	return fmt.Errorf("ontology: invalid business category %q: must be lowercase snake_case starting with a letter", value)
}

// ValidateTypeDefinition checks that a TypeDefinition has all required fields
// and valid values.
func ValidateTypeDefinition(def TypeDefinition) error {
	if def.Type == "" {
		return fmt.Errorf("ontology: type definition has empty type field")
	}
	if def.Label == "" {
		return fmt.Errorf("ontology: type definition %q has empty label", def.Type)
	}
	if def.Description == "" {
		return fmt.Errorf("ontology: type definition %q has empty description", def.Type)
	}
	if def.CreatedAt == "" {
		return fmt.Errorf("ontology: type definition %q has empty createdAt", def.Type)
	}
	if _, ok := validStatuses[def.Status]; !ok {
		return fmt.Errorf("ontology: type definition %q has invalid status %q: must be active, proposed, or deprecated", def.Type, def.Status)
	}
	return nil
}
