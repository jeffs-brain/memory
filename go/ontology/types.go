// SPDX-License-Identifier: Apache-2.0
package ontology

import (
	"fmt"
	"sync"
)

// BuiltInNodeTypes is the authoritative list of 30 built-in node types.
var BuiltInNodeTypes = [30]string{
	"entity.customer",
	"entity.supplier",
	"entity.product",
	"entity.department",
	"entity.person",
	"entity.document",
	"rule.constraint",
	"rule.validation",
	"rule.threshold",
	"rule.policy",
	"rule.classification",
	"rule.matching",
	"rule.mapping",
	"rule.extraction",
	"rule.routing",
	"rule.fallback",
	"rule.priority",
	"rule.requirement",
	"exception.workaround",
	"exception.override",
	"exception.special_case",
	"decision.branch",
	"decision.escalation",
	"decision.table",
	"process.workflow",
	"process.approval_chain",
	"process.procedure",
	"process.stage",
	"process.integration",
	"process.subworkflow",
}

// BuiltInEdgeTypes is the authoritative list of 29 built-in edge types.
var BuiltInEdgeTypes = [29]string{
	"triggers",
	"requires_approval_from",
	"exception_for",
	"overrides",
	"depends_on",
	"belongs_to",
	"escalates_to",
	"constrains",
	"informed_by",
	"produces",
	"precedes",
	"related_to",
	"contradicts",
	"fallback_for",
	"extends",
	"alternative_to",
	"feeds_into",
	"enables",
	"validates",
	"applies_to",
	"contains",
	"assigned_to",
	"implements",
	"created_by",
	"supersedes",
	"derived_from",
	"governs",
	"requires",
	"maps_to",
}

// BusinessCategories is the authoritative list of 8 built-in categories.
var BusinessCategories = [8]string{
	"customer",
	"order",
	"product",
	"address",
	"document",
	"authorization",
	"integration",
	"general",
}

// NodeTypePrefixes are the 5 valid prefixes for node type identifiers.
var NodeTypePrefixes = [5]string{
	"entity.",
	"rule.",
	"exception.",
	"decision.",
	"process.",
}

// nodeTypePrefixes is the mutable backing list of valid prefixes.
// Protected by prefixMu. Starts with the 5 built-in prefixes.
var nodeTypePrefixes = []string{
	"entity.",
	"rule.",
	"exception.",
	"decision.",
	"process.",
}

// prefixMu protects concurrent access to nodeTypePrefixes.
var prefixMu sync.RWMutex

// NodeTypePrefixesList returns a copy of the current valid prefixes.
// The returned slice is safe to iterate without holding any lock.
func NodeTypePrefixesList() []string {
	prefixMu.RLock()
	defer prefixMu.RUnlock()
	out := make([]string, len(nodeTypePrefixes))
	copy(out, nodeTypePrefixes)
	return out
}

// RegisterPrefix adds a custom node type prefix to the valid set.
// The prefix must end with a dot (e.g., "metric."). Returns an error
// if the prefix is empty, does not end with a dot, or is already
// registered. This is safe for concurrent use.
func RegisterPrefix(prefix string) error {
	if prefix == "" {
		return fmt.Errorf("ontology: prefix must not be empty")
	}
	if prefix[len(prefix)-1] != '.' {
		return fmt.Errorf("ontology: prefix %q must end with a dot", prefix)
	}
	prefixMu.Lock()
	defer prefixMu.Unlock()
	for _, existing := range nodeTypePrefixes {
		if existing == prefix {
			return fmt.Errorf("ontology: prefix %q is already registered", prefix)
		}
	}
	nodeTypePrefixes = append(nodeTypePrefixes, prefix)
	return nil
}

// resetPrefixes restores the built-in prefixes only (for testing).
func resetPrefixes() {
	prefixMu.Lock()
	defer prefixMu.Unlock()
	nodeTypePrefixes = []string{
		"entity.",
		"rule.",
		"exception.",
		"decision.",
		"process.",
	}
}

// Scope determines where a type definition lives in the resolution hierarchy.
type Scope string

const (
	ScopeBuiltIn      Scope = "built-in"
	ScopeOrganisation Scope = "organisation"
	ScopeProject      Scope = "project"
	ScopeBrain        Scope = "brain"
)

// TypeStatus represents the lifecycle state of a type definition.
type TypeStatus string

const (
	TypeStatusActive     TypeStatus = "active"
	TypeStatusProposed   TypeStatus = "proposed"
	TypeStatusDeprecated TypeStatus = "deprecated"
)

// TypeDefinition describes a registered ontology type (node or edge).
type TypeDefinition struct {
	Type           string     `json:"type"`
	Label          string     `json:"label"`
	Description    string     `json:"description"`
	DiscoveredFrom string     `json:"discoveredFrom,omitempty"`
	CreatedAt      string     `json:"createdAt"`
	Status         TypeStatus `json:"status"`
}

// builtInNodeTypeSet is a precomputed lookup set for O(1) membership checks.
var builtInNodeTypeSet map[string]struct{}

// builtInEdgeTypeSet is a precomputed lookup set for O(1) membership checks.
var builtInEdgeTypeSet map[string]struct{}

// businessCategorySet is a precomputed lookup set for O(1) membership checks.
var businessCategorySet map[string]struct{}

func init() {
	builtInNodeTypeSet = make(map[string]struct{}, len(BuiltInNodeTypes))
	for _, t := range BuiltInNodeTypes {
		builtInNodeTypeSet[t] = struct{}{}
	}

	builtInEdgeTypeSet = make(map[string]struct{}, len(BuiltInEdgeTypes))
	for _, t := range BuiltInEdgeTypes {
		builtInEdgeTypeSet[t] = struct{}{}
	}

	businessCategorySet = make(map[string]struct{}, len(BusinessCategories))
	for _, c := range BusinessCategories {
		businessCategorySet[c] = struct{}{}
	}
}

// IsBuiltInNodeType reports whether value exactly matches a built-in node type.
func IsBuiltInNodeType(value string) bool {
	_, ok := builtInNodeTypeSet[value]
	return ok
}

// IsBuiltInEdgeType reports whether value exactly matches a built-in edge type.
func IsBuiltInEdgeType(value string) bool {
	_, ok := builtInEdgeTypeSet[value]
	return ok
}

// IsBuiltInBusinessCategory reports whether value exactly matches a built-in category.
func IsBuiltInBusinessCategory(value string) bool {
	_, ok := businessCategorySet[value]
	return ok
}
