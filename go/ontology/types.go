// SPDX-License-Identifier: Apache-2.0
package ontology

// BuiltInNodeTypes is the authoritative list of 31 built-in node types.
var BuiltInNodeTypes = [31]string{
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
	"rule.combined",
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

// BuiltInEdgeTypes is the authoritative list of 19 built-in edge types.
var BuiltInEdgeTypes = [19]string{
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
