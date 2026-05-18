// SPDX-License-Identifier: Apache-2.0
package ontology

// builtInNodeTypeDescriptions maps each built-in node type to its human-readable description.
var builtInNodeTypeDescriptions = map[string]string{
	"entity.customer":        "Customers, clients, accounts, subscribers, or end-users of a service",
	"entity.supplier":        "Vendors, suppliers, partners, or service providers",
	"entity.product":         "Products, goods, SKUs, services offered, or catalogue items",
	"entity.department":      "Departments, teams, business units, or organisational divisions",
	"entity.person":          "Named individuals, contacts, employees, or stakeholders",
	"entity.document":        "Documents, files, templates, forms, contracts, or certificates",
	"rule.constraint":        "General constraints, limitations, or restrictions on operations",
	"rule.validation":        "Data validation checks, format requirements, or input verification",
	"rule.threshold":         "Numeric thresholds, limits, ranges, minimums, maximums, or SLA targets",
	"rule.policy":            "Organisational policies, compliance rules, regulations, or mandates",
	"rule.classification":    "Rules that categorise or classify items into groups",
	"rule.matching":          "Rules for matching, pairing, or reconciling records",
	"rule.mapping":           "Rules for transforming, converting, or mapping data between formats",
	"rule.extraction":        "Rules for extracting specific data from documents or sources",
	"rule.routing":           "Rules that determine where work or data should be directed",
	"rule.fallback":          "Default or backup rules applied when primary rules do not match",
	"rule.priority":          "Rules that establish ordering, ranking, or precedence",
	"rule.requirement":       "Mandatory conditions, prerequisites, or dependencies",
	"exception.workaround":   "Temporary solutions or manual interventions to bypass issues",
	"exception.override":     "Deliberate overrides of existing rules by authorised personnel",
	"exception.special_case": "Unique or one-off handling for specific scenarios",
	"decision.branch":        "Conditional branches, if-then logic, or routing decisions",
	"decision.escalation":    "Escalation paths, priority handling, or SLA breach responses",
	"decision.table":         "Decision matrices, lookup tables, or multi-criteria evaluations",
	"process.workflow":       "End-to-end workflows, automated sequences, or pipelines",
	"process.approval_chain": "Approval processes, sign-off sequences, or authorisation flows",
	"process.procedure":      "Step-by-step procedures, standard operating procedures, or protocols",
	"process.stage":          "Stages, phases, milestones, or checkpoints within a process",
	"process.integration":    "Integration points, API connections, or external system interactions",
	"process.subworkflow":    "Nested workflows, reusable process fragments, or child processes",
}

// builtInEdgeTypeDescriptions maps each built-in edge type to its human-readable description.
var builtInEdgeTypeDescriptions = map[string]string{
	"triggers":               "Source causes target to start or execute",
	"requires_approval_from": "Source needs approval or sign-off from target",
	"exception_for":          "Source is an exception or deviation from target rule",
	"overrides":              "Source takes precedence over or replaces target",
	"depends_on":             "Source requires target to be completed or available first",
	"belongs_to":             "Source is a member, child, or component of target",
	"escalates_to":           "Source escalates issues or decisions to target",
	"constrains":             "Source imposes restrictions or limitations on target",
	"informed_by":            "Source uses information or data provided by target",
	"produces":               "Source generates, creates, or outputs target",
	"precedes":               "Source occurs before target in sequence",
	"related_to":             "Source has a general association with target",
	"contradicts":            "Source conflicts with or opposes target",
	"fallback_for":           "Source serves as a backup or default when target fails",
	"extends":                "Source adds to or builds upon target",
	"alternative_to":         "Source is an interchangeable option with target",
	"feeds_into":             "Source provides input data or context to target",
	"enables":                "Source makes target possible or available",
	"validates":              "Source checks or confirms the correctness of target",
	"applies_to":             "Source rule or policy governs or targets the destination entity",
	"contains":               "Source is the parent or container of target (composition relationship)",
	"assigned_to":            "Source task, document, or responsibility is assigned to target person or team",
	"implements":             "Source concrete process or system realises target abstract rule or policy",
	"created_by":             "Source entity was authored or created by target person or system",
	"supersedes":             "Source is the newer version that replaces target",
	"derived_from":           "Source entity was created or generated from target entity",
	"governs":                "Source policy or rule controls target domain or scope",
	"requires":               "Source entity needs target as a dependency or prerequisite",
	"maps_to":                "Source entity in one system corresponds to target entity in another system",
}

// GetBuiltInNodeTypeDescription returns the human-readable description for a
// built-in node type. Returns a generic fallback if the type is not built-in.
func GetBuiltInNodeTypeDescription(typ string) string {
	if desc, ok := builtInNodeTypeDescriptions[typ]; ok {
		return desc
	}
	return "A business intelligence node type"
}

// GetBuiltInEdgeTypeDescription returns the human-readable description for a
// built-in edge type. Returns a generic fallback if the type is not built-in.
func GetBuiltInEdgeTypeDescription(typ string) string {
	if desc, ok := builtInEdgeTypeDescriptions[typ]; ok {
		return desc
	}
	return "A relationship between intelligence nodes"
}
