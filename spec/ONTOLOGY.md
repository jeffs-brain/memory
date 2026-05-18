# Ontology Type System

This document defines the canonical ontology type catalogue for the memory package. All node types, edge types, business categories, and validation rules are authoritative. Implementations in Go and TypeScript must produce identical results for identical inputs.

## Node Types

Node types use a dotted notation: `prefix.name` where the prefix classifies the domain category and the name identifies the specific type within that category. Names use `snake_case`.

### Entity Types (6)

Types representing real-world entities in the business domain.

| Type | Description |
|------|-------------|
| `entity.customer` | Customers, clients, accounts, subscribers, or end-users of a service |
| `entity.supplier` | Vendors, suppliers, partners, or service providers |
| `entity.product` | Products, goods, SKUs, services offered, or catalogue items |
| `entity.department` | Departments, teams, business units, or organisational divisions |
| `entity.person` | Named individuals, contacts, employees, or stakeholders |
| `entity.document` | Documents, files, templates, forms, contracts, or certificates |

### Rule Types (12)

Types representing business rules, constraints, and logic.

| Type | Description |
|------|-------------|
| `rule.constraint` | General constraints, limitations, or restrictions on operations |
| `rule.validation` | Data validation checks, format requirements, or input verification |
| `rule.threshold` | Numeric thresholds, limits, ranges, minimums, maximums, or SLA targets |
| `rule.policy` | Organisational policies, compliance rules, regulations, or mandates |
| `rule.classification` | Rules that categorise or classify items into groups |
| `rule.matching` | Rules for matching, pairing, or reconciling records |
| `rule.mapping` | Rules for transforming, converting, or mapping data between formats |
| `rule.extraction` | Rules for extracting specific data from documents or sources |
| `rule.routing` | Rules that determine where work or data should be directed |
| `rule.fallback` | Default or backup rules applied when primary rules do not match |
| `rule.priority` | Rules that establish ordering, ranking, or precedence |
| `rule.requirement` | Mandatory conditions, prerequisites, or dependencies |

> **Note:** Combined rules (formerly `rule.combined`) should be expressed as multiple individual rules connected by `extends` or `derived_from` edges.

### Exception Types (3)

Types representing deviations from standard rules or processes.

| Type | Description |
|------|-------------|
| `exception.workaround` | Temporary solutions or manual interventions to bypass issues |
| `exception.override` | Deliberate overrides of existing rules by authorised personnel |
| `exception.special_case` | Unique or one-off handling for specific scenarios |

### Decision Types (3)

Types representing decision points and branching logic.

| Type | Description |
|------|-------------|
| `decision.branch` | Conditional branches, if-then logic, or routing decisions |
| `decision.escalation` | Escalation paths, priority handling, or SLA breach responses |
| `decision.table` | Decision matrices, lookup tables, or multi-criteria evaluations |

### Process Types (6)

Types representing workflows, procedures, and operational sequences.

| Type | Description |
|------|-------------|
| `process.workflow` | End-to-end workflows, automated sequences, or pipelines |
| `process.approval_chain` | Approval processes, sign-off sequences, or authorisation flows |
| `process.procedure` | Step-by-step procedures, standard operating procedures, or protocols |
| `process.stage` | Stages, phases, milestones, or checkpoints within a process |
| `process.integration` | Integration points, API connections, or external system interactions |
| `process.subworkflow` | Nested workflows, reusable process fragments, or child processes |

**Total: 30 built-in node types.**

## Edge Types (29)

Edge types represent relationships between nodes. They use `snake_case` identifiers without a prefix.

| Type | Description |
|------|-------------|
| `triggers` | Source causes target to start or execute |
| `requires_approval_from` | Source needs approval or sign-off from target |
| `exception_for` | Source is an exception or deviation from target rule |
| `overrides` | Source takes precedence over or replaces target |
| `depends_on` | Source requires target to be completed or available first |
| `belongs_to` | Source is a member, child, or component of target |
| `escalates_to` | Source escalates issues or decisions to target |
| `constrains` | Source imposes restrictions or limitations on target |
| `informed_by` | Source uses information or data provided by target |
| `produces` | Source generates, creates, or outputs target |
| `precedes` | Source occurs before target in sequence |
| `related_to` | Source has a general association with target |
| `contradicts` | Source conflicts with or opposes target |
| `fallback_for` | Source serves as a backup or default when target fails |
| `extends` | Source adds to or builds upon target |
| `alternative_to` | Source is an interchangeable option with target |
| `feeds_into` | Source provides input data or context to target |
| `enables` | Source makes target possible or available |
| `validates` | Source checks or confirms the correctness of target |
| `applies_to` | Source rule or policy governs or targets the destination entity |
| `contains` | Source is the parent or container of target (composition relationship) |
| `assigned_to` | Source task, document, or responsibility is assigned to target person or team |
| `implements` | Source concrete process or system realises target abstract rule or policy |
| `created_by` | Source entity was authored or created by target person or system |
| `supersedes` | Source is the newer version that replaces target |
| `derived_from` | Source entity was created or generated from target entity |
| `governs` | Source policy or rule controls target domain or scope |
| `requires` | Source entity needs target as a dependency or prerequisite |
| `maps_to` | Source entity in one system corresponds to target entity in another system |

## Business Categories (8)

Business categories classify nodes by domain area. They use `snake_case` identifiers.

| Category | Purpose |
|----------|---------|
| `customer` | Customer-related entities and processes |
| `order` | Order management and fulfilment |
| `product` | Product catalogue and inventory |
| `address` | Address and location management |
| `document` | Document management and processing |
| `authorization` | Authentication, authorisation, and access control |
| `integration` | External system integrations |
| `general` | Default category for uncategorised items |

## Node Type Prefixes (5)

Valid prefixes for node type identifiers:

| Prefix | Domain |
|--------|--------|
| `entity.` | Real-world entities |
| `rule.` | Business rules and constraints |
| `exception.` | Deviations from standard rules |
| `decision.` | Decision points and branching |
| `process.` | Workflows and procedures |

## OntologyTypeDefinition Schema

A type definition describes a registered ontology type (node or edge):

```json
{
  "type": "string",
  "label": "string",
  "description": "string",
  "discoveredFrom": "string (optional)",
  "createdAt": "string (ISO 8601)",
  "status": "active | proposed | deprecated"
}
```

Fields:

- `type`: The dotted identifier (e.g., `entity.customer`) or edge identifier (e.g., `triggers`)
- `label`: Human-readable display name
- `description`: One-sentence explanation of what the type represents
- `discoveredFrom`: Source document path where the type was first discovered (optional)
- `createdAt`: ISO 8601 timestamp of creation
- `status`: Lifecycle state of the type definition

## Scope Resolution

Types are resolved through a four-level scope hierarchy. Higher scopes override lower scopes when types share the same identifier.

Resolution order (lowest to highest priority):

1. **built-in** -- The 30 node types + 29 edge types defined in this spec
2. **organisation** -- Custom types defined at the organisation level
3. **project** -- Custom types scoped to a specific project
4. **brain** -- Custom types specific to a single brain

When resolving the full ontology:

- Start with all built-in types
- Layer organisation types on top (overriding built-in types with the same identifier)
- Layer project types on top (overriding organisation and built-in)
- Layer brain types on top (overriding all lower scopes)
- Only types with `status: "active"` are included in resolution

## Validation Rules

### Node Type Validation

A string is a valid node type if it satisfies ANY of these conditions:

1. It exactly matches one of the 30 built-in node types
2. It starts with a valid prefix (`entity.`, `rule.`, `exception.`, `decision.`, `process.`) AND has at least one character after the dot AND the name portion matches `^[a-z][a-z0-9]*(_[a-z0-9]+)*$`

### Edge Type Validation

A string is a valid edge type if it satisfies ANY of these conditions:

1. It exactly matches one of the 29 built-in edge types
2. It matches the pattern `^[a-z][a-z0-9]*(_[a-z0-9]+)*$` (lowercase snake_case starting with a letter)

### Business Category Validation

A string is a valid business category if it matches the pattern `^[a-z][a-z0-9]*(_[a-z0-9]+)*$` (lowercase snake_case starting with a letter).

### Type Definition Validation

A TypeDefinition is valid when:

- `type` passes node type or edge type validation (depending on context)
- `label` is non-empty
- `description` is non-empty
- `status` is one of: `active`, `proposed`, `deprecated`
- `createdAt` is a non-empty string (ISO 8601 format)

## Label Formatting

### Node Type Labels

The format function converts a dotted node type to a human-readable label:

- `entity.customer` becomes `Customer (Entity)`
- `process.approval_chain` becomes `Approval Chain (Process)`
- `rule.special_handling` becomes `Special Handling (Rule)`

Algorithm: split on `.`, title-case each word in the name (splitting on `_`), append the prefix in parentheses with title case.

### Edge Type Labels

The format function converts a snake_case edge type to a human-readable label:

- `requires_approval_from` becomes `Requires Approval From`
- `triggers` becomes `Triggers`
- `feeds_into` becomes `Feeds Into`

Algorithm: split on `_`, title-case each word, join with spaces.
