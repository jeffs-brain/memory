// SPDX-License-Identifier: Apache-2.0
package ontology

import "context"

// OntologyStore provides persistence for ontology type definitions.
// Dual-backend: FileOntologyStore (local-first, file-based) and
// PostgresOntologyStore (hosted). This interface covers type CRUD
// operations and 4-layer scope resolution.
type OntologyStore interface {
	// GetType returns a single type definition at the given scope.
	// Returns an error wrapping brain.ErrNotFound if the type does not exist.
	GetType(ctx context.Context, scope Scope, typeID string) (*TypeDefinition, error)

	// ListTypes returns all types at the given scope, optionally filtered.
	ListTypes(ctx context.Context, scope Scope, opts ListTypesOpts) ([]TypeDefinition, error)

	// UpsertType creates or updates a type definition at the given scope.
	UpsertType(ctx context.Context, scope Scope, def TypeDefinition) error

	// DeleteType removes a type from the given scope.
	DeleteType(ctx context.Context, scope Scope, typeID string) error

	// GetResolvedOntology returns the fully merged ontology across all 4 layers.
	// Resolution order: built-in -> organisation -> project -> brain.
	// Higher scopes shadow lower scopes for types sharing the same identifier.
	// Only active types are included in the result.
	GetResolvedOntology(ctx context.Context, brainID, projectID, orgID string) (*ResolvedOntology, error)

	// Close releases any resources held by the store.
	Close() error
}

// ListTypesOpts filters results from ListTypes.
type ListTypesOpts struct {
	Prefix string // filter node types by prefix (e.g., "entity.", "rule.")
	Status string // filter by status: "active", "proposed", "deprecated"
}

// ResolvedType is a TypeDefinition enriched with its resolution scope.
type ResolvedType struct {
	TypeDefinition
	Scope Scope `json:"scope"`
}

// ResolvedOntology is the merged result of 4-layer scope resolution.
// Resolution order: built-in -> organisation -> project -> brain.
// Higher scopes shadow lower scopes for types sharing the same identifier.
type ResolvedOntology struct {
	NodeTypes          []ResolvedType `json:"nodeTypes"`
	EdgeTypes          []ResolvedType `json:"edgeTypes"`
	BusinessCategories []string       `json:"businessCategories"`
}

// StoredOntology is the on-disk JSON shape for custom types at a single scope.
type StoredOntology struct {
	CustomNodeTypes          []TypeDefinition `json:"customNodeTypes"`
	CustomEdgeTypes          []TypeDefinition `json:"customEdgeTypes"`
	CustomBusinessCategories []string         `json:"customBusinessCategories"`
}
