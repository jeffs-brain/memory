// SPDX-License-Identifier: Apache-2.0

// Package ontology defines the built-in type catalogue for the memory
// knowledge graph. It provides the 31 node types, 19 edge types, 8 business
// categories, and 5 node type prefixes that form the core ontology schema.
//
// All validation functions accept both built-in types and custom types that
// follow the naming conventions. Custom node types must start with a valid
// prefix (entity., rule., exception., decision., process.) followed by a
// snake_case name. Custom edge types and business categories must be lowercase
// snake_case identifiers.
//
// Scope resolution follows a four-level hierarchy: built-in, organisation,
// project, brain. Higher scopes override lower scopes for types sharing the
// same identifier.
//
// Industry templates provide pre-built sets of node types, edge types, and
// business categories for common verticals (healthcare, finance, legal,
// ecommerce, education, software).
package ontology
