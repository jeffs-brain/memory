// SPDX-License-Identifier: Apache-2.0

/**
 * Ontology type validation functions. Validates node types, edge types,
 * business categories, and type definitions against the naming conventions.
 *
 * Port of packages/intelligence-database validation logic.
 */

import type { OntologyTypeDefinition, TypeStatus } from './types.js'
import { BUILT_IN_EDGE_TYPES, BUILT_IN_NODE_TYPES, BUSINESS_CATEGORIES, NODE_TYPE_PREFIXES } from './types.js'

const SNAKE_CASE_NAME_RE = /^[a-z][a-z0-9]*(_[a-z0-9]+)*$/

const builtInNodeTypeSet: ReadonlySet<string> = new Set(BUILT_IN_NODE_TYPES)
const builtInEdgeTypeSet: ReadonlySet<string> = new Set(BUILT_IN_EDGE_TYPES)
const businessCategorySet: ReadonlySet<string> = new Set(BUSINESS_CATEGORIES)

const VALID_STATUSES: ReadonlySet<TypeStatus> = new Set(['active', 'proposed', 'deprecated'])

/**
 * Reports whether value exactly matches a built-in node type.
 */
export function isBuiltInNodeType(value: string): boolean {
  return builtInNodeTypeSet.has(value)
}

/**
 * Reports whether value exactly matches a built-in edge type.
 */
export function isBuiltInEdgeType(value: string): boolean {
  return builtInEdgeTypeSet.has(value)
}

/**
 * Reports whether value exactly matches a built-in business category.
 */
export function isBuiltInBusinessCategory(value: string): boolean {
  return businessCategorySet.has(value)
}

/**
 * Reports whether value is a valid node type identifier.
 * A value is valid if it exactly matches a built-in node type, or if it
 * starts with a valid prefix and has a valid snake_case name after the dot.
 */
export function isValidNodeType(value: string): boolean {
  if (builtInNodeTypeSet.has(value)) {
    return true
  }
  for (const prefix of NODE_TYPE_PREFIXES) {
    if (value.startsWith(prefix)) {
      const name = value.slice(prefix.length)
      if (name.length === 0) {
        return false
      }
      return SNAKE_CASE_NAME_RE.test(name)
    }
  }
  return false
}

/**
 * Reports whether value is a valid edge type identifier.
 * A value is valid if it exactly matches a built-in edge type, or if it
 * matches the snake_case pattern.
 */
export function isValidEdgeType(value: string): boolean {
  if (builtInEdgeTypeSet.has(value)) {
    return true
  }
  return SNAKE_CASE_NAME_RE.test(value)
}

/**
 * Reports whether value is a valid business category identifier.
 */
export function isValidBusinessCategory(value: string): boolean {
  return SNAKE_CASE_NAME_RE.test(value)
}

/**
 * Returns the matching prefix for a node type, or undefined if none matches.
 */
export function hasPrefix(nodeType: string): string | undefined {
  for (const prefix of NODE_TYPE_PREFIXES) {
    if (nodeType.startsWith(prefix)) {
      return prefix
    }
  }
  return undefined
}

/**
 * Returns an error message if value is not a valid node type, or undefined if valid.
 */
export function validateNodeType(value: string): string | undefined {
  if (isValidNodeType(value)) {
    return undefined
  }
  return `ontology: invalid node type "${value}": must start with a valid prefix (entity., rule., exception., decision., process.) followed by a snake_case name`
}

/**
 * Returns an error message if value is not a valid edge type, or undefined if valid.
 */
export function validateEdgeType(value: string): string | undefined {
  if (isValidEdgeType(value)) {
    return undefined
  }
  return `ontology: invalid edge type "${value}": must be lowercase snake_case starting with a letter`
}

/**
 * Returns an error message if value is not a valid business category, or undefined if valid.
 */
export function validateBusinessCategory(value: string): string | undefined {
  if (isValidBusinessCategory(value)) {
    return undefined
  }
  return `ontology: invalid business category "${value}": must be lowercase snake_case starting with a letter`
}

/**
 * Validates a type definition has all required fields and valid values.
 * Returns an error message or undefined if valid.
 */
export function validateTypeDefinition(def: OntologyTypeDefinition): string | undefined {
  if (!def.type) {
    return 'ontology: type definition has empty type field'
  }
  if (!def.label) {
    return `ontology: type definition "${def.type}" has empty label`
  }
  if (!def.description) {
    return `ontology: type definition "${def.type}" has empty description`
  }
  if (!def.createdAt) {
    return `ontology: type definition "${def.type}" has empty createdAt`
  }
  if (!VALID_STATUSES.has(def.status)) {
    return `ontology: type definition "${def.type}" has invalid status "${def.status}": must be active, proposed, or deprecated`
  }
  return undefined
}
