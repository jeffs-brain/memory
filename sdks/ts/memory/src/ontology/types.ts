// SPDX-License-Identifier: Apache-2.0

/**
 * Core ontology type definitions for the memory knowledge graph.
 * Defines the 30 built-in node types, 29 edge types, 8 business categories,
 * and 5 node type prefixes that form the ontology schema.
 *
 * Port of packages/intelligence-database/src/types.ts.
 */

export const BUILT_IN_NODE_TYPES = [
  'entity.customer',
  'entity.supplier',
  'entity.product',
  'entity.department',
  'entity.person',
  'entity.document',
  'rule.constraint',
  'rule.validation',
  'rule.threshold',
  'rule.policy',
  'rule.classification',
  'rule.matching',
  'rule.mapping',
  'rule.extraction',
  'rule.routing',
  'rule.fallback',
  'rule.priority',
  'rule.requirement',
  'exception.workaround',
  'exception.override',
  'exception.special_case',
  'decision.branch',
  'decision.escalation',
  'decision.table',
  'process.workflow',
  'process.approval_chain',
  'process.procedure',
  'process.stage',
  'process.integration',
  'process.subworkflow',
] as const

export type BuiltInNodeType = (typeof BUILT_IN_NODE_TYPES)[number]

const BUILT_IN_PREFIXES = [
  'entity.',
  'rule.',
  'exception.',
  'decision.',
  'process.',
] as const

/**
 * Mutable list of valid node type prefixes. Starts with the 5 built-in
 * prefixes and can be extended at runtime via registerPrefix().
 */
const nodeTypePrefixes: string[] = [...BUILT_IN_PREFIXES]

/**
 * Read-only reference to the built-in prefixes for static typing.
 * Runtime validation uses the mutable nodeTypePrefixes list.
 */
export const NODE_TYPE_PREFIXES = BUILT_IN_PREFIXES

export type NodeTypePrefix = (typeof BUILT_IN_PREFIXES)[number]

/**
 * Returns a copy of the current valid node type prefixes, including
 * any custom prefixes added via registerPrefix().
 */
export function getNodeTypePrefixes(): readonly string[] {
  return [...nodeTypePrefixes]
}

/**
 * Internal access to the live prefix list for validation.
 * Not exported -- used by validation.ts.
 */
export function _nodeTypePrefixesRef(): readonly string[] {
  return nodeTypePrefixes
}

/**
 * Registers a custom node type prefix. The prefix must end with a dot
 * (e.g., "metric."). Throws if the prefix is empty, does not end with
 * a dot, or is already registered.
 */
export function registerPrefix(prefix: string): void {
  if (prefix === '') {
    throw new Error('ontology: prefix must not be empty')
  }
  if (!prefix.endsWith('.')) {
    throw new Error(`ontology: prefix "${prefix}" must end with a dot`)
  }
  if (nodeTypePrefixes.includes(prefix)) {
    throw new Error(`ontology: prefix "${prefix}" is already registered`)
  }
  nodeTypePrefixes.push(prefix)
}

/**
 * Resets prefixes to the 5 built-in defaults (for testing).
 */
export function _resetPrefixes(): void {
  nodeTypePrefixes.length = 0
  nodeTypePrefixes.push(...BUILT_IN_PREFIXES)
}

export type NodeType =
  | BuiltInNodeType
  | `entity.${string}`
  | `rule.${string}`
  | `exception.${string}`
  | `decision.${string}`
  | `process.${string}`

export const BUILT_IN_EDGE_TYPES = [
  'triggers',
  'requires_approval_from',
  'exception_for',
  'overrides',
  'depends_on',
  'belongs_to',
  'escalates_to',
  'constrains',
  'informed_by',
  'produces',
  'precedes',
  'related_to',
  'contradicts',
  'fallback_for',
  'extends',
  'alternative_to',
  'feeds_into',
  'enables',
  'validates',
  'applies_to',
  'contains',
  'assigned_to',
  'implements',
  'created_by',
  'supersedes',
  'derived_from',
  'governs',
  'requires',
  'maps_to',
] as const

export type BuiltInEdgeType = (typeof BUILT_IN_EDGE_TYPES)[number]

export type EdgeType = BuiltInEdgeType | (string & {})

export const BUSINESS_CATEGORIES = [
  'customer',
  'order',
  'product',
  'address',
  'document',
  'authorization',
  'integration',
  'general',
] as const

export type BusinessCategory = (typeof BUSINESS_CATEGORIES)[number]

export type OntologyScope = 'built-in' | 'organisation' | 'project' | 'brain'

export type TypeStatus = 'active' | 'proposed' | 'deprecated'

export type OntologyTypeDefinition = {
  readonly type: string
  readonly label: string
  readonly description: string
  readonly discoveredFrom?: string
  readonly createdAt: string
  readonly status: TypeStatus
}

/**
 * TypeDefinition is an alias for OntologyTypeDefinition used by the
 * deduplication system. Kept as a named export for clarity at dedup
 * call sites.
 */
export type TypeDefinition = OntologyTypeDefinition
