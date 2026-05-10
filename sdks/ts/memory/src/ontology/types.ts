// SPDX-License-Identifier: Apache-2.0

/**
 * Core ontology type definitions for the memory knowledge graph.
 * Defines the 31 built-in node types, 19 edge types, 8 business categories,
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
  'rule.combined',
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

export const NODE_TYPE_PREFIXES = [
  'entity.',
  'rule.',
  'exception.',
  'decision.',
  'process.',
] as const

export type NodeTypePrefix = (typeof NODE_TYPE_PREFIXES)[number]

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
