// SPDX-License-Identifier: Apache-2.0

import { afterEach, describe, expect, it } from 'vitest'
import { getBuiltInEdgeTypeDescription, getBuiltInNodeTypeDescription } from './descriptions.js'
import {
  BUILT_IN_EDGE_TYPES,
  BUILT_IN_NODE_TYPES,
  BUSINESS_CATEGORIES,
  NODE_TYPE_PREFIXES,
  getNodeTypePrefixes,
  registerPrefix,
  _resetPrefixes,
} from './types.js'
import { hasPrefix, isValidNodeType } from './validation.js'

describe('BUILT_IN_NODE_TYPES', () => {
  it('has exactly 30 entries', () => {
    expect(BUILT_IN_NODE_TYPES).toHaveLength(30)
  })

  it('contains no duplicates', () => {
    const unique = new Set(BUILT_IN_NODE_TYPES)
    expect(unique.size).toBe(BUILT_IN_NODE_TYPES.length)
  })

  it('all have a valid prefix', () => {
    for (const typ of BUILT_IN_NODE_TYPES) {
      expect(hasPrefix(typ)).toBeDefined()
    }
  })

  it('all have non-empty descriptions', () => {
    for (const typ of BUILT_IN_NODE_TYPES) {
      const desc = getBuiltInNodeTypeDescription(typ)
      expect(desc).not.toBe('A business intelligence node type')
      expect(desc.length).toBeGreaterThan(0)
    }
  })
})

describe('BUILT_IN_EDGE_TYPES', () => {
  it('has exactly 29 entries', () => {
    expect(BUILT_IN_EDGE_TYPES).toHaveLength(29)
  })

  it('contains no duplicates', () => {
    const unique = new Set(BUILT_IN_EDGE_TYPES)
    expect(unique.size).toBe(BUILT_IN_EDGE_TYPES.length)
  })

  it('all have non-empty descriptions', () => {
    for (const typ of BUILT_IN_EDGE_TYPES) {
      const desc = getBuiltInEdgeTypeDescription(typ)
      expect(desc).not.toBe('A relationship between intelligence nodes')
      expect(desc.length).toBeGreaterThan(0)
    }
  })
})

describe('BUSINESS_CATEGORIES', () => {
  it('has exactly 8 entries', () => {
    expect(BUSINESS_CATEGORIES).toHaveLength(8)
  })

  it('contains no duplicates', () => {
    const unique = new Set(BUSINESS_CATEGORIES)
    expect(unique.size).toBe(BUSINESS_CATEGORIES.length)
  })
})

describe('NODE_TYPE_PREFIXES', () => {
  it('has exactly 5 entries', () => {
    expect(NODE_TYPE_PREFIXES).toHaveLength(5)
  })

  it('all end with a dot', () => {
    for (const prefix of NODE_TYPE_PREFIXES) {
      expect(prefix.endsWith('.')).toBe(true)
    }
  })

  it('contains no duplicates', () => {
    const unique = new Set(NODE_TYPE_PREFIXES)
    expect(unique.size).toBe(NODE_TYPE_PREFIXES.length)
  })
})

describe('cross-SDK parity with Go constants', () => {
  it('node types match expected Go values', () => {
    const expectedNodeTypes = [
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
    ]
    expect([...BUILT_IN_NODE_TYPES]).toEqual(expectedNodeTypes)
  })

  it('edge types match expected Go values', () => {
    const expectedEdgeTypes = [
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
    ]
    expect([...BUILT_IN_EDGE_TYPES]).toEqual(expectedEdgeTypes)
  })

  it('business categories match expected Go values', () => {
    const expectedCategories = [
      'customer',
      'order',
      'product',
      'address',
      'document',
      'authorization',
      'integration',
      'general',
    ]
    expect([...BUSINESS_CATEGORIES]).toEqual(expectedCategories)
  })

  it('prefixes match expected Go values', () => {
    const expectedPrefixes = ['entity.', 'rule.', 'exception.', 'decision.', 'process.']
    expect([...NODE_TYPE_PREFIXES]).toEqual(expectedPrefixes)
  })
})

describe('registerPrefix', () => {
  afterEach(() => {
    _resetPrefixes()
  })

  it('adds a custom prefix', () => {
    registerPrefix('metric.')
    const prefixes = getNodeTypePrefixes()
    expect(prefixes).toContain('metric.')
    expect(prefixes).toHaveLength(6)
  })

  it('enables validation of types with the new prefix', () => {
    registerPrefix('metric.')
    expect(isValidNodeType('metric.cpu_usage')).toBe(true)
  })

  it('rejects empty prefix', () => {
    expect(() => registerPrefix('')).toThrow('must not be empty')
  })

  it('rejects prefix without trailing dot', () => {
    expect(() => registerPrefix('noDot')).toThrow('must end with a dot')
  })

  it('rejects duplicate prefix', () => {
    expect(() => registerPrefix('entity.')).toThrow('already registered')
  })
})
