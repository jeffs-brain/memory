// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from 'vitest'
import { formatEdgeTypeLabel, formatNodeTypeLabel } from './format.js'
import type { OntologyTypeDefinition } from './types.js'
import { BUILT_IN_EDGE_TYPES, BUILT_IN_NODE_TYPES, BUSINESS_CATEGORIES } from './types.js'
import {
  hasPrefix,
  isBuiltInBusinessCategory,
  isBuiltInEdgeType,
  isBuiltInNodeType,
  isValidBusinessCategory,
  isValidEdgeType,
  isValidNodeType,
  validateBusinessCategory,
  validateEdgeType,
  validateNodeType,
  validateTypeDefinition,
} from './validation.js'

describe('isValidNodeType', () => {
  it('accepts all 30 built-in node types', () => {
    for (const typ of BUILT_IN_NODE_TYPES) {
      expect(isValidNodeType(typ)).toBe(true)
    }
  })

  it.each([
    ['entity.invoice', true],
    ['rule.custom_thing', true],
    ['exception.timeout_handling', true],
    ['decision.routing_logic', true],
    ['process.batch_job', true],
    ['entity.multi_word_name', true],
  ])('accepts valid custom type %s', (value, expected) => {
    expect(isValidNodeType(value)).toBe(expected)
  })

  it.each([
    ['invalid', false],
    ['wrong.prefix', false],
    ['', false],
    ['entity.', false],
    ['entity.Customer', false],
    ['entity.1thing', false],
    ['entity.my-thing', false],
    ['entity.my thing', false],
    ['entity..customer', false],
    ['entity.name_', false],
    ['entity.name__thing', false],
  ])('rejects invalid type %s', (value, expected) => {
    expect(isValidNodeType(value)).toBe(expected)
  })
})

describe('isValidEdgeType', () => {
  it('accepts all 29 built-in edge types', () => {
    for (const typ of BUILT_IN_EDGE_TYPES) {
      expect(isValidEdgeType(typ)).toBe(true)
    }
  })

  it.each([
    ['ships_to', true],
    ['custom_relation', true],
    ['connects', true],
  ])('accepts valid custom type %s', (value, expected) => {
    expect(isValidEdgeType(value)).toBe(expected)
  })

  it.each([
    ['Invalid', false],
    ['has spaces', false],
    ['123start', false],
    ['', false],
    ['has.dot', false],
    ['has-hyphen', false],
    ['name_', false],
  ])('rejects invalid type %s', (value, expected) => {
    expect(isValidEdgeType(value)).toBe(expected)
  })
})

describe('isValidBusinessCategory', () => {
  it('accepts all 8 built-in categories', () => {
    for (const cat of BUSINESS_CATEGORIES) {
      expect(isValidBusinessCategory(cat)).toBe(true)
    }
  })

  it.each([
    ['custom_category', true],
    ['server_hardware', true],
  ])('accepts valid custom category %s', (value, expected) => {
    expect(isValidBusinessCategory(value)).toBe(expected)
  })

  it.each([
    ['Has Spaces', false],
    ['123', false],
    ['', false],
    ['Customer', false],
    ['my.category', false],
  ])('rejects invalid category %s', (value, expected) => {
    expect(isValidBusinessCategory(value)).toBe(expected)
  })
})

describe('isBuiltInNodeType', () => {
  it('returns true for all built-in types', () => {
    for (const typ of BUILT_IN_NODE_TYPES) {
      expect(isBuiltInNodeType(typ)).toBe(true)
    }
  })

  it('returns false for custom types', () => {
    expect(isBuiltInNodeType('entity.invoice')).toBe(false)
    expect(isBuiltInNodeType('')).toBe(false)
  })
})

describe('isBuiltInEdgeType', () => {
  it('returns true for all built-in types', () => {
    for (const typ of BUILT_IN_EDGE_TYPES) {
      expect(isBuiltInEdgeType(typ)).toBe(true)
    }
  })

  it('returns false for custom types', () => {
    expect(isBuiltInEdgeType('ships_to')).toBe(false)
  })
})

describe('isBuiltInBusinessCategory', () => {
  it('returns true for all built-in categories', () => {
    for (const cat of BUSINESS_CATEGORIES) {
      expect(isBuiltInBusinessCategory(cat)).toBe(true)
    }
  })

  it('returns false for custom categories', () => {
    expect(isBuiltInBusinessCategory('custom_area')).toBe(false)
  })
})

describe('hasPrefix', () => {
  it.each([
    ['entity.customer', 'entity.'],
    ['rule.validation', 'rule.'],
    ['exception.override', 'exception.'],
    ['decision.branch', 'decision.'],
    ['process.workflow', 'process.'],
  ])('detects prefix for %s', (nodeType, expected) => {
    expect(hasPrefix(nodeType)).toBe(expected)
  })

  it.each([
    ['triggers', undefined],
    ['wrong.thing', undefined],
    ['', undefined],
  ])('returns undefined for %s', (nodeType, expected) => {
    expect(hasPrefix(nodeType)).toBe(expected)
  })
})

describe('validateNodeType', () => {
  it('returns undefined for valid types', () => {
    expect(validateNodeType('entity.customer')).toBeUndefined()
    expect(validateNodeType('rule.custom_thing')).toBeUndefined()
  })

  it('returns error message for invalid types', () => {
    const errMsg = validateNodeType('invalid')
    expect(errMsg).toBeDefined()
    expect(errMsg).toContain('invalid node type')
  })
})

describe('validateEdgeType', () => {
  it('returns undefined for valid types', () => {
    expect(validateEdgeType('triggers')).toBeUndefined()
    expect(validateEdgeType('ships_to')).toBeUndefined()
  })

  it('returns error message for invalid types', () => {
    const errMsg = validateEdgeType('INVALID')
    expect(errMsg).toBeDefined()
    expect(errMsg).toContain('invalid edge type')
  })
})

describe('validateBusinessCategory', () => {
  it('returns undefined for valid categories', () => {
    expect(validateBusinessCategory('customer')).toBeUndefined()
    expect(validateBusinessCategory('server_hardware')).toBeUndefined()
  })

  it('returns error message for invalid categories', () => {
    const errMsg = validateBusinessCategory('')
    expect(errMsg).toBeDefined()
    expect(errMsg).toContain('invalid business category')
  })
})

describe('validateTypeDefinition', () => {
  const validDef: OntologyTypeDefinition = {
    type: 'entity.customer',
    label: 'Customer',
    description: 'A customer entity',
    createdAt: '2026-01-01T00:00:00Z',
    status: 'active',
  }

  it('accepts a valid definition', () => {
    expect(validateTypeDefinition(validDef)).toBeUndefined()
  })

  it('accepts proposed status', () => {
    expect(validateTypeDefinition({ ...validDef, status: 'proposed' })).toBeUndefined()
  })

  it('accepts deprecated status', () => {
    expect(validateTypeDefinition({ ...validDef, status: 'deprecated' })).toBeUndefined()
  })

  it('rejects empty type', () => {
    expect(validateTypeDefinition({ ...validDef, type: '' })).toContain('empty type')
  })

  it('rejects empty label', () => {
    expect(validateTypeDefinition({ ...validDef, label: '' })).toContain('empty label')
  })

  it('rejects empty description', () => {
    expect(validateTypeDefinition({ ...validDef, description: '' })).toContain('empty description')
  })

  it('rejects empty createdAt', () => {
    expect(validateTypeDefinition({ ...validDef, createdAt: '' })).toContain('empty createdAt')
  })

  it('rejects invalid status', () => {
    const invalid = { ...validDef, status: 'invalid' as 'active' }
    expect(validateTypeDefinition(invalid)).toContain('invalid status')
  })
})

describe('formatNodeTypeLabel', () => {
  it.each([
    ['entity.customer', 'Customer (Entity)'],
    ['process.approval_chain', 'Approval Chain (Process)'],
    ['rule.classification', 'Classification (Rule)'],
    ['exception.special_case', 'Special Case (Exception)'],
    ['decision.escalation', 'Escalation (Decision)'],
    ['invalid', 'invalid'],
  ])('formats %s correctly', (input, expected) => {
    expect(formatNodeTypeLabel(input)).toBe(expected)
  })
})

describe('formatEdgeTypeLabel', () => {
  it.each([
    ['requires_approval_from', 'Requires Approval From'],
    ['triggers', 'Triggers'],
    ['feeds_into', 'Feeds Into'],
    ['related_to', 'Related To'],
  ])('formats %s correctly', (input, expected) => {
    expect(formatEdgeTypeLabel(input)).toBe(expected)
  })
})
