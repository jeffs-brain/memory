// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from 'vitest'
import { getBuiltInNodeTypeDescription, getBuiltInEdgeTypeDescription } from './descriptions.js'

describe('getBuiltInNodeTypeDescription', () => {
  it('returns the description for entity.customer', () => {
    const desc = getBuiltInNodeTypeDescription('entity.customer')
    expect(desc).toBe('Customers, clients, accounts, subscribers, or end-users of a service')
  })

  it('returns the description for entity.product', () => {
    const desc = getBuiltInNodeTypeDescription('entity.product')
    expect(desc).toBe('Products, goods, SKUs, services offered, or catalogue items')
  })

  it('returns the description for rule.validation', () => {
    const desc = getBuiltInNodeTypeDescription('rule.validation')
    expect(desc).toBe('Data validation checks, format requirements, or input verification')
  })

  it('returns the description for process.workflow', () => {
    const desc = getBuiltInNodeTypeDescription('process.workflow')
    expect(desc).toBe('End-to-end workflows, automated sequences, or pipelines')
  })

  it('returns the description for decision.escalation', () => {
    const desc = getBuiltInNodeTypeDescription('decision.escalation')
    expect(desc).toBe('Escalation paths, priority handling, or SLA breach responses')
  })

  it('returns a generic fallback for an unknown node type', () => {
    const desc = getBuiltInNodeTypeDescription('unknown.type')
    expect(desc).toBe('A business intelligence node type')
  })

  it('returns a generic fallback for an empty string', () => {
    const desc = getBuiltInNodeTypeDescription('')
    expect(desc).toBe('A business intelligence node type')
  })
})

describe('getBuiltInEdgeTypeDescription', () => {
  it('returns the description for triggers', () => {
    const desc = getBuiltInEdgeTypeDescription('triggers')
    expect(desc).toBe('Source causes target to start or execute')
  })

  it('returns the description for depends_on', () => {
    const desc = getBuiltInEdgeTypeDescription('depends_on')
    expect(desc).toBe('Source requires target to be completed or available first')
  })

  it('returns the description for validates', () => {
    const desc = getBuiltInEdgeTypeDescription('validates')
    expect(desc).toBe('Source checks or confirms the correctness of target')
  })

  it('returns the description for overrides', () => {
    const desc = getBuiltInEdgeTypeDescription('overrides')
    expect(desc).toBe('Source takes precedence over or replaces target')
  })

  it('returns a generic fallback for an unknown edge type', () => {
    const desc = getBuiltInEdgeTypeDescription('unknown_edge')
    expect(desc).toBe('A relationship between intelligence nodes')
  })

  it('returns a generic fallback for an empty string', () => {
    const desc = getBuiltInEdgeTypeDescription('')
    expect(desc).toBe('A relationship between intelligence nodes')
  })
})
