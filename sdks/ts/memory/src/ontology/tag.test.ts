// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from 'vitest'
import type { ResolvedOntology, ResolvedType } from './store.js'
import type { ClassificationResult } from './classify.js'
import { tagChunk, tagChunks } from './tag.js'
import { Classifier } from './classify.js'

function sampleOntology(): ResolvedOntology {
  return {
    nodeTypes: [
      {
        type: 'entity.customer',
        label: 'Customer (Entity)',
        description: 'A customer entity',
        createdAt: '2026-01-01T00:00:00Z',
        status: 'active',
        scope: 'built-in',
      } as ResolvedType,
      {
        type: 'entity.product',
        label: 'Product (Entity)',
        description: 'A product entity',
        createdAt: '2026-01-01T00:00:00Z',
        status: 'active',
        scope: 'built-in',
      } as ResolvedType,
      {
        type: 'process.workflow',
        label: 'Workflow (Process)',
        description: 'A workflow process',
        createdAt: '2026-01-01T00:00:00Z',
        status: 'active',
        scope: 'built-in',
      } as ResolvedType,
      {
        type: 'rule.validation',
        label: 'Validation (Rule)',
        description: 'A validation rule',
        createdAt: '2026-01-01T00:00:00Z',
        status: 'active',
        scope: 'built-in',
      } as ResolvedType,
    ],
    edgeTypes: [
      {
        type: 'triggers',
        label: 'Triggers',
        description: 'Triggers relationship',
        createdAt: '2026-01-01T00:00:00Z',
        status: 'active',
        scope: 'built-in',
      } as ResolvedType,
    ],
    businessCategories: ['customer', 'order', 'product', 'general'],
  }
}

describe('tagChunk', () => {
  it('tags with ontology entity types', () => {
    const ont = sampleOntology()
    const classification: ClassificationResult = {
      class: 'unstructured',
      category: 'general',
      confidence: 0.8,
      isStructured: false,
    }

    const content = 'The customer submitted a product order through the workflow.'
    const tag = tagChunk(content, classification, ont)

    expect(tag.documentClass).toBe('unstructured')
    expect(tag.confidence).toBe(0.8)
    expect(tag.entityTypes).toBeDefined()
    expect(tag.entityTypes).toContain('entity.customer')
    expect(tag.entityTypes).toContain('entity.product')
    expect(tag.entityTypes).toContain('process.workflow')
  })

  it('preserves business category from classification', () => {
    const ont = sampleOntology()
    const classification: ClassificationResult = {
      class: 'structured',
      category: 'customer',
      confidence: 0.95,
      isStructured: true,
    }

    const content = 'Customer account details for enterprise clients.'
    const tag = tagChunk(content, classification, ont)
    expect(tag.businessCategory).toBeTruthy()
  })

  it('works with minimal classification (no ontology)', () => {
    const classification: ClassificationResult = {
      class: 'unstructured',
      category: 'general',
      confidence: 0.5,
      isStructured: false,
    }

    const content = 'A simple text with no known ontology terms.'
    const tag = tagChunk(content, classification, undefined)

    expect(tag.documentClass).toBe('unstructured')
    expect(tag.businessCategory).toBe('general')
    expect(tag.entityTypes).toBeUndefined()
  })

  it('returns no entity types when no matches found', () => {
    const ont = sampleOntology()
    const classification: ClassificationResult = {
      class: 'unstructured',
      category: 'general',
      confidence: 0.5,
      isStructured: false,
    }

    const content = 'This text mentions nothing from the ontology at all.'
    const tag = tagChunk(content, classification, ont)
    expect(tag.entityTypes).toBeUndefined()
  })
})

describe('tagChunks', () => {
  it('batch tags multiple chunks', () => {
    const ont = sampleOntology()
    const classification: ClassificationResult = {
      class: 'unstructured',
      category: 'general',
      confidence: 0.7,
      isStructured: false,
    }

    const contents = [
      'The customer placed an order.',
      'Product validation rules apply.',
      'No relevant terms here.',
    ]

    const tags = tagChunks(contents, classification, ont)
    expect(tags).toHaveLength(3)

    // First chunk: customer
    expect(tags[0]!.entityTypes).toContain('entity.customer')

    // Second chunk: product and validation
    expect(tags[1]!.entityTypes).toContain('entity.product')
    expect(tags[1]!.entityTypes).toContain('rule.validation')
  })
})

describe('classify and tag end-to-end', () => {
  it('classifies JSON and tags with entity types', async () => {
    const c = new Classifier({})
    const ont = sampleOntology()

    const content = '{"customer_id": 1, "customer_name": "Acme", "product": "Widget"}'
    const result = await c.classify(content, 'data.json')

    const tag = tagChunk(content, result, ont)
    expect(tag.documentClass).toBe('structured')
    expect(tag.confidence).toBeGreaterThanOrEqual(0.8)
    expect(tag.entityTypes).toContain('entity.customer')
    expect(tag.entityTypes).toContain('entity.product')
  })
})
