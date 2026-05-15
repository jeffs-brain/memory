// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from 'vitest'
import type { Provider, CompletionResponse } from '../llm/types.js'
import type { ResolvedOntology, ResolvedType } from './store.js'
import {
  Classifier,
  isJsonDocument,
  isTabularDocument,
  determineCategory,
  CATEGORY_WINNER_THRESHOLD,
} from './classify.js'

function fakeProvider(response: string): Provider {
  return {
    name: () => 'fake',
    modelName: () => 'fake-model',
    stream: () => {
      throw new Error('not implemented')
    },
    complete: async (): Promise<CompletionResponse> => ({
      content: response,
      toolCalls: [],
      usage: { inputTokens: 0, outputTokens: 0 },
      stopReason: 'end_turn',
    }),
    supportsStructuredDecoding: () => false,
    structured: async () => response,
  }
}

describe('isJsonDocument', () => {
  it('returns true for valid JSON object', () => {
    expect(isJsonDocument('{"rules": [{"name": "discount"}]}')).toBe(true)
  })

  it('returns true for valid JSON array', () => {
    expect(isJsonDocument('[{"name": "Alice"}, {"name": "Bob"}]')).toBe(true)
  })

  it('returns false for prose text', () => {
    expect(isJsonDocument('This is just some regular text content.')).toBe(false)
  })

  it('returns false for bare primitives', () => {
    expect(isJsonDocument('42')).toBe(false)
    expect(isJsonDocument('"hello"')).toBe(false)
  })

  it('returns false for empty string', () => {
    expect(isJsonDocument('')).toBe(false)
  })

  it('returns false for primitive arrays', () => {
    expect(isJsonDocument('[1, 2, 3]')).toBe(false)
    expect(isJsonDocument('["a", "b", "c"]')).toBe(false)
  })

  it('returns false for empty structures', () => {
    expect(isJsonDocument('{}')).toBe(false)
    expect(isJsonDocument('[]')).toBe(false)
  })
})

describe('isTabularDocument', () => {
  it('returns true for CSV content', () => {
    const content = 'name,age,city\nAlice,30,NYC\nBob,25,LA\nCharlie,35,SF\n'
    expect(isTabularDocument(content)).toBe(true)
  })

  it('returns true for pipe-delimited content', () => {
    const content = 'name|age|city\nAlice|30|NYC\nBob|25|LA\nCharlie|35|SF\n'
    expect(isTabularDocument(content)).toBe(true)
  })

  it('returns false for prose text', () => {
    const content = 'This is a paragraph.\nAnother paragraph follows.\nNo delimiters here.\n'
    expect(isTabularDocument(content)).toBe(false)
  })
})

describe('Classifier', () => {
  describe('classify', () => {
    it('detects JSON as structured', async () => {
      const c = new Classifier({})
      const result = await c.classify('{"customers": [{"id": 1}]}', 'data.json')
      expect(result.class).toBe('structured')
      expect(result.isStructured).toBe(true)
      expect(result.confidence).toBeGreaterThanOrEqual(0.9)
    })

    it('detects CSV as tabular', async () => {
      const c = new Classifier({})
      const content = 'name,age,city\nAlice,30,NYC\nBob,25,LA\nCharlie,35,SF\n'
      const result = await c.classify(content, 'contacts.csv')
      expect(result.class).toBe('tabular')
      expect(result.isStructured).toBe(true)
    })

    it('falls back to LLM for unstructured content', async () => {
      const provider = fakeProvider('{"category": "entity", "confidence": 0.85, "reasoning": "customer data"}')
      const c = new Classifier({ provider })
      const content = 'The customer approval process requires manager sign-off for orders above $10,000.'
      const result = await c.classify(content, 'approval-rules.md')
      expect(result.class).toBe('unstructured')
      expect(result.confidence).toBeGreaterThan(0)
    })

    it('returns default without provider', async () => {
      const c = new Classifier({})
      const content = 'The customer approval process requires manager sign-off.'
      const result = await c.classify(content, 'doc.md')
      expect(result.class).toBe('unstructured')
      expect(result.category).toBe('general')
    })

    it('infers customer category from JSON keywords', async () => {
      const c = new Classifier({})
      const content = '{"customer_id": 123, "customer_name": "Acme Corp", "account_type": "enterprise"}'
      const result = await c.classify(content, 'data.json')
      expect(result.category).toBe('customer')
    })
  })
})

describe('determineCategory', () => {
  const sampleOntology: ResolvedOntology = {
    nodeTypes: [
      {
        type: 'entity.customer',
        label: 'Customer (Entity)',
        description: 'Customer entity for customer management',
        createdAt: '2026-01-01T00:00:00Z',
        status: 'active',
        scope: 'built-in',
      } as ResolvedType,
    ],
    edgeTypes: [],
    businessCategories: ['customer', 'order', 'general'],
  }

  it('returns specific category from ontology analysis', () => {
    const content = 'The customer submitted a new order for processing.'
    const category = determineCategory(content, sampleOntology)
    // Content contains "customer" which matches entity.customer whose
    // description contains "customer" (a business category).
    expect(category).toBe('customer')
  })

  it('returns general for undefined ontology', () => {
    expect(determineCategory('some content', undefined)).toBe('general')
  })
})

describe('LLM category mapping', () => {
  it('preserves LLM process category instead of mapping to general', async () => {
    const provider = fakeProvider('{"category": "process", "confidence": 0.9, "reasoning": "describes a workflow"}')
    const c = new Classifier({ provider })
    const content = 'The approval workflow requires two levels of sign-off before any purchase order is released.'
    const result = await c.classify(content, 'workflow.md')
    expect(result.category).toBe('process')
  })

  it('preserves LLM entity category', async () => {
    const provider = fakeProvider('{"category": "entity", "confidence": 0.85, "reasoning": "contains customer data"}')
    const c = new Classifier({ provider })
    const content = 'The customer approval process requires manager sign-off for orders above $10,000.'
    const result = await c.classify(content, 'approval-rules.md')
    expect(result.category).toBe('entity')
  })
})

describe('constants', () => {
  it('CATEGORY_WINNER_THRESHOLD matches intelligence service', () => {
    expect(CATEGORY_WINNER_THRESHOLD).toBe(0.4)
  })
})
