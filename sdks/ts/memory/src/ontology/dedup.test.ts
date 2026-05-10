// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from 'vitest'

import type { Embedder } from '../llm/types.js'
import type { TypeDefinition } from './types.js'

import { cosineSimilarity } from './similarity.js'
import {
  Deduplicator,
  deduplicateType,
  EMBEDDING_AUTO_MERGE_THRESHOLD,
  EMBEDDING_REVIEW_THRESHOLD,
  FUZZY_LABEL_THRESHOLD,
} from './dedup.js'

function makeType(type: string, label: string, description: string): TypeDefinition {
  return { type, label, description, createdAt: '2026-01-01', status: 'active' }
}

function createFakeEmbedder(dimensions: number, vectors: Map<string, number[]>): Embedder {
  const zeroVec = new Array<number>(dimensions).fill(0)
  return {
    name: () => 'fake-embedder',
    model: () => 'fake-model',
    dimension: () => dimensions,
    embed: async (texts: readonly string[]): Promise<number[][]> => {
      return texts.map((text) => vectors.get(text) ?? zeroVec)
    },
  }
}

describe('Deduplicator', () => {
  it('returns exact match when type ID is identical', async () => {
    const dedup = new Deduplicator({})
    const extracted = [makeType('entity.customer', 'Customer', 'A customer entity')]
    const existing = [makeType('entity.customer', 'Customer', 'A customer entity')]

    const result = await dedup.deduplicate(extracted, existing)

    expect(result.autoMerged).toHaveLength(1)
    expect(result.autoMerged[0]!.similarity).toBe(1.0)
    expect(result.autoMerged[0]!.method).toBe('exact')
    expect(result.unique).toHaveLength(0)
  })

  it('fuzzy matches within same prefix', async () => {
    const dedup = new Deduplicator({})
    const extracted = [makeType('entity.customer_record', 'Customer Record', 'A customer record')]
    const existing = [makeType('entity.customer_records', 'Customer Records', 'Customer records')]

    const result = await dedup.deduplicate(extracted, existing)

    expect(result.autoMerged).toHaveLength(1)
    expect(result.autoMerged[0]!.method).toBe('fuzzy_label')
    expect(result.autoMerged[0]!.similarity).toBeGreaterThanOrEqual(FUZZY_LABEL_THRESHOLD)
  })

  it('does not fuzzy match across different prefixes', async () => {
    const dedup = new Deduplicator({})
    const extracted = [makeType('entity.customer_records', 'Customer Records', 'Records')]
    const existing = [makeType('rule.customer_records', 'Customer Records', 'Records rule')]

    const result = await dedup.deduplicate(extracted, existing)

    expect(result.autoMerged).toHaveLength(0)
    expect(result.unique).toHaveLength(1)
  })

  it('returns unique when fuzzy similarity is below threshold', async () => {
    const dedup = new Deduplicator({})
    const extracted = [makeType('entity.product', 'Product', 'A product')]
    const existing = [makeType('entity.customer', 'Customer', 'A customer')]

    const result = await dedup.deduplicate(extracted, existing)

    expect(result.autoMerged).toHaveLength(0)
    expect(result.unique).toHaveLength(1)
  })

  it('semantic auto-merges when cosine >= 0.9', async () => {
    const vectors = new Map<string, number[]>([
      ['Client: A client entity', [0.95, 0.1, 0.05]],
      ['Customer: A customer type', [0.96, 0.1, 0.04]],
    ])
    const embedder = createFakeEmbedder(3, vectors)
    const dedup = new Deduplicator({ embedder })

    const extracted = [makeType('entity.client', 'Client', 'A client entity')]
    const existing = [makeType('entity.customer', 'Customer', 'A customer type')]

    const result = await dedup.deduplicate(extracted, existing)

    expect(result.autoMerged).toHaveLength(1)
    expect(result.autoMerged[0]!.method).toBe('embedding')
    expect(result.autoMerged[0]!.similarity).toBeGreaterThanOrEqual(EMBEDDING_AUTO_MERGE_THRESHOLD)
  })

  it('puts in review zone when cosine is between 0.75 and 0.9', async () => {
    // cosine([0.8, 0.4, 0.2], [0.5, 0.7, 0.5]) ~ 0.855
    const vectors = new Map<string, number[]>([
      ['Workflow: A workflow process', [0.8, 0.4, 0.2]],
      ['Pipeline: A pipeline stage', [0.5, 0.7, 0.5]],
    ])

    const similarity = cosineSimilarity([0.8, 0.4, 0.2], [0.5, 0.7, 0.5])
    if (similarity < EMBEDDING_REVIEW_THRESHOLD || similarity >= EMBEDDING_AUTO_MERGE_THRESHOLD) {
      return
    }

    const embedder = createFakeEmbedder(3, vectors)
    const dedup = new Deduplicator({ embedder })

    const extracted = [makeType('process.workflow', 'Workflow', 'A workflow process')]
    const existing = [makeType('process.pipeline', 'Pipeline', 'A pipeline stage')]

    const result = await dedup.deduplicate(extracted, existing)

    expect(result.reviewCandidates).toHaveLength(1)
    expect(result.reviewCandidates[0]!.method).toBe('embedding')
    expect(result.reviewCandidates[0]!.similarity).toBeGreaterThanOrEqual(
      EMBEDDING_REVIEW_THRESHOLD,
    )
    expect(result.reviewCandidates[0]!.similarity).toBeLessThan(EMBEDDING_AUTO_MERGE_THRESHOLD)
  })

  it('skips semantic tier when no embedder is provided', async () => {
    const dedup = new Deduplicator({})
    const extracted = [makeType('entity.client', 'Client', 'A client entity')]
    const existing = [makeType('entity.customer', 'Customer', 'A customer type')]

    const result = await dedup.deduplicate(extracted, existing)

    expect(result.unique).toHaveLength(1)
    expect(result.autoMerged).toHaveLength(0)
    expect(result.reviewCandidates).toHaveLength(0)
  })

  it('returns empty result for empty extracted list', async () => {
    const dedup = new Deduplicator({})
    const existing = [makeType('entity.customer', 'Customer', 'A customer')]

    const result = await dedup.deduplicate([], existing)

    expect(result.unique).toHaveLength(0)
    expect(result.autoMerged).toHaveLength(0)
    expect(result.reviewCandidates).toHaveLength(0)
  })

  it('treats all as unique when existing list is empty', async () => {
    const dedup = new Deduplicator({})
    const extracted = [
      makeType('entity.customer', 'Customer', 'A customer'),
      makeType('entity.product', 'Product', 'A product'),
    ]

    const result = await dedup.deduplicate(extracted, [])

    expect(result.unique).toHaveLength(2)
    expect(result.autoMerged).toHaveLength(0)
  })
})

describe('deduplicateType', () => {
  it('returns exact when type ID matches', async () => {
    const candidate = makeType('entity.customer', 'Customer', 'A customer')
    const existing = [makeType('entity.customer', 'Customer', 'A customer')]

    const result = await deduplicateType(candidate, existing)

    expect(result.kind).toBe('exact')
    expect(result.similarity).toBe(1.0)
    expect(result.existingType).toBeDefined()
  })

  it('returns fuzzy_match within same prefix', async () => {
    const candidate = makeType('entity.customer_record', 'Customer Record', 'A record')
    const existing = [makeType('entity.customer_records', 'Customer Records', 'Records')]

    const result = await deduplicateType(candidate, existing)

    expect(result.kind).toBe('fuzzy_match')
    expect(result.similarity).toBeGreaterThanOrEqual(FUZZY_LABEL_THRESHOLD)
  })

  it('returns new when prefix differs even if labels match', async () => {
    const candidate = makeType('entity.customer_records', 'Customer Records', 'Records')
    const existing = [makeType('rule.customer_records', 'Customer Records', 'Rule')]

    const result = await deduplicateType(candidate, existing)

    expect(result.kind).toBe('new')
  })

  it('returns semantic_match when cosine >= 0.9', async () => {
    const vectors = new Map<string, number[]>([
      ['Client: A client entity', [0.95, 0.1, 0.05]],
      ['Customer: A customer type', [0.96, 0.1, 0.04]],
    ])
    const embedder = createFakeEmbedder(3, vectors)

    const candidate = makeType('entity.client', 'Client', 'A client entity')
    const existing = [makeType('entity.customer', 'Customer', 'A customer type')]

    const result = await deduplicateType(candidate, existing, embedder)

    expect(result.kind).toBe('semantic_match')
    expect(result.similarity).toBeGreaterThanOrEqual(EMBEDDING_AUTO_MERGE_THRESHOLD)
  })

  it('returns semantic_review when cosine is in [0.75, 0.9)', async () => {
    // cosine([0.8, 0.4, 0.2], [0.5, 0.7, 0.5]) ~ 0.855
    const vectors = new Map<string, number[]>([
      ['Workflow: A workflow process', [0.8, 0.4, 0.2]],
      ['Pipeline: A pipeline stage', [0.5, 0.7, 0.5]],
    ])
    const embedder = createFakeEmbedder(3, vectors)

    const candidate = makeType('process.workflow', 'Workflow', 'A workflow process')
    const existing = [makeType('process.pipeline', 'Pipeline', 'A pipeline stage')]

    const result = await deduplicateType(candidate, existing, embedder)

    expect(result.kind).toBe('semantic_review')
    expect(result.similarity).toBeGreaterThanOrEqual(EMBEDDING_REVIEW_THRESHOLD)
    expect(result.similarity).toBeLessThan(EMBEDDING_AUTO_MERGE_THRESHOLD)
  })

  it('returns new when no embedder and no exact/fuzzy match', async () => {
    const candidate = makeType('entity.client', 'Client', 'A client entity')
    const existing = [makeType('entity.customer', 'Customer', 'A customer type')]

    const result = await deduplicateType(candidate, existing)

    expect(result.kind).toBe('new')
  })

  it('returns new for empty existing list', async () => {
    const candidate = makeType('entity.customer', 'Customer', 'A customer')

    const result = await deduplicateType(candidate, [])

    expect(result.kind).toBe('new')
  })
})
