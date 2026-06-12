// SPDX-License-Identifier: Apache-2.0

import type { Embedder } from '@jeffs-brain/memory/llm'
import { describe, expect, it } from 'vitest'
import {
  type PostgresRetriever,
  type PostgresSearchLike,
  createPostgresRetriever,
} from './retrieval.js'
import type { PgSql } from './store.js'

/**
 * Unit coverage for the query-embedding deadline. `createPostgresRetriever`
 * accepts an injected embedder and search index, so these tests exercise the
 * full retrieve -> embed call chain without a database. They prove the
 * caller-supplied `AbortSignal` is threaded down to `embedder.embed`, so a
 * degraded embeddings service cannot run past the request deadline.
 */

const lexicalOnlyIndex = (): PostgresSearchLike => ({
  searchBM25: async () => [],
  hasEmbeddings: async () => true,
  searchVector: async () => [],
  getChunk: async () => undefined,
})

const buildRetriever = (input: {
  readonly embedder: Embedder
  readonly index?: PostgresSearchLike
}): PostgresRetriever =>
  createPostgresRetriever({
    // The pg client is never touched: the injected index short-circuits all DB
    // access and the injected embedder replaces the network embed call.
    pg: {} as unknown as PgSql,
    tenantId: 'tenant-test',
    brainId: 'brain-test',
    env: {},
    indexFactory: () => input.index ?? lexicalOnlyIndex(),
    embedderFactory: () => input.embedder,
  })

describe('createPostgresRetriever query-embedding deadline', () => {
  it('threads the request signal down to embedder.embed', async () => {
    let received: AbortSignal | undefined
    const embedder: Embedder = {
      name: () => 'spy',
      model: () => 'spy',
      dimension: () => 0,
      embed: async (_texts, signal) => {
        received = signal
        return [[0.1, 0.2, 0.3]]
      },
    }
    const controller = new AbortController()
    const retriever = buildRetriever({ embedder })

    await retriever.retrieve({
      query: 'invoice processing',
      limit: 5,
      mode: 'semantic',
      signal: controller.signal,
    })

    expect(received).toBe(controller.signal)
  })

  it('aborts the in-flight embed promptly when the signal is already aborted', async () => {
    const controller = new AbortController()
    controller.abort()

    let resolved = false
    const embedder: Embedder = {
      name: () => 'honours-signal',
      model: () => 'honours-signal',
      dimension: () => 0,
      // A signal-honouring embedder (TEI/OpenAI both forward the signal to
      // fetch) rejects immediately when the signal is already aborted, rather
      // than running the network round-trip to completion.
      embed: async (_texts, signal) => {
        if (signal?.aborted === true) {
          throw new DOMException('aborted', 'AbortError')
        }
        await new Promise((r) => setTimeout(r, 10_000))
        resolved = true
        return [[0.1, 0.2, 0.3]]
      },
    }
    const retriever = buildRetriever({ embedder })

    const response = await retriever.retrieve({
      query: 'invoice processing',
      limit: 5,
      mode: 'semantic',
      signal: controller.signal,
    })

    // The aborted embed degrades gracefully to lexical (matching the existing
    // catch-and-fall-back behaviour) and never runs the 10s embed to
    // completion. Without the threading the embedder would not see the aborted
    // signal and would hang for the full timeout.
    expect(resolved).toBe(false)
    expect(response.trace.fellBackToLexical).toBe(true)
    expect(response.trace.effectiveMode).toBe('lexical')
  })

  it('omitting the signal leaves embed behaviour unchanged (backwards compatible)', async () => {
    let receivedArgCount = -1
    const embedder: Embedder = {
      name: () => 'spy',
      model: () => 'spy',
      dimension: () => 0,
      embed: async (_texts, signal) => {
        receivedArgCount = signal === undefined ? 1 : 2
        return [[0.1, 0.2, 0.3]]
      },
    }
    const retriever = buildRetriever({ embedder })

    await retriever.retrieve({
      query: 'invoice processing',
      limit: 5,
      mode: 'semantic',
    })

    expect(receivedArgCount).toBe(1)
  })
})
