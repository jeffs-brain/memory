// SPDX-License-Identifier: Apache-2.0

/**
 * Unit tests for the self-healing reconciliation module. Uses MockStore
 * and MockSearchIndex from the shared test helpers, plus a real SQLite
 * search index for integration-style assertions.
 */

import { afterEach, describe, expect, it } from 'vitest'
import { createHashEmbedder } from '../llm/hashembed.js'
import { type SearchIndex, createSearchIndex } from '../search/index.js'
import { type Path, toPath } from '../store/index.js'
import { createMemStore } from '../store/memstore.js'
import { type Reconciler, createReconciler } from './reconcile.js'
import { testLogger } from './test-helpers.js'

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const RAW_DOCUMENTS_PREFIX = 'raw/documents'

const indices: SearchIndex[] = []
const reconcilers: Reconciler[] = []

const fresh = async (dim = 32): Promise<SearchIndex> => {
  const idx = await createSearchIndex({ dbPath: ':memory:', vectorDim: dim })
  indices.push(idx)
  return idx
}

afterEach(async () => {
  for (const r of reconcilers) {
    r.stop()
  }
  reconcilers.length = 0
  while (indices.length > 0) {
    const idx = indices.pop()
    if (idx !== undefined) await idx.close()
  }
})

/**
 * Write a markdown document to the store at the canonical raw/documents
 * path, simulating a successful ingest store step.
 */
const writeStoreDoc = async (
  store: ReturnType<typeof createMemStore>,
  slug: string,
  body: string,
): Promise<string> => {
  const path = `${RAW_DOCUMENTS_PREFIX}/${slug}.md`
  const content = `---\ntitle: "${slug}"\n---\n\n${body}\n`
  await store.write(toPath(path), Buffer.from(content, 'utf8'))
  return path
}

/**
 * Write a document to the store AND index it in the search index,
 * simulating a complete ingest.
 */
const indexDoc = async (
  store: ReturnType<typeof createMemStore>,
  searchIndex: SearchIndex,
  slug: string,
  body: string,
): Promise<string> => {
  const path = await writeStoreDoc(store, slug, body)
  const embedder = createHashEmbedder({ dim: 32 })
  const texts = [body]
  const embeddings = await embedder.embed(texts)
  searchIndex.upsertChunks(
    texts.map((text, i) => ({
      id: `${slug}:${i}`,
      path,
      ordinal: i,
      title: slug,
      content: text,
      metadata: { documentId: slug, brainId: 'test' },
      ...(embeddings[i] !== undefined ? { embedding: embeddings[i] } : {}),
    })),
  )
  return path
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('createReconciler', () => {
  it('detects documents in store but missing from search index', async () => {
    const store = createMemStore()
    const searchIndex = await fresh()
    const reindexed: string[] = []

    await writeStoreDoc(store, 'missing-doc', 'This document was stored but never indexed.')

    const reconciler = createReconciler({
      store,
      searchIndex,
      logger: testLogger,
      reindexFn: async (path) => {
        reindexed.push(path)
      },
    })
    reconcilers.push(reconciler)

    const report = await reconciler.runOnce()

    expect(report.totalDocuments).toBe(1)
    expect(report.driftDetected).toBe(true)
    expect(report.missingReindexed).toBe(1)
    expect(report.orphanedDeleted).toBe(0)
    expect(reindexed).toHaveLength(1)
    expect(reindexed[0]).toContain('missing-doc')
  })

  it('detects orphaned index entries with no corresponding store document', async () => {
    const store = createMemStore()
    const searchIndex = await fresh()

    // Index a document, then remove it from the store.
    const path = await indexDoc(store, searchIndex, 'orphan-doc', 'Content that will be orphaned.')
    await store.delete(toPath(path))

    const reconciler = createReconciler({
      store,
      searchIndex,
      logger: testLogger,
    })
    reconcilers.push(reconciler)

    const report = await reconciler.runOnce()

    expect(report.driftDetected).toBe(true)
    expect(report.orphanedDeleted).toBeGreaterThanOrEqual(1)
    expect(report.missingReindexed).toBe(0)

    // Verify the orphaned entry was removed.
    const remaining = searchIndex.indexedPaths()
    expect(remaining).not.toContain(path)
  })

  it('repairs missing documents by calling reindexFn', async () => {
    const store = createMemStore()
    const searchIndex = await fresh()
    const reindexed: string[] = []

    await writeStoreDoc(store, 'repair-alpha', 'Alpha content for repair test.')
    await writeStoreDoc(store, 'repair-beta', 'Beta content for repair test.')

    const reconciler = createReconciler({
      store,
      searchIndex,
      logger: testLogger,
      reindexFn: async (path) => {
        reindexed.push(path)
        // Simulate re-indexing by actually inserting a chunk.
        const embedder = createHashEmbedder({ dim: 32 })
        const text = (await store.read(toPath(path) as Path)).toString('utf8')
        const embeddings = await embedder.embed([text])
        searchIndex.upsertChunks([
          {
            id: `reindex:${path}:0`,
            path,
            ordinal: 0,
            title: path,
            content: text,
            metadata: { documentId: path, brainId: 'test' },
            ...(embeddings[0] !== undefined ? { embedding: embeddings[0] } : {}),
          },
        ])
      },
    })
    reconcilers.push(reconciler)

    const report = await reconciler.runOnce()

    expect(report.missingReindexed).toBe(2)
    expect(reindexed).toHaveLength(2)
  })

  it('removes orphaned chunks from the index', async () => {
    const store = createMemStore()
    const searchIndex = await fresh()

    const path1 = await indexDoc(store, searchIndex, 'orphan-one', 'First orphan document.')
    const path2 = await indexDoc(store, searchIndex, 'orphan-two', 'Second orphan document.')

    // Delete both documents from the store.
    await store.delete(toPath(path1))
    await store.delete(toPath(path2))

    const reconciler = createReconciler({
      store,
      searchIndex,
      logger: testLogger,
    })
    reconcilers.push(reconciler)

    const report = await reconciler.runOnce()

    expect(report.orphanedDeleted).toBeGreaterThanOrEqual(2)

    const remaining = searchIndex.indexedPaths()
    expect(remaining).not.toContain(path1)
    expect(remaining).not.toContain(path2)
  })

  it('prevents concurrent reconciliation via brain-level lock', async () => {
    const store = createMemStore()
    const searchIndex = await fresh()
    let reindexCount = 0

    await writeStoreDoc(store, 'concurrent-doc', 'Content for concurrency test.')

    const reconciler = createReconciler({
      store,
      searchIndex,
      logger: testLogger,
      reindexFn: async () => {
        // Simulate slow re-indexing.
        await new Promise((resolve) => setTimeout(resolve, 50))
        reindexCount++
      },
    })
    reconcilers.push(reconciler)

    // Launch multiple reconciliation runs in parallel.
    const results = await Promise.all([
      reconciler.runOnce(),
      reconciler.runOnce(),
      reconciler.runOnce(),
    ])

    // At most one should have performed actual work.
    const workingRuns = results.filter((r) => r.driftDetected)
    expect(workingRuns.length).toBeLessThanOrEqual(1)
  })

  it('is idempotent: running twice produces the same result', async () => {
    const store = createMemStore()
    const searchIndex = await fresh()

    await writeStoreDoc(store, 'idempotent-doc', 'Idempotent reconciliation test content.')

    const reconciler = createReconciler({
      store,
      searchIndex,
      logger: testLogger,
      reindexFn: async (path) => {
        // Simulate re-indexing by inserting a chunk.
        const text = (await store.read(toPath(path) as Path)).toString('utf8')
        const embedder = createHashEmbedder({ dim: 32 })
        const embeddings = await embedder.embed([text])
        searchIndex.upsertChunks([
          {
            id: `idempotent:${path}:0`,
            path,
            ordinal: 0,
            title: path,
            content: text,
            metadata: { documentId: path, brainId: 'test' },
            ...(embeddings[0] !== undefined ? { embedding: embeddings[0] } : {}),
          },
        ])
      },
    })
    reconcilers.push(reconciler)

    // First run: should repair the missing document.
    const report1 = await reconciler.runOnce()
    expect(report1.missingReindexed).toBe(1)
    expect(report1.driftDetected).toBe(true)

    // Second run: should find everything aligned.
    const report2 = await reconciler.runOnce()
    expect(report2.driftDetected).toBe(false)
    expect(report2.missingReindexed).toBe(0)
    expect(report2.orphanedDeleted).toBe(0)
  })

  it('reports no drift when store and index are both empty', async () => {
    const store = createMemStore()
    const searchIndex = await fresh()

    const reconciler = createReconciler({
      store,
      searchIndex,
      logger: testLogger,
    })
    reconcilers.push(reconciler)

    const report = await reconciler.runOnce()

    expect(report.totalDocuments).toBe(0)
    expect(report.totalIndexed).toBe(0)
    expect(report.driftDetected).toBe(false)
    expect(report.missingReindexed).toBe(0)
    expect(report.orphanedDeleted).toBe(0)
    expect(report.errors).toBe(0)
  })

  it('respects maxRepairs circuit breaker', async () => {
    const store = createMemStore()
    const searchIndex = await fresh()
    let reindexCount = 0

    // Write 5 documents without indexing.
    for (let i = 0; i < 5; i++) {
      await writeStoreDoc(store, `breaker-${i}`, `Breaker content ${i}.`)
    }

    const reconciler = createReconciler({
      store,
      searchIndex,
      logger: testLogger,
      maxRepairs: 2,
      reindexFn: async () => {
        reindexCount++
      },
    })
    reconcilers.push(reconciler)

    const report = await reconciler.runOnce()

    // Should cap at 2 repairs.
    expect(report.missingReindexed).toBeLessThanOrEqual(2)
    expect(reindexCount).toBeLessThanOrEqual(2)
  })

  it('counts errors when reindexFn throws', async () => {
    const store = createMemStore()
    const searchIndex = await fresh()

    await writeStoreDoc(store, 'error-doc', 'Document that will fail re-indexing.')

    const reconciler = createReconciler({
      store,
      searchIndex,
      logger: testLogger,
      reindexFn: async () => {
        throw new Error('simulated re-index failure')
      },
    })
    reconcilers.push(reconciler)

    const report = await reconciler.runOnce()

    expect(report.driftDetected).toBe(true)
    expect(report.errors).toBe(1)
    expect(report.missingReindexed).toBe(0)
  })

  it('reports accurate drift metrics for mixed missing and orphaned', async () => {
    const store = createMemStore()
    const searchIndex = await fresh()
    const reindexed: string[] = []

    // Create a document in store but not indexed (missing).
    await writeStoreDoc(store, 'mixed-missing', 'Missing from index.')

    // Create a document in both store and index, then delete from store (orphaned).
    const orphanPath = await indexDoc(
      store,
      searchIndex,
      'mixed-orphan',
      'Will become orphaned.',
    )
    await store.delete(toPath(orphanPath))

    const reconciler = createReconciler({
      store,
      searchIndex,
      logger: testLogger,
      reindexFn: async (path) => {
        reindexed.push(path)
      },
    })
    reconcilers.push(reconciler)

    const report = await reconciler.runOnce()

    expect(report.driftDetected).toBe(true)
    expect(report.missingReindexed).toBe(1)
    expect(report.orphanedDeleted).toBeGreaterThanOrEqual(1)
    expect(report.totalDocuments).toBe(1)
    expect(reindexed).toHaveLength(1)
  })
})
