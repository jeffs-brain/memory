// SPDX-License-Identifier: Apache-2.0

import { execSync } from 'node:child_process'
import { readFileSync, readdirSync } from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { toPath } from '@jeffs-brain/memory/store'
import postgres from 'postgres'
import { afterAll, beforeAll, describe, expect, it } from 'vitest'
import { createPostgresSearchIndex } from './search.js'
import { createPostgresStore } from './store.js'
import type { PgSql } from './store.js'

const hasDocker = (() => {
  try {
    execSync('docker info', { stdio: 'ignore' })
    return true
  } catch {
    return false
  }
})()

const maybe = hasDocker ? describe : describe.skip

const makeUnitVector = (dimension: number, activeIndex: number): number[] => {
  const vector = new Array<number>(dimension).fill(0)
  vector[activeIndex] = 1
  return vector
}

maybe('PostgresSearchIndex (testcontainers)', () => {
  type RawSql = ReturnType<typeof postgres>
  let sql: RawSql
  let stop: () => Promise<void>
  const tenantA = '11111111-1111-1111-1111-111111111111'
  const brainA = 'aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa'

  beforeAll(async () => {
    const { PostgreSqlContainer } = await import('@testcontainers/postgresql')
    const container = await new PostgreSqlContainer('pgvector/pgvector:pg17')
      .withDatabase('jeffs_brain')
      .withUsername('postgres')
      .withPassword('postgres')
      .start()

    sql = postgres(container.getConnectionUri(), { max: 2, prepare: false })

    const here = path.dirname(fileURLToPath(import.meta.url))
    const migrationsDir = path.join(here, '..', 'migrations')
    const migrations = readdirSync(migrationsDir)
      .filter((entry) => /^\d+_.*\.sql$/.test(entry))
      .sort()
      .map((entry) => readFileSync(path.join(migrationsDir, entry), 'utf8'))
    for (const migration of migrations) {
      await sql.unsafe(migration).simple()
    }

    await sql.unsafe(
      `insert into platform.tenants (tenant_id, slug, name, region) values
         ('${tenantA}', 'tenant-a', 'Tenant A', 'eu-fsn1')`,
    )
    await sql.unsafe(
      `insert into memory.brains (brain_id, tenant_id, slug, name) values
         ('${brainA}', '${tenantA}', 'brain-a', 'Brain A')`,
    )

    stop = async () => {
      await sql.end({ timeout: 5 })
      await container.stop()
    }
  }, 120_000)

  afterAll(async () => {
    if (stop) await stop()
  })

  it('BM25 + vector + hybrid roundtrip', async () => {
    // Seed a parent document so chunks have a document_id to join on.
    const store = await createPostgresStore({
      sql: sql as unknown as PgSql,
      tenantId: tenantA,
      brainId: brainA,
    })
    await store.write(toPath('doc.md'), Buffer.from('seed'))

    // Pull document_id back.
    const rows = (await sql<{ document_id: string }[]>`
      select document_id::text as document_id from memory.documents
      where brain_id = ${brainA}::uuid and path = 'doc.md'
    `) as ReadonlyArray<{ document_id: string }>
    const documentId = rows[0]?.document_id
    expect(documentId).toBeDefined()

    const index = createPostgresSearchIndex({
      sql: sql as unknown as PgSql,
      tenantId: tenantA,
      brainId: brainA,
    })

    const vectorDim = 1024
    // Five chunks; chunk 3 is the BM25 target, chunk 1 is the vector target.
    const texts = [
      'the quick brown fox jumps over the lazy dog',
      'postgres row level security with tenant isolation',
      'hybrid retrieval blends bm25 and vector search via rrf fusion',
      'cats sleep on warm radiators in winter',
      'a gentle introduction to full text search in sqlite',
    ]
    const baseEmbeddings: number[][] = texts.map((_, i) => makeUnitVector(vectorDim, i % vectorDim))

    for (let i = 0; i < texts.length; i += 1) {
      await index.upsert({
        chunkId: BigInt(i + 1),
        documentId: documentId as string,
        ordinal: i,
        content: texts[i] as string,
        tokens: (texts[i] as string).split(/\s+/).length,
        embedding: baseEmbeddings[i] as number[],
      })
    }

    // BM25 for a distinctive term hits exactly one chunk.
    const bm25 = await index.searchBM25('rrf fusion', 5)
    expect(bm25.length).toBeGreaterThan(0)
    expect(bm25[0]?.chunkId).toBe(3n)

    // Vector query equal to chunk 1's embedding → nearest is chunk 1.
    const query = makeUnitVector(vectorDim, 0)
    const vec = await index.searchVector(query, 5)
    expect(vec.length).toBeGreaterThan(0)
    expect(vec[0]?.chunkId).toBe(1n)

    // Hybrid: produces a non-empty RRF ordering covering both winners.
    const hybrid = await index.searchHybrid({
      query: 'rrf fusion',
      embedding: query,
      limit: 10,
    })
    expect(hybrid.length).toBeGreaterThan(0)
    const ids = hybrid.map((r) => r.chunkId)
    expect(ids).toContain(3n)
    expect(ids).toContain(1n)
    // RRF scores are strictly decreasing (monotone non-increasing).
    for (let i = 1; i < hybrid.length; i += 1) {
      const prev = hybrid[i - 1]?.rrfScore ?? 0
      const cur = hybrid[i]?.rrfScore ?? 0
      expect(cur).toBeLessThanOrEqual(prev)
    }

    await index.close()
    await store.close()
  }, 120_000)

  it('supports the 3072-dim embedding profile', async () => {
    const store = await createPostgresStore({
      sql: sql as unknown as PgSql,
      tenantId: tenantA,
      brainId: brainA,
    })
    await store.write(toPath('profile-3072.md'), Buffer.from('seed'))

    const rows = (await sql<{ document_id: string }[]>`
      select document_id::text as document_id from memory.documents
      where brain_id = ${brainA}::uuid and path = 'profile-3072.md'
    `) as ReadonlyArray<{ document_id: string }>
    const documentId = rows[0]?.document_id
    expect(documentId).toBeDefined()

    const index = createPostgresSearchIndex({
      sql: sql as unknown as PgSql,
      tenantId: tenantA,
      brainId: brainA,
      vectorDim: 3072,
    })

    const vectorDim = 3072
    const query = makeUnitVector(vectorDim, 17)
    const chunkEmbedding = makeUnitVector(vectorDim, 17)

    await index.upsert({
      chunkId: 301n,
      documentId: documentId as string,
      ordinal: 0,
      content: 'large embedding profile support',
      tokens: 4,
      embedding: chunkEmbedding,
    })

    const hits = await index.searchVector(query, 5)
    expect(hits.length).toBeGreaterThan(0)
    expect(hits[0]?.chunkId).toBe(301n)

    await expect(
      index.upsert({
        chunkId: 302n,
        documentId: documentId as string,
        ordinal: 1,
        content: 'mismatched embedding',
        tokens: 2,
        embedding: makeUnitVector(1024, 3),
      }),
    ).rejects.toThrow(/3072/)

    await index.close()
    await store.close()
  }, 120_000)

  it('honours phrase, boolean, and prefix query syntax on the BM25 path', async () => {
    const store = await createPostgresStore({
      sql: sql as unknown as PgSql,
      tenantId: tenantA,
      brainId: brainA,
    })
    await store.write(toPath('syntax.md'), Buffer.from('seed'))

    const rows = (await sql<{ document_id: string }[]>`
      select document_id::text as document_id from memory.documents
      where brain_id = ${brainA}::uuid and path = 'syntax.md'
    `) as ReadonlyArray<{ document_id: string }>
    const documentId = rows[0]?.document_id
    expect(documentId).toBeDefined()

    const index = createPostgresSearchIndex({
      sql: sql as unknown as PgSql,
      tenantId: tenantA,
      brainId: brainA,
    })

    await index.upsertMany([
      {
        chunkId: 101n,
        documentId: documentId as string,
        ordinal: 0,
        content: 'rapid transit planning for Dutch cities',
        tokens: 6,
      },
      {
        chunkId: 102n,
        documentId: documentId as string,
        ordinal: 1,
        content: 'transit planning notes for regional rail',
        tokens: 6,
      },
      {
        chunkId: 103n,
        documentId: documentId as string,
        ordinal: 2,
        content: 'rapid road building project',
        tokens: 4,
      },
    ])

    const phrase = await index.searchBM25('"rapid transit"', 10)
    expect(phrase.map((row) => row.chunkId)).toContain(101n)
    expect(phrase.map((row) => row.chunkId)).not.toContain(103n)

    const prefix = await index.searchBM25('trans*', 10)
    expect(prefix.map((row) => row.chunkId)).toContain(101n)
    expect(prefix.map((row) => row.chunkId)).toContain(102n)

    const boolean = await index.searchBM25('rapid AND NOT road', 10)
    expect(boolean.map((row) => row.chunkId)).toContain(101n)
    expect(boolean.map((row) => row.chunkId)).not.toContain(103n)

    await index.close()
    await store.close()
  }, 120_000)

  it('expands aliases before compiling BM25 queries', async () => {
    const store = await createPostgresStore({
      sql: sql as unknown as PgSql,
      tenantId: tenantA,
      brainId: brainA,
    })
    await store.write(toPath('alias.md'), Buffer.from('seed'))

    const rows = (await sql<{ document_id: string }[]>`
      select document_id::text as document_id from memory.documents
      where brain_id = ${brainA}::uuid and path = 'alias.md'
    `) as ReadonlyArray<{ document_id: string }>
    const documentId = rows[0]?.document_id
    expect(documentId).toBeDefined()

    const index = createPostgresSearchIndex({
      sql: sql as unknown as PgSql,
      tenantId: tenantA,
      brainId: brainA,
      aliases: new Map<string, readonly string[]>([['rbh', ['rbh', 'robert-bosch']]]),
    })

    await index.upsert({
      chunkId: 104n,
      documentId: documentId as string,
      ordinal: 0,
      content: 'Robert Bosch founded this company',
      tokens: 5,
    })

    const hits = await index.searchBM25('rbh', 10)
    expect(hits.map((row) => row.chunkId)).toContain(104n)

    await index.close()
    await store.close()
  }, 120_000)

  it('expands single-alternative aliases before compiling BM25 queries', async () => {
    const store = await createPostgresStore({
      sql: sql as unknown as PgSql,
      tenantId: tenantA,
      brainId: brainA,
    })
    await store.write(toPath('single-alias.md'), Buffer.from('seed'))

    const rows = (await sql<{ document_id: string }[]>`
      select document_id::text as document_id from memory.documents
      where brain_id = ${brainA}::uuid and path = 'single-alias.md'
    `) as ReadonlyArray<{ document_id: string }>
    const documentId = rows[0]?.document_id
    expect(documentId).toBeDefined()

    const index = createPostgresSearchIndex({
      sql: sql as unknown as PgSql,
      tenantId: tenantA,
      brainId: brainA,
      aliases: new Map<string, readonly string[]>([['dude', ['oude']]]),
    })

    await index.upsert({
      chunkId: 105n,
      documentId: documentId as string,
      ordinal: 0,
      content: 'Oude station notes for Amsterdam travel',
      tokens: 6,
    })

    const hits = await index.searchBM25('dude', 10)
    expect(hits.map((row) => row.chunkId)).toContain(105n)

    await index.close()
    await store.close()
  }, 120_000)
})
