// SPDX-License-Identifier: Apache-2.0

import { execSync } from 'node:child_process'
import { readFileSync, readdirSync } from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import postgres from 'postgres'
import { afterAll, beforeAll, describe, expect, it } from 'vitest'
import {
  computeAllEdgesForDocument,
  computeDocumentCentroid,
  computeDocumentOntologyEdges,
  computeEpisodeHeuristicEdges,
  computeSameSessionEdges,
  computeSessionEpisodeEdges,
  computeSharedFolderEdges,
  computeSharedTagEdges,
  computeSupersedesEdges,
  computeWikilinkEdges,
  deleteDocumentEdges,
  findSimilarDocuments,
  upsertDocumentEdges,
} from './edges.js'
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

const vector = (a: number, b: number): string => `[${a},${b}${',0'.repeat(1022)}]`

maybe('document edges integration', () => {
  type RawSql = ReturnType<typeof postgres>
  let sql: RawSql
  let stop: () => Promise<void>
  const tenantId = '11111111-1111-1111-1111-111111111111'
  const brainId = 'aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa'

  const docs = {
    source: '00000000-0000-0000-0000-000000000001',
    similar: '00000000-0000-0000-0000-000000000002',
    folder: '00000000-0000-0000-0000-000000000003',
    session: '00000000-0000-0000-0000-000000000004',
    sourceNoMeta: '00000000-0000-0000-0000-000000000005',
    sourceNoChunks: '00000000-0000-0000-0000-000000000006',
    targetSuperseded: '00000000-0000-0000-0000-000000000007',
    episodeS1: '00000000-0000-0000-0000-000000000008',
    heuristicS1: '00000000-0000-0000-0000-000000000009',
    ontologyCustomer: '00000000-0000-0000-0000-000000000010',
    memoryCustomer: '00000000-0000-0000-0000-000000000011',
    linkSource: '00000000-0000-0000-0000-000000000012',
  } as const

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
      `insert into platform.tenants (tenant_id, slug, name, region) values ('${tenantId}', 'tenant-a', 'Tenant A', 'eu-fsn1')`,
    )
    await sql.unsafe(
      `insert into memory.brains (brain_id, tenant_id, slug, name) values ('${brainId}', '${tenantId}', 'brain-a', 'Brain A')`,
    )

    await sql.unsafe(`
      insert into memory.documents (document_id, brain_id, tenant_id, path, content_hash, content, size, source, metadata)
      values
      ('${docs.source}', '${brainId}', '${tenantId}', 'memory/project/a/source.md', decode('00','hex'), ''::bytea, 10, 'test', '{"tags":["alpha","beta"],"session_id":"s1"}'::jsonb),
      ('${docs.similar}', '${brainId}', '${tenantId}', 'memory/project/a/similar.md', decode('00','hex'), ''::bytea, 11, 'test', '{"tags":["alpha","gamma"],"session_id":"s2"}'::jsonb),
      ('${docs.folder}', '${brainId}', '${tenantId}', 'memory/project/a/folder.md', decode('00','hex'), ''::bytea, 12, 'test', '{"tags":["delta"],"session_id":"s1"}'::jsonb),
      ('${docs.session}', '${brainId}', '${tenantId}', 'memory/project/b/session.md', decode('00','hex'), ''::bytea, 13, 'test', '{"tags":["beta"],"session_id":"s1"}'::jsonb),
      ('${docs.sourceNoMeta}', '${brainId}', '${tenantId}', 'memory/project/c/no-meta.md', decode('00','hex'), ''::bytea, 14, 'test', '{}'::jsonb),
      ('${docs.sourceNoChunks}', '${brainId}', '${tenantId}', 'memory/project/c/no-chunks.md', decode('00','hex'), ''::bytea, 15, 'test', '{"tags":"not-an-array"}'::jsonb),
      ('${docs.targetSuperseded}', '${brainId}', '${tenantId}', 'memory/project/a/legacy-note.md', decode('00','hex'), ''::bytea, 16, 'test', '{}'::jsonb),
      ('${docs.episodeS1}', '${brainId}', '${tenantId}', 'episodes/session-s1.md', decode('00','hex'), ''::bytea, 17, 'test', '{"session_id":"s1"}'::jsonb),
      ('${docs.heuristicS1}', '${brainId}', '${tenantId}', 'memory/project/a/heuristic-review.md', decode('00','hex'), ''::bytea, 18, 'test', '{"sessionId":"s1"}'::jsonb),
      ('${docs.ontologyCustomer}', '${brainId}', '${tenantId}', 'ontology/types/customer.md', decode('00','hex'), ''::bytea, 19, 'test', '{"ontology_type":"customer","ontology_label":"Customer"}'::jsonb),
      ('${docs.memoryCustomer}', '${brainId}', '${tenantId}', 'memory/project/a/customer-note.md', decode('00','hex'), ''::bytea, 20, 'test', '{"ontologyType":"customer"}'::jsonb),
      ('${docs.linkSource}', '${brainId}', '${tenantId}', 'wiki/link-source.md', decode('00','hex'), convert_to('Links: [[memory/project/a/similar]] [customer](/ontology/types/customer.md) [folder](memory/project/a/folder.md) ![ignored](/memory/project/b/session.md)', 'UTF8'), 21, 'test', '{"session_id":"s1","supersedes":"legacy-note"}'::jsonb)
    `)

    await sql.unsafe(`
      insert into memory.chunks (chunk_id, tenant_id, document_id, ordinal, content, tokens, embedding)
      values
      (1, '${tenantId}', '${docs.source}', 0, 'source chunk', 2, '${vector(1, 0)}'::halfvec),
      (2, '${tenantId}', '${docs.source}', 1, 'source chunk two', 3, '${vector(1, 0)}'::halfvec),
      (3, '${tenantId}', '${docs.similar}', 0, 'similar chunk', 2, '${vector(0.95, 0.1)}'::halfvec),
      (4, '${tenantId}', '${docs.folder}', 0, 'folder chunk', 2, '${vector(0.1, 0.9)}'::halfvec),
      (5, '${tenantId}', '${docs.session}', 0, 'session chunk', 2, '${vector(0.1, 0.8)}'::halfvec)
    `)

    stop = async () => {
      await sql.end({ timeout: 5 })
      await container.stop()
    }
  }, 120_000)

  afterAll(async () => {
    if (stop) await stop()
  })

  it('computes centroid and finds similar documents with threshold/limit', async () => {
    const centroid = await computeDocumentCentroid(sql as unknown as PgSql, docs.source, tenantId)
    expect(centroid).not.toBeNull()
    expect((centroid ?? [])[0]).toBeCloseTo(1, 5)

    const similar = await findSimilarDocuments(
      sql as unknown as PgSql,
      docs.source,
      brainId,
      tenantId,
      {
        threshold: 0.2,
        limit: 1,
      },
    )
    expect(similar).toHaveLength(1)
    expect(similar[0]?.targetDocId).toBe(docs.similar)
  }, 120_000)

  it('returns null centroid and no semantic edges when source has no chunks', async () => {
    const centroid = await computeDocumentCentroid(
      sql as unknown as PgSql,
      docs.sourceNoChunks,
      tenantId,
    )
    expect(centroid).toBeNull()

    const similar = await findSimilarDocuments(
      sql as unknown as PgSql,
      docs.sourceNoChunks,
      brainId,
      tenantId,
      {
        threshold: 0,
        limit: 10,
      },
    )
    expect(similar).toEqual([])
  }, 120_000)

  it('computes shared tag, folder, and session edges', async () => {
    const tags = await computeSharedTagEdges(
      sql as unknown as PgSql,
      docs.source,
      brainId,
      tenantId,
    )
    const tagToSimilar = tags.find((row) => row.targetDocId === docs.similar)
    expect(tagToSimilar).toBeDefined()
    expect(tagToSimilar?.weight).toBeCloseTo(1 / 3, 5)
    expect(tagToSimilar?.label).toContain('alpha')

    const folders = await computeSharedFolderEdges(
      sql as unknown as PgSql,
      docs.source,
      brainId,
      tenantId,
    )
    expect(folders.some((row) => row.targetDocId === docs.similar && row.weight === 0.3)).toBe(true)
    expect(folders.some((row) => row.targetDocId === docs.folder && row.weight === 0.3)).toBe(true)

    const sessions = await computeSameSessionEdges(
      sql as unknown as PgSql,
      docs.source,
      brainId,
      tenantId,
    )
    expect(sessions.some((row) => row.targetDocId === docs.folder && row.weight === 0.5)).toBe(true)
    expect(sessions.some((row) => row.targetDocId === docs.session && row.weight === 0.5)).toBe(
      true,
    )
  }, 120_000)

  it('handles no tags/session and computes added edge types', async () => {
    const noTagRows = await computeSharedTagEdges(
      sql as unknown as PgSql,
      docs.sourceNoChunks,
      brainId,
      tenantId,
    )
    expect(noTagRows).toEqual([])

    const noSessionRows = await computeSameSessionEdges(
      sql as unknown as PgSql,
      docs.sourceNoMeta,
      brainId,
      tenantId,
    )
    expect(noSessionRows).toEqual([])

    const supersedes = await computeSupersedesEdges(
      sql as unknown as PgSql,
      docs.linkSource,
      brainId,
      tenantId,
    )
    expect(
      supersedes.some((row) => row.targetDocId === docs.targetSuperseded && row.weight === 1),
    ).toBe(true)

    const sessionEpisode = await computeSessionEpisodeEdges(
      sql as unknown as PgSql,
      docs.source,
      brainId,
      tenantId,
    )
    expect(
      sessionEpisode.some((row) => row.targetDocId === docs.episodeS1 && row.weight === 0.8),
    ).toBe(true)

    const episodeHeuristic = await computeEpisodeHeuristicEdges(
      sql as unknown as PgSql,
      docs.episodeS1,
      brainId,
      tenantId,
    )
    expect(
      episodeHeuristic.some((row) => row.targetDocId === docs.heuristicS1 && row.weight === 0.7),
    ).toBe(true)

    const documentOntology = await computeDocumentOntologyEdges(
      sql as unknown as PgSql,
      docs.memoryCustomer,
      brainId,
      tenantId,
    )
    expect(
      documentOntology.some(
        (row) => row.targetDocId === docs.ontologyCustomer && row.label === 'customer',
      ),
    ).toBe(true)

    const wikilinks = await computeWikilinkEdges(
      sql as unknown as PgSql,
      docs.linkSource,
      brainId,
      tenantId,
    )
    expect(
      wikilinks.some(
        (row) => row.targetDocId === docs.similar && row.label === 'memory/project/a/similar',
      ),
    ).toBe(true)
    expect(
      wikilinks.some(
        (row) =>
          row.targetDocId === docs.ontologyCustomer && row.label === '/ontology/types/customer.md',
      ),
    ).toBe(true)
    expect(
      wikilinks.some(
        (row) => row.targetDocId === docs.folder && row.label === 'memory/project/a/folder.md',
      ),
    ).toBe(true)
    expect(wikilinks.some((row) => row.targetDocId === docs.session)).toBe(false)
  }, 120_000)

  it('upserts and deletes document edges', async () => {
    await upsertDocumentEdges(sql as unknown as PgSql, brainId, tenantId, [
      {
        sourceDocId: docs.source,
        targetDocId: docs.similar,
        edgeType: 'shared_folder',
        weight: 0.3,
      },
      {
        sourceDocId: docs.source,
        targetDocId: docs.similar,
        edgeType: 'shared_folder',
        weight: 0.7,
        label: 'updated',
      },
    ])

    const edges = (await sql<{ weight: number; label: string | null }[]>`
      select weight, label
      from memory.document_edges
      where brain_id = ${brainId}::uuid
        and tenant_id = ${tenantId}::uuid
        and source_doc_id = ${docs.source}::uuid
        and target_doc_id = ${docs.similar}::uuid
        and edge_type = 'shared_folder'
    `) as ReadonlyArray<{ weight: number; label: string | null }>
    expect(edges).toHaveLength(1)
    expect(edges[0]?.weight).toBeCloseTo(0.7, 5)
    expect(edges[0]?.label).toBe('updated')

    await deleteDocumentEdges(sql as unknown as PgSql, docs.source, brainId, tenantId)
    const afterDelete = (await sql<{ count: number }[]>`
      select count(*)::int as count
      from memory.document_edges
      where brain_id = ${brainId}::uuid
        and tenant_id = ${tenantId}::uuid
        and (source_doc_id = ${docs.source}::uuid or target_doc_id = ${docs.source}::uuid)
    `) as ReadonlyArray<{ count: number }>
    expect(afterDelete[0]?.count).toBe(0)
  }, 120_000)

  it('computes and persists all edges for a document', async () => {
    await computeAllEdgesForDocument(sql as unknown as PgSql, docs.source, brainId, tenantId, {
      similarityThreshold: 0.2,
      similarityLimit: 10,
    })

    const rows = (await sql<{ edge_type: string; count: number }[]>`
      select edge_type, count(*)::int as count
      from memory.document_edges
      where brain_id = ${brainId}::uuid
        and tenant_id = ${tenantId}::uuid
        and source_doc_id = ${docs.source}::uuid
      group by edge_type
    `) as ReadonlyArray<{ edge_type: string; count: number }>

    const map = new Map(rows.map((row) => [row.edge_type, row.count]))
    expect((map.get('semantic_similarity') ?? 0) > 0).toBe(true)
    expect((map.get('shared_tag') ?? 0) > 0).toBe(true)
    expect((map.get('shared_folder') ?? 0) > 0).toBe(true)
    expect((map.get('same_session') ?? 0) > 0).toBe(true)

    await computeAllEdgesForDocument(sql as unknown as PgSql, docs.linkSource, brainId, tenantId, {
      similarityThreshold: 0.2,
      similarityLimit: 10,
    })
    const addedTypeRows = (await sql<{ edge_type: string; count: number }[]>`
      select edge_type, count(*)::int as count
      from memory.document_edges
      where brain_id = ${brainId}::uuid
        and tenant_id = ${tenantId}::uuid
        and source_doc_id = ${docs.linkSource}::uuid
      group by edge_type
    `) as ReadonlyArray<{ edge_type: string; count: number }>
    const addedTypeMap = new Map(addedTypeRows.map((row) => [row.edge_type, row.count]))
    expect((addedTypeMap.get('supersedes') ?? 0) > 0).toBe(true)
    expect((addedTypeMap.get('session_episode') ?? 0) > 0).toBe(true)
    expect((addedTypeMap.get('wikilink') ?? 0) > 0).toBe(true)

    await computeAllEdgesForDocument(
      sql as unknown as PgSql,
      docs.memoryCustomer,
      brainId,
      tenantId,
      {
        similarityThreshold: 0.2,
        similarityLimit: 10,
      },
    )
    const ontologyTypeRows = (await sql<{ edge_type: string; count: number }[]>`
      select edge_type, count(*)::int as count
      from memory.document_edges
      where brain_id = ${brainId}::uuid
        and tenant_id = ${tenantId}::uuid
        and source_doc_id = ${docs.memoryCustomer}::uuid
      group by edge_type
    `) as ReadonlyArray<{ edge_type: string; count: number }>
    const ontologyTypeMap = new Map(ontologyTypeRows.map((row) => [row.edge_type, row.count]))
    expect((ontologyTypeMap.get('document_ontology') ?? 0) > 0).toBe(true)

    await computeAllEdgesForDocument(sql as unknown as PgSql, docs.episodeS1, brainId, tenantId, {
      similarityThreshold: 0.2,
      similarityLimit: 10,
    })
    const episodeTypeRows = (await sql<{ edge_type: string; count: number }[]>`
      select edge_type, count(*)::int as count
      from memory.document_edges
      where brain_id = ${brainId}::uuid
        and tenant_id = ${tenantId}::uuid
        and source_doc_id = ${docs.episodeS1}::uuid
      group by edge_type
    `) as ReadonlyArray<{ edge_type: string; count: number }>
    const episodeTypeMap = new Map(episodeTypeRows.map((row) => [row.edge_type, row.count]))
    expect((episodeTypeMap.get('episode_heuristic') ?? 0) > 0).toBe(true)
  }, 120_000)
})
