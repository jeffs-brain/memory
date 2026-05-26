// SPDX-License-Identifier: Apache-2.0

import { execSync } from 'node:child_process'
import { readFileSync, readdirSync } from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import postgres from 'postgres'
import { afterAll, beforeAll, describe, expect, it } from 'vitest'
import { getDocumentGraph, getDocumentStats } from './graph.js'
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

maybe('document graph integration', () => {
  type RawSql = ReturnType<typeof postgres>
  let sql: RawSql
  let stop: () => Promise<void>
  const tenantId = '11111111-1111-1111-1111-111111111111'
  const brainId = 'aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa'
  const emptyBrainId = 'bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb'
  const mappingBrainId = 'cccccccc-cccc-cccc-cccc-cccccccccccc'

  const d1 = '00000000-0000-0000-0000-000000000011'
  const d2 = '00000000-0000-0000-0000-000000000012'
  const d3 = '00000000-0000-0000-0000-000000000013'
  const d4 = '00000000-0000-0000-0000-000000000014'
  const d5 = '00000000-0000-0000-0000-000000000015'
  const d6 = '00000000-0000-0000-0000-000000000016'

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
    await sql.unsafe(
      `insert into memory.brains (brain_id, tenant_id, slug, name) values ('${emptyBrainId}', '${tenantId}', 'brain-empty', 'Empty Brain')`,
    )
    await sql.unsafe(
      `insert into memory.brains (brain_id, tenant_id, slug, name) values ('${mappingBrainId}', '${tenantId}', 'brain-mapping', 'Mapping Brain')`,
    )

    await sql.unsafe(`
      insert into memory.documents (document_id, brain_id, tenant_id, path, content_hash, content, size, source, metadata)
      values
      ('${d1}', '${brainId}', '${tenantId}', 'memory/project/me/note.md', decode('00','hex'), ''::bytea, 100, 'test', '{}'::jsonb),
      ('${d2}', '${brainId}', '${tenantId}', 'wiki/article.md', decode('00','hex'), ''::bytea, 200, 'test', '{}'::jsonb),
      ('${d3}', '${brainId}', '${tenantId}', 'episodes/session-1.md', decode('00','hex'), ''::bytea, 300, 'test', '{}'::jsonb)
    `)

    await sql.unsafe(`
      insert into memory.chunks (chunk_id, tenant_id, document_id, ordinal, content, tokens)
      values
      (101, '${tenantId}', '${d1}', 0, 'a', 1),
      (102, '${tenantId}', '${d1}', 1, 'b', 1),
      (103, '${tenantId}', '${d2}', 0, 'c', 1)
    `)

    await sql.unsafe(`
      insert into memory.document_edges (brain_id, tenant_id, source_doc_id, target_doc_id, edge_type, weight, label)
      values
      ('${brainId}', '${tenantId}', '${d1}', '${d2}', 'shared_tag', 0.7, 'alpha'),
      ('${brainId}', '${tenantId}', '${d1}', '${d3}', 'same_session', 0.5, null),
      ('${brainId}', '${tenantId}', '${d2}', '${d3}', 'semantic_similarity', 0.15, null)
    `)

    await sql.unsafe(`
      insert into memory.documents (document_id, brain_id, tenant_id, path, content_hash, content, size, source, metadata)
      values
      ('${d4}', '${mappingBrainId}', '${tenantId}', 'memory/project/me/heuristic-prioritization.md', decode('00','hex'), ''::bytea, 25, 'test', '{}'::jsonb),
      ('${d5}', '${mappingBrainId}', '${tenantId}', 'memory/_procedural/review-flow.md', decode('00','hex'), ''::bytea, 30, 'test', '{}'::jsonb),
      ('${d6}', '${mappingBrainId}', '${tenantId}', 'ontology/types/customer.md', decode('00','hex'), ''::bytea, 35, 'test', '{"ontology_type":"customer","ontology_label":"Customer"}'::jsonb)
    `)

    stop = async () => {
      await sql.end({ timeout: 5 })
      await container.stop()
    }
  }, 120_000)

  afterAll(async () => {
    if (stop) await stop()
  })

  it('returns graph nodes, filtered edges, and meta counts', async () => {
    const graph = await getDocumentGraph(sql as unknown as PgSql, brainId, tenantId, {
      minWeight: 0.2,
      maxNodes: 10,
      maxEdges: 10,
    })

    expect(graph.nodes).toHaveLength(3)
    expect(graph.edges).toHaveLength(2)
    expect(graph.meta.totalNodes).toBe(3)
    expect(graph.meta.totalEdges).toBe(2)
    expect(graph.meta.entityTypeCounts.memory).toBe(1)
    expect(graph.meta.entityTypeCounts.article).toBe(1)
    expect(graph.meta.entityTypeCounts.episode).toBe(1)
    expect(graph.meta.edgeTypeCounts.shared_tag).toBe(1)
    expect(graph.meta.edgeTypeCounts.same_session).toBe(1)
  }, 120_000)

  it('applies edgeTypes and entityTypes filters with truncation limits', async () => {
    const graph = await getDocumentGraph(sql as unknown as PgSql, brainId, tenantId, {
      minWeight: 0,
      edgeTypes: ['shared_tag'],
      entityTypes: ['memory', 'article'],
      maxNodes: 2,
      maxEdges: 1,
    })

    expect(graph.nodes.every((node) => node.entityType === 'memory' || node.entityType === 'article')).toBe(true)
    expect(graph.edges.every((edge) => edge.edgeType === 'shared_tag')).toBe(true)
    expect(graph.meta.truncated).toBe(true)
  }, 120_000)

  it('returns aggregate stats for documents, chunks, edges, and last activity', async () => {
    const stats = await getDocumentStats(sql as unknown as PgSql, brainId, tenantId)
    expect(stats.documentCount).toBe(3)
    expect(stats.chunkCount).toBe(3)
    expect(stats.edgeCount).toBe(3)
    expect(stats.entityTypeCounts.memory).toBe(1)
    expect(stats.entityTypeCounts.article).toBe(1)
    expect(stats.entityTypeCounts.episode).toBe(1)
    expect(stats.edgeTypeCounts.shared_tag).toBe(1)
    expect(stats.edgeTypeCounts.same_session).toBe(1)
    expect(stats.lastActivityAt).not.toBeNull()
  }, 120_000)

  it('returns empty graph and stats for a brain with no documents', async () => {
    const graph = await getDocumentGraph(sql as unknown as PgSql, emptyBrainId, tenantId)
    expect(graph.nodes).toHaveLength(0)
    expect(graph.edges).toHaveLength(0)
    expect(graph.communities).toEqual([])
    expect(graph.meta.totalNodes).toBe(0)
    expect(graph.meta.totalEdges).toBe(0)

    const stats = await getDocumentStats(sql as unknown as PgSql, emptyBrainId, tenantId)
    expect(stats.documentCount).toBe(0)
    expect(stats.chunkCount).toBe(0)
    expect(stats.edgeCount).toBe(0)
    expect(stats.entityTypeCounts).toEqual({})
    expect(stats.edgeTypeCounts).toEqual({})
    expect(stats.lastActivityAt).toBeNull()
  }, 120_000)

  it('maps heuristic and ontology documents with optional ontology fields', async () => {
    const graph = await getDocumentGraph(sql as unknown as PgSql, mappingBrainId, tenantId)

    const byPath = new Map(graph.nodes.map((node) => [node.path, node]))
    const heuristicNode = byPath.get('memory/project/me/heuristic-prioritization.md')
    const proceduralNode = byPath.get('memory/_procedural/review-flow.md')
    const ontologyNode = byPath.get('ontology/types/customer.md')

    expect(heuristicNode?.entityType).toBe('heuristic')
    expect(proceduralNode?.entityType).toBe('procedural')
    expect(ontologyNode?.entityType).toBe('ontology')
    expect(ontologyNode?.ontologyType).toBe('customer')
    expect(ontologyNode?.ontologyLabel).toBe('Customer')
  }, 120_000)

  it('computes entity stats with SQL classification ordering', async () => {
    const stats = await getDocumentStats(sql as unknown as PgSql, mappingBrainId, tenantId)

    expect(stats.documentCount).toBe(3)
    expect(stats.entityTypeCounts.heuristic).toBe(1)
    expect(stats.entityTypeCounts.procedural).toBe(1)
    expect(stats.entityTypeCounts.ontology).toBe(1)
  }, 120_000)
})
