// SPDX-License-Identifier: Apache-2.0

import type { PgSql } from './store.js'

export const DOCUMENT_EDGE_TYPES = [
  'semantic_similarity',
  'shared_tag',
  'shared_folder',
  'same_session',
  'supersedes',
  'session_episode',
  'episode_heuristic',
  'document_ontology',
  'wikilink',
] as const

export type DocumentEdgeType = (typeof DOCUMENT_EDGE_TYPES)[number]

export type GraphNode = {
  id: string
  path: string
  label: string
  entityType: 'memory' | 'article' | 'episode' | 'heuristic' | 'procedural' | 'ontology' | 'ingested'
  ontologyType?: string
  ontologyLabel?: string
  scope: string
  size: number
  modTime: string
  chunkCount: number
  metadata: Record<string, unknown>
}

export type GraphEdge = {
  source: string
  target: string
  edgeType: DocumentEdgeType
  weight: number
  label?: string
}

export type GraphCommunity = {
  id: string
  nodeIds: string[]
  label?: string
}

export type GraphMeta = {
  totalNodes: number
  totalEdges: number
  entityTypeCounts: Record<string, number>
  edgeTypeCounts: Record<string, number>
  truncated: boolean
}

export type GraphResponse = {
  nodes: GraphNode[]
  edges: GraphEdge[]
  communities: GraphCommunity[]
  meta: GraphMeta
}

export type GraphQueryOptions = {
  minWeight?: number
  edgeTypes?: DocumentEdgeType[]
  entityTypes?: string[]
  maxNodes?: number
  maxEdges?: number
}

export type StatsResponse = {
  documentCount: number
  chunkCount: number
  edgeCount: number
  entityTypeCounts: Record<string, number>
  edgeTypeCounts: Record<string, number>
  lastActivityAt: string | null
}

type NodeRow = {
  document_id: string
  path: string
  size: number
  updated_at: Date
  chunk_count: number
  metadata: Record<string, unknown> | null
}

type EdgeRow = {
  source_doc_id: string
  target_doc_id: string
  edge_type: DocumentEdgeType
  weight: number
  label: string | null
}

const lastPathSegment = (path: string): string => {
  const pieces = path.split('/')
  const last = pieces[pieces.length - 1]
  return last === undefined || last === '' ? path : last
}

const isHeuristicPath = (path: string): boolean =>
  /^memory\/.*\/(heuristic-|anti-pattern-)/.test(path)

const resolveEntityAndScope = (path: string): Pick<GraphNode, 'entityType' | 'scope'> => {
  if (isHeuristicPath(path)) return { entityType: 'heuristic', scope: 'other' }
  if (path.startsWith('memory/_procedural/')) return { entityType: 'procedural', scope: 'agent' }
  if (path.startsWith('memory/project/')) return { entityType: 'memory', scope: 'project' }
  if (path.startsWith('memory/global/')) return { entityType: 'memory', scope: 'global' }
  if (path.startsWith('memory/agent/')) return { entityType: 'memory', scope: 'agent' }
  if (path.startsWith('ontology/')) return { entityType: 'ontology', scope: 'other' }
  if (path.startsWith('wiki/')) return { entityType: 'article', scope: 'other' }
  if (path.startsWith('drafts/')) return { entityType: 'article', scope: 'other' }
  if (path.startsWith('episodes/')) return { entityType: 'episode', scope: 'other' }
  if (path.startsWith('raw/')) return { entityType: 'ingested', scope: 'other' }
  if (path.startsWith('reflections/')) return { entityType: 'episode', scope: 'other' }
  return { entityType: 'memory', scope: 'other' }
}

const buildEntityCounts = (nodes: readonly GraphNode[]): Record<string, number> => {
  const counts: Record<string, number> = {}
  for (const node of nodes) {
    counts[node.entityType] = (counts[node.entityType] ?? 0) + 1
  }
  return counts
}

const buildEdgeCounts = (edges: readonly GraphEdge[]): Record<string, number> => {
  const counts: Record<string, number> = {}
  for (const edge of edges) {
    counts[edge.edgeType] = (counts[edge.edgeType] ?? 0) + 1
  }
  return counts
}

/** Fetch the full graph (nodes + pre-computed edges). */
export async function getDocumentGraph(
  sql: PgSql,
  brainId: string,
  tenantId: string,
  options: GraphQueryOptions = {},
): Promise<GraphResponse> {
  const minWeight = options.minWeight ?? 0.2
  const maxNodes = options.maxNodes ?? 10_000
  const maxEdges = options.maxEdges ?? 50_000

  const nodeRows = (await sql<NodeRow>`
    select d.document_id::text as document_id,
           d.path,
           d.size,
           d.updated_at,
           d.metadata,
           count(c.chunk_id)::int as chunk_count
    from memory.documents d
    left join memory.chunks c
      on c.document_id = d.document_id
     and c.tenant_id = d.tenant_id
    where d.brain_id = ${brainId}::uuid
      and d.tenant_id = ${tenantId}::uuid
    group by d.document_id, d.path, d.size, d.updated_at, d.metadata
    order by d.updated_at desc
    limit ${maxNodes}
  `) as ReadonlyArray<NodeRow>

  const allNodes = nodeRows.map((row) => {
    const resolved = resolveEntityAndScope(row.path)
    const ontologyType =
      typeof row.metadata?.ontology_type === 'string'
        ? row.metadata.ontology_type
        : typeof row.metadata?.ontologyType === 'string'
          ? row.metadata.ontologyType
          : undefined
    const ontologyLabel =
      typeof row.metadata?.ontology_label === 'string'
        ? row.metadata.ontology_label
        : typeof row.metadata?.ontologyLabel === 'string'
          ? row.metadata.ontologyLabel
          : undefined
    return {
      id: row.document_id,
      path: row.path,
      label: lastPathSegment(row.path),
      entityType: resolved.entityType,
      ...(ontologyType === undefined ? {} : { ontologyType }),
      ...(ontologyLabel === undefined ? {} : { ontologyLabel }),
      scope: resolved.scope,
      size: row.size,
      modTime: row.updated_at.toISOString(),
      chunkCount: row.chunk_count,
      metadata: row.metadata ?? {},
    } satisfies GraphNode
  })

  const entityFilterSet = options.entityTypes === undefined ? undefined : new Set(options.entityTypes)
  const nodes =
    entityFilterSet === undefined
      ? allNodes
      : allNodes.filter((node) => entityFilterSet.has(node.entityType))

  if (nodes.length === 0) {
    return {
      nodes: [],
      edges: [],
      communities: [],
      meta: {
        totalNodes: 0,
        totalEdges: 0,
        entityTypeCounts: {},
        edgeTypeCounts: {},
        truncated: nodeRows.length >= maxNodes,
      },
    }
  }

  const nodeIds = nodes.map((node) => node.id)
  const selectedEdgeTypes = options.edgeTypes ?? []
  const rows = (await sql<EdgeRow>`
    select source_doc_id::text as source_doc_id,
           target_doc_id::text as target_doc_id,
           edge_type,
           weight,
           label
    from memory.document_edges
    where brain_id = ${brainId}::uuid
      and tenant_id = ${tenantId}::uuid
      and weight >= ${minWeight}
      and source_doc_id = any(${nodeIds}::uuid[])
      and target_doc_id = any(${nodeIds}::uuid[])
      and (${selectedEdgeTypes.length} = 0 or edge_type = any(${selectedEdgeTypes}::text[]))
    order by weight desc
    limit ${maxEdges}
  `) as ReadonlyArray<EdgeRow>

  const edges = rows.map((row) => ({
    source: row.source_doc_id,
    target: row.target_doc_id,
    edgeType: row.edge_type,
    weight: Number(row.weight),
    ...(row.label === null ? {} : { label: row.label }),
  }))

  return {
    nodes,
    edges,
    communities: [],
    meta: {
      totalNodes: nodes.length,
      totalEdges: edges.length,
      entityTypeCounts: buildEntityCounts(nodes),
      edgeTypeCounts: buildEdgeCounts(edges),
      truncated: nodeRows.length >= maxNodes || rows.length >= maxEdges,
    },
  }
}

/** Fetch aggregate stats. */
export async function getDocumentStats(
  sql: PgSql,
  brainId: string,
  tenantId: string,
): Promise<StatsResponse> {
  const [countsRows, entityTypeRows, edgeTypeRows] = await Promise.all([
    sql<{
      document_count: number
      chunk_count: number
      edge_count: number
      last_activity_at: Date | null
    }>`
      with docs as (
        select document_id, tenant_id, updated_at
        from memory.documents
        where brain_id = ${brainId}::uuid
          and tenant_id = ${tenantId}::uuid
      )
      select (select count(*)::int from docs) as document_count,
             (select count(*)::int
              from memory.chunks c
              join docs d
                on d.document_id = c.document_id
               and d.tenant_id = c.tenant_id) as chunk_count,
             (select count(*)::int
              from memory.document_edges e
              where e.brain_id = ${brainId}::uuid
                and e.tenant_id = ${tenantId}::uuid) as edge_count,
             (select max(updated_at) from docs) as last_activity_at
    `,
    sql<{ entity_type: string; count: number }>`
      select entity_type, count(*)::int as count
      from memory.documents
      cross join lateral (
        select case
          when path ~ '^memory/.*/(heuristic-|anti-pattern-)' then 'heuristic'
          when path like 'memory/_procedural/%' then 'procedural'
          when path like 'memory/project/%' then 'memory'
          when path like 'memory/global/%' then 'memory'
          when path like 'memory/agent/%' then 'memory'
          when path like 'ontology/%' then 'ontology'
          when path like 'wiki/%' then 'article'
          when path like 'drafts/%' then 'article'
          when path like 'episodes/%' then 'episode'
          when path like 'raw/%' then 'ingested'
          when path like 'reflections/%' then 'episode'
          else 'memory'
        end as entity_type
      ) classified
      where brain_id = ${brainId}::uuid
        and tenant_id = ${tenantId}::uuid
      group by entity_type
    `,
    sql<{ edge_type: DocumentEdgeType; count: number }>`
      select edge_type, count(*)::int as count
      from memory.document_edges
      where brain_id = ${brainId}::uuid
        and tenant_id = ${tenantId}::uuid
      group by edge_type
    `,
  ])

  const docs = countsRows as ReadonlyArray<{
    document_count: number
    chunk_count: number
    edge_count: number
    last_activity_at: Date | null
  }>
  const entities = entityTypeRows as ReadonlyArray<{ entity_type: string; count: number }>
  const edges = edgeTypeRows as ReadonlyArray<{ edge_type: DocumentEdgeType; count: number }>

  const entityTypeCounts: Record<string, number> = {}
  for (const row of entities) {
    entityTypeCounts[row.entity_type] = row.count
  }

  const edgeTypeCounts: Record<string, number> = {}
  for (const row of edges) {
    edgeTypeCounts[row.edge_type] = row.count
  }

  return {
    documentCount: docs[0]?.document_count ?? 0,
    chunkCount: docs[0]?.chunk_count ?? 0,
    edgeCount: docs[0]?.edge_count ?? 0,
    entityTypeCounts,
    edgeTypeCounts,
    lastActivityAt: docs[0]?.last_activity_at?.toISOString() ?? null,
  }
}
