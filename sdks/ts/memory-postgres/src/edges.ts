// SPDX-License-Identifier: Apache-2.0

import type { DocumentEdgeType } from './graph.js'
import type { PgSql } from './store.js'

export type EmbeddingDim = 1024 | 3072

const DEFAULT_SIMILARITY_THRESHOLD = 0.2
const DEFAULT_SIMILARITY_LIMIT = 10
const DEFAULT_RELATION_LIMIT = 1_000
const SHARED_FOLDER_WEIGHT = 0.3
const SAME_SESSION_WEIGHT = 0.5

type SimilarDocumentRow = {
  target_doc_id: string
  similarity: number
}

type SharedTagRow = {
  target_doc_id: string
  weight: number
  label: string
}

type SharedFolderRow = {
  target_doc_id: string
}

type SameSessionRow = {
  target_doc_id: string
}

type SupersedesRow = {
  target_doc_id: string
  label: string
}

type SessionEpisodeRow = {
  target_doc_id: string
}

type EpisodeHeuristicRow = {
  target_doc_id: string
}

type DocumentOntologyRow = {
  target_doc_id: string
  label: string
}

type WikilinkRow = {
  target_doc_id: string
  label: string
}

type UpsertEdge = {
  sourceDocId: string
  targetDocId: string
  edgeType: DocumentEdgeType
  weight: number
  label?: string
}

const parseCentroidJson = (value: unknown): number[] | null => {
  if (typeof value === 'string') {
    const trimmed = value.trim()
    const jsonParsed = (() => {
      try {
        return JSON.parse(trimmed) as unknown
      } catch {
        return trimmed.startsWith('[') && trimmed.endsWith(']')
          ? trimmed
              .slice(1, -1)
              .split(',')
              .map((item) => Number(item.trim()))
          : null
      }
    })()
    return parseCentroidJson(jsonParsed)
  }
  if (!Array.isArray(value)) return null
  const parsed = value.map((item) => Number(item))
  return parsed.every((item) => Number.isFinite(item)) ? parsed : null
}

/**
 * Compute the average embedding (centroid) for one document.
 *
 * @param sql Postgres tagged-template client.
 * @param documentId Source document UUID.
 * @param tenantId Tenant UUID.
 * @returns Parsed centroid vector or `null` when no chunk embeddings exist.
 */
export async function computeDocumentCentroid(
  sql: PgSql,
  documentId: string,
  tenantId: string,
  options: { embeddingDim?: EmbeddingDim } = {},
): Promise<number[] | null> {
  const embeddingDim = options.embeddingDim ?? 1024
  const rows =
    embeddingDim === 3072
      ? ((await sql<{ centroid_json: unknown }>`
          select to_json(avg(embedding_3072::vector)) as centroid_json
          from memory.chunks
          where document_id = ${documentId}::uuid
            and tenant_id = ${tenantId}::uuid
            and embedding_3072 is not null
        `) as ReadonlyArray<{ centroid_json: unknown }>)
      : ((await sql<{ centroid_json: unknown }>`
          select to_json(avg(embedding::vector)) as centroid_json
          from memory.chunks
          where document_id = ${documentId}::uuid
            and tenant_id = ${tenantId}::uuid
            and embedding is not null
        `) as ReadonlyArray<{ centroid_json: unknown }>)
  return parseCentroidJson(rows[0]?.centroid_json)
}

/**
 * Find nearest documents by centroid cosine similarity.
 *
 * @param sql Postgres tagged-template client.
 * @param documentId Source document UUID.
 * @param brainId Brain UUID.
 * @param tenantId Tenant UUID.
 * @param options Optional similarity threshold and maximum result size.
 * @returns Candidate document ids with cosine similarity scores in `[0, 1]`.
 */
export async function findSimilarDocuments(
  sql: PgSql,
  documentId: string,
  brainId: string,
  tenantId: string,
  options: { threshold?: number; limit?: number; embeddingDim?: EmbeddingDim } = {},
): Promise<{ targetDocId: string; similarity: number }[]> {
  const threshold = options.threshold ?? DEFAULT_SIMILARITY_THRESHOLD
  const limit = options.limit ?? DEFAULT_SIMILARITY_LIMIT
  const embeddingDim = options.embeddingDim ?? 1024

  // Known limitation: this computes target centroids on demand (O(N) per source document).
  // We keep this until a persisted centroid column + index lands in schema.

  const rows =
    embeddingDim === 3072
      ? ((await sql<SimilarDocumentRow>`
          with source_centroid as (
            select avg(c.embedding_3072::vector)::halfvec(3072) as centroid
            from memory.chunks c
            where c.document_id = ${documentId}::uuid
              and c.tenant_id = ${tenantId}::uuid
              and c.embedding_3072 is not null
          ),
          target_centroids as (
            select d.document_id,
                   avg(c.embedding_3072::vector)::halfvec(3072) as centroid
            from memory.documents d
            join memory.chunks c
              on c.document_id = d.document_id
             and c.tenant_id = d.tenant_id
            where d.brain_id = ${brainId}::uuid
              and d.tenant_id = ${tenantId}::uuid
              and d.document_id != ${documentId}::uuid
              and c.embedding_3072 is not null
            group by d.document_id
          )
          select tc.document_id::text as target_doc_id,
                 1 - (tc.centroid <=> sc.centroid) as similarity
          from target_centroids tc
          cross join source_centroid sc
          where sc.centroid is not null
            and 1 - (tc.centroid <=> sc.centroid) >= ${threshold}
          order by tc.centroid <=> sc.centroid asc
          limit ${limit}
        `) as ReadonlyArray<SimilarDocumentRow>)
      : ((await sql<SimilarDocumentRow>`
          with source_centroid as (
            select avg(c.embedding::vector)::halfvec(1024) as centroid
            from memory.chunks c
            where c.document_id = ${documentId}::uuid
              and c.tenant_id = ${tenantId}::uuid
              and c.embedding is not null
          ),
          target_centroids as (
            select d.document_id,
                   avg(c.embedding::vector)::halfvec(1024) as centroid
            from memory.documents d
            join memory.chunks c
              on c.document_id = d.document_id
             and c.tenant_id = d.tenant_id
            where d.brain_id = ${brainId}::uuid
              and d.tenant_id = ${tenantId}::uuid
              and d.document_id != ${documentId}::uuid
              and c.embedding is not null
            group by d.document_id
          )
          select tc.document_id::text as target_doc_id,
                 1 - (tc.centroid <=> sc.centroid) as similarity
          from target_centroids tc
          cross join source_centroid sc
          where sc.centroid is not null
            and 1 - (tc.centroid <=> sc.centroid) >= ${threshold}
          order by tc.centroid <=> sc.centroid asc
          limit ${limit}
        `) as ReadonlyArray<SimilarDocumentRow>)

  return rows.map((row) => ({
    targetDocId: row.target_doc_id,
    similarity: Number(row.similarity),
  }))
}

/**
 * Compute Jaccard-weighted edges for shared frontmatter tags.
 *
 * @param sql Postgres tagged-template client.
 * @param documentId Source document UUID.
 * @param brainId Brain UUID.
 * @param tenantId Tenant UUID.
 * @returns Shared-tag edges with tag list labels.
 */
export async function computeSharedTagEdges(
  sql: PgSql,
  documentId: string,
  brainId: string,
  tenantId: string,
): Promise<{ targetDocId: string; weight: number; label: string }[]> {
  const rows = (await sql<SharedTagRow>`
    with source_tags as (
      select distinct jsonb_array_elements_text(
        case
          when jsonb_typeof(d.metadata->'tags') = 'array' then d.metadata->'tags'
          else '[]'::jsonb
        end
      ) as tag
      from memory.documents d
      where d.document_id = ${documentId}::uuid
        and d.brain_id = ${brainId}::uuid
        and d.tenant_id = ${tenantId}::uuid
    ),
    target_tags as (
      select d.document_id,
             jsonb_array_elements_text(
               case
                 when jsonb_typeof(d.metadata->'tags') = 'array' then d.metadata->'tags'
                 else '[]'::jsonb
               end
             ) as tag
      from memory.documents d
      where d.brain_id = ${brainId}::uuid
        and d.tenant_id = ${tenantId}::uuid
        and d.document_id != ${documentId}::uuid
    ),
    source_count as (
      select count(*)::real as cnt from source_tags
    ),
    target_counts as (
      select tt.document_id, count(distinct tt.tag)::real as cnt
      from target_tags tt
      group by tt.document_id
    ),
    intersections as (
      select tt.document_id,
             count(distinct tt.tag)::real as intersection_count,
             string_agg(distinct tt.tag, ',' order by tt.tag) as shared_tags
      from target_tags tt
      join source_tags st on st.tag = tt.tag
      group by tt.document_id
    )
    select i.document_id::text as target_doc_id,
           (i.intersection_count / nullif((sc.cnt + tc.cnt - i.intersection_count), 0)) as weight,
           i.shared_tags as label
    from intersections i
    cross join source_count sc
    join target_counts tc on tc.document_id = i.document_id
    where i.intersection_count > 0
  `) as ReadonlyArray<SharedTagRow>

  return rows.map((row) => ({
    targetDocId: row.target_doc_id,
    weight: Number(row.weight),
    label: row.label,
  }))
}

/**
 * Compute folder-co-location edges.
 *
 * @param sql Postgres tagged-template client.
 * @param documentId Source document UUID.
 * @param brainId Brain UUID.
 * @param tenantId Tenant UUID.
 * @returns Fixed-weight edges to documents in the same directory.
 */
export async function computeSharedFolderEdges(
  sql: PgSql,
  documentId: string,
  brainId: string,
  tenantId: string,
  options: { limit?: number } = {},
): Promise<{ targetDocId: string; weight: number }[]> {
  const limit = options.limit ?? DEFAULT_RELATION_LIMIT
  const rows = (await sql<SharedFolderRow>`
    with source as (
       select regexp_replace(path, '/[^/]*$', '') as directory
      from memory.documents
      where document_id = ${documentId}::uuid
        and brain_id = ${brainId}::uuid
        and tenant_id = ${tenantId}::uuid
    )
    select d.document_id::text as target_doc_id
    from memory.documents d
    cross join source s
    where d.brain_id = ${brainId}::uuid
      and d.tenant_id = ${tenantId}::uuid
      and d.document_id != ${documentId}::uuid
       and regexp_replace(d.path, '/[^/]*$', '') = s.directory
     limit ${limit}
  `) as ReadonlyArray<SharedFolderRow>

  return rows.map((row) => ({ targetDocId: row.target_doc_id, weight: SHARED_FOLDER_WEIGHT }))
}

/**
 * Compute edges for documents sharing the same session id.
 *
 * @param sql Postgres tagged-template client.
 * @param documentId Source document UUID.
 * @param brainId Brain UUID.
 * @param tenantId Tenant UUID.
 * @returns Fixed-weight edges to documents with matching `session_id`/`sessionId`.
 */
export async function computeSameSessionEdges(
  sql: PgSql,
  documentId: string,
  brainId: string,
  tenantId: string,
  options: { limit?: number } = {},
): Promise<{ targetDocId: string; weight: number }[]> {
  const limit = options.limit ?? DEFAULT_RELATION_LIMIT
  const rows = (await sql<SameSessionRow>`
    with source as (
       select coalesce(metadata->>'session_id', metadata->>'sessionId') as session_id
      from memory.documents
      where document_id = ${documentId}::uuid
        and brain_id = ${brainId}::uuid
        and tenant_id = ${tenantId}::uuid
    )
    select d.document_id::text as target_doc_id
    from memory.documents d
    cross join source s
    where d.brain_id = ${brainId}::uuid
      and d.tenant_id = ${tenantId}::uuid
      and d.document_id != ${documentId}::uuid
      and s.session_id is not null
       and coalesce(d.metadata->>'session_id', d.metadata->>'sessionId') = s.session_id
     limit ${limit}
  `) as ReadonlyArray<SameSessionRow>

  return rows.map((row) => ({ targetDocId: row.target_doc_id, weight: SAME_SESSION_WEIGHT }))
}

/**
 * Compute `supersedes` edges from frontmatter metadata.
 *
 * @param sql Postgres tagged-template client.
 * @param documentId Source document UUID.
 * @param brainId Brain UUID.
 * @param tenantId Tenant UUID.
 * @returns Exact replacement links to superseded documents.
 */
export async function computeSupersedesEdges(
  sql: PgSql,
  documentId: string,
  brainId: string,
  tenantId: string,
): Promise<{ targetDocId: string; weight: number; label: string }[]> {
  const rows = (await sql<SupersedesRow>`
    with source as (
      select nullif(trim(coalesce(metadata->>'supersedes', '')), '') as supersedes
      from memory.documents
      where document_id = ${documentId}::uuid
        and brain_id = ${brainId}::uuid
        and tenant_id = ${tenantId}::uuid
    )
    select d.document_id::text as target_doc_id,
           s.supersedes as label
    from memory.documents d
    cross join source s
    where d.brain_id = ${brainId}::uuid
      and d.tenant_id = ${tenantId}::uuid
      and d.document_id != ${documentId}::uuid
      and s.supersedes is not null
      and (
        d.path = s.supersedes
        or d.path = s.supersedes || '.md'
        or d.path like '%/' || replace(replace(replace(s.supersedes, '\\', '\\\\'), '%', '\\%'), '_', '\\_') escape '\\'
        or d.path like '%/' || replace(replace(replace(s.supersedes, '\\', '\\\\'), '%', '\\%'), '_', '\\_') || '.md' escape '\\'
      )
  `) as ReadonlyArray<SupersedesRow>

  return rows.map((row) => ({ targetDocId: row.target_doc_id, weight: 1, label: row.label }))
}

/**
 * Compute edges from session documents to episode notes.
 *
 * @param sql Postgres tagged-template client.
 * @param documentId Source document UUID.
 * @param brainId Brain UUID.
 * @param tenantId Tenant UUID.
 * @returns Fixed-weight edges to matching `episodes/*` session documents.
 */
export async function computeSessionEpisodeEdges(
  sql: PgSql,
  documentId: string,
  brainId: string,
  tenantId: string,
): Promise<{ targetDocId: string; weight: number }[]> {
  const rows = (await sql<SessionEpisodeRow>`
    with source as (
      select coalesce(metadata->>'session_id', metadata->>'sessionId') as session_id
      from memory.documents
      where document_id = ${documentId}::uuid
        and brain_id = ${brainId}::uuid
        and tenant_id = ${tenantId}::uuid
    )
    select d.document_id::text as target_doc_id
    from memory.documents d
    cross join source s
    where d.brain_id = ${brainId}::uuid
      and d.tenant_id = ${tenantId}::uuid
      and d.document_id != ${documentId}::uuid
      and s.session_id is not null
      and d.path like 'episodes/%'
      and (
        coalesce(d.metadata->>'session_id', d.metadata->>'sessionId') = s.session_id
        or d.path = 'episodes/' || s.session_id || '.md'
        or d.path = 'episodes/session-' || s.session_id || '.md'
      )
  `) as ReadonlyArray<SessionEpisodeRow>

  return rows.map((row) => ({ targetDocId: row.target_doc_id, weight: 0.8 }))
}

/**
 * Compute edges connecting episodes and heuristic notes.
 *
 * @param sql Postgres tagged-template client.
 * @param documentId Source document UUID.
 * @param brainId Brain UUID.
 * @param tenantId Tenant UUID.
 * @returns Fixed-weight edges between related episode/heuristic docs.
 */
export async function computeEpisodeHeuristicEdges(
  sql: PgSql,
  documentId: string,
  brainId: string,
  tenantId: string,
): Promise<{ targetDocId: string; weight: number }[]> {
  const rows = (await sql<EpisodeHeuristicRow>`
    with source as (
      select path,
             coalesce(metadata->>'session_id', metadata->>'sessionId') as session_id
      from memory.documents
      where document_id = ${documentId}::uuid
        and brain_id = ${brainId}::uuid
        and tenant_id = ${tenantId}::uuid
    )
    select d.document_id::text as target_doc_id
    from memory.documents d
    cross join source s
    where d.brain_id = ${brainId}::uuid
      and d.tenant_id = ${tenantId}::uuid
      and d.document_id != ${documentId}::uuid
      and (
        (
          s.path like 'episodes/%'
          and (d.path like 'memory/%/heuristic-%' or d.path like 'memory/%/anti-pattern-%')
        )
        or (
          (s.path like 'memory/%/heuristic-%' or s.path like 'memory/%/anti-pattern-%')
          and d.path like 'episodes/%'
        )
      )
      and s.session_id is not null
      and coalesce(d.metadata->>'session_id', d.metadata->>'sessionId') = s.session_id
  `) as ReadonlyArray<EpisodeHeuristicRow>

  return rows.map((row) => ({ targetDocId: row.target_doc_id, weight: 0.7 }))
}

/**
 * Compute edges between ontology type docs and typed documents.
 *
 * @param sql Postgres tagged-template client.
 * @param documentId Source document UUID.
 * @param brainId Brain UUID.
 * @param tenantId Tenant UUID.
 * @returns Type-linked edges with the resolved ontology type as label.
 */
export async function computeDocumentOntologyEdges(
  sql: PgSql,
  documentId: string,
  brainId: string,
  tenantId: string,
): Promise<{ targetDocId: string; weight: number; label: string }[]> {
  const rows = (await sql<DocumentOntologyRow>`
    with source as (
      select path,
             nullif(trim(coalesce(metadata->>'ontology_type', metadata->>'ontologyType', '')), '') as ontology_type
      from memory.documents
      where document_id = ${documentId}::uuid
        and brain_id = ${brainId}::uuid
        and tenant_id = ${tenantId}::uuid
    )
    select d.document_id::text as target_doc_id,
           s.ontology_type as label
    from memory.documents d
    cross join source s
    where d.brain_id = ${brainId}::uuid
      and d.tenant_id = ${tenantId}::uuid
      and d.document_id != ${documentId}::uuid
      and s.ontology_type is not null
      and (
        (
          s.path like 'ontology/%'
          and coalesce(d.metadata->>'ontology_type', d.metadata->>'ontologyType') = s.ontology_type
        )
        or (
          s.path not like 'ontology/%'
          and d.path like 'ontology/%'
          and coalesce(d.metadata->>'ontology_type', d.metadata->>'ontologyType') = s.ontology_type
        )
      )
  `) as ReadonlyArray<DocumentOntologyRow>

  return rows.map((row) => ({ targetDocId: row.target_doc_id, weight: 0.9, label: row.label }))
}

/**
 * Compute edges for explicit `[[wikilink]]` and OKF markdown-link references in content.
 *
 * @param sql Postgres tagged-template client.
 * @param documentId Source document UUID.
 * @param brainId Brain UUID.
 * @param tenantId Tenant UUID.
 * @returns Link edges to resolved document paths with original link labels.
 */
export async function computeWikilinkEdges(
  sql: PgSql,
  documentId: string,
  brainId: string,
  tenantId: string,
): Promise<{ targetDocId: string; weight: number; label: string }[]> {
  const rows = (await sql<WikilinkRow>`
    with source as (
      select path,
             regexp_replace(path, '/[^/]*$', '') as directory,
             convert_from(content, 'UTF8') as body
      from memory.documents
      where document_id = ${documentId}::uuid
        and brain_id = ${brainId}::uuid
        and tenant_id = ${tenantId}::uuid
    ),
    raw_links as (
      select distinct trim(split_part(matches[1], '|', 1)) as target,
             trim(split_part(matches[1], '|', 1)) as label
      from source s,
      lateral regexp_matches(s.body, '\\[\\[([^\\]]+)\\]\\]', 'g') as matches
      where trim(split_part(matches[1], '|', 1)) <> ''

      union

      select distinct trim(matches[2]) as target,
             trim(matches[2]) as label
      from source s,
      lateral regexp_matches(s.body, '(^|[^!])\\[[^\\]\\n]+\\]\\(([^)\\s]+)(?:\\s+"[^"]*")?\\)', 'g') as matches
      where trim(matches[2]) <> ''
    ),
    links as (
      select distinct
             case
               when cleaned.target like '/%' then ltrim(cleaned.target, '/')
               when cleaned.target like './%' and s.directory <> s.path then s.directory || '/' || substring(cleaned.target from 3)
               when cleaned.target like './%' then substring(cleaned.target from 3)
               else cleaned.target
             end as target,
             cleaned.label as label
      from source s,
      lateral (
        select regexp_replace(rl.target, '[#?].*$', '') as target,
               rl.label as label
        from raw_links rl
      ) as cleaned
      where cleaned.target <> ''
        and cleaned.target not like '#%'
        and cleaned.target !~* '^[a-z][a-z0-9+.-]*:'
        and cleaned.target not like '../%'
    )
    select d.document_id::text as target_doc_id,
           l.label as label
    from memory.documents d
    join links l
      on d.path = l.target
      or d.path = l.target || '.md'
      or d.path like '%/' || replace(replace(replace(l.target, '\\', '\\\\'), '%', '\\%'), '_', '\\_') escape '\\'
      or d.path like '%/' || replace(replace(replace(l.target, '\\', '\\\\'), '%', '\\%'), '_', '\\_') || '.md' escape '\\'
    where d.brain_id = ${brainId}::uuid
      and d.tenant_id = ${tenantId}::uuid
      and d.document_id != ${documentId}::uuid
  `) as ReadonlyArray<WikilinkRow>

  return rows.map((row) => ({ targetDocId: row.target_doc_id, weight: 0.7, label: row.label }))
}

/**
 * Upsert edge rows into `memory.document_edges`.
 *
 * @param sql Postgres tagged-template client.
 * @param brainId Brain UUID.
 * @param tenantId Tenant UUID.
 * @param edges Edge rows to insert or update.
 */
export async function upsertDocumentEdges(
  sql: PgSql,
  brainId: string,
  tenantId: string,
  edges: UpsertEdge[],
): Promise<void> {
  if (edges.length === 0) return
  const dedupedMap = new Map<string, UpsertEdge>()
  for (const edge of edges) {
    const key = `${edge.sourceDocId}:${edge.targetDocId}:${edge.edgeType}`
    dedupedMap.set(key, edge)
  }
  const deduped = [...dedupedMap.values()]
  const sourceDocIds = deduped.map((edge) => edge.sourceDocId)
  const targetDocIds = deduped.map((edge) => edge.targetDocId)
  const edgeTypes = deduped.map((edge) => edge.edgeType)
  const weights = deduped.map((edge) => edge.weight)
  const labels = deduped.map((edge) => edge.label ?? null)
  await sql`
    insert into memory.document_edges (
      brain_id,
      tenant_id,
      source_doc_id,
      target_doc_id,
      edge_type,
      weight,
      label
    )
    select ${brainId}::uuid,
           ${tenantId}::uuid,
           u.source_doc_id,
           u.target_doc_id,
           u.edge_type,
           u.weight,
           u.label
    from unnest(
      ${sourceDocIds}::uuid[],
      ${targetDocIds}::uuid[],
      ${edgeTypes}::text[],
      ${weights}::real[],
      ${labels}::text[]
    ) as u(source_doc_id, target_doc_id, edge_type, weight, label)
    on conflict (brain_id, source_doc_id, target_doc_id, edge_type)
    do update set
      weight = excluded.weight,
      label = excluded.label
  `
}

/**
 * Delete all edges touching a document.
 *
 * @param sql Postgres tagged-template client.
 * @param documentId Document UUID.
 * @param brainId Brain UUID.
 * @param tenantId Tenant UUID.
 */
export async function deleteDocumentEdges(
  sql: PgSql,
  documentId: string,
  brainId: string,
  tenantId: string,
  options: { edgeTypes?: readonly DocumentEdgeType[] } = {},
): Promise<void> {
  const edgeTypes = options.edgeTypes ?? []
  await sql`
    delete from memory.document_edges
    where brain_id = ${brainId}::uuid
      and tenant_id = ${tenantId}::uuid
      and (source_doc_id = ${documentId}::uuid or target_doc_id = ${documentId}::uuid)
      and (${edgeTypes.length} = 0 or edge_type = any(${edgeTypes}::text[]))
  `
}

/**
 * Recompute and persist all edge types for one document.
 *
 * @param sql Postgres tagged-template client.
 * @param documentId Source document UUID.
 * @param brainId Brain UUID.
 * @param tenantId Tenant UUID.
 * @param options Optional semantic similarity tuning.
 */
export async function computeAllEdgesForDocument(
  sql: PgSql,
  documentId: string,
  brainId: string,
  tenantId: string,
  options: {
    similarityThreshold?: number
    similarityLimit?: number
    relationLimit?: number
    embeddingDim?: EmbeddingDim
  } = {},
): Promise<void> {
  const threshold = options.similarityThreshold ?? DEFAULT_SIMILARITY_THRESHOLD
  const limit = options.similarityLimit ?? DEFAULT_SIMILARITY_LIMIT
  const relationLimit = options.relationLimit ?? DEFAULT_RELATION_LIMIT
  const embeddingDim = options.embeddingDim ?? 1024

  const recomputedEdgeTypes: readonly DocumentEdgeType[] = [
    'semantic_similarity',
    'shared_tag',
    'shared_folder',
    'same_session',
    'supersedes',
    'session_episode',
    'episode_heuristic',
    'document_ontology',
    'wikilink',
  ]

  await sql.begin(async (tx) => {
    await deleteDocumentEdges(tx, documentId, brainId, tenantId, { edgeTypes: recomputedEdgeTypes })

    const [
      similar,
      sharedTag,
      sharedFolder,
      sameSession,
      supersedes,
      sessionEpisode,
      episodeHeuristic,
      documentOntology,
      wikilink,
    ] = await Promise.all([
      findSimilarDocuments(tx, documentId, brainId, tenantId, { threshold, limit, embeddingDim }),
      computeSharedTagEdges(tx, documentId, brainId, tenantId),
      computeSharedFolderEdges(tx, documentId, brainId, tenantId, { limit: relationLimit }),
      computeSameSessionEdges(tx, documentId, brainId, tenantId, { limit: relationLimit }),
      computeSupersedesEdges(tx, documentId, brainId, tenantId),
      computeSessionEpisodeEdges(tx, documentId, brainId, tenantId),
      computeEpisodeHeuristicEdges(tx, documentId, brainId, tenantId),
      computeDocumentOntologyEdges(tx, documentId, brainId, tenantId),
      computeWikilinkEdges(tx, documentId, brainId, tenantId),
    ])

    const edges: UpsertEdge[] = [
      ...similar.map((item) => ({
        sourceDocId: documentId,
        targetDocId: item.targetDocId,
        edgeType: 'semantic_similarity' as const,
        weight: item.similarity,
      })),
      ...sharedTag.map((item) => ({
        sourceDocId: documentId,
        targetDocId: item.targetDocId,
        edgeType: 'shared_tag' as const,
        weight: item.weight,
        label: item.label,
      })),
      ...sharedFolder.map((item) => ({
        sourceDocId: documentId,
        targetDocId: item.targetDocId,
        edgeType: 'shared_folder' as const,
        weight: item.weight,
      })),
      ...sameSession.map((item) => ({
        sourceDocId: documentId,
        targetDocId: item.targetDocId,
        edgeType: 'same_session' as const,
        weight: item.weight,
      })),
      ...supersedes.map((item) => ({
        sourceDocId: documentId,
        targetDocId: item.targetDocId,
        edgeType: 'supersedes' as const,
        weight: item.weight,
        label: item.label,
      })),
      ...sessionEpisode.map((item) => ({
        sourceDocId: documentId,
        targetDocId: item.targetDocId,
        edgeType: 'session_episode' as const,
        weight: item.weight,
      })),
      ...episodeHeuristic.map((item) => ({
        sourceDocId: documentId,
        targetDocId: item.targetDocId,
        edgeType: 'episode_heuristic' as const,
        weight: item.weight,
      })),
      ...documentOntology.map((item) => ({
        sourceDocId: documentId,
        targetDocId: item.targetDocId,
        edgeType: 'document_ontology' as const,
        weight: item.weight,
        label: item.label,
      })),
      ...wikilink.map((item) => ({
        sourceDocId: documentId,
        targetDocId: item.targetDocId,
        edgeType: 'wikilink' as const,
        weight: item.weight,
        label: item.label,
      })),
    ]

    await upsertDocumentEdges(tx, brainId, tenantId, edges)
  })
}
