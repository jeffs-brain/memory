// SPDX-License-Identifier: Apache-2.0

package graph

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/lib/pq"
)

const (
	defaultSimilarityThreshold = 0.2
	defaultSimilarityLimit     = 10
	defaultRelationLimit       = 1000

	sharedFolderWeight     = 0.3
	sameSessionWeight      = 0.5
	supersedesWeight       = 1.0
	sessionEpisodeWeight   = 0.8
	episodeHeuristicWeight = 0.7
	documentOntologyWeight = 0.9
	wikilinkWeight         = 0.7
)

// SimilarityOptions controls nearest-neighbour semantic edge computation.
type SimilarityOptions struct {
	Threshold    float64
	Limit        int
	EmbeddingDim EmbeddingDim
}

// SimilarDocument holds one semantic neighbour match.
type SimilarDocument struct {
	TargetDocID string
	Similarity  float64
}

// WeightedEdge holds a target document and edge weight.
type WeightedEdge struct {
	TargetDocID string
	Weight      float64
}

// LabeledWeightedEdge holds a target document, weight, and label.
type LabeledWeightedEdge struct {
	TargetDocID string
	Weight      float64
	Label       string
}

// ComputeAllEdgesOptions controls full recomputation behavior.
type ComputeAllEdgesOptions struct {
	SimilarityThreshold float64
	SimilarityLimit     int
	RelationLimit       int
	EmbeddingDim        EmbeddingDim
}

type queryer interface {
	QueryContext(context.Context, string, ...any) (*sql.Rows, error)
	QueryRowContext(context.Context, string, ...any) *sql.Row
}

type execer interface {
	ExecContext(context.Context, string, ...any) (sql.Result, error)
}

// ComputeDocumentCentroid computes one document's average chunk embedding.
func ComputeDocumentCentroid(ctx context.Context, db *sql.DB, documentID, tenantID string, dim EmbeddingDim) ([]float64, error) {
	if db == nil {
		return nil, fmt.Errorf("compute document centroid: nil db")
	}
	return computeDocumentCentroid(ctx, db, documentID, tenantID, dim)
}

// FindSimilarDocuments finds semantic-neighbour documents by centroid cosine similarity.
func FindSimilarDocuments(ctx context.Context, db *sql.DB, documentID, brainID, tenantID string, opts SimilarityOptions) ([]SimilarDocument, error) {
	if db == nil {
		return nil, fmt.Errorf("find similar documents: nil db")
	}
	return findSimilarDocuments(ctx, db, documentID, brainID, tenantID, opts)
}

// ComputeSharedTagEdges computes Jaccard-weighted shared-tag edges.
func ComputeSharedTagEdges(ctx context.Context, db *sql.DB, documentID, brainID, tenantID string) ([]LabeledWeightedEdge, error) {
	if db == nil {
		return nil, fmt.Errorf("compute shared tag edges: nil db")
	}
	return computeSharedTagEdges(ctx, db, documentID, brainID, tenantID)
}

// ComputeSharedFolderEdges computes fixed-weight same-folder edges.
func ComputeSharedFolderEdges(ctx context.Context, db *sql.DB, documentID, brainID, tenantID string, limit int) ([]WeightedEdge, error) {
	if db == nil {
		return nil, fmt.Errorf("compute shared folder edges: nil db")
	}
	return computeSharedFolderEdges(ctx, db, documentID, brainID, tenantID, limit)
}

// ComputeSameSessionEdges computes fixed-weight same-session edges.
func ComputeSameSessionEdges(ctx context.Context, db *sql.DB, documentID, brainID, tenantID string, limit int) ([]WeightedEdge, error) {
	if db == nil {
		return nil, fmt.Errorf("compute same session edges: nil db")
	}
	return computeSameSessionEdges(ctx, db, documentID, brainID, tenantID, limit)
}

// ComputeSupersedesEdges computes explicit supersedes relationships.
func ComputeSupersedesEdges(ctx context.Context, db *sql.DB, documentID, brainID, tenantID string) ([]LabeledWeightedEdge, error) {
	if db == nil {
		return nil, fmt.Errorf("compute supersedes edges: nil db")
	}
	return computeSupersedesEdges(ctx, db, documentID, brainID, tenantID)
}

// ComputeSessionEpisodeEdges computes edges from session docs to episode docs.
func ComputeSessionEpisodeEdges(ctx context.Context, db *sql.DB, documentID, brainID, tenantID string, limit int) ([]WeightedEdge, error) {
	if db == nil {
		return nil, fmt.Errorf("compute session episode edges: nil db")
	}
	return computeSessionEpisodeEdges(ctx, db, documentID, brainID, tenantID, limit)
}

// ComputeEpisodeHeuristicEdges computes edges linking episodes and heuristics for a session.
func ComputeEpisodeHeuristicEdges(ctx context.Context, db *sql.DB, documentID, brainID, tenantID string, limit int) ([]WeightedEdge, error) {
	if db == nil {
		return nil, fmt.Errorf("compute episode heuristic edges: nil db")
	}
	return computeEpisodeHeuristicEdges(ctx, db, documentID, brainID, tenantID, limit)
}

// ComputeDocumentOntologyEdges computes edges between ontology docs and typed docs.
func ComputeDocumentOntologyEdges(ctx context.Context, db *sql.DB, documentID, brainID, tenantID string) ([]LabeledWeightedEdge, error) {
	if db == nil {
		return nil, fmt.Errorf("compute document ontology edges: nil db")
	}
	return computeDocumentOntologyEdges(ctx, db, documentID, brainID, tenantID)
}

// ComputeWikilinkEdges computes edges from explicit [[wikilink]] references.
func ComputeWikilinkEdges(ctx context.Context, db *sql.DB, documentID, brainID, tenantID string, limit int) ([]LabeledWeightedEdge, error) {
	if db == nil {
		return nil, fmt.Errorf("compute wikilink edges: nil db")
	}
	return computeWikilinkEdges(ctx, db, documentID, brainID, tenantID, limit)
}

// UpsertDocumentEdges inserts or updates edge rows in memory.document_edges.
func UpsertDocumentEdges(ctx context.Context, tx *sql.Tx, brainID, tenantID string, edges []UpsertEdge) error {
	if tx == nil {
		return fmt.Errorf("upsert document edges: nil tx")
	}
	if len(edges) == 0 {
		return nil
	}

	deduped := dedupeEdges(edges)
	sourceIDs := make([]string, 0, len(deduped))
	targetIDs := make([]string, 0, len(deduped))
	edgeTypes := make([]string, 0, len(deduped))
	weights := make([]float64, 0, len(deduped))
	labels := make([]string, 0, len(deduped))
	for _, edge := range deduped {
		sourceIDs = append(sourceIDs, edge.SourceDocID)
		targetIDs = append(targetIDs, edge.TargetDocID)
		edgeTypes = append(edgeTypes, string(edge.EdgeType))
		weights = append(weights, edge.Weight)
		labels = append(labels, edge.Label)
	}

	_, err := tx.ExecContext(ctx, `
		insert into memory.document_edges (
			brain_id,
			tenant_id,
			source_doc_id,
			target_doc_id,
			edge_type,
			weight,
			label
		)
		select $1::uuid,
		       $2::uuid,
		       u.source_doc_id,
		       u.target_doc_id,
		       u.edge_type,
		       u.weight,
		       nullif(u.label, '')
		from unnest($3::uuid[], $4::uuid[], $5::text[], $6::real[], $7::text[])
		as u(source_doc_id, target_doc_id, edge_type, weight, label)
		on conflict (brain_id, source_doc_id, target_doc_id, edge_type)
		do update set
			weight = excluded.weight,
			label = excluded.label`,
		brainID,
		tenantID,
		pq.Array(sourceIDs),
		pq.Array(targetIDs),
		pq.Array(edgeTypes),
		pq.Array(weights),
		pq.Array(labels),
	)
	if err != nil {
		return fmt.Errorf("upsert document edges: %w", err)
	}
	return nil
}

// DeleteDocumentEdges deletes all edges touching a document, optionally filtered by edge type.
func DeleteDocumentEdges(ctx context.Context, tx *sql.Tx, documentID, brainID, tenantID string, edgeTypes []DocumentEdgeType) error {
	if tx == nil {
		return fmt.Errorf("delete document edges: nil tx")
	}
	types := make([]string, 0, len(edgeTypes))
	for _, edgeType := range edgeTypes {
		types = append(types, string(edgeType))
	}
	_, err := tx.ExecContext(ctx, `
		delete from memory.document_edges
		where brain_id = $1::uuid
		  and tenant_id = $2::uuid
		  and (source_doc_id = $3::uuid or target_doc_id = $3::uuid)
		  and ($4 = 0 or edge_type = any($5::text[]))`,
		brainID,
		tenantID,
		documentID,
		len(types),
		pq.Array(types),
	)
	if err != nil {
		return fmt.Errorf("delete document edges: %w", err)
	}
	return nil
}

// ComputeAllEdgesForDocument recomputes and persists all edge types for one document.
func ComputeAllEdgesForDocument(ctx context.Context, db *sql.DB, documentID, brainID, tenantID string, opts ComputeAllEdgesOptions) error {
	if db == nil {
		return fmt.Errorf("compute all edges for document: nil db")
	}

	tx, err := db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("compute all edges for document: begin tx: %w", err)
	}
	defer func() { _ = tx.Rollback() }()

	recomputed := []DocumentEdgeType{
		EdgeSemanticSimilarity,
		EdgeSharedTag,
		EdgeSharedFolder,
		EdgeSameSession,
		EdgeSupersedes,
		EdgeSessionEpisode,
		EdgeEpisodeHeuristic,
		EdgeDocumentOntology,
		EdgeWikilink,
	}

	if err := DeleteDocumentEdges(ctx, tx, documentID, brainID, tenantID, recomputed); err != nil {
		return fmt.Errorf("compute all edges for document: delete existing edges: %w", err)
	}

	similar, err := findSimilarDocuments(ctx, tx, documentID, brainID, tenantID, SimilarityOptions{
		Threshold:    opts.SimilarityThreshold,
		Limit:        opts.SimilarityLimit,
		EmbeddingDim: opts.EmbeddingDim,
	})
	if err != nil {
		return fmt.Errorf("compute all edges for document: semantic similarity: %w", err)
	}
	sharedTag, err := computeSharedTagEdges(ctx, tx, documentID, brainID, tenantID)
	if err != nil {
		return fmt.Errorf("compute all edges for document: shared tag: %w", err)
	}
	relationLimit := intOrDefault(opts.RelationLimit, defaultRelationLimit)
	sharedFolder, err := computeSharedFolderEdges(ctx, tx, documentID, brainID, tenantID, relationLimit)
	if err != nil {
		return fmt.Errorf("compute all edges for document: shared folder: %w", err)
	}
	sameSession, err := computeSameSessionEdges(ctx, tx, documentID, brainID, tenantID, relationLimit)
	if err != nil {
		return fmt.Errorf("compute all edges for document: same session: %w", err)
	}
	supersedes, err := computeSupersedesEdges(ctx, tx, documentID, brainID, tenantID)
	if err != nil {
		return fmt.Errorf("compute all edges for document: supersedes: %w", err)
	}
	sessionEpisode, err := computeSessionEpisodeEdges(ctx, tx, documentID, brainID, tenantID, relationLimit)
	if err != nil {
		return fmt.Errorf("compute all edges for document: session episode: %w", err)
	}
	episodeHeuristic, err := computeEpisodeHeuristicEdges(ctx, tx, documentID, brainID, tenantID, relationLimit)
	if err != nil {
		return fmt.Errorf("compute all edges for document: episode heuristic: %w", err)
	}
	documentOntology, err := computeDocumentOntologyEdges(ctx, tx, documentID, brainID, tenantID)
	if err != nil {
		return fmt.Errorf("compute all edges for document: document ontology: %w", err)
	}
	wikilink, err := computeWikilinkEdges(ctx, tx, documentID, brainID, tenantID, relationLimit)
	if err != nil {
		return fmt.Errorf("compute all edges for document: wikilink: %w", err)
	}

	edges := make([]UpsertEdge, 0, len(similar)+len(sharedTag)+len(sharedFolder)+len(sameSession)+len(supersedes)+len(sessionEpisode)+len(episodeHeuristic)+len(documentOntology)+len(wikilink))
	edges = append(edges, toSemanticEdges(documentID, similar)...)
	edges = append(edges, toLabeledEdges(documentID, sharedTag, EdgeSharedTag)...)
	edges = append(edges, toWeightedEdges(documentID, sharedFolder, EdgeSharedFolder)...)
	edges = append(edges, toWeightedEdges(documentID, sameSession, EdgeSameSession)...)
	edges = append(edges, toLabeledEdges(documentID, supersedes, EdgeSupersedes)...)
	edges = append(edges, toWeightedEdges(documentID, sessionEpisode, EdgeSessionEpisode)...)
	edges = append(edges, toWeightedEdges(documentID, episodeHeuristic, EdgeEpisodeHeuristic)...)
	edges = append(edges, toLabeledEdges(documentID, documentOntology, EdgeDocumentOntology)...)
	edges = append(edges, toLabeledEdges(documentID, wikilink, EdgeWikilink)...)

	if err := UpsertDocumentEdges(ctx, tx, brainID, tenantID, edges); err != nil {
		return fmt.Errorf("compute all edges for document: upsert edges: %w", err)
	}
	if err := tx.Commit(); err != nil {
		return fmt.Errorf("compute all edges for document: commit: %w", err)
	}
	return nil
}

func computeDocumentCentroid(ctx context.Context, q queryer, documentID, tenantID string, dim EmbeddingDim) ([]float64, error) {
	queryByDim := map[EmbeddingDim]string{
		Dim3072: `select to_json(avg(embedding_3072::vector)) from memory.chunks where document_id = $1::uuid and tenant_id = $2::uuid and embedding_3072 is not null`,
		Dim1024: `select to_json(avg(embedding::vector)) from memory.chunks where document_id = $1::uuid and tenant_id = $2::uuid and embedding is not null`,
	}
	query := queryByDim[normalizeDim(dim)]
	var raw []byte
	if err := q.QueryRowContext(ctx, query, documentID, tenantID).Scan(&raw); err != nil {
		return nil, fmt.Errorf("compute document centroid: query centroid: %w", err)
	}
	if len(raw) == 0 || string(raw) == "null" {
		return nil, nil
	}
	var parsed []float64
	if err := json.Unmarshal(raw, &parsed); err != nil {
		return nil, fmt.Errorf("compute document centroid: parse centroid json: %w", err)
	}
	return parsed, nil
}

func findSimilarDocuments(ctx context.Context, q queryer, documentID, brainID, tenantID string, opts SimilarityOptions) ([]SimilarDocument, error) {
	threshold := floatOrDefault(opts.Threshold, defaultSimilarityThreshold)
	limit := intOrDefault(opts.Limit, defaultSimilarityLimit)
	// Known limitation: this computes target centroids on demand (O(N) per source document).
	// We keep this until a persisted centroid column + index lands in schema.
	queryByDim := map[EmbeddingDim]string{
		Dim3072: `
			with source_centroid as (
				select avg(c.embedding_3072::vector)::halfvec(3072) as centroid
				from memory.chunks c
				where c.document_id = $1::uuid
				  and c.tenant_id = $3::uuid
				  and c.embedding_3072 is not null
			),
			target_centroids as (
				select d.document_id, avg(c.embedding_3072::vector)::halfvec(3072) as centroid
				from memory.documents d
				join memory.chunks c
				  on c.document_id = d.document_id
				 and c.tenant_id = d.tenant_id
				where d.brain_id = $2::uuid
				  and d.tenant_id = $3::uuid
				  and d.document_id != $1::uuid
				  and c.embedding_3072 is not null
				group by d.document_id
			)
			select tc.document_id::text, 1 - (tc.centroid <=> sc.centroid)
			from target_centroids tc
			cross join source_centroid sc
			where sc.centroid is not null
			  and 1 - (tc.centroid <=> sc.centroid) >= $4
			order by tc.centroid <=> sc.centroid asc
			limit $5`,
		Dim1024: `
			with source_centroid as (
				select avg(c.embedding::vector)::halfvec(1024) as centroid
				from memory.chunks c
				where c.document_id = $1::uuid
				  and c.tenant_id = $3::uuid
				  and c.embedding is not null
			),
			target_centroids as (
				select d.document_id, avg(c.embedding::vector)::halfvec(1024) as centroid
				from memory.documents d
				join memory.chunks c
				  on c.document_id = d.document_id
				 and c.tenant_id = d.tenant_id
				where d.brain_id = $2::uuid
				  and d.tenant_id = $3::uuid
				  and d.document_id != $1::uuid
				  and c.embedding is not null
				group by d.document_id
			)
			select tc.document_id::text, 1 - (tc.centroid <=> sc.centroid)
			from target_centroids tc
			cross join source_centroid sc
			where sc.centroid is not null
			  and 1 - (tc.centroid <=> sc.centroid) >= $4
			order by tc.centroid <=> sc.centroid asc
			limit $5`,
	}
	rows, err := q.QueryContext(ctx, queryByDim[normalizeDim(opts.EmbeddingDim)], documentID, brainID, tenantID, threshold, limit)
	if err != nil {
		return nil, fmt.Errorf("find similar documents: query: %w", err)
	}
	defer func() { _ = rows.Close() }()

	out := make([]SimilarDocument, 0)
	for rows.Next() {
		var item SimilarDocument
		if err := rows.Scan(&item.TargetDocID, &item.Similarity); err != nil {
			return nil, fmt.Errorf("find similar documents: scan row: %w", err)
		}
		out = append(out, item)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("find similar documents: iterate rows: %w", err)
	}
	return out, nil
}

func computeSharedTagEdges(ctx context.Context, q queryer, documentID, brainID, tenantID string) ([]LabeledWeightedEdge, error) {
	return runLabeledWeightedQuery(ctx, q, `
		with source_tags as (
			select distinct jsonb_array_elements_text(case when jsonb_typeof(d.metadata->'tags') = 'array' then d.metadata->'tags' else '[]'::jsonb end) as tag
			from memory.documents d
			where d.document_id = $1::uuid and d.brain_id = $2::uuid and d.tenant_id = $3::uuid
		), target_tags as (
			select d.document_id, jsonb_array_elements_text(case when jsonb_typeof(d.metadata->'tags') = 'array' then d.metadata->'tags' else '[]'::jsonb end) as tag
			from memory.documents d
			where d.brain_id = $2::uuid and d.tenant_id = $3::uuid and d.document_id != $1::uuid
		), source_count as (
			select count(*)::real as cnt from source_tags
		), target_counts as (
			select tt.document_id, count(distinct tt.tag)::real as cnt from target_tags tt group by tt.document_id
		), intersections as (
			select tt.document_id, count(distinct tt.tag)::real as intersection_count, string_agg(distinct tt.tag, ',' order by tt.tag) as shared_tags
			from target_tags tt join source_tags st on st.tag = tt.tag group by tt.document_id
		)
		select i.document_id::text, (i.intersection_count / nullif((sc.cnt + tc.cnt - i.intersection_count), 0)), i.shared_tags
		from intersections i cross join source_count sc join target_counts tc on tc.document_id = i.document_id
		where i.intersection_count > 0
		order by i.intersection_count desc, i.document_id
		limit $4`, documentID, brainID, tenantID, defaultRelationLimit)
}

func computeSharedFolderEdges(ctx context.Context, q queryer, documentID, brainID, tenantID string, limit int) ([]WeightedEdge, error) {
	return runWeightedQuery(ctx, q, `
		with source as (
			select regexp_replace(path, '/[^/]*$', '') as directory
			from memory.documents
			where document_id = $1::uuid and brain_id = $2::uuid and tenant_id = $3::uuid
		)
		select d.document_id::text, $5::double precision
		from memory.documents d cross join source s
		where d.brain_id = $2::uuid and d.tenant_id = $3::uuid and d.document_id != $1::uuid
		  and regexp_replace(d.path, '/[^/]*$', '') = s.directory
		limit $4`, documentID, brainID, tenantID, intOrDefault(limit, defaultRelationLimit), sharedFolderWeight)
}

func computeSameSessionEdges(ctx context.Context, q queryer, documentID, brainID, tenantID string, limit int) ([]WeightedEdge, error) {
	return runWeightedQuery(ctx, q, `
		with source as (
			select coalesce(metadata->>'session_id', metadata->>'sessionId') as session_id
			from memory.documents
			where document_id = $1::uuid and brain_id = $2::uuid and tenant_id = $3::uuid
		)
		select d.document_id::text, $5::double precision
		from memory.documents d cross join source s
		where d.brain_id = $2::uuid and d.tenant_id = $3::uuid and d.document_id != $1::uuid
		  and s.session_id is not null
		  and coalesce(d.metadata->>'session_id', d.metadata->>'sessionId') = s.session_id
		limit $4`, documentID, brainID, tenantID, intOrDefault(limit, defaultRelationLimit), sameSessionWeight)
}

func computeSupersedesEdges(ctx context.Context, q queryer, documentID, brainID, tenantID string) ([]LabeledWeightedEdge, error) {
	return runLabeledWeightedQuery(ctx, q, `
		with source as (
			select nullif(trim(coalesce(metadata->>'supersedes', '')), '') as supersedes
			from memory.documents
			where document_id = $1::uuid and brain_id = $2::uuid and tenant_id = $3::uuid
		)
		select d.document_id::text, $4::double precision, s.supersedes
		from memory.documents d cross join source s
		where d.brain_id = $2::uuid and d.tenant_id = $3::uuid and d.document_id != $1::uuid
		  and s.supersedes is not null
		  and (
			d.path = s.supersedes
			or d.path = s.supersedes || '.md'
			or d.path like '%/' || replace(replace(replace(s.supersedes, '\\', '\\\\'), '%', '\\%'), '_', '\\_') escape '\\'
			or d.path like '%/' || replace(replace(replace(s.supersedes, '\\', '\\\\'), '%', '\\%'), '_', '\\_') || '.md' escape '\\'
		  )
		limit $5`, documentID, brainID, tenantID, supersedesWeight, defaultRelationLimit)
}

func computeSessionEpisodeEdges(ctx context.Context, q queryer, documentID, brainID, tenantID string, limit int) ([]WeightedEdge, error) {
	return runWeightedQuery(ctx, q, `
		with source as (
			select coalesce(metadata->>'session_id', metadata->>'sessionId') as session_id
			from memory.documents
			where document_id = $1::uuid and brain_id = $2::uuid and tenant_id = $3::uuid
		)
		select d.document_id::text, $4::double precision
		from memory.documents d cross join source s
		where d.brain_id = $2::uuid and d.tenant_id = $3::uuid and d.document_id != $1::uuid
		  and s.session_id is not null and d.path like 'episodes/%'
		  and (coalesce(d.metadata->>'session_id', d.metadata->>'sessionId') = s.session_id or d.path = 'episodes/' || s.session_id || '.md' or d.path = 'episodes/session-' || s.session_id || '.md')
		limit $5`,
		documentID, brainID, tenantID, sessionEpisodeWeight, intOrDefault(limit, defaultRelationLimit))
}

func computeEpisodeHeuristicEdges(ctx context.Context, q queryer, documentID, brainID, tenantID string, limit int) ([]WeightedEdge, error) {
	return runWeightedQuery(ctx, q, `
		with source as (
			select path, coalesce(metadata->>'session_id', metadata->>'sessionId') as session_id
			from memory.documents
			where document_id = $1::uuid and brain_id = $2::uuid and tenant_id = $3::uuid
		)
		select d.document_id::text, $4::double precision
		from memory.documents d cross join source s
		where d.brain_id = $2::uuid and d.tenant_id = $3::uuid and d.document_id != $1::uuid
		  and ((s.path like 'episodes/%' and (d.path like 'memory/%/heuristic-%' or d.path like 'memory/%/anti-pattern-%'))
		       or ((s.path like 'memory/%/heuristic-%' or s.path like 'memory/%/anti-pattern-%') and d.path like 'episodes/%'))
		  and s.session_id is not null
		  and coalesce(d.metadata->>'session_id', d.metadata->>'sessionId') = s.session_id
		limit $5`,
		documentID, brainID, tenantID, episodeHeuristicWeight, intOrDefault(limit, defaultRelationLimit))
}

func computeDocumentOntologyEdges(ctx context.Context, q queryer, documentID, brainID, tenantID string) ([]LabeledWeightedEdge, error) {
	return runLabeledWeightedQuery(ctx, q, `
		with source as (
			select path, nullif(trim(coalesce(metadata->>'ontology_type', metadata->>'ontologyType', '')), '') as ontology_type
			from memory.documents
			where document_id = $1::uuid and brain_id = $2::uuid and tenant_id = $3::uuid
		)
		select d.document_id::text, $4::double precision, s.ontology_type
		from memory.documents d cross join source s
		where d.brain_id = $2::uuid and d.tenant_id = $3::uuid and d.document_id != $1::uuid
		  and s.ontology_type is not null
		  and ((s.path like 'ontology/%' and coalesce(d.metadata->>'ontology_type', d.metadata->>'ontologyType') = s.ontology_type)
		       or (s.path not like 'ontology/%' and d.path like 'ontology/%' and coalesce(d.metadata->>'ontology_type', d.metadata->>'ontologyType') = s.ontology_type))
		limit $5`,
		documentID, brainID, tenantID, documentOntologyWeight, defaultRelationLimit)
}

func computeWikilinkEdges(ctx context.Context, q queryer, documentID, brainID, tenantID string, limit int) ([]LabeledWeightedEdge, error) {
	return runLabeledWeightedQuery(ctx, q, `
		with source as (
			select coalesce(convert_from(content, 'UTF8'), '') as body
			from memory.documents
			where document_id = $1::uuid and brain_id = $2::uuid and tenant_id = $3::uuid
		), links as (
			select distinct trim(split_part(matches[1], '|', 1)) as target
			from source s,
			lateral regexp_matches(s.body, '\\[\\[([^\\]]+)\\]\\]', 'g') as matches
			where trim(split_part(matches[1], '|', 1)) <> ''
		)
		select d.document_id::text, $4::double precision, l.target
		from memory.documents d
		join links l
		  on d.path = l.target
		  or d.path = l.target || '.md'
		  or d.path like '%/' || replace(replace(replace(l.target, '\\', '\\\\'), '%', '\\%'), '_', '\\_') escape '\\'
		  or d.path like '%/' || replace(replace(replace(l.target, '\\', '\\\\'), '%', '\\%'), '_', '\\_') || '.md' escape '\\'
		where d.brain_id = $2::uuid and d.tenant_id = $3::uuid and d.document_id != $1::uuid
		limit $5`,
		documentID, brainID, tenantID, wikilinkWeight, intOrDefault(limit, defaultRelationLimit))
}

func runWeightedQuery(ctx context.Context, q queryer, sqlQuery string, args ...any) ([]WeightedEdge, error) {
	rows, err := q.QueryContext(ctx, sqlQuery, args...)
	if err != nil {
		return nil, fmt.Errorf("run weighted query: %w", err)
	}
	defer func() { _ = rows.Close() }()
	out := make([]WeightedEdge, 0)
	for rows.Next() {
		var edge WeightedEdge
		if err := rows.Scan(&edge.TargetDocID, &edge.Weight); err != nil {
			return nil, fmt.Errorf("run weighted query: scan row: %w", err)
		}
		out = append(out, edge)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("run weighted query: iterate rows: %w", err)
	}
	return out, nil
}

func runLabeledWeightedQuery(ctx context.Context, q queryer, sqlQuery string, args ...any) ([]LabeledWeightedEdge, error) {
	rows, err := q.QueryContext(ctx, sqlQuery, args...)
	if err != nil {
		return nil, fmt.Errorf("run labelled weighted query: %w", err)
	}
	defer func() { _ = rows.Close() }()
	out := make([]LabeledWeightedEdge, 0)
	for rows.Next() {
		var edge LabeledWeightedEdge
		if err := rows.Scan(&edge.TargetDocID, &edge.Weight, &edge.Label); err != nil {
			return nil, fmt.Errorf("run labelled weighted query: scan row: %w", err)
		}
		out = append(out, edge)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("run labelled weighted query: iterate rows: %w", err)
	}
	return out, nil
}

func dedupeEdges(edges []UpsertEdge) []UpsertEdge {
	positions := make(map[string]int, len(edges))
	out := make([]UpsertEdge, 0, len(edges))
	for _, edge := range edges {
		key := strings.Join([]string{edge.SourceDocID, edge.TargetDocID, string(edge.EdgeType)}, ":")
		idx, ok := positions[key]
		if !ok {
			positions[key] = len(out)
			out = append(out, edge)
			continue
		}
		out[idx] = edge
	}
	return out
}

func toSemanticEdges(sourceID string, values []SimilarDocument) []UpsertEdge {
	out := make([]UpsertEdge, 0, len(values))
	for _, item := range values {
		out = append(out, UpsertEdge{SourceDocID: sourceID, TargetDocID: item.TargetDocID, EdgeType: EdgeSemanticSimilarity, Weight: item.Similarity})
	}
	return out
}

func toWeightedEdges(sourceID string, values []WeightedEdge, edgeType DocumentEdgeType) []UpsertEdge {
	out := make([]UpsertEdge, 0, len(values))
	for _, item := range values {
		out = append(out, UpsertEdge{SourceDocID: sourceID, TargetDocID: item.TargetDocID, EdgeType: edgeType, Weight: item.Weight})
	}
	return out
}

func toLabeledEdges(sourceID string, values []LabeledWeightedEdge, edgeType DocumentEdgeType) []UpsertEdge {
	out := make([]UpsertEdge, 0, len(values))
	for _, item := range values {
		out = append(out, UpsertEdge{SourceDocID: sourceID, TargetDocID: item.TargetDocID, EdgeType: edgeType, Weight: item.Weight, Label: item.Label})
	}
	return out
}

func normalizeDim(dim EmbeddingDim) EmbeddingDim {
	if dim == Dim3072 {
		return Dim3072
	}
	return Dim1024
}

func intOrDefault(v int, d int) int {
	if v > 0 {
		return v
	}
	return d
}

func floatOrDefault(v float64, d float64) float64 {
	if v > 0 {
		return v
	}
	return d
}
