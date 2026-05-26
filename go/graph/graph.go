// SPDX-License-Identifier: Apache-2.0

package graph

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"time"

	"github.com/lib/pq"
)

const (
	defaultGraphMinWeight = 0.2
	defaultGraphMaxNodes  = 10000
	defaultGraphMaxEdges  = 50000
)

type graphNodeRow struct {
	id        string
	path      string
	size      int
	updatedAt time.Time
	chunks    int
	metadata  []byte
}

type graphEdgeRow struct {
	source string
	target string
	kind   DocumentEdgeType
	weight float64
	label  sql.NullString
}

// GetDocumentGraph fetches the full graph for one brain.
func GetDocumentGraph(
	ctx context.Context,
	db *sql.DB,
	brainID string,
	tenantID string,
	opts GraphQueryOptions,
) (GraphResponse, error) {
	if db == nil {
		return GraphResponse{}, fmt.Errorf("get document graph: nil db")
	}
	minWeight := defaultGraphMinWeight
	if opts.MinWeight > 0 {
		minWeight = opts.MinWeight
	}
	maxNodes := defaultGraphMaxNodes
	if opts.MaxNodes > 0 {
		maxNodes = opts.MaxNodes
	}
	maxEdges := defaultGraphMaxEdges
	if opts.MaxEdges > 0 {
		maxEdges = opts.MaxEdges
	}

	nodeRows, err := db.QueryContext(ctx, `
		select d.document_id::text,
		       d.path,
		       d.size,
		       d.updated_at,
		       d.metadata,
		       count(c.chunk_id)::int
		from memory.documents d
		left join memory.chunks c
		  on c.document_id = d.document_id
		 and c.tenant_id = d.tenant_id
		where d.brain_id = $1::uuid
		  and d.tenant_id = $2::uuid
		group by d.document_id, d.path, d.size, d.updated_at, d.metadata
		order by d.updated_at desc
		limit $3`,
		brainID,
		tenantID,
		maxNodes,
	)
	if err != nil {
		return GraphResponse{}, fmt.Errorf("get document graph: query nodes: %w", err)
	}
	defer func() { _ = nodeRows.Close() }()

	nodes := make([]GraphNode, 0)
	nodeIDs := make([]string, 0)
	loadedNodeCount := 0
	for nodeRows.Next() {
		loadedNodeCount++
		var row graphNodeRow
		if err := nodeRows.Scan(&row.id, &row.path, &row.size, &row.updatedAt, &row.metadata, &row.chunks); err != nil {
			return GraphResponse{}, fmt.Errorf("get document graph: scan node: %w", err)
		}
		metadata := map[string]any{}
		if len(row.metadata) > 0 {
			if err := json.Unmarshal(row.metadata, &metadata); err != nil {
				return GraphResponse{}, fmt.Errorf("get document graph: unmarshal metadata: %w", err)
			}
		}
		resolved := resolveEntityAndScope(row.path)
		node := GraphNode{
			ID:         row.id,
			Path:       row.path,
			Label:      fileLabel(row.path),
			EntityType: resolved.entity,
			Scope:      resolved.scope,
			Size:       row.size,
			ModTime:    row.updatedAt.UTC().Format(time.RFC3339Nano),
			ChunkCount: row.chunks,
			Metadata:   metadata,
		}
		if ontologyType, ok := metadata["ontology_type"].(string); ok {
			node.OntologyType = ontologyType
		}
		if ontologyType, ok := metadata["ontologyType"].(string); ok && node.OntologyType == "" {
			node.OntologyType = ontologyType
		}
		if ontologyLabel, ok := metadata["ontology_label"].(string); ok {
			node.OntologyLabel = ontologyLabel
		}
		if ontologyLabel, ok := metadata["ontologyLabel"].(string); ok && node.OntologyLabel == "" {
			node.OntologyLabel = ontologyLabel
		}
		nodes = append(nodes, node)
		nodeIDs = append(nodeIDs, row.id)
	}
	if err := nodeRows.Err(); err != nil {
		return GraphResponse{}, fmt.Errorf("get document graph: iterate nodes: %w", err)
	}

	entityFilter := map[EntityType]struct{}{}
	for _, entity := range opts.EntityTypes {
		entityFilter[entity] = struct{}{}
	}
	if len(entityFilter) > 0 {
		filtered := make([]GraphNode, 0, len(nodes))
		filteredIDs := make([]string, 0, len(nodeIDs))
		for _, node := range nodes {
			if _, ok := entityFilter[node.EntityType]; ok {
				filtered = append(filtered, node)
				filteredIDs = append(filteredIDs, node.ID)
			}
		}
		nodes = filtered
		nodeIDs = filteredIDs
	}

	if len(nodes) == 0 {
		return GraphResponse{
			Nodes:       []GraphNode{},
			Edges:       []GraphEdge{},
			Communities: []GraphCommunity{},
			Meta: GraphMeta{
				TotalNodes:       0,
				TotalEdges:       0,
				EntityTypeCounts: map[string]int{},
				EdgeTypeCounts:   map[string]int{},
				Truncated:        loadedNodeCount >= maxNodes,
			},
		}, nil
	}

	edgeTypes := make([]string, 0, len(opts.EdgeTypes))
	for _, edgeType := range opts.EdgeTypes {
		edgeTypes = append(edgeTypes, string(edgeType))
	}

	edgeRows, err := db.QueryContext(ctx, `
		select source_doc_id::text,
		       target_doc_id::text,
		       edge_type,
		       weight,
		       label
		from memory.document_edges
		where brain_id = $1::uuid
		  and tenant_id = $2::uuid
		  and weight >= $3
		  and source_doc_id = any($4::uuid[])
		  and target_doc_id = any($4::uuid[])
		  and ($5 = 0 or edge_type = any($6::text[]))
		order by weight desc
		limit $7`,
		brainID,
		tenantID,
		minWeight,
		pq.Array(nodeIDs),
		len(edgeTypes),
		pq.Array(edgeTypes),
		maxEdges,
	)
	if err != nil {
		return GraphResponse{}, fmt.Errorf("get document graph: query edges: %w", err)
	}
	defer func() { _ = edgeRows.Close() }()

	edges := make([]GraphEdge, 0)
	for edgeRows.Next() {
		var row graphEdgeRow
		if err := edgeRows.Scan(&row.source, &row.target, &row.kind, &row.weight, &row.label); err != nil {
			return GraphResponse{}, fmt.Errorf("get document graph: scan edge: %w", err)
		}
		edge := GraphEdge{Source: row.source, Target: row.target, EdgeType: row.kind, Weight: row.weight}
		if row.label.Valid {
			edge.Label = row.label.String
		}
		edges = append(edges, edge)
	}
	if err := edgeRows.Err(); err != nil {
		return GraphResponse{}, fmt.Errorf("get document graph: iterate edges: %w", err)
	}

	entityCounts := map[string]int{}
	for _, node := range nodes {
		entityCounts[string(node.EntityType)]++
	}
	edgeCounts := map[string]int{}
	for _, edge := range edges {
		edgeCounts[string(edge.EdgeType)]++
	}

	return GraphResponse{
		Nodes:       nodes,
		Edges:       edges,
		Communities: []GraphCommunity{},
		Meta: GraphMeta{
			TotalNodes:       len(nodes),
			TotalEdges:       len(edges),
			EntityTypeCounts: entityCounts,
			EdgeTypeCounts:   edgeCounts,
			Truncated:        loadedNodeCount >= maxNodes || len(edges) >= maxEdges,
		},
	}, nil
}

// GetDocumentStats fetches aggregate graph statistics.
func GetDocumentStats(ctx context.Context, db *sql.DB, brainID string, tenantID string) (StatsResponse, error) {
	if db == nil {
		return StatsResponse{}, fmt.Errorf("get document stats: nil db")
	}

	var out StatsResponse
	var lastActivity sql.NullTime
	if err := db.QueryRowContext(ctx, `
		with docs as (
			select document_id, tenant_id, updated_at
			from memory.documents
			where brain_id = $1::uuid
			  and tenant_id = $2::uuid
		)
		select (select count(*)::int from docs),
		       (select count(*)::int
		        from memory.chunks c
		        join docs d
		          on d.document_id = c.document_id
		         and d.tenant_id = c.tenant_id),
		       (select count(*)::int
		        from memory.document_edges e
		        where e.brain_id = $1::uuid
		          and e.tenant_id = $2::uuid),
		       (select max(updated_at) from docs)`,
		brainID,
		tenantID,
	).Scan(&out.DocumentCount, &out.ChunkCount, &out.EdgeCount, &lastActivity); err != nil {
		return StatsResponse{}, fmt.Errorf("get document stats: counts query: %w", err)
	}

	entityRows, err := db.QueryContext(ctx, `
		select path
		from memory.documents
		where brain_id = $1::uuid
		  and tenant_id = $2::uuid`,
		brainID,
		tenantID,
	)
	if err != nil {
		return StatsResponse{}, fmt.Errorf("get document stats: entity counts query: %w", err)
	}
	defer func() { _ = entityRows.Close() }()
	out.EntityTypeCounts = map[string]int{}
	for entityRows.Next() {
		var path string
		if err := entityRows.Scan(&path); err != nil {
			return StatsResponse{}, fmt.Errorf("get document stats: entity path scan: %w", err)
		}
		resolved := resolveEntityAndScope(path)
		out.EntityTypeCounts[string(resolved.entity)]++
	}
	if err := entityRows.Err(); err != nil {
		return StatsResponse{}, fmt.Errorf("get document stats: entity counts iterate: %w", err)
	}

	edgeRows, err := db.QueryContext(ctx, `
		select edge_type, count(*)::int
		from memory.document_edges
		where brain_id = $1::uuid
		  and tenant_id = $2::uuid
		group by edge_type`,
		brainID,
		tenantID,
	)
	if err != nil {
		return StatsResponse{}, fmt.Errorf("get document stats: edge counts query: %w", err)
	}
	defer func() { _ = edgeRows.Close() }()
	out.EdgeTypeCounts = map[string]int{}
	for edgeRows.Next() {
		var key string
		var count int
		if err := edgeRows.Scan(&key, &count); err != nil {
			return StatsResponse{}, fmt.Errorf("get document stats: edge count scan: %w", err)
		}
		out.EdgeTypeCounts[key] = count
	}
	if err := edgeRows.Err(); err != nil {
		return StatsResponse{}, fmt.Errorf("get document stats: edge counts iterate: %w", err)
	}

	if lastActivity.Valid {
		iso := lastActivity.Time.UTC().Format(time.RFC3339Nano)
		out.LastActivityAt = &iso
	}
	return out, nil
}
