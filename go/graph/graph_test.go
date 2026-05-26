// SPDX-License-Identifier: Apache-2.0

package graph

import (
	"context"
	"testing"
	"time"
)

func TestGetDocumentGraph_ReturnsExpectedShapeAndValues(t *testing.T) {
	db := requireTestDB(t)
	resetGraphSchema(t, db)

	ctx := context.Background()
	brainID := "11111111-1111-1111-1111-111111111111"
	tenantID := "22222222-2222-2222-2222-222222222222"
	doc1 := "33333333-3333-3333-3333-333333333333"
	doc2 := "44444444-4444-4444-4444-444444444444"
	doc3 := "55555555-5555-5555-5555-555555555555"

	now := time.Now().UTC()
	if _, err := db.ExecContext(ctx, `
		insert into memory.documents (document_id, brain_id, tenant_id, path, size, updated_at, metadata, content)
		values
		($1::uuid, $4::uuid, $5::uuid, 'memory/project/note-a.md', 10, $6::timestamptz, '{"ontology_type":"concept","ontology_label":"Note A"}'::jsonb, E'[[wiki/topic-a]]'),
		($2::uuid, $4::uuid, $5::uuid, 'ontology/concept.md', 20, $7::timestamptz, '{"ontologyType":"concept","ontologyLabel":"Concept"}'::jsonb, null),
		($3::uuid, $4::uuid, $5::uuid, 'wiki/topic-a.md', 30, $8::timestamptz, '{}'::jsonb, null)`,
		doc1, doc2, doc3, brainID, tenantID, now, now.Add(-time.Minute), now.Add(-2*time.Minute),
	); err != nil {
		t.Fatalf("insert documents: %v", err)
	}

	if _, err := db.ExecContext(ctx, `
		insert into memory.chunks (chunk_id, document_id, tenant_id)
		values
		('66666666-6666-6666-6666-666666666661'::uuid, $1::uuid, $4::uuid),
		('66666666-6666-6666-6666-666666666662'::uuid, $1::uuid, $4::uuid),
		('66666666-6666-6666-6666-666666666663'::uuid, $2::uuid, $4::uuid)`,
		doc1, doc2, doc3, tenantID,
	); err != nil {
		t.Fatalf("insert chunks: %v", err)
	}

	if _, err := db.ExecContext(ctx, `
		insert into memory.document_edges (brain_id, tenant_id, source_doc_id, target_doc_id, edge_type, weight, label)
		values
		($1::uuid, $2::uuid, $3::uuid, $4::uuid, 'semantic_similarity', 0.91, null),
		($1::uuid, $2::uuid, $3::uuid, $5::uuid, 'wikilink', 0.70, 'topic-a'),
		($1::uuid, $2::uuid, $4::uuid, $5::uuid, 'shared_folder', 0.10, null)`,
		brainID, tenantID, doc1, doc2, doc3,
	); err != nil {
		t.Fatalf("insert edges: %v", err)
	}

	graphResp, err := GetDocumentGraph(ctx, db, brainID, tenantID, GraphQueryOptions{
		MinWeight: 0.5,
		EdgeTypes: []DocumentEdgeType{EdgeSemanticSimilarity, EdgeWikilink},
		MaxNodes:  10,
		MaxEdges:  10,
	})
	if err != nil {
		t.Fatalf("GetDocumentGraph: %v", err)
	}

	if len(graphResp.Nodes) != 3 {
		t.Fatalf("expected 3 nodes, got %d", len(graphResp.Nodes))
	}
	if len(graphResp.Edges) != 2 {
		t.Fatalf("expected 2 edges after filtering, got %d", len(graphResp.Edges))
	}

	nodesByID := map[string]GraphNode{}
	for _, node := range graphResp.Nodes {
		nodesByID[node.ID] = node
	}

	node1, ok := nodesByID[doc1]
	if !ok {
		t.Fatalf("missing node for doc1")
	}
	if node1.Label != "note-a.md" || node1.EntityType != EntityMemory || node1.Scope != "project" {
		t.Fatalf("unexpected node1 core fields: %+v", node1)
	}
	if node1.ChunkCount != 2 || node1.OntologyType != "concept" || node1.OntologyLabel != "Note A" {
		t.Fatalf("unexpected node1 metadata fields: %+v", node1)
	}

	node2, ok := nodesByID[doc2]
	if !ok {
		t.Fatalf("missing node for doc2")
	}
	if node2.EntityType != EntityOntology || node2.Scope != "other" || node2.ChunkCount != 1 {
		t.Fatalf("unexpected node2 fields: %+v", node2)
	}
	if node2.OntologyType != "concept" || node2.OntologyLabel != "Concept" {
		t.Fatalf("expected fallback ontology metadata keys to populate fields: %+v", node2)
	}

	if graphResp.Meta.TotalNodes != 3 || graphResp.Meta.TotalEdges != 2 || graphResp.Meta.Truncated {
		t.Fatalf("unexpected meta totals: %+v", graphResp.Meta)
	}
	if graphResp.Meta.EntityTypeCounts[string(EntityMemory)] != 1 || graphResp.Meta.EntityTypeCounts[string(EntityOntology)] != 1 || graphResp.Meta.EntityTypeCounts[string(EntityArticle)] != 1 {
		t.Fatalf("unexpected entity counts: %+v", graphResp.Meta.EntityTypeCounts)
	}
	if graphResp.Meta.EdgeTypeCounts[string(EdgeSemanticSimilarity)] != 1 || graphResp.Meta.EdgeTypeCounts[string(EdgeWikilink)] != 1 {
		t.Fatalf("unexpected edge counts: %+v", graphResp.Meta.EdgeTypeCounts)
	}
}

func TestGetDocumentGraph_EntityFilterRestrictsNodesAndEdges(t *testing.T) {
	db := requireTestDB(t)
	resetGraphSchema(t, db)

	ctx := context.Background()
	brainID := "11111111-1111-1111-1111-111111111111"
	tenantID := "22222222-2222-2222-2222-222222222222"
	docMemory := "33333333-3333-3333-3333-333333333333"
	docWiki := "44444444-4444-4444-4444-444444444444"

	if _, err := db.ExecContext(ctx, `
		insert into memory.documents (document_id, brain_id, tenant_id, path, size, updated_at, metadata)
		values
		($1::uuid, $3::uuid, $4::uuid, 'memory/global/a.md', 1, now(), '{}'::jsonb),
		($2::uuid, $3::uuid, $4::uuid, 'wiki/b.md', 1, now(), '{}'::jsonb)`,
		docMemory, docWiki, brainID, tenantID,
	); err != nil {
		t.Fatalf("insert docs: %v", err)
	}
	if _, err := db.ExecContext(ctx, `
		insert into memory.document_edges (brain_id, tenant_id, source_doc_id, target_doc_id, edge_type, weight)
		values ($1::uuid, $2::uuid, $3::uuid, $4::uuid, 'wikilink', 0.8)`,
		brainID, tenantID, docMemory, docWiki,
	); err != nil {
		t.Fatalf("insert edge: %v", err)
	}

	graphResp, err := GetDocumentGraph(ctx, db, brainID, tenantID, GraphQueryOptions{EntityTypes: []EntityType{EntityMemory}})
	if err != nil {
		t.Fatalf("GetDocumentGraph: %v", err)
	}
	if len(graphResp.Nodes) != 1 || graphResp.Nodes[0].ID != docMemory {
		t.Fatalf("expected only memory node, got %+v", graphResp.Nodes)
	}
	if len(graphResp.Edges) != 0 {
		t.Fatalf("expected no edges when only one node survives filtering, got %+v", graphResp.Edges)
	}
}

func TestGetDocumentStats_ReturnsAggregateCounts(t *testing.T) {
	db := requireTestDB(t)
	resetGraphSchema(t, db)

	ctx := context.Background()
	brainID := "11111111-1111-1111-1111-111111111111"
	tenantID := "22222222-2222-2222-2222-222222222222"
	doc1 := "33333333-3333-3333-3333-333333333333"
	doc2 := "44444444-4444-4444-4444-444444444444"

	if _, err := db.ExecContext(ctx, `
		insert into memory.documents (document_id, brain_id, tenant_id, path, size, updated_at, metadata)
		values
		($1::uuid, $3::uuid, $4::uuid, 'memory/agent/a.md', 1, now() - interval '1 hour', '{}'::jsonb),
		($2::uuid, $3::uuid, $4::uuid, 'wiki/b.md', 1, now(), '{}'::jsonb)`,
		doc1, doc2, brainID, tenantID,
	); err != nil {
		t.Fatalf("insert docs: %v", err)
	}
	if _, err := db.ExecContext(ctx, `
		insert into memory.chunks (chunk_id, document_id, tenant_id)
		values
		('66666666-6666-6666-6666-666666666661'::uuid, $1::uuid, $3::uuid),
		('66666666-6666-6666-6666-666666666662'::uuid, $2::uuid, $3::uuid)`,
		doc1, doc2, tenantID,
	); err != nil {
		t.Fatalf("insert chunks: %v", err)
	}
	if _, err := db.ExecContext(ctx, `
		insert into memory.document_edges (brain_id, tenant_id, source_doc_id, target_doc_id, edge_type, weight)
		values ($1::uuid, $2::uuid, $3::uuid, $4::uuid, 'shared_folder', 0.3)`,
		brainID, tenantID, doc1, doc2,
	); err != nil {
		t.Fatalf("insert edges: %v", err)
	}

	stats, err := GetDocumentStats(ctx, db, brainID, tenantID)
	if err != nil {
		t.Fatalf("GetDocumentStats: %v", err)
	}
	if stats.DocumentCount != 2 || stats.ChunkCount != 2 || stats.EdgeCount != 1 {
		t.Fatalf("unexpected totals: %+v", stats)
	}
	if stats.EntityTypeCounts[string(EntityMemory)] != 1 || stats.EntityTypeCounts[string(EntityArticle)] != 1 {
		t.Fatalf("unexpected entity type counts: %+v", stats.EntityTypeCounts)
	}
	if stats.EdgeTypeCounts[string(EdgeSharedFolder)] != 1 {
		t.Fatalf("unexpected edge type counts: %+v", stats.EdgeTypeCounts)
	}
	if stats.LastActivityAt == nil || *stats.LastActivityAt == "" {
		t.Fatalf("expected last activity timestamp, got %+v", stats.LastActivityAt)
	}
}

func TestGraphNilDBErrors(t *testing.T) {
	ctx := context.Background()
	if _, err := GetDocumentGraph(ctx, nil, "b", "t", GraphQueryOptions{}); err == nil {
		t.Fatal("expected GetDocumentGraph nil-db error")
	}
	if _, err := GetDocumentStats(ctx, nil, "b", "t"); err == nil {
		t.Fatal("expected GetDocumentStats nil-db error")
	}
}

func TestGetDocumentGraph_EmptyBrainTruncationSignal(t *testing.T) {
	db := requireTestDB(t)
	resetGraphSchema(t, db)

	ctx := context.Background()
	brainID := "11111111-1111-1111-1111-111111111111"
	tenantID := "22222222-2222-2222-2222-222222222222"

	graphResp, err := GetDocumentGraph(ctx, db, brainID, tenantID, GraphQueryOptions{MaxNodes: 1, MaxEdges: 1})
	if err != nil {
		t.Fatalf("GetDocumentGraph: %v", err)
	}
	if len(graphResp.Nodes) != 0 || len(graphResp.Edges) != 0 {
		t.Fatalf("expected empty graph, got nodes=%d edges=%d", len(graphResp.Nodes), len(graphResp.Edges))
	}
	if graphResp.Meta.Truncated {
		t.Fatalf("expected non-truncated empty graph, got %+v", graphResp.Meta)
	}
}
