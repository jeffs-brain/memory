// SPDX-License-Identifier: Apache-2.0

package graph

import (
	"context"
	"database/sql"
	"os"
	"strings"
	"testing"

	_ "github.com/jackc/pgx/v5/stdlib"
)

func requireTestDB(t *testing.T) *sql.DB {
	t.Helper()

	pgURL := strings.TrimSpace(os.Getenv("MEMORY_TEST_PG_URL"))
	if pgURL == "" {
		t.Skip("MEMORY_TEST_PG_URL is unset; skipping integration test")
	}

	db, err := sql.Open("pgx", pgURL)
	if err != nil {
		t.Fatalf("sql.Open(pgx): %v", err)
	}
	t.Cleanup(func() { _ = db.Close() })

	if err := db.PingContext(context.Background()); err != nil {
		t.Fatalf("db ping: %v", err)
	}
	return db
}

func resetGraphSchema(t *testing.T, db *sql.DB) {
	t.Helper()

	ctx := context.Background()
	statements := []string{
		`drop schema if exists memory cascade`,
		`create schema memory`,
		`create table memory.documents (
			document_id uuid primary key,
			brain_id uuid not null,
			tenant_id uuid not null,
			path text not null,
			size int not null default 0,
			updated_at timestamptz not null,
			metadata jsonb,
			content bytea
		)`,
		`create table memory.chunks (
			chunk_id uuid primary key,
			document_id uuid not null,
			tenant_id uuid not null,
			embedding halfvec(1024),
			embedding_3072 halfvec(3072)
		)`,
		`create table memory.document_edges (
			brain_id uuid not null,
			tenant_id uuid not null,
			source_doc_id uuid not null,
			target_doc_id uuid not null,
			edge_type text not null,
			weight real not null,
			label text,
			primary key (brain_id, source_doc_id, target_doc_id, edge_type)
		)`,
	}
	for _, statement := range statements {
		if _, err := db.ExecContext(ctx, statement); err != nil {
			t.Fatalf("schema setup failed for %q: %v", statement, err)
		}
	}
}

func TestUpsertDocumentEdges_DedupesAndUpdates(t *testing.T) {
	db := requireTestDB(t)
	resetGraphSchema(t, db)

	ctx := context.Background()
	tx, err := db.BeginTx(ctx, nil)
	if err != nil {
		t.Fatalf("begin tx: %v", err)
	}

	brainID := "11111111-1111-1111-1111-111111111111"
	tenantID := "22222222-2222-2222-2222-222222222222"
	sourceID := "33333333-3333-3333-3333-333333333333"
	targetID := "44444444-4444-4444-4444-444444444444"

	if err := UpsertDocumentEdges(ctx, tx, brainID, tenantID, []UpsertEdge{
		{SourceDocID: sourceID, TargetDocID: targetID, EdgeType: EdgeSharedTag, Weight: 0.25, Label: "   first   "},
		{SourceDocID: sourceID, TargetDocID: targetID, EdgeType: EdgeSharedTag, Weight: 0.9, Label: "  updated  "},
	}); err != nil {
		t.Fatalf("upsert edges: %v", err)
	}
	if err := tx.Commit(); err != nil {
		t.Fatalf("commit tx: %v", err)
	}

	var count int
	if err := db.QueryRowContext(ctx, `select count(*) from memory.document_edges`).Scan(&count); err != nil {
		t.Fatalf("count edges: %v", err)
	}
	if count != 1 {
		t.Fatalf("expected 1 deduped edge row, got %d", count)
	}

	var weight float64
	var label sql.NullString
	if err := db.QueryRowContext(ctx, `
		select weight, label
		from memory.document_edges
		where brain_id = $1::uuid and tenant_id = $2::uuid and source_doc_id = $3::uuid and target_doc_id = $4::uuid and edge_type = $5`,
		brainID, tenantID, sourceID, targetID, string(EdgeSharedTag),
	).Scan(&weight, &label); err != nil {
		t.Fatalf("select edge: %v", err)
	}
	if weight != 0.9 {
		t.Fatalf("expected updated weight 0.9, got %v", weight)
	}
	if !label.Valid || label.String != "  updated  " {
		t.Fatalf("expected untrimmed label '  updated  ', got valid=%v value=%q", label.Valid, label.String)
	}
}

func TestDeleteDocumentEdges_FiltersByEdgeType(t *testing.T) {
	db := requireTestDB(t)
	resetGraphSchema(t, db)

	ctx := context.Background()
	tx, err := db.BeginTx(ctx, nil)
	if err != nil {
		t.Fatalf("begin tx: %v", err)
	}

	brainID := "11111111-1111-1111-1111-111111111111"
	tenantID := "22222222-2222-2222-2222-222222222222"
	sourceID := "33333333-3333-3333-3333-333333333333"
	targetA := "44444444-4444-4444-4444-444444444444"
	targetB := "55555555-5555-5555-5555-555555555555"

	seed := []UpsertEdge{
		{SourceDocID: sourceID, TargetDocID: targetA, EdgeType: EdgeSharedTag, Weight: 0.5, Label: "tag"},
		{SourceDocID: sourceID, TargetDocID: targetB, EdgeType: EdgeSharedFolder, Weight: 0.3},
		{SourceDocID: targetA, TargetDocID: sourceID, EdgeType: EdgeWikilink, Weight: 0.7, Label: "note"},
	}
	if err := UpsertDocumentEdges(ctx, tx, brainID, tenantID, seed); err != nil {
		t.Fatalf("seed edges: %v", err)
	}
	if err := DeleteDocumentEdges(ctx, tx, sourceID, brainID, tenantID, []DocumentEdgeType{EdgeSharedTag, EdgeWikilink}); err != nil {
		t.Fatalf("delete edges: %v", err)
	}
	if err := tx.Commit(); err != nil {
		t.Fatalf("commit tx: %v", err)
	}

	rows, err := db.QueryContext(ctx, `select edge_type from memory.document_edges order by edge_type`)
	if err != nil {
		t.Fatalf("query remaining edges: %v", err)
	}
	defer func() { _ = rows.Close() }()

	types := make([]string, 0)
	for rows.Next() {
		var edgeType string
		if err := rows.Scan(&edgeType); err != nil {
			t.Fatalf("scan remaining edge: %v", err)
		}
		types = append(types, edgeType)
	}
	if err := rows.Err(); err != nil {
		t.Fatalf("iterate remaining edges: %v", err)
	}
	if len(types) != 1 || types[0] != string(EdgeSharedFolder) {
		t.Fatalf("expected only %q to remain, got %v", EdgeSharedFolder, types)
	}
}

func TestEdgeConversionHelpers(t *testing.T) {
	semantic := toSemanticEdges("src", []SimilarDocument{{TargetDocID: "a", Similarity: 0.8}})
	if len(semantic) != 1 || semantic[0].EdgeType != EdgeSemanticSimilarity || semantic[0].Weight != 0.8 {
		t.Fatalf("unexpected semantic edges: %+v", semantic)
	}

	weighted := toWeightedEdges("src", []WeightedEdge{{TargetDocID: "b", Weight: 0.3}}, EdgeSharedFolder)
	if len(weighted) != 1 || weighted[0].EdgeType != EdgeSharedFolder || weighted[0].TargetDocID != "b" {
		t.Fatalf("unexpected weighted edges: %+v", weighted)
	}

	labelled := toLabeledEdges("src", []LabeledWeightedEdge{{TargetDocID: "c", Weight: 0.5, Label: "x"}}, EdgeWikilink)
	if len(labelled) != 1 || labelled[0].EdgeType != EdgeWikilink || labelled[0].Label != "x" {
		t.Fatalf("unexpected labelled edges: %+v", labelled)
	}

	deduped := dedupeEdges([]UpsertEdge{
		{SourceDocID: "s", TargetDocID: "t", EdgeType: EdgeSharedTag, Weight: 0.1},
		{SourceDocID: "s", TargetDocID: "t", EdgeType: EdgeSharedTag, Weight: 0.9},
	})
	if len(deduped) != 1 || deduped[0].Weight != 0.9 {
		t.Fatalf("expected last value to win during dedupe, got %+v", deduped)
	}
}

func TestDeleteDocumentEdges_DeleteAllForDocument(t *testing.T) {
	db := requireTestDB(t)
	resetGraphSchema(t, db)

	ctx := context.Background()
	tx, err := db.BeginTx(ctx, nil)
	if err != nil {
		t.Fatalf("begin tx: %v", err)
	}

	brainID := "11111111-1111-1111-1111-111111111111"
	tenantID := "22222222-2222-2222-2222-222222222222"
	sourceID := "33333333-3333-3333-3333-333333333333"
	targetA := "44444444-4444-4444-4444-444444444444"
	targetB := "55555555-5555-5555-5555-555555555555"

	seed := []UpsertEdge{
		{SourceDocID: sourceID, TargetDocID: targetA, EdgeType: EdgeSharedTag, Weight: 0.5, Label: ""},
		{SourceDocID: targetB, TargetDocID: sourceID, EdgeType: EdgeWikilink, Weight: 0.7, Label: "note"},
	}
	if err := UpsertDocumentEdges(ctx, tx, brainID, tenantID, seed); err != nil {
		t.Fatalf("seed edges: %v", err)
	}
	if err := DeleteDocumentEdges(ctx, tx, sourceID, brainID, tenantID, nil); err != nil {
		t.Fatalf("delete all edges: %v", err)
	}
	if err := tx.Commit(); err != nil {
		t.Fatalf("commit tx: %v", err)
	}

	var count int
	if err := db.QueryRowContext(ctx, `select count(*) from memory.document_edges`).Scan(&count); err != nil {
		t.Fatalf("count edges: %v", err)
	}
	if count != 0 {
		t.Fatalf("expected all document edges deleted, got %d rows", count)
	}
}
