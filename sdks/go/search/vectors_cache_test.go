// SPDX-License-Identifier: Apache-2.0

package search

import (
	"context"
	"database/sql"
	"testing"
	"time"

	_ "modernc.org/sqlite"
)

// TestVectorIndex_SchemaEvolvesToIncludeTitleSummary opens an
// in-memory SQLite handle pre-populated with the pre-hydrated
// schema (no title / summary / topic columns) and a single row of
// historical data, then runs NewVectorIndex to exercise the ALTER
// TABLE migration inside Schema. The pre-existing row must still
// be queryable afterwards with empty strings in the new fields,
// and a second Schema call must be a no-op.
func TestVectorIndex_SchemaEvolvesToIncludeTitleSummary(t *testing.T) {
	db, err := sql.Open("sqlite", ":memory:")
	if err != nil {
		t.Fatalf("opening in-memory db: %v", err)
	}
	t.Cleanup(func() { db.Close() })

	legacy := `
CREATE TABLE knowledge_embeddings (
    path       TEXT NOT NULL,
    checksum   TEXT NOT NULL,
    dim        INTEGER NOT NULL,
    model      TEXT NOT NULL,
    vector     BLOB NOT NULL,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (path, model)
);
CREATE INDEX idx_embeddings_model ON knowledge_embeddings(model);
`
	if _, err := db.Exec(legacy); err != nil {
		t.Fatalf("applying legacy schema: %v", err)
	}

	legacyVec := []float32{0.1, 0.2, 0.3, 0.4}
	if _, err := db.Exec(
		`INSERT INTO knowledge_embeddings (path, checksum, dim, model, vector)
		 VALUES (?, ?, ?, ?, ?)`,
		"wiki/legacy.md",
		"cs-legacy",
		len(legacyVec),
		testModel,
		packFloat32(legacyVec),
	); err != nil {
		t.Fatalf("seeding legacy row: %v", err)
	}

	vi, err := NewVectorIndex(db)
	if err != nil {
		t.Fatalf("NewVectorIndex with legacy data: %v", err)
	}

	if err := vi.Schema(context.Background()); err != nil {
		t.Fatalf("second Schema call: %v", err)
	}

	cols, err := vi.columnSet(context.Background())
	if err != nil {
		t.Fatalf("columnSet: %v", err)
	}
	for _, col := range hydratedColumns {
		if _, ok := cols[col]; !ok {
			t.Errorf("after Schema, column %q missing from knowledge_embeddings", col)
		}
	}

	loaded, err := vi.LoadAll(context.Background(), testModel)
	if err != nil {
		t.Fatalf("LoadAll after migration: %v", err)
	}
	if len(loaded) != 1 {
		t.Fatalf("got %d entries, want 1", len(loaded))
	}
	entry := loaded[0]
	if entry.Path != "wiki/legacy.md" {
		t.Errorf("path = %q, want wiki/legacy.md", entry.Path)
	}
	if entry.Checksum != "cs-legacy" {
		t.Errorf("checksum = %q, want cs-legacy", entry.Checksum)
	}
	if entry.Title != "" || entry.Summary != "" || entry.Topic != "" {
		t.Errorf("migrated row has non-empty hydrated fields: title=%q summary=%q topic=%q",
			entry.Title, entry.Summary, entry.Topic)
	}
	if len(entry.Vector) != len(legacyVec) {
		t.Fatalf("vector length = %d, want %d", len(entry.Vector), len(legacyVec))
	}
	for i := range legacyVec {
		if entry.Vector[i] != legacyVec[i] {
			t.Errorf("vec[%d] = %v, want %v", i, entry.Vector[i], legacyVec[i])
		}
	}

	fresh := VectorEntry{
		Path:     "wiki/fresh.md",
		Checksum: "cs-fresh",
		Model:    testModel,
		Vector:   []float32{1, 0, 0, 0},
		Title:    "Fresh Article",
		Summary:  "A new hydrated row",
		Topic:    "wiki",
	}
	if err := vi.Store(context.Background(), fresh); err != nil {
		t.Fatalf("Store hydrated entry: %v", err)
	}

	all, err := vi.LoadAll(context.Background(), testModel)
	if err != nil {
		t.Fatalf("LoadAll after hydrated store: %v", err)
	}
	if len(all) != 2 {
		t.Fatalf("got %d entries, want 2", len(all))
	}
	var freshLoaded *VectorEntry
	for i := range all {
		if all[i].Path == "wiki/fresh.md" {
			freshLoaded = &all[i]
			break
		}
	}
	if freshLoaded == nil {
		t.Fatalf("fresh entry missing after reload")
	}
	if freshLoaded.Title != "Fresh Article" {
		t.Errorf("title = %q, want %q", freshLoaded.Title, "Fresh Article")
	}
	if freshLoaded.Summary != "A new hydrated row" {
		t.Errorf("summary = %q, want %q", freshLoaded.Summary, "A new hydrated row")
	}
	if freshLoaded.Topic != "wiki" {
		t.Errorf("topic = %q, want %q", freshLoaded.Topic, "wiki")
	}
}

// TestVectorIndex_CacheHit proves that a second LoadAll call for
// the same model does not touch SQLite at all. The test drops the
// underlying table after the first load, so any SQLite read on the
// second call would surface as an error. A cached second read
// succeeds.
func TestVectorIndex_CacheHit(t *testing.T) {
	db, vi := openVectorDB(t)
	ctx := context.Background()

	entries := []VectorEntry{
		{Path: "wiki/alpha.md", Checksum: "a", Model: testModel, Vector: []float32{1, 0, 0}, Title: "Alpha", Summary: "first", Topic: "wiki"},
		{Path: "wiki/bravo.md", Checksum: "b", Model: testModel, Vector: []float32{0, 1, 0}, Title: "Bravo", Summary: "second", Topic: "wiki"},
	}
	if err := vi.StoreBatch(ctx, entries); err != nil {
		t.Fatalf("StoreBatch: %v", err)
	}

	first, err := vi.LoadAll(ctx, testModel)
	if err != nil {
		t.Fatalf("first LoadAll: %v", err)
	}
	if len(first) != 2 {
		t.Fatalf("first LoadAll returned %d rows, want 2", len(first))
	}

	if _, err := db.ExecContext(ctx, "DROP TABLE knowledge_embeddings"); err != nil {
		t.Fatalf("dropping table: %v", err)
	}

	start := time.Now()
	second, err := vi.LoadAll(ctx, testModel)
	if err != nil {
		t.Fatalf("second LoadAll (cache miss would error): %v", err)
	}
	elapsed := time.Since(start)

	if len(second) != 2 {
		t.Fatalf("second LoadAll returned %d rows, want 2 (cache lost rows?)", len(second))
	}
	if elapsed > 5*time.Millisecond {
		t.Errorf("cached LoadAll took %v, expected sub-millisecond", elapsed)
	}
	if second[0].Title != "Alpha" || second[1].Title != "Bravo" {
		t.Errorf("cached titles = %q / %q, want Alpha / Bravo", second[0].Title, second[1].Title)
	}
}

// TestVectorIndex_CacheInvalidatedOnStore populates the cache,
// writes a new entry via Store, and asserts the follow-up LoadAll
// sees the new row.
func TestVectorIndex_CacheInvalidatedOnStore(t *testing.T) {
	_, vi := openVectorDB(t)
	ctx := context.Background()

	seed := []VectorEntry{
		{Path: "wiki/one.md", Checksum: "1", Model: testModel, Vector: []float32{1, 0, 0}},
		{Path: "wiki/two.md", Checksum: "2", Model: testModel, Vector: []float32{0, 1, 0}},
	}
	if err := vi.StoreBatch(ctx, seed); err != nil {
		t.Fatalf("seed StoreBatch: %v", err)
	}
	if _, err := vi.LoadAll(ctx, testModel); err != nil {
		t.Fatalf("warm LoadAll: %v", err)
	}

	fresh := VectorEntry{
		Path:     "wiki/three.md",
		Checksum: "3",
		Model:    testModel,
		Vector:   []float32{0, 0, 1},
		Title:    "Three",
	}
	if err := vi.Store(ctx, fresh); err != nil {
		t.Fatalf("Store: %v", err)
	}

	after, err := vi.LoadAll(ctx, testModel)
	if err != nil {
		t.Fatalf("LoadAll after store: %v", err)
	}
	if len(after) != 3 {
		t.Fatalf("after store got %d rows, want 3 (cache not invalidated)", len(after))
	}
	var found bool
	for _, e := range after {
		if e.Path == "wiki/three.md" && e.Title == "Three" {
			found = true
			break
		}
	}
	if !found {
		t.Errorf("new entry missing from post-store LoadAll: %+v", pathsOfEntries(after))
	}

	batch := []VectorEntry{{
		Path:     "wiki/four.md",
		Checksum: "4",
		Model:    testModel,
		Vector:   []float32{1, 1, 0},
	}}
	if err := vi.StoreBatch(ctx, batch); err != nil {
		t.Fatalf("StoreBatch after warm cache: %v", err)
	}
	afterBatch, err := vi.LoadAll(ctx, testModel)
	if err != nil {
		t.Fatalf("LoadAll after batch: %v", err)
	}
	if len(afterBatch) != 4 {
		t.Errorf("after StoreBatch got %d rows, want 4", len(afterBatch))
	}
}

// TestVectorIndex_CacheInvalidatedOnDelete covers DeleteByPath and
// DeleteByModel. After each delete the follow-up LoadAll must
// reflect the removed rows.
func TestVectorIndex_CacheInvalidatedOnDelete(t *testing.T) {
	_, vi := openVectorDB(t)
	ctx := context.Background()

	entries := []VectorEntry{
		{Path: "wiki/keep.md", Checksum: "k", Model: testModel, Vector: []float32{1, 0}},
		{Path: "wiki/drop.md", Checksum: "d", Model: testModel, Vector: []float32{0, 1}},
		{Path: "wiki/alpha.md", Checksum: "a", Model: "other-model", Vector: []float32{1, 1}},
	}
	if err := vi.StoreBatch(ctx, entries); err != nil {
		t.Fatalf("StoreBatch: %v", err)
	}

	if _, err := vi.LoadAll(ctx, testModel); err != nil {
		t.Fatalf("warm test model: %v", err)
	}
	if _, err := vi.LoadAll(ctx, "other-model"); err != nil {
		t.Fatalf("warm other model: %v", err)
	}

	if err := vi.DeleteByPath(ctx, "wiki/drop.md"); err != nil {
		t.Fatalf("DeleteByPath: %v", err)
	}

	afterPath, err := vi.LoadAll(ctx, testModel)
	if err != nil {
		t.Fatalf("LoadAll after DeleteByPath: %v", err)
	}
	if len(afterPath) != 1 || afterPath[0].Path != "wiki/keep.md" {
		t.Errorf("after DeleteByPath got %+v, want just wiki/keep.md", pathsOfEntries(afterPath))
	}

	if err := vi.DeleteByModel(ctx, "other-model"); err != nil {
		t.Fatalf("DeleteByModel: %v", err)
	}
	afterModel, err := vi.LoadAll(ctx, "other-model")
	if err != nil {
		t.Fatalf("LoadAll after DeleteByModel: %v", err)
	}
	if len(afterModel) != 0 {
		t.Errorf("after DeleteByModel got %+v, want empty", pathsOfEntries(afterModel))
	}

	vi.ClearCache()
	recovered, err := vi.LoadAll(ctx, testModel)
	if err != nil {
		t.Fatalf("LoadAll after ClearCache: %v", err)
	}
	if len(recovered) != 1 {
		t.Errorf("after ClearCache got %d rows, want 1", len(recovered))
	}
}

// TestVectorIndex_SearchReturnsHydratedHits proves Search copies
// the stored Title / Summary fields onto each VectorHit so hybrid
// retrieval no longer needs to read the frontmatter from disk.
func TestVectorIndex_SearchReturnsHydratedHits(t *testing.T) {
	_, vi := openVectorDB(t)
	ctx := context.Background()

	entries := []VectorEntry{
		{
			Path:     "clients/bosch.md",
			Checksum: "bosch",
			Model:    testModel,
			Vector:   []float32{1, 0, 0},
			Title:    "Bosch Power Tools",
			Summary:  "Manufacturing automation partner",
			Topic:    "clients",
		},
		{
			Path:     "clients/heineken.md",
			Checksum: "heineken",
			Model:    testModel,
			Vector:   []float32{0.9, 0.1, 0},
			Title:    "Heineken",
			Summary:  "Beverage distribution pipelines",
			Topic:    "clients",
		},
	}
	if err := vi.StoreBatch(ctx, entries); err != nil {
		t.Fatalf("StoreBatch: %v", err)
	}

	hits, err := vi.Search(ctx, []float32{1, 0, 0}, testModel, 2)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(hits) != 2 {
		t.Fatalf("got %d hits, want 2", len(hits))
	}

	byPath := map[string]VectorHit{}
	for _, h := range hits {
		byPath[h.Path] = h
	}

	bosch, ok := byPath["clients/bosch.md"]
	if !ok {
		t.Fatalf("bosch hit missing from results")
	}
	if bosch.Title != "Bosch Power Tools" {
		t.Errorf("bosch title = %q, want %q", bosch.Title, "Bosch Power Tools")
	}
	if bosch.Summary != "Manufacturing automation partner" {
		t.Errorf("bosch summary = %q", bosch.Summary)
	}

	heineken, ok := byPath["clients/heineken.md"]
	if !ok {
		t.Fatalf("heineken hit missing from results")
	}
	if heineken.Title != "Heineken" {
		t.Errorf("heineken title = %q, want Heineken", heineken.Title)
	}
	if heineken.Summary != "Beverage distribution pipelines" {
		t.Errorf("heineken summary = %q", heineken.Summary)
	}
}
