// SPDX-License-Identifier: Apache-2.0

package search

import (
	"context"
	"database/sql"
	"math"
	"testing"

	_ "modernc.org/sqlite"
)

// openVectorDB returns an in-memory SQLite handle with the
// embedding schema applied.
func openVectorDB(t *testing.T) (*sql.DB, *VectorIndex) {
	t.Helper()
	db, err := sql.Open("sqlite", ":memory:")
	if err != nil {
		t.Fatalf("opening in-memory db: %v", err)
	}
	t.Cleanup(func() { db.Close() })

	vi, err := NewVectorIndex(db)
	if err != nil {
		t.Fatalf("NewVectorIndex: %v", err)
	}
	return db, vi
}

const testModel = "bge-m3-test"

func TestVectorIndex_SchemaIdempotent(t *testing.T) {
	_, vi := openVectorDB(t)
	ctx := context.Background()

	if err := vi.Schema(ctx); err != nil {
		t.Fatalf("first Schema: %v", err)
	}
	if err := vi.Schema(ctx); err != nil {
		t.Fatalf("second Schema: %v", err)
	}

	entry := VectorEntry{
		Path:     "wiki/alpha.md",
		Checksum: "cs-alpha",
		Model:    testModel,
		Vector:   []float32{1, 0, 0, 0},
	}
	if err := vi.Store(ctx, entry); err != nil {
		t.Fatalf("Store: %v", err)
	}

	n, err := vi.Count(ctx, testModel)
	if err != nil {
		t.Fatalf("Count: %v", err)
	}
	if n != 1 {
		t.Fatalf("Count = %d, want 1 (schema re-apply should not duplicate rows)", n)
	}
}

func TestVectorIndex_StoreAndLoad(t *testing.T) {
	_, vi := openVectorDB(t)
	ctx := context.Background()

	entries := []VectorEntry{
		{Path: "wiki/a.md", Checksum: "cs-a", Model: testModel, Vector: []float32{0.1, 0.2, 0.3, 0.4}},
		{Path: "wiki/b.md", Checksum: "cs-b", Model: testModel, Vector: []float32{-0.5, 0.5, -0.5, 0.5}},
		{Path: "wiki/c.md", Checksum: "cs-c", Model: testModel, Vector: []float32{1, 2, 3, 4}},
	}
	if err := vi.StoreBatch(ctx, entries); err != nil {
		t.Fatalf("StoreBatch: %v", err)
	}

	loaded, err := vi.LoadAll(ctx, testModel)
	if err != nil {
		t.Fatalf("LoadAll: %v", err)
	}
	if len(loaded) != len(entries) {
		t.Fatalf("LoadAll: got %d entries, want %d", len(loaded), len(entries))
	}
	for i, want := range entries {
		got := loaded[i]
		if got.Path != want.Path {
			t.Errorf("entry %d: path %q, want %q", i, got.Path, want.Path)
		}
		if got.Checksum != want.Checksum {
			t.Errorf("entry %d: checksum %q, want %q", i, got.Checksum, want.Checksum)
		}
		if got.Model != want.Model {
			t.Errorf("entry %d: model %q, want %q", i, got.Model, want.Model)
		}
		if got.Dim != len(want.Vector) {
			t.Errorf("entry %d: dim %d, want %d", i, got.Dim, len(want.Vector))
		}
		if len(got.Vector) != len(want.Vector) {
			t.Fatalf("entry %d: vector len %d, want %d", i, len(got.Vector), len(want.Vector))
		}
		for j := range got.Vector {
			if got.Vector[j] != want.Vector[j] {
				t.Errorf("entry %d vec[%d] = %v, want %v (byte-exact)", i, j, got.Vector[j], want.Vector[j])
			}
		}
		if got.Norm == 0 {
			t.Errorf("entry %d: norm not populated", i)
		}
	}

	loadedOne, err := vi.LoadByPath(ctx, "wiki/b.md")
	if err != nil {
		t.Fatalf("LoadByPath: %v", err)
	}
	if loadedOne == nil {
		t.Fatalf("LoadByPath returned nil")
	}
	if loadedOne.Checksum != "cs-b" {
		t.Errorf("LoadByPath checksum = %q, want cs-b", loadedOne.Checksum)
	}

	missing, err := vi.LoadByPath(ctx, "wiki/nope.md")
	if err != nil {
		t.Fatalf("LoadByPath missing: %v", err)
	}
	if missing != nil {
		t.Errorf("LoadByPath for missing path = %+v, want nil", missing)
	}
}

func TestVectorIndex_StoreBatchReplacesExisting(t *testing.T) {
	_, vi := openVectorDB(t)
	ctx := context.Background()

	first := VectorEntry{
		Path:     "wiki/replace.md",
		Checksum: "cs-old",
		Model:    testModel,
		Vector:   []float32{1, 0, 0, 0},
	}
	if err := vi.Store(ctx, first); err != nil {
		t.Fatalf("first Store: %v", err)
	}

	second := VectorEntry{
		Path:     "wiki/replace.md",
		Checksum: "cs-new",
		Model:    testModel,
		Vector:   []float32{0, 1, 0, 0},
	}
	if err := vi.Store(ctx, second); err != nil {
		t.Fatalf("second Store: %v", err)
	}

	loaded, err := vi.LoadAll(ctx, testModel)
	if err != nil {
		t.Fatalf("LoadAll: %v", err)
	}
	if len(loaded) != 1 {
		t.Fatalf("LoadAll: got %d entries, want 1 (upsert should not duplicate)", len(loaded))
	}
	if loaded[0].Checksum != "cs-new" {
		t.Errorf("checksum = %q, want cs-new (second store should win)", loaded[0].Checksum)
	}
	if loaded[0].Vector[1] != 1 {
		t.Errorf("vector not replaced: got %v", loaded[0].Vector)
	}
}

func TestVectorIndex_Search_ExactMatch(t *testing.T) {
	_, vi := openVectorDB(t)
	ctx := context.Background()

	entries := []VectorEntry{
		{Path: "wiki/x.md", Checksum: "x", Model: testModel, Vector: []float32{1, 0, 0}},
		{Path: "wiki/y.md", Checksum: "y", Model: testModel, Vector: []float32{0, 1, 0}},
		{Path: "wiki/z.md", Checksum: "z", Model: testModel, Vector: []float32{0, 0, 1}},
	}
	if err := vi.StoreBatch(ctx, entries); err != nil {
		t.Fatalf("StoreBatch: %v", err)
	}

	hits, err := vi.Search(ctx, []float32{1, 0, 0}, testModel, 3)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(hits) != 3 {
		t.Fatalf("got %d hits, want 3", len(hits))
	}
	if hits[0].Path != "wiki/x.md" {
		t.Errorf("top hit = %q, want wiki/x.md", hits[0].Path)
	}
	if math.Abs(float64(hits[0].Similarity)-1.0) > 1e-6 {
		t.Errorf("top similarity = %v, want ~1.0", hits[0].Similarity)
	}
	if math.Abs(float64(hits[1].Similarity)) > 1e-6 {
		t.Errorf("orthogonal similarity = %v, want 0", hits[1].Similarity)
	}
}

func TestVectorIndex_Search_RankingOrder(t *testing.T) {
	_, vi := openVectorDB(t)
	ctx := context.Background()

	entries := []VectorEntry{
		{Path: "wiki/one.md", Checksum: "1", Model: testModel, Vector: []float32{1, 0, 0}},
		{Path: "wiki/two.md", Checksum: "2", Model: testModel, Vector: []float32{0.8, 0.6, 0}},
		{Path: "wiki/three.md", Checksum: "3", Model: testModel, Vector: []float32{0.6, 0.8, 0}},
		{Path: "wiki/four.md", Checksum: "4", Model: testModel, Vector: []float32{0, 1, 0}},
	}
	if err := vi.StoreBatch(ctx, entries); err != nil {
		t.Fatalf("StoreBatch: %v", err)
	}

	query := []float32{0.6, 0.8, 0}
	hits, err := vi.Search(ctx, query, testModel, 4)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(hits) != 4 {
		t.Fatalf("got %d hits, want 4", len(hits))
	}
	if hits[0].Path != "wiki/three.md" {
		t.Errorf("top = %q, want wiki/three.md", hits[0].Path)
	}
	order := []string{"wiki/three.md", "wiki/two.md", "wiki/four.md", "wiki/one.md"}
	for i, want := range order {
		if hits[i].Path != want {
			t.Errorf("hits[%d] = %q, want %q (full order %v)", i, hits[i].Path, want, pathsOf(hits))
		}
	}
	for i := 1; i < len(hits); i++ {
		if hits[i].Similarity > hits[i-1].Similarity {
			t.Errorf("ranking not descending at %d: %v > %v", i, hits[i].Similarity, hits[i-1].Similarity)
		}
	}
}

func pathsOf(hits []VectorHit) []string {
	out := make([]string, len(hits))
	for i, h := range hits {
		out[i] = h.Path
	}
	return out
}

func TestVectorIndex_Search_KLimit(t *testing.T) {
	_, vi := openVectorDB(t)
	ctx := context.Background()

	entries := make([]VectorEntry, 10)
	for i := range entries {
		entries[i] = VectorEntry{
			Path:     vecPath("k-limit", i),
			Checksum: "c",
			Model:    testModel,
			Vector:   []float32{float32(i + 1), 0, 0},
		}
	}
	if err := vi.StoreBatch(ctx, entries); err != nil {
		t.Fatalf("StoreBatch: %v", err)
	}

	hits, err := vi.Search(ctx, []float32{1, 0, 0}, testModel, 3)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(hits) != 3 {
		t.Fatalf("got %d hits, want 3", len(hits))
	}

	allHits, err := vi.Search(ctx, []float32{1, 0, 0}, testModel, 0)
	if err != nil {
		t.Fatalf("Search default k: %v", err)
	}
	if len(allHits) != 10 {
		t.Errorf("default k returned %d hits, want 10 (corpus < default)", len(allHits))
	}
}

func TestVectorIndex_Search_ZeroVector(t *testing.T) {
	_, vi := openVectorDB(t)
	ctx := context.Background()

	entries := []VectorEntry{
		{Path: "wiki/p.md", Checksum: "p", Model: testModel, Vector: []float32{1, 2, 3}},
		{Path: "wiki/q.md", Checksum: "q", Model: testModel, Vector: []float32{4, 5, 6}},
	}
	if err := vi.StoreBatch(ctx, entries); err != nil {
		t.Fatalf("StoreBatch: %v", err)
	}

	hits, err := vi.Search(ctx, []float32{0, 0, 0}, testModel, 5)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(hits) != 2 {
		t.Fatalf("got %d hits, want 2", len(hits))
	}
	for _, h := range hits {
		if math.IsNaN(float64(h.Similarity)) {
			t.Errorf("NaN similarity for %s", h.Path)
		}
		if h.Similarity != 0 {
			t.Errorf("zero-query similarity for %s = %v, want 0", h.Path, h.Similarity)
		}
	}
}

func TestVectorIndex_Search_ModelFilter(t *testing.T) {
	_, vi := openVectorDB(t)
	ctx := context.Background()

	entries := []VectorEntry{
		{Path: "wiki/bge.md", Checksum: "b", Model: "bge-m3", Vector: []float32{1, 0, 0}},
		{Path: "wiki/nomic.md", Checksum: "n", Model: "nomic-embed-text", Vector: []float32{1, 0, 0}},
		{Path: "wiki/shared.md", Checksum: "s", Model: "bge-m3", Vector: []float32{0, 1, 0}},
	}
	if err := vi.StoreBatch(ctx, entries); err != nil {
		t.Fatalf("StoreBatch: %v", err)
	}

	hits, err := vi.Search(ctx, []float32{1, 0, 0}, "bge-m3", 10)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(hits) != 2 {
		t.Fatalf("bge-m3 got %d hits, want 2", len(hits))
	}
	for _, h := range hits {
		if h.Path == "wiki/nomic.md" {
			t.Errorf("model filter leaked: %+v", h)
		}
	}

	nomicHits, err := vi.Search(ctx, []float32{1, 0, 0}, "nomic-embed-text", 10)
	if err != nil {
		t.Fatalf("Search nomic: %v", err)
	}
	if len(nomicHits) != 1 || nomicHits[0].Path != "wiki/nomic.md" {
		t.Errorf("nomic hits = %+v, want just wiki/nomic.md", nomicHits)
	}

	missHits, err := vi.Search(ctx, []float32{1, 0, 0}, "unknown-model", 10)
	if err != nil {
		t.Fatalf("Search unknown: %v", err)
	}
	if len(missHits) != 0 {
		t.Errorf("unknown model got %d hits, want 0", len(missHits))
	}
}

func TestVectorIndex_DeleteByPath(t *testing.T) {
	_, vi := openVectorDB(t)
	ctx := context.Background()

	entries := []VectorEntry{
		{Path: "wiki/keep.md", Checksum: "k", Model: testModel, Vector: []float32{1, 0}},
		{Path: "wiki/drop.md", Checksum: "d", Model: testModel, Vector: []float32{0, 1}},
	}
	if err := vi.StoreBatch(ctx, entries); err != nil {
		t.Fatalf("StoreBatch: %v", err)
	}

	if err := vi.DeleteByPath(ctx, "wiki/drop.md"); err != nil {
		t.Fatalf("DeleteByPath: %v", err)
	}

	loaded, err := vi.LoadAll(ctx, testModel)
	if err != nil {
		t.Fatalf("LoadAll: %v", err)
	}
	if len(loaded) != 1 || loaded[0].Path != "wiki/keep.md" {
		t.Errorf("after delete got %+v, want just wiki/keep.md", pathsOfEntries(loaded))
	}
}

func pathsOfEntries(e []VectorEntry) []string {
	out := make([]string, len(e))
	for i, x := range e {
		out[i] = x.Path
	}
	return out
}

func TestVectorIndex_DeleteByModel(t *testing.T) {
	_, vi := openVectorDB(t)
	ctx := context.Background()

	entries := []VectorEntry{
		{Path: "wiki/alpha.md", Checksum: "a", Model: "bge-m3", Vector: []float32{1, 0}},
		{Path: "wiki/alpha.md", Checksum: "a", Model: "nomic-embed-text", Vector: []float32{0, 1}},
		{Path: "wiki/beta.md", Checksum: "b", Model: "bge-m3", Vector: []float32{1, 1}},
	}
	if err := vi.StoreBatch(ctx, entries); err != nil {
		t.Fatalf("StoreBatch: %v", err)
	}

	if err := vi.DeleteByModel(ctx, "bge-m3"); err != nil {
		t.Fatalf("DeleteByModel: %v", err)
	}

	bge, err := vi.LoadAll(ctx, "bge-m3")
	if err != nil {
		t.Fatalf("LoadAll bge: %v", err)
	}
	if len(bge) != 0 {
		t.Errorf("bge-m3 rows after delete = %d, want 0", len(bge))
	}

	nomic, err := vi.LoadAll(ctx, "nomic-embed-text")
	if err != nil {
		t.Fatalf("LoadAll nomic: %v", err)
	}
	if len(nomic) != 1 || nomic[0].Path != "wiki/alpha.md" {
		t.Errorf("nomic rows after delete = %+v, want just wiki/alpha.md", pathsOfEntries(nomic))
	}
}

func TestVectorIndex_Count(t *testing.T) {
	_, vi := openVectorDB(t)
	ctx := context.Background()

	n, err := vi.Count(ctx, testModel)
	if err != nil {
		t.Fatalf("Count empty: %v", err)
	}
	if n != 0 {
		t.Errorf("empty count = %d, want 0", n)
	}

	entries := make([]VectorEntry, 5)
	for i := range entries {
		entries[i] = VectorEntry{
			Path:     vecPath("count", i),
			Checksum: "c",
			Model:    testModel,
			Vector:   []float32{float32(i), 1, 2},
		}
	}
	if err := vi.StoreBatch(ctx, entries); err != nil {
		t.Fatalf("StoreBatch: %v", err)
	}

	n, err = vi.Count(ctx, testModel)
	if err != nil {
		t.Fatalf("Count: %v", err)
	}
	if n != 5 {
		t.Errorf("count = %d, want 5", n)
	}

	other, err := vi.Count(ctx, "some-other-model")
	if err != nil {
		t.Fatalf("Count other: %v", err)
	}
	if other != 0 {
		t.Errorf("other-model count = %d, want 0", other)
	}
}

func vecPath(prefix string, i int) string {
	return "wiki/" + prefix + "-" + string(rune('a'+i)) + ".md"
}

func TestPackUnpackFloat32(t *testing.T) {
	cases := [][]float32{
		{},
		{0},
		{1, -1, 0.5, -0.25},
		{float32(math.Pi), float32(math.E), float32(math.SqrtPi), float32(math.MaxFloat32)},
		{1e-20, -1e-20, 1e20, -1e20},
	}
	for i, want := range cases {
		blob := packFloat32(want)
		if len(blob) != 4*len(want) {
			t.Errorf("case %d: blob len %d, want %d", i, len(blob), 4*len(want))
		}
		got, err := unpackFloat32(blob)
		if err != nil {
			t.Fatalf("case %d: unpack: %v", i, err)
		}
		if len(got) != len(want) {
			t.Fatalf("case %d: got %d floats, want %d", i, len(got), len(want))
		}
		for j := range want {
			if math.Float32bits(got[j]) != math.Float32bits(want[j]) {
				t.Errorf("case %d idx %d: got %v, want %v (bits differ)", i, j, got[j], want[j])
			}
		}
	}

	if _, err := unpackFloat32([]byte{1, 2, 3}); err == nil {
		t.Error("expected error on 3-byte blob")
	}
}

func TestCosineSimilarity(t *testing.T) {
	cases := []struct {
		name  string
		a, b  []float32
		wantF float32
	}{
		{"identical unit", []float32{1, 0, 0}, []float32{1, 0, 0}, 1},
		{"identical non-unit", []float32{2, 2, 2}, []float32{2, 2, 2}, 1},
		{"orthogonal", []float32{1, 0, 0}, []float32{0, 1, 0}, 0},
		{"opposite", []float32{1, 0, 0}, []float32{-1, 0, 0}, -1},
		{"zero norm a", []float32{0, 0, 0}, []float32{1, 2, 3}, 0},
		{"zero norm b", []float32{1, 2, 3}, []float32{0, 0, 0}, 0},
		{"both zero", []float32{0, 0, 0}, []float32{0, 0, 0}, 0},
		{"45 deg", []float32{1, 0}, []float32{1, 1}, float32(1.0 / math.Sqrt2)},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			normA := l2Norm(tc.a)
			normB := l2Norm(tc.b)
			got := cosineSimilarity(tc.a, tc.b, normA, normB)
			if math.IsNaN(float64(got)) {
				t.Fatalf("got NaN, must never NaN")
			}
			if math.Abs(float64(got-tc.wantF)) > 1e-6 {
				t.Errorf("got %v, want %v", got, tc.wantF)
			}
		})
	}
}

func TestL2Norm(t *testing.T) {
	cases := []struct {
		name string
		in   []float32
		want float32
	}{
		{"zero", []float32{0, 0, 0}, 0},
		{"unit", []float32{1, 0, 0}, 1},
		{"three-four-five", []float32{3, 4}, 5},
		{"negative", []float32{-3, -4}, 5},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := l2Norm(tc.in)
			if math.Abs(float64(got-tc.want)) > 1e-6 {
				t.Errorf("got %v, want %v", got, tc.want)
			}
		})
	}
}
