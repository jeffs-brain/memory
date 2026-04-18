// SPDX-License-Identifier: Apache-2.0

package search

import (
	"context"
	"database/sql"
	"fmt"
	"math/rand"
	"testing"
	"time"

	_ "modernc.org/sqlite"
)

// TestVectorIndex_Perf is skipped under `go test -short`. It
// produces realistic-data timings for 5k articles at 1024
// dimensions. Run with `go test -run Perf` to regenerate.
func TestVectorIndex_Perf(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping perf probe under -short")
	}
	// Also skip by default so the test suite stays fast. Run
	// explicitly with `go test -run TestVectorIndex_Perf` to
	// execute.
	t.Skip("perf probe: run explicitly via -run TestVectorIndex_Perf")

	db, err := sql.Open("sqlite", ":memory:")
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	t.Cleanup(func() { db.Close() })
	vi, err := NewVectorIndex(db)
	if err != nil {
		t.Fatalf("NewVectorIndex: %v", err)
	}

	const n = 5000
	const dim = 1024
	rng := rand.New(rand.NewSource(42))

	entries := make([]VectorEntry, n)
	for i := range entries {
		vec := make([]float32, dim)
		for j := range vec {
			vec[j] = rng.Float32()
		}
		entries[i] = VectorEntry{
			Path:     fmt.Sprintf("wiki/perf-%04d.md", i),
			Checksum: fmt.Sprintf("c-%04d", i),
			Model:    "bge-m3",
			Vector:   vec,
		}
	}

	ctx := context.Background()

	start := time.Now()
	if err := vi.StoreBatch(ctx, entries); err != nil {
		t.Fatalf("StoreBatch: %v", err)
	}
	storeDur := time.Since(start)

	start = time.Now()
	loaded, err := vi.LoadAll(ctx, "bge-m3")
	if err != nil {
		t.Fatalf("LoadAll: %v", err)
	}
	loadDur := time.Since(start)
	if len(loaded) != n {
		t.Fatalf("LoadAll returned %d, want %d", len(loaded), n)
	}

	query := make([]float32, dim)
	for j := range query {
		query[j] = rng.Float32()
	}

	var searchDurs [3]time.Duration
	for i := range searchDurs {
		start = time.Now()
		hits, err := vi.Search(ctx, query, "bge-m3", 20)
		if err != nil {
			t.Fatalf("Search: %v", err)
		}
		searchDurs[i] = time.Since(start)
		if len(hits) != 20 {
			t.Fatalf("Search returned %d hits, want 20", len(hits))
		}
	}

	t.Logf("perf: StoreBatch %dx%dd = %v", n, dim, storeDur)
	t.Logf("perf: LoadAll %d entries = %v", n, loadDur)
	t.Logf("perf: Search top-20 runs = %v %v %v", searchDurs[0], searchDurs[1], searchDurs[2])
}

// benchVectorIndex seeds a realistic 5k-row / 1024-dim index
// against an in-memory SQLite database. Shared by the cached +
// uncached search benchmarks below so the seed cost is not charged
// to either.
func benchVectorIndex(b *testing.B) (*VectorIndex, []float32) {
	b.Helper()
	db, err := sql.Open("sqlite", ":memory:")
	if err != nil {
		b.Fatalf("open: %v", err)
	}
	b.Cleanup(func() { db.Close() })

	vi, err := NewVectorIndex(db)
	if err != nil {
		b.Fatalf("NewVectorIndex: %v", err)
	}

	const n = 5000
	const dim = 1024
	rng := rand.New(rand.NewSource(7))

	entries := make([]VectorEntry, n)
	for i := range entries {
		vec := make([]float32, dim)
		for j := range vec {
			vec[j] = rng.Float32()
		}
		entries[i] = VectorEntry{
			Path:     fmt.Sprintf("wiki/bench-%04d.md", i),
			Checksum: fmt.Sprintf("c-%04d", i),
			Model:    "bge-m3",
			Vector:   vec,
			Title:    fmt.Sprintf("Bench article %d", i),
			Summary:  "synthetic summary for cache benchmark",
			Topic:    "bench",
		}
	}
	if err := vi.StoreBatch(context.Background(), entries); err != nil {
		b.Fatalf("StoreBatch: %v", err)
	}

	query := make([]float32, dim)
	for j := range query {
		query[j] = rng.Float32()
	}
	return vi, query
}

// BenchmarkVectorIndex_SearchCached measures the steady-state hot
// path: every iteration hits the in-memory cache because the first
// LoadAll has already populated it.
func BenchmarkVectorIndex_SearchCached(b *testing.B) {
	vi, query := benchVectorIndex(b)
	ctx := context.Background()
	if _, err := vi.LoadAll(ctx, "bge-m3"); err != nil {
		b.Fatalf("warm LoadAll: %v", err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := vi.Search(ctx, query, "bge-m3", 20); err != nil {
			b.Fatalf("Search: %v", err)
		}
	}
}

// BenchmarkVectorIndex_SearchUncached clears the cache on every
// iteration so the reported number is the old, pre-cache cost.
func BenchmarkVectorIndex_SearchUncached(b *testing.B) {
	vi, query := benchVectorIndex(b)
	ctx := context.Background()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		b.StopTimer()
		vi.ClearCache()
		b.StartTimer()
		if _, err := vi.Search(ctx, query, "bge-m3", 20); err != nil {
			b.Fatalf("Search: %v", err)
		}
	}
}
