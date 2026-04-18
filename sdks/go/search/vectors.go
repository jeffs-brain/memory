// SPDX-License-Identifier: Apache-2.0

package search

import (
	"context"
	"database/sql"
	"encoding/binary"
	"fmt"
	"math"
	"sort"
	"strings"
	"sync"
	"time"

	_ "modernc.org/sqlite"
)

// TODO(sqlite-vec): once the CGo build variant lands, a parallel
// implementation using github.com/asg017/sqlite-vec-go-bindings/cgo
// should live behind a `sqlite_vec` build tag and push cosine ranking
// into SQLite via `vec_distance_cosine`. The pure-Go path below is
// the default so the library builds under `CGO_ENABLED=0` and stays
// portable. Keep both paths behind the same public API so callers do
// not need to branch.

// vectorSchema creates the knowledge_embeddings table and its model
// index. The composite primary key lets us keep vectors for multiple
// models side by side, so A/B testing bge-m3 against nomic-embed-text
// does not require tearing the table down.
//
// The title / summary / topic columns are hydrated at embed time by
// upstream callers so hybrid search can return enriched hits without
// touching the filesystem. They are added via ALTER TABLE inside
// [VectorIndex.Schema] so an existing deployment's BLOB payload is
// preserved across upgrades.
const vectorSchema = `
CREATE TABLE IF NOT EXISTS knowledge_embeddings (
    path       TEXT NOT NULL,
    checksum   TEXT NOT NULL,
    dim        INTEGER NOT NULL,
    model      TEXT NOT NULL,
    vector     BLOB NOT NULL,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (path, model)
);

CREATE INDEX IF NOT EXISTS idx_embeddings_model ON knowledge_embeddings(model);
`

// hydratedColumns lists the nullable text columns [VectorIndex.Schema]
// adds on top of [vectorSchema] via ALTER TABLE. Populated by the
// embed pipeline from article frontmatter so [VectorIndex.Search] can
// return hydrated [VectorHit]s without touching the wiki files on
// disk. Existing rows are left untouched; LoadAll returns empty
// strings for any NULL.
var hydratedColumns = []string{"title", "summary", "topic"}

// defaultVectorSearchK is the fallback top-K used when callers pass
// k <= 0 to [VectorIndex.Search]. Matches the FTS5 default in
// [defaultMaxResults] so hybrid fusion sees balanced slate sizes.
const defaultVectorSearchK = 20

// VectorEntry is one article's embedding in the knowledge_embeddings
// table. Vector is stored as little-endian packed float32 bytes.
// Title, Summary, and Topic are hydrated at embed time from the
// article's frontmatter (first path segment for Topic) so hybrid
// search can return enriched hits without reading the wiki file on
// every query.
type VectorEntry struct {
	Path     string
	Checksum string
	Dim      int
	Model    string
	Vector   []float32
	Norm     float32 // precomputed L2 norm, populated by StoreBatch
	Title    string
	Summary  string
	Topic    string
}

// vectorCacheEntry holds the full slice of entries for a single model
// so repeated LoadAll calls do not re-read the BLOBs from SQLite.
type vectorCacheEntry struct {
	entries  []VectorEntry
	loadedAt time.Time
}

// VectorIndex wraps a *sql.DB and exposes embedding storage plus
// cosine search. It is a sibling of [Index] (the FTS5 path) sharing
// the same underlying SQLite file. Both construct via [OpenDB] at
// startup so the WAL writer lock is serialised across paths.
//
// A small process-lifetime cache keyed by model avoids re-reading the
// vector BLOB payload on every search. The cache is invalidated by
// any Store/StoreBatch/DeleteByPath/DeleteByModel call affecting that
// model. Writes are rare relative to reads in practice (embed runs
// are incremental), so the invalidate-on-write policy is cheaper
// than any TTL or versioning scheme.
type VectorIndex struct {
	db    *sql.DB
	mu    sync.RWMutex
	cache map[string]vectorCacheEntry
}

// VectorHit is one cosine-ranked result. Similarity is in [-1, 1]
// but normalised embedding models produce positive vectors so
// real-world scores are typically [0, 1]. Higher is better. Title
// and Summary are hydrated from the matching [VectorEntry] so
// callers do not need to re-read article frontmatter at query time.
type VectorHit struct {
	Path       string
	Similarity float32
	Title      string
	Summary    string
}

// NewVectorIndex ensures the knowledge_embeddings schema exists and
// returns a handle ready for Store/Load/Search calls. The caller
// retains ownership of db: closing the VectorIndex is not required.
func NewVectorIndex(db *sql.DB) (*VectorIndex, error) {
	v := &VectorIndex{
		db:    db,
		cache: make(map[string]vectorCacheEntry),
	}
	if err := v.Schema(context.Background()); err != nil {
		return nil, err
	}
	return v, nil
}

// Schema creates the knowledge_embeddings table idempotently and
// runs the column-evolution migration for the hydrated title /
// summary / topic fields. Running Schema twice on an already-evolved
// database is a no-op.
func (v *VectorIndex) Schema(ctx context.Context) error {
	if _, err := v.db.ExecContext(ctx, vectorSchema); err != nil {
		return fmt.Errorf("creating knowledge_embeddings schema: %w", err)
	}

	existing, err := v.columnSet(ctx)
	if err != nil {
		return err
	}
	for _, col := range hydratedColumns {
		if _, ok := existing[col]; ok {
			continue
		}
		stmt := fmt.Sprintf("ALTER TABLE knowledge_embeddings ADD COLUMN %s TEXT", col)
		if _, err := v.db.ExecContext(ctx, stmt); err != nil {
			// Another process racing on the same DB file may add
			// the column between our PRAGMA and our ALTER. SQLite
			// surfaces this as "duplicate column name: <col>";
			// treat it as a harmless no-op.
			if strings.Contains(strings.ToLower(err.Error()), "duplicate column name") {
				continue
			}
			return fmt.Errorf("adding %s column: %w", col, err)
		}
	}
	return nil
}

// columnSet returns the set of column names currently on the
// knowledge_embeddings table via PRAGMA table_info. Used by Schema
// to decide which hydrated columns still need an ALTER TABLE.
func (v *VectorIndex) columnSet(ctx context.Context) (map[string]struct{}, error) {
	rows, err := v.db.QueryContext(ctx, "PRAGMA table_info(knowledge_embeddings)")
	if err != nil {
		return nil, fmt.Errorf("reading knowledge_embeddings columns: %w", err)
	}
	defer rows.Close()

	set := make(map[string]struct{})
	for rows.Next() {
		var (
			cid       int
			name      string
			typ       string
			notnull   int
			dfltValue sql.NullString
			pk        int
		)
		if err := rows.Scan(&cid, &name, &typ, &notnull, &dfltValue, &pk); err != nil {
			return nil, fmt.Errorf("scanning pragma row: %w", err)
		}
		set[name] = struct{}{}
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterating pragma rows: %w", err)
	}
	return set, nil
}

// Store inserts or replaces a single entry. Computes and stores the
// L2 norm so [VectorIndex.Search] can skip recomputing it per query.
func (v *VectorIndex) Store(ctx context.Context, entry VectorEntry) error {
	return v.StoreBatch(ctx, []VectorEntry{entry})
}

// StoreBatch inserts or replaces many entries in one transaction.
// This is the hot path for bulk embed runs.
func (v *VectorIndex) StoreBatch(ctx context.Context, entries []VectorEntry) error {
	if len(entries) == 0 {
		return nil
	}

	tx, err := v.db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("beginning vector batch: %w", err)
	}

	stmt, err := tx.PrepareContext(ctx,
		`INSERT INTO knowledge_embeddings (path, checksum, dim, model, vector, title, summary, topic, updated_at)
		 VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
		 ON CONFLICT(path, model) DO UPDATE SET
		     checksum   = excluded.checksum,
		     dim        = excluded.dim,
		     vector     = excluded.vector,
		     title      = excluded.title,
		     summary    = excluded.summary,
		     topic      = excluded.topic,
		     updated_at = CURRENT_TIMESTAMP`,
	)
	if err != nil {
		_ = tx.Rollback()
		return fmt.Errorf("preparing vector upsert: %w", err)
	}
	defer stmt.Close()

	affectedModels := make(map[string]struct{}, 1)
	for i := range entries {
		entry := &entries[i]
		if entry.Path == "" {
			_ = tx.Rollback()
			return fmt.Errorf("vector entry at index %d has empty path", i)
		}
		if entry.Model == "" {
			_ = tx.Rollback()
			return fmt.Errorf("vector entry %q has empty model", entry.Path)
		}
		if len(entry.Vector) == 0 {
			_ = tx.Rollback()
			return fmt.Errorf("vector entry %q has empty vector", entry.Path)
		}
		if entry.Dim == 0 {
			entry.Dim = len(entry.Vector)
		}
		if entry.Dim != len(entry.Vector) {
			_ = tx.Rollback()
			return fmt.Errorf("vector entry %q: dim=%d but len(vector)=%d", entry.Path, entry.Dim, len(entry.Vector))
		}
		entry.Norm = l2Norm(entry.Vector)
		blob := packFloat32(entry.Vector)
		if _, err := stmt.ExecContext(ctx,
			entry.Path,
			entry.Checksum,
			entry.Dim,
			entry.Model,
			blob,
			entry.Title,
			entry.Summary,
			entry.Topic,
		); err != nil {
			_ = tx.Rollback()
			return fmt.Errorf("storing vector for %s: %w", entry.Path, err)
		}
		affectedModels[entry.Model] = struct{}{}
	}

	if err := tx.Commit(); err != nil {
		return fmt.Errorf("committing vector batch: %w", err)
	}

	v.invalidateModels(affectedModels)
	return nil
}

// LoadAll returns every entry for the given model. Used by Search to
// build an in-memory candidate set. Subsequent calls for the same
// model are served from an in-memory cache until a write invalidates
// it, so hot searches pay zero SQLite cost.
func (v *VectorIndex) LoadAll(ctx context.Context, model string) ([]VectorEntry, error) {
	if cached, ok := v.cachedEntries(model); ok {
		return cached, nil
	}

	rows, err := v.db.QueryContext(ctx,
		`SELECT path, checksum, dim, model, vector, title, summary, topic
		 FROM knowledge_embeddings
		 WHERE model = ?
		 ORDER BY rowid`,
		model,
	)
	if err != nil {
		return nil, fmt.Errorf("querying knowledge_embeddings: %w", err)
	}
	defer rows.Close()

	var out []VectorEntry
	for rows.Next() {
		entry, err := scanVectorEntry(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, entry)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterating knowledge_embeddings: %w", err)
	}

	v.mu.Lock()
	v.cache[model] = vectorCacheEntry{entries: out, loadedAt: time.Now()}
	v.mu.Unlock()

	return out, nil
}

// cachedEntries returns a cached slice under a read lock. The
// returned slice is shared with subsequent callers; callers must
// treat the entries as read-only.
func (v *VectorIndex) cachedEntries(model string) ([]VectorEntry, bool) {
	v.mu.RLock()
	defer v.mu.RUnlock()
	if v.cache == nil {
		return nil, false
	}
	entry, ok := v.cache[model]
	if !ok {
		return nil, false
	}
	return entry.entries, true
}

// invalidateModels drops the cached slice for every model in the
// set. Called by Store/StoreBatch once the transaction has
// committed.
func (v *VectorIndex) invalidateModels(models map[string]struct{}) {
	if len(models) == 0 {
		return
	}
	v.mu.Lock()
	defer v.mu.Unlock()
	for model := range models {
		delete(v.cache, model)
	}
}

// ClearCache wipes the full model-to-entries cache. Exposed for
// tests that need to force a reload from SQLite to assert behaviour.
func (v *VectorIndex) ClearCache() {
	v.mu.Lock()
	defer v.mu.Unlock()
	v.cache = make(map[string]vectorCacheEntry)
}

// LoadByPath returns a single entry by path. Used to probe whether
// an article needs re-embedding. When multiple models coexist this
// returns the row with the lowest rowid; callers that care about
// the active model should filter by calling [VectorIndex.LoadAll]
// instead.
func (v *VectorIndex) LoadByPath(ctx context.Context, path string) (*VectorEntry, error) {
	row := v.db.QueryRowContext(ctx,
		`SELECT path, checksum, dim, model, vector, title, summary, topic
		 FROM knowledge_embeddings
		 WHERE path = ?
		 ORDER BY rowid
		 LIMIT 1`,
		path,
	)
	entry, err := scanVectorEntry(row)
	if err != nil {
		if err == sql.ErrNoRows {
			return nil, nil
		}
		return nil, err
	}
	return &entry, nil
}

// vectorRow is the intersection of *sql.Row and *sql.Rows so
// [scanVectorEntry] can handle both without generics.
type vectorRow interface {
	Scan(dest ...any) error
}

func scanVectorEntry(row vectorRow) (VectorEntry, error) {
	var (
		entry   VectorEntry
		blob    []byte
		title   sql.NullString
		summary sql.NullString
		topic   sql.NullString
	)
	if err := row.Scan(
		&entry.Path,
		&entry.Checksum,
		&entry.Dim,
		&entry.Model,
		&blob,
		&title,
		&summary,
		&topic,
	); err != nil {
		return VectorEntry{}, err
	}
	vec, err := unpackFloat32(blob)
	if err != nil {
		return VectorEntry{}, fmt.Errorf("decoding vector for %s: %w", entry.Path, err)
	}
	if entry.Dim != 0 && entry.Dim != len(vec) {
		return VectorEntry{}, fmt.Errorf("vector %s: stored dim=%d but blob decoded to %d floats", entry.Path, entry.Dim, len(vec))
	}
	entry.Vector = vec
	entry.Norm = l2Norm(vec)
	if title.Valid {
		entry.Title = title.String
	}
	if summary.Valid {
		entry.Summary = summary.String
	}
	if topic.Valid {
		entry.Topic = topic.String
	}
	return entry, nil
}

// Search computes cosine similarity between queryVec and every
// stored entry for the given model, returning the top-k ordered by
// similarity descending. k <= 0 defaults to [defaultVectorSearchK].
// Returns an empty slice when the table has no entries for that
// model. Does NOT hit the network or an LLM.
//
// Safe to call from multiple goroutines: the method only reads from
// the database and maintains no mutable state on the receiver.
func (v *VectorIndex) Search(ctx context.Context, queryVec []float32, model string, k int) ([]VectorHit, error) {
	if k <= 0 {
		k = defaultVectorSearchK
	}
	if len(queryVec) == 0 {
		return nil, fmt.Errorf("search: empty query vector")
	}

	entries, err := v.LoadAll(ctx, model)
	if err != nil {
		return nil, err
	}
	if len(entries) == 0 {
		return []VectorHit{}, nil
	}

	queryNorm := l2Norm(queryVec)
	hits := make([]VectorHit, 0, len(entries))
	for i := range entries {
		e := &entries[i]
		if len(e.Vector) != len(queryVec) {
			// Dimension mismatch means the stored vector comes from a
			// different model version; skip rather than poison the
			// ranking. Callers that care can re-embed from scratch.
			continue
		}
		sim := cosineSimilarity(queryVec, e.Vector, queryNorm, e.Norm)
		hits = append(hits, VectorHit{
			Path:       e.Path,
			Similarity: sim,
			Title:      e.Title,
			Summary:    e.Summary,
		})
	}

	sort.SliceStable(hits, func(i, j int) bool {
		return hits[i].Similarity > hits[j].Similarity
	})

	if len(hits) > k {
		hits = hits[:k]
	}
	return hits, nil
}

// Count returns the number of rows in the table for a given model.
func (v *VectorIndex) Count(ctx context.Context, model string) (int, error) {
	var n int
	err := v.db.QueryRowContext(ctx,
		"SELECT count(*) FROM knowledge_embeddings WHERE model = ?",
		model,
	).Scan(&n)
	if err != nil {
		return 0, fmt.Errorf("counting knowledge_embeddings: %w", err)
	}
	return n, nil
}

// DeleteByPath removes every entry for the given path across all
// models. Used by the watcher path when an article is deleted. The
// cache is dropped wholesale because a single path may span
// multiple models and we do not know up front which model caches
// need to invalidate.
func (v *VectorIndex) DeleteByPath(ctx context.Context, path string) error {
	if _, err := v.db.ExecContext(ctx,
		"DELETE FROM knowledge_embeddings WHERE path = ?",
		path,
	); err != nil {
		return fmt.Errorf("deleting vector for %s: %w", path, err)
	}
	v.ClearCache()
	return nil
}

// DeleteByModel removes every entry for a given model. Used when
// changing the configured embedding model: a full rebuild is the
// only correct response because dimensions and semantics both
// differ.
func (v *VectorIndex) DeleteByModel(ctx context.Context, model string) error {
	if _, err := v.db.ExecContext(ctx,
		"DELETE FROM knowledge_embeddings WHERE model = ?",
		model,
	); err != nil {
		return fmt.Errorf("deleting vectors for model %s: %w", model, err)
	}
	v.invalidateModels(map[string]struct{}{model: {}})
	return nil
}

// packFloat32 encodes a slice of float32 as little-endian bytes.
// Wire format matches encoding/binary.LittleEndian.PutUint32 on
// math.Float32bits. Four bytes per float.
func packFloat32(v []float32) []byte {
	out := make([]byte, 4*len(v))
	for i, f := range v {
		binary.LittleEndian.PutUint32(out[i*4:], math.Float32bits(f))
	}
	return out
}

// unpackFloat32 decodes a little-endian float32 BLOB. Returns an
// error if len(data) is not a multiple of 4.
func unpackFloat32(data []byte) ([]float32, error) {
	if len(data)%4 != 0 {
		return nil, fmt.Errorf("vector blob length %d is not a multiple of 4", len(data))
	}
	out := make([]float32, len(data)/4)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[i*4:]))
	}
	return out, nil
}

// cosineSimilarity returns dot(a, b) / (|a| * |b|). Assumes
// len(a) == len(b); callers must validate. Zero norms return 0 (not
// NaN) to avoid poisoning the ranking.
func cosineSimilarity(a, b []float32, normA, normB float32) float32 {
	if normA == 0 || normB == 0 {
		return 0
	}
	var dot float32
	for i := range a {
		dot += a[i] * b[i]
	}
	return dot / (normA * normB)
}

// l2Norm returns the Euclidean norm of v.
func l2Norm(v []float32) float32 {
	var sum float64
	for _, f := range v {
		sum += float64(f) * float64(f)
	}
	return float32(math.Sqrt(sum))
}
