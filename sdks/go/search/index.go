// SPDX-License-Identifier: Apache-2.0

package search

import (
	"context"
	"crypto/sha256"
	"database/sql"
	"errors"
	"fmt"
	"log/slog"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/jeffs-brain/memory/go/brain"

	_ "modernc.org/sqlite"
)

// openDBs is a process-wide, path-keyed cache of search index SQLite
// handles. Multiple callers (CLI + concurrent HTTP handlers) share a
// single *sql.DB so they don't multiply connection pools fighting
// for the WAL writer lock. Entries are refcounted by OpenDB /
// CloseDB — the underlying handle only closes when the last
// reference is released.
var (
	openDBsMu sync.Mutex
	openDBs   = map[string]*sharedDB{}
)

type sharedDB struct {
	db   *sql.DB
	refs int
	path string
}

// OpenDB opens (or returns a shared handle for) the search index
// SQLite database at dbPath. Pragmas are applied via DSN query
// parameters so every connection in the pool is configured
// identically.
//
// The pool is capped at a single connection. SQLite permits only one
// writer at a time; a single connection funnels all writes through
// the same lock holder and eliminates pool-induced contention while
// WAL snapshot reads remain available to other processes.
//
// Callers must pair each OpenDB with a CloseDB so the refcount can
// drop to zero and the underlying handle can be released.
func OpenDB(dbPath string) (*sql.DB, error) {
	absPath, err := filepath.Abs(dbPath)
	if err != nil {
		return nil, fmt.Errorf("resolving search db path: %w", err)
	}

	openDBsMu.Lock()
	defer openDBsMu.Unlock()

	if shared, ok := openDBs[absPath]; ok {
		shared.refs++
		return shared.db, nil
	}

	dsn := "file:" + absPath +
		"?_pragma=journal_mode(WAL)" +
		"&_pragma=busy_timeout(10000)" +
		"&_pragma=synchronous(NORMAL)" +
		"&_txlock=immediate"

	db, err := sql.Open("sqlite", dsn)
	if err != nil {
		return nil, fmt.Errorf("opening search db: %w", err)
	}
	db.SetMaxOpenConns(1)
	db.SetMaxIdleConns(1)
	db.SetConnMaxLifetime(0)

	openDBs[absPath] = &sharedDB{db: db, refs: 1, path: absPath}
	return db, nil
}

// CloseDB releases one reference to a handle returned by OpenDB.
// The underlying *sql.DB is only closed when the final reference is
// released. A nil db is ignored.
func CloseDB(db *sql.DB) error {
	if db == nil {
		return nil
	}
	openDBsMu.Lock()
	defer openDBsMu.Unlock()

	for path, shared := range openDBs {
		if shared.db != db {
			continue
		}
		shared.refs--
		if shared.refs > 0 {
			return nil
		}
		delete(openDBs, path)
		return shared.db.Close()
	}
	return db.Close()
}

// refreshInterval is the minimum time between automatic index
// refreshes. Searches that hit a stale index trigger an Update()
// before querying.
const refreshInterval = 30 * time.Second

const ftsSchema = `
CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_fts USING fts5(
    path,
    title,
    summary,
    tags,
    content,
    scope,
    project_slug,
    session_date UNINDEXED,
    tokenize='porter unicode61'
);

CREATE TABLE IF NOT EXISTS knowledge_index_state (
    path TEXT PRIMARY KEY,
    checksum TEXT NOT NULL,
    indexed_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS knowledge_index_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
`

// metaLastRefresh is the knowledge_index_meta key that holds the
// timestamp of the most recent Rebuild/Update. Persisted so
// cross-process callers can report staleness without relying on
// in-memory state.
const metaLastRefresh = "last_refresh_at"

// metaSchemaVersion is the knowledge_index_meta key that records
// the FTS column layout in use. Bumped when the schema shape
// changes so on-disk indices written by an earlier binary get
// dropped and rebuilt on the next open rather than silently
// erroring out with "no such column".
const metaSchemaVersion = "schema_version"

// currentSchemaVersion is the expected value of [metaSchemaVersion].
// Incrementing it triggers an automatic drop + recreate on any
// index whose persisted version is lower.
const currentSchemaVersion = "2"

// staleWikiRatio is the minimum ratio of indexed wiki rows to
// on-disk wiki files below which [Index.RebuildIfStale] forces a
// full rebuild. Chosen so a large compilation run that only
// partially propagates heals on the next startup.
const staleWikiRatio = 0.5

const defaultMaxResults = 20
const insertBatchSize = 100

// Index provides full-text search over memory and wiki files via
// SQLite FTS5. It discovers and reads files through a [brain.Store]
// so the same index logic works against any backend (filesystem or
// git-backed). Paths stored in the index are logical [brain.Path]
// values — portable across machines but always valid identifiers
// for the same [brain.Store].
type Index struct {
	db          *sql.DB
	store       brain.Store
	lastRefresh time.Time
	refreshMu   sync.Mutex
	// OnIndex, when non-nil, is invoked once per file as it is
	// successfully written to the FTS index. Used by the reindex
	// CLI for verbose output. The callback runs inside the write
	// transaction, so it should be fast and must not touch the
	// database itself.
	OnIndex func(p brain.Path)
}

// SearchOpts controls filtering and pagination for searches.
type SearchOpts struct {
	Scope       string // "" for all, "wiki", "global_memory", "project_memory", "raw_document", "sources"
	ProjectSlug string // filter by project (only for project_memory scope)
	MaxResults  int    // default 20
	Sort        SortMode

	// DateFrom / DateTo narrow results to hits whose session date
	// falls inside the inclusive range. Zero values disable that
	// bound. Hits without a parseable date are excluded from
	// date-filtered queries.
	DateFrom time.Time
	DateTo   time.Time

	// IncludeSuperseded keeps memory rows whose frontmatter
	// carries a superseded_by pointer. Default (false) drops them
	// because the newer replacement is already in the corpus.
	// Callers that want a time-travel view (history, audits) set
	// this true.
	IncludeSuperseded bool
}

// HasDateFilter reports whether either bound is set.
func (o SearchOpts) HasDateFilter() bool {
	return !o.DateFrom.IsZero() || !o.DateTo.IsZero()
}

// SortMode controls how search hits are ordered after FTS5 returns
// candidates. SortRelevance (the zero value) preserves the default
// BM25 ranking. The recency modes re-order candidates by the
// Modified timestamp parsed from the source file's frontmatter so
// questions about current state retrieve the freshest fact rather
// than whatever ranks highest lexically.
type SortMode int

const (
	// SortRelevance orders by FTS5 BM25 rank. Default.
	SortRelevance SortMode = iota
	// SortRecency orders purely by Modified timestamp, descending.
	// Dateless hits fall to the end. FTS5 rank is ignored.
	SortRecency
	// SortRelevanceThenRecency takes a wider candidate pool from
	// FTS5, then re-sorts by Modified descending to break ties
	// toward newer hits while still filtering by lexical match.
	SortRelevanceThenRecency
)

// String returns a human-readable name for the sort mode.
func (m SortMode) String() string {
	switch m {
	case SortRecency:
		return "recency"
	case SortRelevanceThenRecency:
		return "relevance_then_recency"
	default:
		return "relevance"
	}
}

// ParseSortMode converts a string to a SortMode. Empty / unknown
// values default to SortRelevance so misspelled inputs never change
// behaviour silently.
func ParseSortMode(s string) SortMode {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "recency":
		return SortRecency
	case "relevance_then_recency", "relevance-then-recency":
		return SortRelevanceThenRecency
	default:
		return SortRelevance
	}
}

// SearchResult represents a single search hit.
type SearchResult struct {
	Path     string
	Title    string
	Summary  string
	Snippet  string  // FTS5 snippet highlight
	Score    float64 // BM25 rank (lower is better in FTS5)
	Scope    string
	Modified time.Time // zero when no Modified frontmatter was parseable
	// SessionDate is the ISO YYYY-MM-DD string carried in the
	// session_date FTS column, populated at index time with the
	// precedence session_date > observed_on > modified. Empty when
	// none of those frontmatter fields are parseable.
	SessionDate string
}

// NewIndex creates the FTS5 tables if they don't exist and binds
// the index to the supplied [brain.Store]. A nil store returns an
// error: the Go SDK does not fall back to an autodetected default,
// callers must be explicit about which brain they are indexing.
func NewIndex(db *sql.DB, store brain.Store) (*Index, error) {
	if store == nil {
		return nil, errors.New("search: NewIndex requires a non-nil brain store")
	}
	return newIndex(db, store)
}

// newIndex is the shared construction path. Exported constructors
// wrap it so nil-store handling can vary by caller (for example,
// tests that want to exercise the SQLite paths without a store).
func newIndex(db *sql.DB, store brain.Store) (*Index, error) {
	// Belt-and-braces: ensure WAL + busy_timeout are set even if
	// the caller supplied a plain handle (e.g. tests that open via
	// sql.Open directly). Idempotent when OpenDB already set them.
	for _, pragma := range []string{
		"PRAGMA journal_mode=WAL",
		"PRAGMA busy_timeout=10000",
	} {
		if _, err := db.Exec(pragma); err != nil {
			return nil, fmt.Errorf("setting %s: %w", pragma, err)
		}
	}
	// If an index from an earlier schema version exists on disk,
	// drop the FTS tables so the fresh CREATE below picks up the
	// current column layout. The knowledge_index_state rows go
	// with it so the next Update path reindexes everything from
	// the brain store.
	if err := migrateSchemaIfNeeded(db); err != nil {
		return nil, fmt.Errorf("migrating FTS schema: %w", err)
	}
	if _, err := db.Exec(ftsSchema); err != nil {
		return nil, fmt.Errorf("creating FTS5 schema: %w", err)
	}
	if err := recordSchemaVersion(db); err != nil {
		return nil, fmt.Errorf("recording FTS schema version: %w", err)
	}
	// Configure the FTS5 rank expression to use our custom column
	// weights. This MUST be set via the config INSERT pattern
	// rather than in the SELECT's ORDER BY clause, because SQLite
	// FTS5 can short-circuit ORDER BY rank LIMIT N but cannot
	// short-circuit an explicit bm25() call — the latter forces
	// bm25 to be computed for every matching row before the sort.
	//
	// Columns in order match ftsSchema: path, title, summary, tags,
	// content, scope, project_slug, session_date. session_date is
	// UNINDEXED so its weight is irrelevant to MATCH; we still pass
	// 1.0 so the argument count matches the column count.
	rankExpr := "'bm25(3.0, 10.0, 5.0, 4.0, 1.0, 1.0, 1.0, 1.0)'"
	if _, err := db.Exec(`INSERT INTO knowledge_fts(knowledge_fts, rank) VALUES('rank', ` + rankExpr + `)`); err != nil {
		return nil, fmt.Errorf("configuring FTS5 rank weights: %w", err)
	}
	return &Index{db: db, store: store, lastRefresh: time.Now()}, nil
}

// refreshIfStale triggers an incremental index update if the last
// refresh was more than refreshInterval ago. This ensures
// cross-process writes become visible to the search path.
func (idx *Index) refreshIfStale() {
	idx.refreshMu.Lock()
	defer idx.refreshMu.Unlock()

	if time.Since(idx.lastRefresh) < refreshInterval {
		return
	}
	if err := idx.Update(context.Background()); err != nil {
		slog.Debug("search: auto-refresh failed", "err", err)
	}
	idx.lastRefresh = time.Now()
}

// Search performs a full-text search with BM25 ranking. The query
// is first passed through [sanitiseQuery] to strip stop words,
// preserve phrases, and convert the default operator to `OR`.
// Callers that already hold a hand-built FTS5 expression should use
// [Index.SearchRaw] instead.
func (idx *Index) Search(query string, opts SearchOpts) ([]SearchResult, error) {
	sanitised := sanitiseQuery(query)
	if sanitised == "" {
		return nil, nil
	}
	return idx.SearchRaw(sanitised, opts)
}

// SearchRaw runs a pre-built FTS5 expression directly against the
// index, skipping [sanitiseQuery]. Used for callers with their own
// FTS5 expression building logic. Scope filtering and the
// stale-index auto-refresh still apply.
func (idx *Index) SearchRaw(expr string, opts SearchOpts) ([]SearchResult, error) {
	idx.refreshIfStale()

	expr = strings.TrimSpace(expr)
	if expr == "" {
		return nil, nil
	}

	maxResults := opts.MaxResults
	if maxResults <= 0 {
		maxResults = defaultMaxResults
	}

	// Fetch more than needed so we can filter by scope in Go.
	// Recency modes ask for a wider pool so the post-sort has
	// enough candidates to re-order meaningfully.
	fetchLimit := max(maxResults*3, 100)
	if opts.Sort == SortRecency || opts.Sort == SortRelevanceThenRecency {
		fetchLimit = max(maxResults*5, 200)
	}

	query, args := buildSearchSQL(expr, opts.DateFrom, opts.DateTo, fetchLimit)
	rows, err := idx.db.Query(query, args...)
	if err != nil {
		return nil, fmt.Errorf("executing FTS5 search: %w", err)
	}
	defer rows.Close()

	var results []SearchResult
	for rows.Next() {
		var (
			r              SearchResult
			projectSlug    string
			sessionDateRaw sql.NullString
		)
		if err := rows.Scan(&r.Path, &r.Title, &r.Summary, &r.Snippet, &r.Score, &r.Scope, &projectSlug, &sessionDateRaw); err != nil {
			return nil, fmt.Errorf("scanning search result: %w", err)
		}
		if sessionDateRaw.Valid {
			r.SessionDate = sessionDateRaw.String
		}

		// Apply scope filter.
		if opts.Scope != "" && r.Scope != opts.Scope {
			continue
		}
		if opts.ProjectSlug != "" && projectSlug != opts.ProjectSlug {
			continue
		}

		results = append(results, r)
		// For recency modes we still need the full pool before
		// truncation so the post-sort has enough candidates to work
		// with. Relevance mode keeps the original early-exit at
		// maxResults; the date filter no longer requires a wider
		// pool because SQL has already applied it.
		if opts.Sort == SortRelevance && len(results) >= maxResults {
			break
		}
	}

	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterating search results: %w", err)
	}

	if opts.Sort != SortRelevance {
		idx.hydrateModified(results)
	}
	if !opts.IncludeSuperseded {
		results = filterSuperseded(idx, results)
	}
	if opts.Sort != SortRelevance {
		results = applySort(results, opts.Sort, maxResults)
	} else if len(results) > maxResults {
		results = results[:maxResults]
	}

	return results, nil
}

// buildSearchSQL assembles the FTS5 SELECT with an optional
// session_date range pushed into the WHERE clause. Returns the
// final query string and the matching positional arguments.
func buildSearchSQL(expr string, from, to time.Time, fetchLimit int) (string, []any) {
	var clauses []string
	args := []any{expr}
	clauses = append(clauses, "knowledge_fts MATCH ?")

	if !from.IsZero() {
		clauses = append(clauses, "session_date >= ?")
		args = append(args, from.UTC().Format("2006-01-02"))
	}
	if !to.IsZero() {
		clauses = append(clauses, "session_date <= ?")
		args = append(args, to.UTC().Format("2006-01-02"))
	}
	if !from.IsZero() || !to.IsZero() {
		// Exclude rows without a parseable session_date from any
		// ranged query — a dateless row cannot honour a between
		// contract. Empty string covers rows indexed before the
		// frontmatter carried a date at all.
		clauses = append(clauses, "session_date IS NOT NULL AND session_date <> ''")
	}

	query := `SELECT path, title, summary,
	        snippet(knowledge_fts, 4, '<mark>', '</mark>', '...', 64) AS snippet,
	        rank AS score,
	        scope, project_slug, session_date
	 FROM knowledge_fts
	 WHERE ` + strings.Join(clauses, " AND ") + `
	 ORDER BY rank
	 LIMIT ?`
	args = append(args, fetchLimit)
	return query, args
}

// filterSuperseded drops memory rows whose frontmatter carries a
// superseded_by pointer. Wiki rows are left alone because the
// concept does not apply to articles; the parser only looks at
// memory files. A read failure keeps the row (fail open): we would
// rather return a slightly-stale result than drop a valid hit on a
// transient I/O error.
func filterSuperseded(idx *Index, in []SearchResult) []SearchResult {
	if idx == nil || idx.store == nil || len(in) == 0 {
		return in
	}
	ctx := context.Background()
	out := in[:0]
	for _, r := range in {
		if r.Scope != "global_memory" && r.Scope != "project_memory" {
			out = append(out, r)
			continue
		}
		data, err := idx.store.Read(ctx, brain.Path(r.Path))
		if err != nil {
			out = append(out, r)
			continue
		}
		fm, _ := parseMemoryFrontmatter(string(data))
		if fm.SupersededBy != "" {
			continue
		}
		out = append(out, r)
	}
	return out
}

// hydrateModified reads each hit's source file through the bound
// brain.Store and parses its Modified frontmatter. Populates each
// SearchResult.Modified in place. Failures are silent: a hit that
// cannot be read keeps its zero Modified, which sorts last.
func (idx *Index) hydrateModified(results []SearchResult) {
	ctx := context.Background()
	for i := range results {
		if idx.store == nil {
			return
		}
		data, err := idx.store.Read(ctx, brain.Path(results[i].Path))
		if err != nil {
			continue
		}
		if mod, ok := extractModified(data, results[i].Scope); ok {
			results[i].Modified = mod
		}
	}
}

// extractModified pulls the `modified:` field out of either a
// memory frontmatter block or a wiki frontmatter block. Returns
// zero + false when the field is absent or unparseable.
func extractModified(raw []byte, scope string) (time.Time, bool) {
	switch scope {
	case "wiki":
		fm, _ := parseWikiFrontmatter(string(raw))
		return parseModifiedString(fm.Modified)
	default:
		fm, _ := parseMemoryFrontmatter(string(raw))
		return parseModifiedString(fm.Modified)
	}
}

// sessionDateKeyRe matches the top-level frontmatter fields that
// carry a session or observation date. Scoped by the (?m) multiline
// flag so a stray occurrence deeper in the body is ignored.
var sessionDateKeyRe = regexp.MustCompile(`(?m)^(session_date|observed_on):\s*(\S.*?)\s*$`)
var dateLiteralRe = regexp.MustCompile(`\b(\d{4})[/-](\d{2})[/-](\d{2})\b`)

// extractSessionDate returns the ISO YYYY-MM-DD string stored into
// the FTS session_date column. Precedence: frontmatter session_date,
// then observed_on, then modified. Returns the empty string when no
// date is parseable — ranged queries exclude such rows because a
// dateless row cannot honour a between contract.
func extractSessionDate(raw, scope string) string {
	// session_date and observed_on are not part of the wiki or
	// memory frontmatter structs, so scan the raw frontmatter
	// block with a targeted regex. Cap the scan at the closing
	// `---` to avoid matching body text.
	if closeIdx := frontmatterEnd(raw); closeIdx > 0 {
		head := raw[:closeIdx]
		for _, m := range sessionDateKeyRe.FindAllStringSubmatch(head, -1) {
			if iso := normaliseISODate(strings.Trim(m[2], `"'`)); iso != "" {
				return iso
			}
		}
	}
	// Fall back to the structured `modified:` field parsed by the
	// per-scope frontmatter parsers.
	if mod, ok := extractModified([]byte(raw), scope); ok {
		return mod.UTC().Format("2006-01-02")
	}
	return ""
}

func extractDateSearchTags(raw, scope string) []string {
	var tokens []string
	if closeIdx := frontmatterEnd(raw); closeIdx > 0 {
		head := raw[:closeIdx]
		for _, m := range sessionDateKeyRe.FindAllStringSubmatch(head, -1) {
			tokens = append(tokens, dateSearchTokens(strings.Trim(m[2], `"'`))...)
		}
	}
	if mod, ok := extractModified([]byte(raw), scope); ok {
		tokens = append(tokens, dateSearchTokens(mod.UTC().Format("2006-01-02"))...)
	}
	return dedupeStringSlice(tokens)
}

func dateSearchTokens(value string) []string {
	trimmed := strings.TrimSpace(value)
	if trimmed == "" {
		return nil
	}
	if parsed, ok := parseModifiedString(trimmed); ok {
		utc := parsed.UTC()
		return dedupeStringSlice([]string{
			utc.Format("2006-01-02"),
			utc.Format("2006/01/02"),
			utc.Format("2006"),
			utc.Weekday().String(),
			utc.Month().String(),
		})
	}
	match := dateLiteralRe.FindStringSubmatch(trimmed)
	if len(match) != 4 {
		return nil
	}
	year, month, day := match[1], match[2], match[3]
	return dedupeStringSlice([]string{
		year + "-" + month + "-" + day,
		year + "/" + month + "/" + day,
		year,
	})
}

func dedupeStringSlice(values []string) []string {
	if len(values) == 0 {
		return nil
	}
	seen := make(map[string]bool, len(values))
	out := make([]string, 0, len(values))
	for _, value := range values {
		trimmed := strings.TrimSpace(value)
		if trimmed == "" || seen[trimmed] {
			continue
		}
		seen[trimmed] = true
		out = append(out, trimmed)
	}
	return out
}

// frontmatterEnd returns the byte offset of the closing `---` line
// of a YAML frontmatter block, or -1 when the file does not start
// with frontmatter.
func frontmatterEnd(raw string) int {
	if !strings.HasPrefix(raw, "---") {
		return -1
	}
	rest := raw[3:]
	if len(rest) == 0 {
		return -1
	}
	nl := strings.Index(rest, "\n")
	if nl < 0 {
		return -1
	}
	search := rest[nl+1:]
	idx := strings.Index(search, "\n---")
	if idx < 0 {
		return -1
	}
	return 3 + nl + 1 + idx
}

// normaliseISODate turns the common frontmatter date shapes into a
// canonical YYYY-MM-DD. Accepts the same layouts as
// parseModifiedString. Returns "" when nothing parses.
func normaliseISODate(s string) string {
	s = strings.TrimSpace(s)
	if s == "" {
		return ""
	}
	if t, ok := parseModifiedString(s); ok {
		return t.UTC().Format("2006-01-02")
	}
	return ""
}

func parseModifiedString(s string) (time.Time, bool) {
	s = strings.TrimSpace(s)
	if s == "" {
		return time.Time{}, false
	}
	for _, layout := range []string{
		time.RFC3339,
		time.RFC3339Nano,
		"2006-01-02 15:04:05",
		"2006-01-02",
		"2006/01/02",
		"2006/01/02 (Mon) 15:04",
	} {
		if t, err := time.Parse(layout, s); err == nil {
			return t, true
		}
	}
	return time.Time{}, false
}

// applySort re-orders FTS5 results according to mode and truncates
// to maxResults. Stable: ties preserve the FTS5 input order.
func applySort(in []SearchResult, mode SortMode, maxResults int) []SearchResult {
	if len(in) == 0 {
		return in
	}
	if mode == SortRelevance || maxResults <= 0 {
		if len(in) > maxResults && maxResults > 0 {
			return in[:maxResults]
		}
		return in
	}

	// Split dated and undated so undated hits always fall to the
	// end.
	dated := make([]SearchResult, 0, len(in))
	undated := make([]SearchResult, 0, len(in))
	for _, r := range in {
		if r.Modified.IsZero() {
			undated = append(undated, r)
		} else {
			dated = append(dated, r)
		}
	}

	sortByRecency(dated)

	out := make([]SearchResult, 0, len(in))
	out = append(out, dated...)
	out = append(out, undated...)
	if len(out) > maxResults {
		out = out[:maxResults]
	}
	return out
}

func sortByRecency(results []SearchResult) {
	sort.SliceStable(results, func(i, j int) bool {
		return results[i].Modified.After(results[j].Modified)
	})
}

// RebuildStats summarises a single [Index.Rebuild] or [Index.Update]
// invocation: what was discovered on disk, what was actually written
// to FTS, and what stale entries were pruned. Used by the CLI
// diagnostics to produce meaningful human output.
type RebuildStats struct {
	FilesScanned   int
	FilesIndexed   int
	FilesRemoved   int
	RowsBefore     int
	RowsAfter      int
	DurationMillis int64
}

// Rebuild clears and rebuilds the entire FTS5 index from the brain
// store.
func (idx *Index) Rebuild(ctx context.Context) error {
	_, err := idx.rebuild(ctx)
	return err
}

// RebuildWithStats behaves like [Index.Rebuild] but returns a
// populated [RebuildStats].
func (idx *Index) RebuildWithStats(ctx context.Context) (RebuildStats, error) {
	return idx.rebuild(ctx)
}

func (idx *Index) rebuild(ctx context.Context) (RebuildStats, error) {
	start := time.Now()
	stats := RebuildStats{}
	stats.RowsBefore, _ = idx.rowCount(ctx)

	defer func() {
		idx.lastRefresh = time.Now()
		_ = idx.persistRefresh(ctx, idx.lastRefresh)
	}()

	if _, err := idx.db.ExecContext(ctx, "DELETE FROM knowledge_fts"); err != nil {
		return stats, fmt.Errorf("clearing FTS index: %w", err)
	}
	if _, err := idx.db.ExecContext(ctx, "DELETE FROM knowledge_index_state"); err != nil {
		return stats, fmt.Errorf("clearing index state: %w", err)
	}

	files := idx.discoverFiles(ctx)
	stats.FilesScanned = len(files)
	stats.FilesRemoved = stats.RowsBefore // full rebuild wipes everything

	if err := idx.indexFiles(ctx, files); err != nil {
		return stats, err
	}
	stats.FilesIndexed = len(files)
	stats.RowsAfter, _ = idx.rowCount(ctx)
	stats.DurationMillis = time.Since(start).Milliseconds()
	return stats, nil
}

// Update incrementally indexes new or changed files and removes
// stale entries.
func (idx *Index) Update(ctx context.Context) error {
	_, err := idx.update(ctx)
	return err
}

// UpdateWithStats behaves like [Index.Update] but returns a
// populated [RebuildStats].
func (idx *Index) UpdateWithStats(ctx context.Context) (RebuildStats, error) {
	return idx.update(ctx)
}

func (idx *Index) update(ctx context.Context) (RebuildStats, error) {
	start := time.Now()
	stats := RebuildStats{}
	stats.RowsBefore, _ = idx.rowCount(ctx)

	defer func() {
		idx.lastRefresh = time.Now()
		_ = idx.persistRefresh(ctx, idx.lastRefresh)
	}()

	files := idx.discoverFiles(ctx)
	stats.FilesScanned = len(files)

	// Build a set of current file paths (logical) for stale
	// detection.
	currentPaths := make(map[string]struct{}, len(files))
	for _, f := range files {
		currentPaths[string(f.path)] = struct{}{}
	}

	// Remove stale entries (files that no longer exist in the
	// brain).
	existingPaths, err := idx.allIndexedPaths()
	if err != nil {
		return stats, fmt.Errorf("listing indexed paths: %w", err)
	}
	for _, p := range existingPaths {
		if _, exists := currentPaths[p]; !exists {
			if err := idx.Remove(p); err != nil {
				return stats, fmt.Errorf("removing stale entry %s: %w", p, err)
			}
			stats.FilesRemoved++
		}
	}

	// Filter to only new or changed files.
	var toIndex []discoveredFile
	for _, f := range files {
		data, err := idx.store.Read(ctx, f.path)
		if err != nil {
			continue
		}
		checksum := checksumBytes(data)

		var existing string
		err = idx.db.QueryRowContext(ctx,
			"SELECT checksum FROM knowledge_index_state WHERE path = ?",
			string(f.path),
		).Scan(&existing)

		if errors.Is(err, sql.ErrNoRows) || existing != checksum {
			f.content = data
			f.checksum = checksum
			toIndex = append(toIndex, f)
		}
	}

	stats.FilesIndexed = len(toIndex)

	if len(toIndex) > 0 {
		if err := idx.indexFiles(ctx, toIndex); err != nil {
			return stats, err
		}
	}

	stats.RowsAfter, _ = idx.rowCount(ctx)
	stats.DurationMillis = time.Since(start).Milliseconds()
	return stats, nil
}

// Subscribe attaches the index to a [brain.Store] so every
// successful mutation triggers an incremental update to the FTS
// rows. The returned function unsubscribes the sink; callers
// typically defer it for the lifetime of the process.
//
// Events for paths outside the indexed scopes (memory and wiki
// markdown files) are ignored. Read and index errors are logged but
// not propagated — the next [Index.Update] or [Index.RebuildIfEmpty]
// call heals any inconsistency.
func (idx *Index) Subscribe(store brain.Store) func() {
	if store == nil {
		return func() {}
	}
	return store.Subscribe(brain.EventSinkFunc(func(evt brain.ChangeEvent) {
		idx.handleChange(context.Background(), store, evt)
	}))
}

// handleChange applies a single [brain.ChangeEvent] to the FTS
// index.
func (idx *Index) handleChange(ctx context.Context, store brain.Store, evt brain.ChangeEvent) {
	switch evt.Kind {
	case brain.ChangeDeleted:
		if !isIndexablePath(evt.Path) {
			return
		}
		if err := idx.Remove(string(evt.Path)); err != nil {
			slog.Warn("search: remove on brain delete failed", "path", evt.Path, "err", err)
		}
	case brain.ChangeRenamed:
		if isIndexablePath(evt.OldPath) {
			if err := idx.Remove(string(evt.OldPath)); err != nil {
				slog.Warn("search: remove on brain rename failed", "path", evt.OldPath, "err", err)
			}
		}
		if isIndexablePath(evt.Path) {
			if err := idx.reindexOne(ctx, store, evt.Path); err != nil {
				slog.Warn("search: reindex on brain rename failed", "path", evt.Path, "err", err)
			}
		}
	case brain.ChangeCreated, brain.ChangeUpdated:
		if !isIndexablePath(evt.Path) {
			return
		}
		if err := idx.reindexOne(ctx, store, evt.Path); err != nil {
			slog.Warn("search: reindex on brain change failed", "path", evt.Path, "err", err)
		}
	}
}

// reindexOne reads a single file via the store and upserts it into
// the FTS index.
func (idx *Index) reindexOne(ctx context.Context, store brain.Store, p brain.Path) error {
	scope, projectSlug, ok := classifyPath(p)
	if !ok {
		return nil
	}
	data, err := store.Read(ctx, p)
	if err != nil {
		if errors.Is(err, brain.ErrNotFound) {
			return idx.Remove(string(p))
		}
		return fmt.Errorf("reading %s: %w", p, err)
	}
	return idx.indexFiles(ctx, []discoveredFile{{
		path:        p,
		scope:       scope,
		projectSlug: projectSlug,
		content:     data,
		checksum:    checksumBytes(data),
	}})
}

// RebuildIfEmpty triggers a full [Index.Rebuild] when the FTS table
// has no rows but the brain has indexable content. Intended to be
// called once at startup from a background goroutine so the app
// does not block on a cold cache. A nil store or an empty brain is
// a no-op.
func (idx *Index) RebuildIfEmpty(ctx context.Context, store brain.Store) error {
	if store == nil {
		return nil
	}
	count, err := idx.rowCount(ctx)
	if err != nil {
		return fmt.Errorf("counting FTS rows: %w", err)
	}
	if count > 0 {
		return nil
	}
	// Swap in the supplied store so Rebuild walks the right
	// backend when this Index was constructed with a different
	// default.
	previous := idx.store
	idx.store = store
	defer func() { idx.store = previous }()
	return idx.Rebuild(ctx)
}

// RebuildIfStale triggers a full [Index.Rebuild] when the FTS wiki
// row count is less than [staleWikiRatio] times the supplied
// on-disk wiki file count. Acts as a safety net for cases where
// Subscribe missed writes during a large compilation run. A nil
// store or a non-positive wikiFileCount is a no-op.
func (idx *Index) RebuildIfStale(ctx context.Context, store brain.Store, wikiFileCount int) error {
	if store == nil || wikiFileCount <= 0 {
		return nil
	}
	indexed, err := idx.wikiRowCount(ctx)
	if err != nil {
		return fmt.Errorf("counting wiki FTS rows: %w", err)
	}
	threshold := int(float64(wikiFileCount) * staleWikiRatio)
	if indexed >= threshold {
		return nil
	}
	slog.Info("search: wiki index stale, forcing rebuild",
		"indexed", indexed,
		"onDisk", wikiFileCount,
		"threshold", threshold,
	)
	previous := idx.store
	idx.store = store
	defer func() { idx.store = previous }()
	return idx.Rebuild(ctx)
}

// RowCount returns the current number of rows in knowledge_fts.
func (idx *Index) RowCount(ctx context.Context) (int, error) {
	return idx.rowCount(ctx)
}

// RowCountByScope returns the number of rows in knowledge_fts
// grouped by scope. Used by diagnostics to flag scope-specific
// anomalies.
func (idx *Index) RowCountByScope(ctx context.Context) (map[string]int, error) {
	rows, err := idx.db.QueryContext(ctx, "SELECT scope, count(*) FROM knowledge_fts GROUP BY scope")
	if err != nil {
		return nil, fmt.Errorf("counting FTS rows by scope: %w", err)
	}
	defer rows.Close()

	out := map[string]int{}
	for rows.Next() {
		var scope string
		var count int
		if err := rows.Scan(&scope, &count); err != nil {
			return nil, err
		}
		out[scope] = count
	}
	return out, rows.Err()
}

// StateRowCount returns the number of rows in
// knowledge_index_state. Equal to RowCount in a healthy index; a
// mismatch indicates the two tables drifted and a full rebuild is
// warranted.
func (idx *Index) StateRowCount(ctx context.Context) (int, error) {
	var count int
	err := idx.db.QueryRowContext(ctx, "SELECT count(*) FROM knowledge_index_state").Scan(&count)
	return count, err
}

// LastRefresh returns the wall-clock time of the most recent
// Rebuild or Update invocation. Returns the zero time if the index
// has not been touched since the process started.
func (idx *Index) LastRefresh() time.Time {
	return idx.lastRefresh
}

// PersistedRefresh returns the last_refresh_at timestamp persisted
// to knowledge_index_meta. Prefer this over [Index.LastRefresh] for
// cross-process diagnostics: the in-memory value is only populated
// once this process has itself performed a refresh.
func (idx *Index) PersistedRefresh(ctx context.Context) (time.Time, error) {
	var raw string
	err := idx.db.QueryRowContext(ctx,
		"SELECT value FROM knowledge_index_meta WHERE key = ?",
		metaLastRefresh,
	).Scan(&raw)
	if errors.Is(err, sql.ErrNoRows) {
		return time.Time{}, nil
	}
	if err != nil {
		return time.Time{}, err
	}
	t, parseErr := time.Parse(time.RFC3339Nano, raw)
	if parseErr != nil {
		return time.Time{}, parseErr
	}
	return t, nil
}

func (idx *Index) rowCount(ctx context.Context) (int, error) {
	var count int
	err := idx.db.QueryRowContext(ctx, "SELECT count(*) FROM knowledge_fts").Scan(&count)
	return count, err
}

func (idx *Index) wikiRowCount(ctx context.Context) (int, error) {
	var count int
	err := idx.db.QueryRowContext(ctx,
		"SELECT count(*) FROM knowledge_fts WHERE scope = 'wiki'",
	).Scan(&count)
	return count, err
}

// migrateSchemaIfNeeded drops the FTS tables when the persisted
// schema version is absent or lower than currentSchemaVersion. The
// meta table itself is preserved across migrations so unrelated
// keys survive. knowledge_index_state is dropped alongside the FTS
// table so the next Update reindexes from the brain store rather
// than trusting stale checksums.
func migrateSchemaIfNeeded(db *sql.DB) error {
	var existing string
	err := db.QueryRow(`SELECT value FROM knowledge_index_meta WHERE key = ?`, metaSchemaVersion).Scan(&existing)
	switch {
	case errors.Is(err, sql.ErrNoRows):
		// No meta table yet, or no row — harmless: the fresh
		// schema below is version 2 already, and empty databases
		// have nothing to drop.
		return nil
	case err != nil:
		// The meta table does not exist yet on brand-new databases.
		if strings.Contains(err.Error(), "no such table") {
			return nil
		}
		return err
	}
	if existing == "" || existing == currentSchemaVersion {
		return nil
	}
	// Older version present: drop FTS and state so they get rebuilt
	// with the new column layout.
	for _, stmt := range []string{
		`DROP TABLE IF EXISTS knowledge_fts`,
		`DROP TABLE IF EXISTS knowledge_index_state`,
	} {
		if _, err := db.Exec(stmt); err != nil {
			return err
		}
	}
	return nil
}

// recordSchemaVersion upserts the current schema version into the
// meta table so subsequent opens compare against it.
func recordSchemaVersion(db *sql.DB) error {
	_, err := db.Exec(
		`INSERT INTO knowledge_index_meta (key, value)
		 VALUES (?, ?)
		 ON CONFLICT(key) DO UPDATE SET value = excluded.value`,
		metaSchemaVersion, currentSchemaVersion,
	)
	return err
}

func (idx *Index) persistRefresh(ctx context.Context, t time.Time) error {
	_, err := idx.db.ExecContext(ctx,
		`INSERT INTO knowledge_index_meta (key, value)
		 VALUES (?, ?)
		 ON CONFLICT(key) DO UPDATE SET value = excluded.value`,
		metaLastRefresh, t.UTC().Format(time.RFC3339Nano),
	)
	return err
}

// isIndexablePath reports whether a logical path falls under one of
// the scopes the FTS index covers (memory or wiki) and is a
// markdown file.
func isIndexablePath(p brain.Path) bool {
	_, _, ok := classifyPath(p)
	return ok
}

// classifyPath maps a logical [brain.Path] to the FTS scope and
// project slug it belongs to. Returns ok=false for paths the index
// does not cover, or for non-markdown files.
func classifyPath(p brain.Path) (scope, projectSlug string, ok bool) {
	s := string(p)
	if !strings.HasSuffix(s, ".md") {
		return "", "", false
	}
	switch {
	case strings.HasPrefix(s, "memory/global/"):
		return "global_memory", "", true
	case strings.HasPrefix(s, "memory/project/"):
		rest := strings.TrimPrefix(s, "memory/project/")
		slug, _, found := strings.Cut(rest, "/")
		if !found || slug == "" {
			return "", "", false
		}
		return "project_memory", slug, true
	case strings.HasPrefix(s, "wiki/"):
		return "wiki", "", true
	case strings.HasPrefix(s, "raw/documents/"):
		return "raw_document", "", true
	case strings.HasPrefix(s, "raw/.sources/"):
		return "sources", "", true
	case strings.HasPrefix(s, "raw/lme/"):
		// LongMemEval bulk-ingested session transcripts. Indexed so the
		// tri-SDK eval pipeline can fall back to raw session retrieval
		// when the extraction pass produced no facts (e.g. small
		// extraction models that repeatedly return `{"memories": []}`).
		return "raw_lme", "", true
	}
	return "", "", false
}

// Remove deletes a file from the FTS index and state table.
func (idx *Index) Remove(path string) error {
	if _, err := idx.db.Exec("DELETE FROM knowledge_fts WHERE path = ?", path); err != nil {
		return fmt.Errorf("removing from FTS: %w", err)
	}
	if _, err := idx.db.Exec("DELETE FROM knowledge_index_state WHERE path = ?", path); err != nil {
		return fmt.Errorf("removing from index state: %w", err)
	}
	return nil
}

// discoveredFile holds metadata about a markdown file identified
// by its logical [brain.Path]. Content and checksum are populated
// lazily during indexing so discovery remains cheap.
type discoveredFile struct {
	path        brain.Path
	scope       string
	projectSlug string
	content     []byte
	checksum    string
}

// discoverFiles walks the brain via [brain.Store.List] and returns
// every markdown file in memory, projects, and wiki. The listing
// includes generated files because the FTS index covers them too.
func (idx *Index) discoverFiles(ctx context.Context) []discoveredFile {
	var files []discoveredFile

	// Global memory.
	files = append(files, idx.walkPrefix(ctx, brain.MemoryGlobalPrefix(), "global_memory", "")...)

	// Project memories — enumerate slugs by listing the projects
	// prefix shallowly, then walk each project's memory subtree.
	projects, err := idx.store.List(ctx, brain.MemoryProjectsPrefix(), brain.ListOpts{IncludeGenerated: true})
	if err == nil {
		for _, p := range projects {
			if !p.IsDir {
				continue
			}
			slug := trailingSegment(string(p.Path))
			files = append(files, idx.walkPrefix(ctx, brain.MemoryProjectPrefix(slug), "project_memory", slug)...)
		}
	}

	// Wiki.
	files = append(files, idx.walkPrefix(ctx, brain.WikiPrefix(), "wiki", "")...)

	// Raw ingested documents (knowledge.Base.Ingest writes under this
	// prefix). Indexing here keeps BM25 search honest with what the
	// ingest pipeline persists.
	files = append(files, idx.walkPrefix(ctx, brain.RawDocumentsPrefix(), "raw_document", "")...)

	// LME replay ingest (memory eval lme run --ingest-mode=replay
	// persists session markdown under this prefix). Indexing here lets
	// the daemon surface replay content to the actor at question time.
	files = append(files, idx.walkPrefix(ctx, brain.RawLMEPrefix(), "raw_lme", "")...)

	// Compiled sources (originals preserved after compilation).
	files = append(files, idx.walkPrefix(ctx, brain.SourcesPrefix(), "sources", "")...)

	return files
}

// walkPrefix returns discoveredFile entries for every .md file
// under the given logical prefix. Uses the store's recursive
// listing to flatten nested topic directories.
func (idx *Index) walkPrefix(ctx context.Context, prefix brain.Path, scope, projectSlug string) []discoveredFile {
	entries, err := idx.store.List(ctx, prefix, brain.ListOpts{
		Recursive:        true,
		IncludeGenerated: true,
	})
	if err != nil {
		if errors.Is(err, brain.ErrNotFound) {
			return nil
		}
		return nil
	}
	var files []discoveredFile
	for _, e := range entries {
		if e.IsDir {
			continue
		}
		if !strings.HasSuffix(string(e.Path), ".md") {
			continue
		}
		// Skip underscore-prefixed markdown files (_log.md,
		// _index.md, _health.md, _concepts.md and friends). These
		// are generated topic/summary files and should never
		// appear in search results. The check applies only to the
		// file basename so sibling files inside a topic directory
		// still get indexed.
		if strings.HasPrefix(trailingSegment(string(e.Path)), "_") {
			continue
		}
		files = append(files, discoveredFile{
			path:        e.Path,
			scope:       scope,
			projectSlug: projectSlug,
		})
	}
	return files
}

// indexFiles reads, parses, and inserts files into the FTS index in
// batches.
func (idx *Index) indexFiles(ctx context.Context, files []discoveredFile) error {
	for i := 0; i < len(files); i += insertBatchSize {
		end := min(i+insertBatchSize, len(files))
		batch := files[i:end]

		tx, err := idx.db.BeginTx(ctx, nil)
		if err != nil {
			return fmt.Errorf("beginning batch transaction: %w", err)
		}

		for _, f := range batch {
			if err := idx.indexOneFile(ctx, tx, f); err != nil {
				_ = tx.Rollback()
				return err
			}
		}

		if err := tx.Commit(); err != nil {
			return fmt.Errorf("committing batch: %w", err)
		}
	}

	return nil
}

// indexOneFile parses a single file and inserts it into the FTS
// index. Content is loaded via the brain store so the indexer works
// against any backend.
func (idx *Index) indexOneFile(ctx context.Context, tx *sql.Tx, f discoveredFile) error {
	data := f.content
	if data == nil {
		var err error
		data, err = idx.store.Read(ctx, f.path)
		if err != nil {
			return nil // Skip unreadable files — a missing entry at index time will be caught by the next full rebuild.
		}
	}
	checksum := f.checksum
	if checksum == "" {
		checksum = checksumBytes(data)
	}

	raw := string(data)
	title, summary, tags, body := parseFrontmatterGeneric(raw, f.scope)
	sessionDate := extractSessionDate(raw, f.scope)
	tags = dedupeStringSlice(append(tags, extractDateSearchTags(raw, f.scope)...))

	pathStr := string(f.path)

	if _, err := tx.ExecContext(ctx, "DELETE FROM knowledge_fts WHERE path = ?", pathStr); err != nil {
		return fmt.Errorf("deleting old FTS entry for %s: %w", pathStr, err)
	}

	if _, err := tx.ExecContext(ctx,
		`INSERT INTO knowledge_fts (path, title, summary, tags, content, scope, project_slug, session_date)
		 VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
		pathStr, title, summary, strings.Join(tags, " "), body, f.scope, f.projectSlug, sessionDate,
	); err != nil {
		return fmt.Errorf("inserting FTS entry for %s: %w", pathStr, err)
	}

	if _, err := tx.ExecContext(ctx,
		`INSERT INTO knowledge_index_state (path, checksum, indexed_at)
		 VALUES (?, ?, CURRENT_TIMESTAMP)
		 ON CONFLICT(path) DO UPDATE SET checksum = excluded.checksum, indexed_at = CURRENT_TIMESTAMP`,
		pathStr, checksum,
	); err != nil {
		return fmt.Errorf("upserting index state for %s: %w", pathStr, err)
	}

	if idx.OnIndex != nil {
		idx.OnIndex(f.path)
	}

	return nil
}

// parseFrontmatterGeneric delegates to the appropriate parser based
// on scope and returns a uniform (title, summary, tags, body)
// tuple.
func parseFrontmatterGeneric(content, scope string) (title, summary string, tags []string, body string) {
	switch scope {
	case "wiki":
		fm, b := parseWikiFrontmatter(content)
		return fm.Title, fm.Summary, fm.Tags, b
	default:
		fm, b := parseMemoryFrontmatter(content)
		return fm.Name, fm.Description, fm.Tags, b
	}
}

// allIndexedPaths returns every path currently in the index state
// table.
func (idx *Index) allIndexedPaths() ([]string, error) {
	rows, err := idx.db.Query("SELECT path FROM knowledge_index_state")
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var paths []string
	for rows.Next() {
		var p string
		if err := rows.Scan(&p); err != nil {
			return nil, err
		}
		paths = append(paths, p)
	}
	return paths, rows.Err()
}

// IndexedPaths exports [allIndexedPaths] for callers outside the package
// (e.g. the daemon's vector backfill) that need to walk every FTS-known
// path without re-scanning the store.
func (idx *Index) IndexedPaths() ([]string, error) {
	return idx.allIndexedPaths()
}

// IndexedChecksums returns path -> checksum for every entry in the
// index state table. Callers compare checksums against the vector
// store to detect drift without reading file bodies.
func (idx *Index) IndexedChecksums() (map[string]string, error) {
	rows, err := idx.db.Query("SELECT path, checksum FROM knowledge_index_state")
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	out := map[string]string{}
	for rows.Next() {
		var p, c string
		if err := rows.Scan(&p, &c); err != nil {
			return nil, err
		}
		out[p] = c
	}
	return out, rows.Err()
}

// sanitiseQuery converts a natural-language user query into a valid
// FTS5 MATCH expression with stop words stripped and bare tokens
// joined by `OR`. Quoted phrases and explicit boolean operators are
// preserved. If every meaningful token is a stop word, the raw
// input is returned (lightly scrubbed) so the caller still gets a
// best-effort match instead of nothing at all.
func sanitiseQuery(query string) string {
	query = strings.TrimSpace(query)
	if query == "" {
		return ""
	}
	tokens := ParseQuery(query)
	if len(tokens) == 0 {
		// Fall back to a scrubbed version of the raw input so a
		// "the and or" query still does something reasonable.
		return fallbackSanitise(query)
	}
	return BuildFTS5Expr(tokens)
}

// fallbackSanitise is the belt-and-braces path when [ParseQuery]
// strips every meaningful token.
func fallbackSanitise(query string) string {
	replacer := strings.NewReplacer(
		"*", "",
		"(", "",
		")", "",
		":", "",
		"^", "",
		"+", "",
		`"`, "",
		"?", "",
		"!", "",
		",", "",
		";", "",
		"$", "",
		"#", "",
		"@", "",
		"%", "",
		"=", "",
	)
	query = replacer.Replace(query)
	words := strings.Fields(query)
	if len(words) == 0 {
		return ""
	}
	return strings.Join(words, " ")
}

// checksumBytes returns a hex-encoded SHA-256 of the data.
func checksumBytes(data []byte) string {
	h := sha256.Sum256(data)
	return fmt.Sprintf("%x", h)
}
