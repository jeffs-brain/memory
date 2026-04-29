// SPDX-License-Identifier: Apache-2.0

package search

import (
	"context"
	"database/sql"
	"testing"
	"time"

	"github.com/jeffs-brain/memory/go/brain"
)

// seedRawStore writes arbitrary raw markdown content (including
// its own frontmatter block) to a test store at the given logical
// paths. Unlike newWikiTestStore this helper does NOT prepend a
// frontmatter wrapper, so tests can assert the exact bytes the
// index sees.
func seedRawStore(t *testing.T, docs map[string]string) brain.Store {
	t.Helper()
	store := newTestStore()
	t.Cleanup(func() { _ = store.Close() })
	ctx := context.Background()
	for path, content := range docs {
		if err := store.Write(ctx, brain.Path(path), []byte(content)); err != nil {
			t.Fatalf("write %s: %v", path, err)
		}
	}
	return store
}

// TestSearchOpts_HasDateFilter guards the small helper that the
// public search path consults before widening the FTS fetch pool
// or attaching the WHERE session_date clause.
func TestSearchOpts_HasDateFilter(t *testing.T) {
	if (SearchOpts{}).HasDateFilter() {
		t.Error("empty opts must report no date filter")
	}
	if !(SearchOpts{DateFrom: mustTime("2024-01-01")}).HasDateFilter() {
		t.Error("DateFrom alone must count as a date filter")
	}
	if !(SearchOpts{DateTo: mustTime("2024-01-01")}).HasDateFilter() {
		t.Error("DateTo alone must count as a date filter")
	}
}

// TestSearch_EmptyDateRangeReturnsUnfiltered is the regression
// guard: the zero DateFrom / DateTo pair must NOT drop rows with
// an empty session_date column.
func TestSearch_EmptyDateRangeReturnsUnfiltered(t *testing.T) {
	db, idx := newIndexEmpty(t)

	insertDateRow(t, db, "wiki/with.md", "With Date", "alpha beta", "wiki", "2024-03-15")
	insertDateRow(t, db, "wiki/without.md", "No Date", "alpha beta", "wiki", "")

	results, err := idx.Search("alpha beta", SearchOpts{MaxResults: 10})
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(results) != 2 {
		t.Fatalf("expected 2 results (dateless should survive), got %d: %v", len(results), searchResultPaths(results))
	}
}

// TestSearch_DateRangePushedToSQL seeds five rows with distinct
// session_dates and asserts a range query returns exactly the rows
// inside the inclusive bounds.
func TestSearch_DateRangePushedToSQL(t *testing.T) {
	db, idx := newIndexEmpty(t)

	insertDateRow(t, db, "wiki/a.md", "A", "shared term", "wiki", "2024-02-01")
	insertDateRow(t, db, "wiki/b.md", "B", "shared term", "wiki", "2024-03-01")
	insertDateRow(t, db, "wiki/c.md", "C", "shared term", "wiki", "2024-03-15")
	insertDateRow(t, db, "wiki/d.md", "D", "shared term", "wiki", "2024-04-01")
	insertDateRow(t, db, "wiki/e.md", "E", "shared term", "wiki", "")

	results, err := idx.Search("shared term", SearchOpts{
		MaxResults: 10,
		DateFrom:   mustTime("2024-03-01"),
		DateTo:     mustTime("2024-03-31"),
	})
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	got := searchResultPaths(results)
	want := []string{"wiki/b.md", "wiki/c.md"}
	if !sameSet(got, want) {
		t.Errorf("range hits = %v, want %v", got, want)
	}

	results, err = idx.Search("shared term", SearchOpts{
		MaxResults: 10,
		DateFrom:   mustTime("2024-03-15"),
	})
	if err != nil {
		t.Fatalf("Search from-only: %v", err)
	}
	got = searchResultPaths(results)
	want = []string{"wiki/c.md", "wiki/d.md"}
	if !sameSet(got, want) {
		t.Errorf("from-only hits = %v, want %v", got, want)
	}

	results, err = idx.Search("shared term", SearchOpts{
		MaxResults: 10,
		DateTo:     mustTime("2024-02-28"),
	})
	if err != nil {
		t.Fatalf("Search to-only: %v", err)
	}
	got = searchResultPaths(results)
	want = []string{"wiki/a.md"}
	if !sameSet(got, want) {
		t.Errorf("to-only hits = %v, want %v", got, want)
	}
}

func TestSearch_SessionIDsPushedToSQL(t *testing.T) {
	db, idx := newIndexEmpty(t)

	insertDateRow(t, db, "memory/project/app/a.md", "A", "shared term", "project_memory", "2024-03-01")
	insertDateRow(t, db, "memory/project/app/b.md", "B", "shared term", "project_memory", "2024-03-02")
	insertDateRow(t, db, "memory/project/app/c.md", "C", "shared term", "project_memory", "2024-03-03")
	insertSessionMetadata(t, db, "memory/project/app/a.md", "session-a")
	insertSessionMetadata(t, db, "memory/project/app/b.md", "session-b")
	insertSessionMetadata(t, db, "memory/project/app/c.md", "session-c")

	results, err := idx.Search("shared term", SearchOpts{
		MaxResults: 10,
		Scope:      "project_memory",
		SessionIDs: []string{"session-b", "session-c"},
	})
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	got := searchResultPaths(results)
	want := []string{"memory/project/app/b.md", "memory/project/app/c.md"}
	if !sameSet(got, want) {
		t.Errorf("session hits = %v, want %v", got, want)
	}
}

// TestSearch_DateRangeComposesWithScope asserts the range filter
// intersects cleanly with the scope filter.
func TestSearch_DateRangeComposesWithScope(t *testing.T) {
	db, idx := newIndexEmpty(t)

	insertDateRow(t, db, "wiki/x.md", "Wiki X", "combine", "wiki", "2024-03-10")
	insertDateRow(t, db, "wiki/y.md", "Wiki Y", "combine", "wiki", "2024-05-10")
	insertDateRow(t, db, "memory/global/x.md", "Mem X", "combine", "global_memory", "2024-03-10")
	insertDateRow(t, db, "memory/global/y.md", "Mem Y", "combine", "global_memory", "2024-05-10")

	results, err := idx.Search("combine", SearchOpts{
		MaxResults: 10,
		Scope:      "wiki",
		DateFrom:   mustTime("2024-03-01"),
		DateTo:     mustTime("2024-03-31"),
	})
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	got := searchResultPaths(results)
	want := []string{"wiki/x.md"}
	if !sameSet(got, want) {
		t.Errorf("scope+range intersection = %v, want %v", got, want)
	}
}

// TestSearch_DateRangeComposesWithSupersession asserts the range
// filter runs alongside the default supersession filter without
// double-dropping rows.
func TestSearch_DateRangeComposesWithSupersession(t *testing.T) {
	store := seedRawStore(t, map[string]string{
		"memory/global/old.md":   "---\nname: old fact\nsession_date: 2024-03-10\nsuperseded_by: new.md\n---\nstale body\n",
		"memory/global/new.md":   "---\nname: new fact\nsession_date: 2024-03-20\n---\nfresh body\n",
		"memory/global/other.md": "---\nname: other fact\nsession_date: 2024-05-01\n---\nother body\n",
	})

	db := openTestDB(t)
	idx, err := NewIndex(db, store)
	if err != nil {
		t.Fatalf("NewIndex: %v", err)
	}
	if err := idx.Rebuild(context.Background()); err != nil {
		t.Fatalf("Rebuild: %v", err)
	}

	results, err := idx.Search("body", SearchOpts{
		MaxResults: 10,
		DateFrom:   mustTime("2024-03-01"),
		DateTo:     mustTime("2024-03-31"),
	})
	if err != nil {
		t.Fatalf("Search: %v", err)
	}

	got := searchResultPaths(results)
	want := []string{"memory/global/new.md"}
	if !sameSet(got, want) {
		t.Errorf("range + supersession = %v, want %v", got, want)
	}
}

// TestExtractSessionDate_PrecedenceOrder checks the precedence
// rule documented on extractSessionDate.
func TestExtractSessionDate_PrecedenceOrder(t *testing.T) {
	cases := []struct {
		name string
		raw  string
		want string
	}{
		{
			name: "session_date wins",
			raw:  "---\nsession_date: 2024-03-15\nobserved_on: 2024-02-10\nmodified: 2024-01-05\n---\nbody",
			want: "2024-03-15",
		},
		{
			name: "observed_on beats modified",
			raw:  "---\nobserved_on: 2024-02-10\nmodified: 2024-01-05\n---\nbody",
			want: "2024-02-10",
		},
		{
			name: "modified as fallback",
			raw:  "---\nname: x\nmodified: 2024-01-05\n---\nbody",
			want: "2024-01-05",
		},
		{
			name: "none parseable",
			raw:  "---\nname: x\n---\nbody",
			want: "",
		},
		{
			name: "non-ISO session_date normalised",
			raw:  "---\nsession_date: 2024/03/15 (Mon) 10:00\n---\nbody",
			want: "2024-03-15",
		},
		{
			name: "no frontmatter at all",
			raw:  "just a body",
			want: "",
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := extractSessionDate(c.raw, "global_memory")
			if got != c.want {
				t.Errorf("extractSessionDate = %q, want %q", got, c.want)
			}
		})
	}
}

// TestIndexOneFile_PopulatesSessionDate asserts the index-write
// pipeline materialises the session_date column using the
// documented precedence.
func TestIndexOneFile_PopulatesSessionDate(t *testing.T) {
	store := seedRawStore(t, map[string]string{
		"memory/global/a.md": "---\nname: alpha\nsession_date: 2024-03-15\n---\nalpha body\n",
		"memory/global/b.md": "---\nname: beta\nobserved_on: 2024-02-10\n---\nbeta body\n",
		"memory/global/c.md": "---\nname: gamma\nmodified: 2024-01-05\n---\ngamma body\n",
		"memory/global/d.md": "---\nname: delta\n---\nno frontmatter date body\n",
	})

	db := openTestDB(t)
	idx, err := NewIndex(db, store)
	if err != nil {
		t.Fatalf("NewIndex: %v", err)
	}
	if err := idx.Rebuild(context.Background()); err != nil {
		t.Fatalf("Rebuild: %v", err)
	}

	want := map[string]string{
		"memory/global/a.md": "2024-03-15",
		"memory/global/b.md": "2024-02-10",
		"memory/global/c.md": "2024-01-05",
		"memory/global/d.md": "",
	}

	rows, err := db.Query(`SELECT path, session_date FROM knowledge_fts ORDER BY path`)
	if err != nil {
		t.Fatalf("select: %v", err)
	}
	defer rows.Close()
	got := map[string]string{}
	for rows.Next() {
		var p, sd string
		if err := rows.Scan(&p, &sd); err != nil {
			t.Fatalf("scan: %v", err)
		}
		got[p] = sd
	}
	for path, expected := range want {
		if got[path] != expected {
			t.Errorf("%s: session_date = %q, want %q", path, got[path], expected)
		}
	}
}

// Helpers.

func mustTime(s string) time.Time {
	t, err := time.Parse("2006-01-02", s)
	if err != nil {
		panic(err)
	}
	return t
}

func searchResultPaths(results []SearchResult) []string {
	out := make([]string, len(results))
	for i, r := range results {
		out[i] = r.Path
	}
	return out
}

// sameSet reports whether two string slices contain the same
// elements, ignoring order.
func sameSet(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	bag := make(map[string]int, len(a))
	for _, s := range a {
		bag[s]++
	}
	for _, s := range b {
		bag[s]--
		if bag[s] < 0 {
			return false
		}
	}
	return true
}

// insertDateRow drops an FTS row directly with a controlled
// session_date.
func insertDateRow(t *testing.T, db *sql.DB, path, title, body, scope, sessionDate string) {
	t.Helper()
	if _, err := db.Exec(
		`INSERT INTO knowledge_fts (path, title, summary, tags, content, scope, project_slug, session_date)
		 VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
		path, title, "", "", body, scope, "", sessionDate,
	); err != nil {
		t.Fatalf("insert row %s: %v", path, err)
	}
}

func insertSearchTestRow(t *testing.T, db *sql.DB, path, title, body, scope, projectSlug string) {
	t.Helper()
	if _, err := db.Exec(
		`INSERT INTO knowledge_fts (path, title, summary, tags, content, scope, project_slug)
		 VALUES (?, ?, ?, ?, ?, ?, ?)`,
		path, title, "", "", body, scope, projectSlug,
	); err != nil {
		t.Fatalf("insert row %s: %v", path, err)
	}
}

func insertSessionMetadata(t *testing.T, db *sql.DB, path, sessionID string) {
	t.Helper()
	if _, err := db.Exec(
		`INSERT INTO knowledge_index_metadata (path, session_id)
		 VALUES (?, ?)`,
		path, sessionID,
	); err != nil {
		t.Fatalf("insert metadata %s: %v", path, err)
	}
}
