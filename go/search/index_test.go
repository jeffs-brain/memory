// SPDX-License-Identifier: Apache-2.0

package search

import (
	"context"
	"database/sql"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/jeffs-brain/memory/go/brain"

	_ "modernc.org/sqlite"
)

// openTestDB returns an in-memory SQLite handle scheduled for
// cleanup.
func openTestDB(t *testing.T) *sql.DB {
	t.Helper()
	db, err := sql.Open("sqlite", ":memory:")
	if err != nil {
		t.Fatalf("opening in-memory db: %v", err)
	}
	t.Cleanup(func() { db.Close() })
	return db
}

// newIndexEmpty opens an in-memory SQLite DB and builds an Index
// against an empty test store. Matches the jeff-era NewIndex(db)
// convenience for tests that only exercise the SQL path.
func newIndexEmpty(t *testing.T) (*sql.DB, *Index) {
	t.Helper()
	db := openTestDB(t)
	store := newTestStore()
	t.Cleanup(func() { _ = store.Close() })
	idx, err := NewIndex(db, store)
	if err != nil {
		t.Fatalf("NewIndex: %v", err)
	}
	return db, idx
}

// wikiTestDoc is a lightweight shape used by [newWikiTestStore] to
// seed a test store with wiki or memory markdown documents. Supply
// either [title] (wiki) or [name] (memory) depending on the target
// scope.
type wikiTestDoc struct {
	path  string
	title string
	name  string
	body  string
}

// newWikiTestStore creates an empty testStore, writes the supplied
// markdown documents to it, and returns it.
func newWikiTestStore(t *testing.T, docs ...wikiTestDoc) brain.Store {
	t.Helper()
	store := newTestStore()
	t.Cleanup(func() { _ = store.Close() })
	ctx := context.Background()
	for _, d := range docs {
		var fm string
		switch {
		case d.title != "":
			fm = "---\ntitle: " + d.title + "\n---\n"
		case d.name != "":
			fm = "---\nname: " + d.name + "\n---\n"
		}
		content := fm + d.body
		if !strings.HasSuffix(content, "\n") {
			content += "\n"
		}
		if err := store.Write(ctx, brain.Path(d.path), []byte(content)); err != nil {
			t.Fatalf("write %s: %v", d.path, err)
		}
	}
	return store
}

// memstoreEmpty returns an empty testStore scheduled for cleanup.
func memstoreEmpty(t *testing.T) brain.Store {
	t.Helper()
	store := newTestStore()
	t.Cleanup(func() { _ = store.Close() })
	return store
}

// walkDir is a direct-filesystem scanner retained only for tests
// that stage files in arbitrary temp directories outside the
// brain's logical structure. Production code paths go through
// [Index.walkPrefix] which reads via the configured [brain.Store].
func walkDir(dir, scope, projectSlug string) []discoveredFile {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil
	}
	var files []discoveredFile
	for _, entry := range entries {
		if entry.IsDir() {
			files = append(files, walkDir(filepath.Join(dir, entry.Name()), scope, projectSlug)...)
			continue
		}
		if !strings.HasSuffix(entry.Name(), ".md") {
			continue
		}
		absPath := filepath.Join(dir, entry.Name())
		data, err := os.ReadFile(absPath)
		if err != nil {
			continue
		}
		files = append(files, discoveredFile{
			path:        brain.Path(absPath),
			scope:       scope,
			projectSlug: projectSlug,
			content:     data,
			checksum:    checksumBytes(data),
		})
	}
	return files
}

func TestNewIndex_CreatesTables(t *testing.T) {
	db, idx := newIndexEmpty(t)
	if idx == nil {
		t.Fatal("expected non-nil index")
	}

	if _, err := db.Exec("SELECT count(*) FROM knowledge_fts"); err != nil {
		t.Fatalf("knowledge_fts table not created: %v", err)
	}

	if _, err := db.Exec("SELECT count(*) FROM knowledge_index_state"); err != nil {
		t.Fatalf("knowledge_index_state table not created: %v", err)
	}
}

func TestNewIndex_DropsLegacySchemaWithoutVersion(t *testing.T) {
	db := openTestDB(t)
	if _, err := db.Exec(`CREATE VIRTUAL TABLE knowledge_fts USING fts5(
		path,
		title,
		summary,
		tags,
		content,
		scope,
		project_slug,
		tokenize='porter unicode61'
	)`); err != nil {
		t.Fatalf("create legacy fts: %v", err)
	}
	if _, err := db.Exec(`CREATE TABLE knowledge_index_meta (
		key TEXT PRIMARY KEY,
		value TEXT NOT NULL
	)`); err != nil {
		t.Fatalf("create legacy meta: %v", err)
	}
	if _, err := db.Exec(`CREATE TABLE knowledge_index_state (
		path TEXT PRIMARY KEY,
		checksum TEXT NOT NULL
	)`); err != nil {
		t.Fatalf("create legacy state: %v", err)
	}

	idx, err := NewIndex(db, memstoreEmpty(t))
	if err != nil {
		t.Fatalf("NewIndex: %v", err)
	}
	if idx == nil {
		t.Fatal("expected non-nil index")
	}

	if _, err := db.Exec(`INSERT INTO knowledge_fts
		(path, title, summary, tags, content, scope, project_slug, session_date)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
		"wiki/current.md", "Current", "", "", "body", "wiki", "", "2024-03-15",
	); err != nil {
		t.Fatalf("insert with current schema: %v", err)
	}
}

// writeTestFile creates a markdown file in the given directory.
func writeTestFile(t *testing.T, dir, name, content string) string {
	t.Helper()
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatalf("creating dir %s: %v", dir, err)
	}
	path := filepath.Join(dir, name)
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatalf("writing %s: %v", path, err)
	}
	return path
}

func TestRebuild_IndexesFiles(t *testing.T) {
	_, idx := newIndexEmpty(t)

	tmpDir := t.TempDir()
	memDir := filepath.Join(tmpDir, "memory")
	writeTestFile(t, memDir, "golang.md", `---
name: Go Patterns
description: Common Go patterns
tags: go, patterns
---
This document covers common Go patterns including error handling and concurrency.
`)
	writeTestFile(t, memDir, "docker.md", `---
name: Docker Tips
description: Container management tips
tags: docker
---
Docker tips for building efficient container images and managing deployments.
`)

	files := walkDir(memDir, "global_memory", "")
	if len(files) != 2 {
		t.Fatalf("expected 2 files, got %d", len(files))
	}

	if err := idx.indexFiles(context.Background(), files); err != nil {
		t.Fatalf("indexFiles: %v", err)
	}

	results, err := idx.Search("Go patterns", SearchOpts{})
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(results) == 0 {
		t.Fatal("expected at least one result for 'Go patterns'")
	}
	if results[0].Title != "Go Patterns" {
		t.Errorf("expected title 'Go Patterns', got %q", results[0].Title)
	}
}

func TestSearch_FindsByTitle(t *testing.T) {
	db, idx := newIndexEmpty(t)

	if _, err := db.Exec(
		`INSERT INTO knowledge_fts (path, title, summary, tags, content, scope, project_slug)
		 VALUES (?, ?, ?, ?, ?, ?, ?)`,
		"/test/kubernetes.md", "Kubernetes Guide", "How to deploy apps",
		"k8s containers", "Full guide to deploying applications on Kubernetes.",
		"wiki", "",
	); err != nil {
		t.Fatalf("inserting test doc: %v", err)
	}

	results, err := idx.Search("Kubernetes", SearchOpts{})
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(results) == 0 {
		t.Fatal("expected results for 'Kubernetes'")
	}
	if results[0].Title != "Kubernetes Guide" {
		t.Errorf("expected title 'Kubernetes Guide', got %q", results[0].Title)
	}
}

func TestSearch_FindsByContent(t *testing.T) {
	db, idx := newIndexEmpty(t)

	if _, err := db.Exec(
		`INSERT INTO knowledge_fts (path, title, summary, tags, content, scope, project_slug)
		 VALUES (?, ?, ?, ?, ?, ?, ?)`,
		"/test/networking.md", "Networking Basics", "Introduction to networking",
		"networking", "Understanding TCP/IP protocols and how packets traverse the internet.",
		"global_memory", "",
	); err != nil {
		t.Fatalf("inserting test doc: %v", err)
	}

	results, err := idx.Search("TCP protocols", SearchOpts{})
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(results) == 0 {
		t.Fatal("expected results for 'TCP protocols'")
	}
	if results[0].Path != "/test/networking.md" {
		t.Errorf("expected path '/test/networking.md', got %q", results[0].Path)
	}
}

func TestSearch_BM25Ranking(t *testing.T) {
	db, idx := newIndexEmpty(t)

	if _, err := db.Exec(
		`INSERT INTO knowledge_fts (path, title, summary, tags, content, scope, project_slug)
		 VALUES (?, ?, ?, ?, ?, ?, ?)`,
		"/test/rust-guide.md", "Rust Programming", "Comprehensive Rust guide",
		"rust programming", "Rust is a systems programming language. Rust provides memory safety. Rust is fast.",
		"wiki", "",
	); err != nil {
		t.Fatalf("inserting relevant doc: %v", err)
	}

	if _, err := db.Exec(
		`INSERT INTO knowledge_fts (path, title, summary, tags, content, scope, project_slug)
		 VALUES (?, ?, ?, ?, ?, ?, ?)`,
		"/test/languages.md", "Programming Languages", "Overview of languages",
		"languages", "Python, JavaScript, Rust, and Go are popular languages.",
		"wiki", "",
	); err != nil {
		t.Fatalf("inserting less relevant doc: %v", err)
	}

	results, err := idx.Search("Rust", SearchOpts{})
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(results) < 2 {
		t.Fatalf("expected at least 2 results, got %d", len(results))
	}

	if results[0].Path != "/test/rust-guide.md" {
		t.Errorf("expected rust-guide.md to rank first, got %q", results[0].Path)
	}
}

func TestSearch_RespectsMaxResults(t *testing.T) {
	db, idx := newIndexEmpty(t)

	for i := range 5 {
		if _, err := db.Exec(
			`INSERT INTO knowledge_fts (path, title, summary, tags, content, scope, project_slug)
			 VALUES (?, ?, ?, ?, ?, ?, ?)`,
			fmt.Sprintf("/test/doc-%d.md", i),
			fmt.Sprintf("Document %d", i),
			"A test document",
			"test",
			"This test document contains searchable content about testing.",
			"wiki", "",
		); err != nil {
			t.Fatalf("inserting doc %d: %v", i, err)
		}
	}

	results, err := idx.Search("test document", SearchOpts{MaxResults: 2})
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(results) != 2 {
		t.Errorf("expected 2 results, got %d", len(results))
	}
}

func TestSearch_ScopeFilter(t *testing.T) {
	db, idx := newIndexEmpty(t)

	if _, err := db.Exec(
		`INSERT INTO knowledge_fts (path, title, summary, tags, content, scope, project_slug)
		 VALUES (?, ?, ?, ?, ?, ?, ?)`,
		"/wiki/terraform.md", "Terraform Guide", "IaC with Terraform",
		"terraform", "Infrastructure as code using Terraform.",
		"wiki", "",
	); err != nil {
		t.Fatalf("inserting wiki doc: %v", err)
	}

	if _, err := db.Exec(
		`INSERT INTO knowledge_fts (path, title, summary, tags, content, scope, project_slug)
		 VALUES (?, ?, ?, ?, ?, ?, ?)`,
		"/memory/terraform-notes.md", "Terraform Notes", "Personal notes on Terraform",
		"terraform", "Notes about using Terraform in production.",
		"global_memory", "",
	); err != nil {
		t.Fatalf("inserting memory doc: %v", err)
	}

	results, err := idx.Search("Terraform", SearchOpts{Scope: "wiki"})
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(results) != 1 {
		t.Fatalf("expected 1 wiki result, got %d", len(results))
	}
	if results[0].Scope != "wiki" {
		t.Errorf("expected scope 'wiki', got %q", results[0].Scope)
	}

	results, err = idx.Search("Terraform", SearchOpts{Scope: "global_memory"})
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(results) != 1 {
		t.Fatalf("expected 1 global_memory result, got %d", len(results))
	}
	if results[0].Scope != "global_memory" {
		t.Errorf("expected scope 'global_memory', got %q", results[0].Scope)
	}
}

func TestUpdate_IndexesNewFiles(t *testing.T) {
	_, idx := newIndexEmpty(t)

	tmpDir := t.TempDir()
	memDir := filepath.Join(tmpDir, "memory")
	writeTestFile(t, memDir, "new-topic.md", `---
name: New Topic
description: A brand new topic
tags: new
---
Content about the brand new topic we just discovered.
`)

	files := walkDir(memDir, "global_memory", "")
	if err := idx.indexFiles(context.Background(), files); err != nil {
		t.Fatalf("indexFiles: %v", err)
	}

	results, err := idx.Search("brand new topic", SearchOpts{})
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(results) == 0 {
		t.Fatal("expected results after indexing new file")
	}
}

func TestLookupRows_ReturnsIndexedFrontmatterMetadata(t *testing.T) {
	_, idx := newIndexEmpty(t)

	content := []byte(`---
name: Metadata Fact
session_id: session-123
observed_on: 2024-03-14
modified: 2024-03-15T09:30:00Z
source_role: user
event_date: 2024-03-13
evidence_kind: atomic
evidence_group: atomic|money|purchase|usd-1200|2024-03-13
state_key: state.owned_item_set.bikes
state_kind: owned_item_set
state_subject: bikes
state_value: commuter bike and road bike
claim_key: claim.user.bikes
claim_status: asserted
valid_from: 2024-03-13
valid_to: 2024-04-01
artefact_type: markdown_table
artefact_ordinal: 2
artefact_section: Inventory
artefact_descriptor: Bike table
---
The user completed the migration checklist.
`)
	path := brain.Path("memory/global/metadata-fact.md")
	if err := idx.indexFiles(context.Background(), []discoveredFile{
		{
			path:     path,
			scope:    "global_memory",
			content:  content,
			checksum: checksumBytes(content),
		},
	}); err != nil {
		t.Fatalf("indexFiles: %v", err)
	}

	rows, err := idx.LookupRows(context.Background(), []string{string(path)})
	if err != nil {
		t.Fatalf("LookupRows: %v", err)
	}
	if len(rows) != 1 {
		t.Fatalf("rows = %d, want 1", len(rows))
	}
	row := rows[0]
	if row.SessionDate != "2024-03-14" {
		t.Fatalf("SessionDate = %q, want observed date", row.SessionDate)
	}
	if row.SessionID != "session-123" ||
		row.ObservedOn != "2024-03-14" ||
		row.Modified != "2024-03-15T09:30:00Z" ||
		row.SourceRole != "user" ||
		row.EventDate != "2024-03-13" ||
		row.EvidenceKind != "atomic" ||
		row.StateKey != "state.owned_item_set.bikes" ||
		row.ClaimKey != "claim.user.bikes" ||
		row.ValidFrom != "2024-03-13" ||
		row.ArtefactType != "markdown_table" ||
		row.ArtefactOrdinal != "2" {
		t.Fatalf("metadata not preserved: %+v", row)
	}

	metadataRows, err := idx.AllMetadataRows(context.Background())
	if err != nil {
		t.Fatalf("AllMetadataRows: %v", err)
	}
	if len(metadataRows) != 1 {
		t.Fatalf("metadata rows = %d, want 1", len(metadataRows))
	}
	meta := metadataRows[0]
	if meta.Content != "" || meta.Title != "" || meta.Summary != "" {
		t.Fatalf("metadata-only row loaded body fields: %+v", meta)
	}
	if meta.Path != string(path) || meta.SessionID != "session-123" || meta.SourceRole != "user" || meta.StateKey != "state.owned_item_set.bikes" {
		t.Fatalf("metadata-only row lost filter fields: %+v", meta)
	}

	sessionRows, err := idx.SessionRows(context.Background(), []string{"session-123"})
	if err != nil {
		t.Fatalf("SessionRows: %v", err)
	}
	if len(sessionRows) != 1 {
		t.Fatalf("session rows = %d, want 1", len(sessionRows))
	}
	if sessionRows[0].Path != string(path) || sessionRows[0].SessionID != "session-123" {
		t.Fatalf("SessionRows lost row identity: %+v", sessionRows[0])
	}

	results, err := idx.Search("Wednesday", SearchOpts{})
	if err != nil {
		t.Fatalf("Search event date: %v", err)
	}
	if len(results) == 0 || results[0].Path != string(path) {
		t.Fatalf("event_date was not indexed as a searchable date tag: %+v", results)
	}
}

func TestUpdate_DetectsChanges(t *testing.T) {
	_, idx := newIndexEmpty(t)

	tmpDir := t.TempDir()
	memDir := filepath.Join(tmpDir, "memory")
	filePath := writeTestFile(t, memDir, "changing.md", `---
name: Original Title
description: Original content
---
The original content of this file.
`)

	files := walkDir(memDir, "global_memory", "")
	if err := idx.indexFiles(context.Background(), files); err != nil {
		t.Fatalf("initial indexFiles: %v", err)
	}

	results, err := idx.Search("Original Title", SearchOpts{})
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(results) == 0 {
		t.Fatal("expected results for original content")
	}

	if err := os.WriteFile(filePath, []byte(`---
name: Updated Title
description: Updated content
---
The completely revised content of this file about quantum computing.
`), 0o644); err != nil {
		t.Fatalf("updating file: %v", err)
	}

	files = walkDir(memDir, "global_memory", "")
	if err := idx.indexFiles(context.Background(), files); err != nil {
		t.Fatalf("re-indexFiles: %v", err)
	}

	results, err = idx.Search("quantum computing", SearchOpts{})
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(results) == 0 {
		t.Fatal("expected results for updated content")
	}
	if results[0].Title != "Updated Title" {
		t.Errorf("expected title 'Updated Title', got %q", results[0].Title)
	}
}

func TestUpdate_RemovesDeleted(t *testing.T) {
	_, idx := newIndexEmpty(t)

	tmpDir := t.TempDir()
	memDir := filepath.Join(tmpDir, "memory")
	filePath := writeTestFile(t, memDir, "ephemeral.md", `---
name: Ephemeral Note
description: This will be deleted
---
Temporary content that should be removed from the index.
`)

	files := walkDir(memDir, "global_memory", "")
	if err := idx.indexFiles(context.Background(), files); err != nil {
		t.Fatalf("indexFiles: %v", err)
	}

	results, err := idx.Search("Ephemeral", SearchOpts{})
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(results) == 0 {
		t.Fatal("expected results before deletion")
	}

	if err := os.Remove(filePath); err != nil {
		t.Fatalf("removing file: %v", err)
	}

	if err := idx.Remove(filePath); err != nil {
		t.Fatalf("Remove: %v", err)
	}

	results, err = idx.Search("Ephemeral", SearchOpts{})
	if err != nil {
		t.Fatalf("Search after removal: %v", err)
	}
	if len(results) != 0 {
		t.Errorf("expected no results after removal, got %d", len(results))
	}
}

func TestRemove_DeletesEntry(t *testing.T) {
	db, idx := newIndexEmpty(t)

	if _, err := db.Exec(
		`INSERT INTO knowledge_fts (path, title, summary, tags, content, scope, project_slug)
		 VALUES (?, ?, ?, ?, ?, ?, ?)`,
		"/test/to-remove.md", "Remove Me", "Should be removed",
		"remove", "Content that will be explicitly removed.",
		"wiki", "",
	); err != nil {
		t.Fatalf("inserting doc: %v", err)
	}
	if _, err := db.Exec(
		`INSERT INTO knowledge_index_state (path, checksum) VALUES (?, ?)`,
		"/test/to-remove.md", "abc123",
	); err != nil {
		t.Fatalf("inserting state: %v", err)
	}

	results, err := idx.Search("Remove Me", SearchOpts{})
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(results) == 0 {
		t.Fatal("expected results before removal")
	}

	if err := idx.Remove("/test/to-remove.md"); err != nil {
		t.Fatalf("Remove: %v", err)
	}

	results, err = idx.Search("Remove Me", SearchOpts{})
	if err != nil {
		t.Fatalf("Search after removal: %v", err)
	}
	if len(results) != 0 {
		t.Errorf("expected no FTS results, got %d", len(results))
	}

	var count int
	if err := db.QueryRow("SELECT count(*) FROM knowledge_index_state WHERE path = ?", "/test/to-remove.md").Scan(&count); err != nil {
		t.Fatalf("querying state: %v", err)
	}
	if count != 0 {
		t.Errorf("expected state entry to be removed, count=%d", count)
	}
}

func TestSearch_EmptyQuery(t *testing.T) {
	_, idx := newIndexEmpty(t)

	results, err := idx.Search("", SearchOpts{})
	if err != nil {
		t.Fatalf("Search with empty query: %v", err)
	}
	if results != nil {
		t.Errorf("expected nil results for empty query, got %d", len(results))
	}
}

func TestSearch_SpecialCharacters(t *testing.T) {
	db, idx := newIndexEmpty(t)

	if _, err := db.Exec(
		`INSERT INTO knowledge_fts (path, title, summary, tags, content, scope, project_slug)
		 VALUES (?, ?, ?, ?, ?, ?, ?)`,
		"/test/special.md", "Special Chars", "Testing special characters",
		"test", "Content with special terms like C++ and node.js frameworks.",
		"wiki", "",
	); err != nil {
		t.Fatalf("inserting doc: %v", err)
	}

	results, err := idx.Search("C++ node*", SearchOpts{})
	if err != nil {
		t.Fatalf("Search with special chars should not error: %v", err)
	}
	_ = results
}

// TestSearch_RanksTitleHitsFirst inserts three documents that each
// mention the same term in a different column and asserts the title
// hit ranks ahead of the summary hit, which in turn ranks ahead of
// the content-only hit.
func TestSearch_RanksTitleHitsFirst(t *testing.T) {
	db, idx := newIndexEmpty(t)

	insert := func(path, title, summary, content string) {
		t.Helper()
		if _, err := db.Exec(
			`INSERT INTO knowledge_fts (path, title, summary, tags, content, scope, project_slug)
			 VALUES (?, ?, ?, ?, ?, ?, ?)`,
			path, title, summary, "", content, "wiki", "",
		); err != nil {
			t.Fatalf("insert %s: %v", path, err)
		}
	}

	insert("/w/content-only.md", "Unrelated Heading", "Unrelated summary text", "The term magicword appears in the body only.")
	insert("/w/summary-only.md", "Another Heading", "Notes about magicword in the summary", "Body text about other things entirely.")
	insert("/w/title-only.md", "Magicword Headline", "Totally different summary", "Body text about other things entirely.")

	results, err := idx.Search("magicword", SearchOpts{Scope: "wiki"})
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(results) < 3 {
		t.Fatalf("expected 3 results, got %d", len(results))
	}

	if results[0].Path != "/w/title-only.md" {
		t.Errorf("expected title-only.md to rank first, got %q", results[0].Path)
	}
	if results[1].Path != "/w/summary-only.md" {
		t.Errorf("expected summary-only.md to rank second, got %q", results[1].Path)
	}
	if results[2].Path != "/w/content-only.md" {
		t.Errorf("expected content-only.md to rank third, got %q", results[2].Path)
	}
}

func TestSearch_ProjectSlugFilter(t *testing.T) {
	db, idx := newIndexEmpty(t)

	if _, err := db.Exec(
		`INSERT INTO knowledge_fts (path, title, summary, tags, content, scope, project_slug)
		 VALUES (?, ?, ?, ?, ?, ?, ?)`,
		"/projects/jeff/memory/api.md", "Jeff API Notes", "API design notes",
		"api", "Notes about the Jeff API design patterns.",
		"project_memory", "jeff",
	); err != nil {
		t.Fatalf("inserting jeff doc: %v", err)
	}

	if _, err := db.Exec(
		`INSERT INTO knowledge_fts (path, title, summary, tags, content, scope, project_slug)
		 VALUES (?, ?, ?, ?, ?, ?, ?)`,
		"/projects/lleverage/memory/api.md", "Lleverage API Notes", "API design notes",
		"api", "Notes about the Lleverage API design patterns.",
		"project_memory", "lleverage",
	); err != nil {
		t.Fatalf("inserting lleverage doc: %v", err)
	}

	results, err := idx.Search("API", SearchOpts{Scope: "project_memory", ProjectSlug: "jeff"})
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(results) != 1 {
		t.Fatalf("expected 1 result for jeff project, got %d", len(results))
	}
	if results[0].Title != "Jeff API Notes" {
		t.Errorf("expected 'Jeff API Notes', got %q", results[0].Title)
	}
}

func TestSearch_MemoryScopeProjectFilterIncludesGlobalAndMatchingProject(t *testing.T) {
	db, idx := newIndexEmpty(t)

	insertSearchTestRow(t, db,
		"memory/global/api.md", "Global API", "Shared API memory",
		"global_memory", "",
	)
	insertSearchTestRow(t, db,
		"memory/project/jeff/api.md", "Jeff API", "Project API memory",
		"project_memory", "jeff",
	)
	insertSearchTestRow(t, db,
		"memory/project/other/api.md", "Other API", "Other project API memory",
		"project_memory", "other",
	)
	insertSearchTestRow(t, db,
		"raw/lme/api.md", "Raw API", "Raw API transcript",
		"raw_lme", "",
	)

	results, err := idx.Search("API memory transcript", SearchOpts{
		Scope:       "memory",
		ProjectSlug: "jeff",
		MaxResults:  10,
	})
	if err != nil {
		t.Fatalf("Search memory scope: %v", err)
	}
	got := searchResultPaths(results)
	want := []string{"memory/global/api.md", "memory/project/jeff/api.md"}
	if !sameSet(got, want) {
		t.Fatalf("memory scoped project hits = %v, want %v", got, want)
	}
}

func TestSearch_RawScopeIncludesRawDocumentsAndRawLME(t *testing.T) {
	db, idx := newIndexEmpty(t)

	insertSearchTestRow(t, db,
		"raw/documents/api.md", "Raw Document API", "Shared raw api document",
		"raw_document", "",
	)
	insertSearchTestRow(t, db,
		"raw/lme/session.md", "Raw LME API", "Shared raw api transcript",
		"raw_lme", "",
	)
	insertSearchTestRow(t, db,
		"wiki/api.md", "Wiki API", "Shared raw api wiki",
		"wiki", "",
	)

	results, err := idx.Search("raw api", SearchOpts{
		Scope:      "raw",
		MaxResults: 10,
	})
	if err != nil {
		t.Fatalf("Search raw scope: %v", err)
	}
	got := searchResultPaths(results)
	want := []string{"raw/documents/api.md", "raw/lme/session.md"}
	if !sameSet(got, want) {
		t.Fatalf("raw scoped hits = %v, want %v", got, want)
	}
}

// TestRebuildWithStats verifies that the stats returned by
// RebuildWithStats match the size and shape of the brain that was
// walked.
func TestRebuildWithStats(t *testing.T) {
	store := newWikiTestStore(t,
		wikiTestDoc{path: "wiki/platform/go.md", title: "Go", body: "Go is a fast systems language."},
		wikiTestDoc{path: "wiki/platform/rust.md", title: "Rust", body: "Rust is memory safe."},
		wikiTestDoc{path: "memory/global/habits.md", name: "Habits", body: "Write things down."},
	)

	db := openTestDB(t)
	idx, err := NewIndex(db, store)
	if err != nil {
		t.Fatalf("NewIndex: %v", err)
	}

	stats, err := idx.RebuildWithStats(context.Background())
	if err != nil {
		t.Fatalf("RebuildWithStats: %v", err)
	}
	if stats.FilesScanned != 3 {
		t.Errorf("FilesScanned = %d, want 3", stats.FilesScanned)
	}
	if stats.FilesIndexed != 3 {
		t.Errorf("FilesIndexed = %d, want 3", stats.FilesIndexed)
	}
	if stats.RowsAfter != 3 {
		t.Errorf("RowsAfter = %d, want 3", stats.RowsAfter)
	}
	if stats.DurationMillis < 0 {
		t.Errorf("DurationMillis must be non-negative, got %d", stats.DurationMillis)
	}
}

// TestUpdateWithStats verifies that an incremental update reports
// the number of files actually written (not just scanned).
func TestUpdateWithStats(t *testing.T) {
	store := newWikiTestStore(t,
		wikiTestDoc{path: "wiki/a.md", title: "A", body: "alpha"},
		wikiTestDoc{path: "wiki/b.md", title: "B", body: "beta"},
	)

	db := openTestDB(t)
	idx, err := NewIndex(db, store)
	if err != nil {
		t.Fatalf("NewIndex: %v", err)
	}

	first, err := idx.UpdateWithStats(context.Background())
	if err != nil {
		t.Fatalf("UpdateWithStats first: %v", err)
	}
	if first.FilesIndexed != 2 {
		t.Errorf("first FilesIndexed = %d, want 2", first.FilesIndexed)
	}

	second, err := idx.UpdateWithStats(context.Background())
	if err != nil {
		t.Fatalf("UpdateWithStats second: %v", err)
	}
	if second.FilesIndexed != 0 {
		t.Errorf("second FilesIndexed = %d, want 0", second.FilesIndexed)
	}
	if second.FilesScanned != 2 {
		t.Errorf("second FilesScanned = %d, want 2", second.FilesScanned)
	}
}

// TestRebuildIfStale verifies that the stale-detection heuristic
// triggers a rebuild when the wiki rows in the index are under half
// the on-disk file count.
func TestRebuildIfStale(t *testing.T) {
	store := newWikiTestStore(t,
		wikiTestDoc{path: "wiki/a.md", title: "A", body: "alpha"},
		wikiTestDoc{path: "wiki/b.md", title: "B", body: "beta"},
		wikiTestDoc{path: "wiki/c.md", title: "C", body: "gamma"},
		wikiTestDoc{path: "wiki/d.md", title: "D", body: "delta"},
	)

	db := openTestDB(t)
	idx, err := NewIndex(db, store)
	if err != nil {
		t.Fatalf("NewIndex: %v", err)
	}

	ctx := context.Background()

	if err := idx.RebuildIfStale(ctx, store, 10); err != nil {
		t.Fatalf("RebuildIfStale empty: %v", err)
	}
	wiki, err := idx.wikiRowCount(ctx)
	if err != nil {
		t.Fatalf("wikiRowCount: %v", err)
	}
	if wiki != 4 {
		t.Errorf("after stale rebuild wiki rows = %d, want 4", wiki)
	}

	if _, err := db.Exec(
		`INSERT INTO knowledge_fts (path, title, summary, tags, content, scope, project_slug)
		 VALUES (?, ?, ?, ?, ?, ?, ?)`,
		"sentinel", "sentinel", "", "", "", "wiki", "",
	); err != nil {
		t.Fatalf("insert sentinel: %v", err)
	}
	if err := idx.RebuildIfStale(ctx, store, 4); err != nil {
		t.Fatalf("RebuildIfStale healthy: %v", err)
	}
	rows, err := idx.RowCount(ctx)
	if err != nil {
		t.Fatalf("RowCount: %v", err)
	}
	if rows != 5 {
		t.Errorf("healthy RowCount = %d, want 5 (no rebuild expected)", rows)
	}
}

// TestLastRefresh_PersistsAcrossProcesses verifies the
// knowledge_index_meta timestamp round-trips through
// PersistedRefresh.
func TestLastRefresh_PersistsAcrossProcesses(t *testing.T) {
	store := newWikiTestStore(t,
		wikiTestDoc{path: "wiki/a.md", title: "A", body: "alpha"},
	)

	db := openTestDB(t)
	idx, err := NewIndex(db, store)
	if err != nil {
		t.Fatalf("NewIndex: %v", err)
	}

	ctx := context.Background()

	before, err := idx.PersistedRefresh(ctx)
	if err != nil {
		t.Fatalf("PersistedRefresh before: %v", err)
	}
	if !before.IsZero() {
		t.Errorf("before rebuild PersistedRefresh = %v, want zero", before)
	}

	if err := idx.Rebuild(ctx); err != nil {
		t.Fatalf("Rebuild: %v", err)
	}

	after, err := idx.PersistedRefresh(ctx)
	if err != nil {
		t.Fatalf("PersistedRefresh after: %v", err)
	}
	if after.IsZero() {
		t.Error("after rebuild PersistedRefresh is zero")
	}
	delta := idx.LastRefresh().Sub(after)
	if delta < -5*time.Second || delta > 5*time.Second {
		t.Errorf("PersistedRefresh drift = %v, want <5s", delta)
	}
}

// TestRowCountByScope verifies the scope breakdown used by
// diagnostics.
func TestRowCountByScope(t *testing.T) {
	db := openTestDB(t)
	idx, err := NewIndex(db, memstoreEmpty(t))
	if err != nil {
		t.Fatalf("NewIndex: %v", err)
	}

	insert := func(path, scope string) {
		t.Helper()
		if _, err := db.Exec(
			`INSERT INTO knowledge_fts (path, title, summary, tags, content, scope, project_slug)
			 VALUES (?, ?, ?, ?, ?, ?, ?)`,
			path, "", "", "", "", scope, "",
		); err != nil {
			t.Fatalf("insert %s: %v", path, err)
		}
	}
	insert("wiki/a.md", "wiki")
	insert("wiki/b.md", "wiki")
	insert("memory/global/c.md", "global_memory")

	got, err := idx.RowCountByScope(context.Background())
	if err != nil {
		t.Fatalf("RowCountByScope: %v", err)
	}
	if got["wiki"] != 2 {
		t.Errorf("wiki = %d, want 2", got["wiki"])
	}
	if got["global_memory"] != 1 {
		t.Errorf("global_memory = %d, want 1", got["global_memory"])
	}
}

// TestRebuild_IndexesRawDocuments verifies that markdown files
// persisted under raw/documents/ (where [knowledge.Base.Ingest]
// writes) are discovered, indexed, and returned by BM25 search.
func TestRebuild_IndexesRawDocuments(t *testing.T) {
	store := newWikiTestStore(t,
		wikiTestDoc{path: "raw/documents/hedgehogs.md", title: "Hedgehogs", body: "Hedgehogs live in hedgerows and gardens across Europe."},
		wikiTestDoc{path: "wiki/platform/go.md", title: "Go", body: "Go is a fast systems language."},
	)

	db := openTestDB(t)
	idx, err := NewIndex(db, store)
	if err != nil {
		t.Fatalf("NewIndex: %v", err)
	}

	ctx := context.Background()
	stats, err := idx.RebuildWithStats(ctx)
	if err != nil {
		t.Fatalf("RebuildWithStats: %v", err)
	}
	if stats.FilesScanned != 2 {
		t.Errorf("FilesScanned = %d, want 2", stats.FilesScanned)
	}
	if stats.FilesIndexed != 2 {
		t.Errorf("FilesIndexed = %d, want 2", stats.FilesIndexed)
	}

	results, err := idx.Search("hedgehogs", SearchOpts{})
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(results) == 0 {
		t.Fatal("expected a raw/documents hit for 'hedgehogs'")
	}
	var found bool
	for _, r := range results {
		if r.Path == "raw/documents/hedgehogs.md" {
			found = true
			if r.Scope != "raw_document" {
				t.Errorf("scope = %q, want %q", r.Scope, "raw_document")
			}
			break
		}
	}
	if !found {
		t.Fatalf("raw/documents/hedgehogs.md missing from results: %+v", results)
	}

	// Scope filter should isolate raw documents.
	scoped, err := idx.Search("hedgehogs", SearchOpts{Scope: "raw_document"})
	if err != nil {
		t.Fatalf("Search scope filter: %v", err)
	}
	if len(scoped) != 1 {
		t.Fatalf("scoped results = %d, want 1", len(scoped))
	}
	if scoped[0].Path != "raw/documents/hedgehogs.md" {
		t.Errorf("scoped path = %q, want %q", scoped[0].Path, "raw/documents/hedgehogs.md")
	}
}

// TestSubscribe_IndexesRawDocument verifies the event sink path
// covers the raw/documents/ tree so writes via the brain store land
// in the FTS index without an explicit Rebuild.
func TestSubscribe_IndexesRawDocument(t *testing.T) {
	store := newTestStore()
	t.Cleanup(func() { _ = store.Close() })

	db := openTestDB(t)
	idx, err := NewIndex(db, store)
	if err != nil {
		t.Fatalf("NewIndex: %v", err)
	}
	unsub := idx.Subscribe(store)
	t.Cleanup(unsub)

	ctx := context.Background()
	if err := store.Write(ctx, "raw/documents/hedgehogs.md", []byte(`---
title: Hedgehogs
summary: Hedgehog primer
---
Hedgehogs live in hedgerows and gardens.`)); err != nil {
		t.Fatalf("write: %v", err)
	}

	results, err := idx.Search("Hedgehogs", SearchOpts{})
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(results) == 0 {
		t.Fatal("expected a hit after a raw/documents write")
	}
	if results[0].Path != "raw/documents/hedgehogs.md" {
		t.Errorf("path = %q, want %q", results[0].Path, "raw/documents/hedgehogs.md")
	}
	if results[0].Scope != "raw_document" {
		t.Errorf("scope = %q, want %q", results[0].Scope, "raw_document")
	}
}

// TestDiscoverFiles_ExcludesUnderscorePrefixed verifies that
// markdown files whose basename starts with an underscore are
// filtered out of the FTS discovery path.
func TestDiscoverFiles_ExcludesUnderscorePrefixed(t *testing.T) {
	store := newWikiTestStore(t,
		wikiTestDoc{path: "wiki/companies/bosch.md", title: "Bosch", body: "Bosch is a manufacturer."},
		wikiTestDoc{path: "wiki/companies/_index.md", title: "Companies", body: "Index of companies."},
		wikiTestDoc{path: "wiki/_log.md", title: "Wiki Log", body: "Change log entries."},
		wikiTestDoc{path: "wiki/companies/_health.md", title: "Companies Health", body: "Health report."},
		wikiTestDoc{path: "wiki/companies/_concepts.md", title: "Companies Concepts", body: "Concepts summary."},
	)

	db := openTestDB(t)
	idx, err := NewIndex(db, store)
	if err != nil {
		t.Fatalf("NewIndex: %v", err)
	}

	ctx := context.Background()
	stats, err := idx.RebuildWithStats(ctx)
	if err != nil {
		t.Fatalf("RebuildWithStats: %v", err)
	}

	if stats.FilesScanned != 1 {
		t.Errorf("FilesScanned = %d, want 1 (only bosch.md)", stats.FilesScanned)
	}
	if stats.FilesIndexed != 1 {
		t.Errorf("FilesIndexed = %d, want 1 (only bosch.md)", stats.FilesIndexed)
	}

	rows, err := db.QueryContext(ctx, "SELECT path FROM knowledge_fts ORDER BY path")
	if err != nil {
		t.Fatalf("querying knowledge_fts: %v", err)
	}
	defer rows.Close()
	var indexed []string
	for rows.Next() {
		var p string
		if err := rows.Scan(&p); err != nil {
			t.Fatalf("scan: %v", err)
		}
		indexed = append(indexed, p)
	}
	if err := rows.Err(); err != nil {
		t.Fatalf("rows.Err: %v", err)
	}

	want := []string{"wiki/companies/bosch.md"}
	if len(indexed) != len(want) || indexed[0] != want[0] {
		t.Errorf("indexed paths = %v, want %v", indexed, want)
	}

	for _, banned := range []string{
		"wiki/_log.md",
		"wiki/companies/_index.md",
		"wiki/companies/_health.md",
		"wiki/companies/_concepts.md",
	} {
		var count int
		if err := db.QueryRowContext(ctx, "SELECT COUNT(*) FROM knowledge_fts WHERE path = ?", banned).Scan(&count); err != nil {
			t.Fatalf("count %s: %v", banned, err)
		}
		if count != 0 {
			t.Errorf("%s should not be indexed, got count = %d", banned, count)
		}
	}
}
