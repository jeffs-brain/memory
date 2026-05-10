// SPDX-License-Identifier: Apache-2.0
package ingest

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/jeffs-brain/memory/go/brain"
)

// migrationStatePath is the file-based state location for tracking
// BLAKE3 migration progress. Compatible with FsStore backends.
const migrationStatePath = "raw/.pipeline-state/.blake3-migration.json"

// defaultBatchSize is used when MigrateOpts.BatchSize is zero.
const defaultBatchSize = 100

// MigrateOpts controls a single migration pass.
type MigrateOpts struct {
	// BatchSize is the maximum number of documents to process per call.
	// Zero defaults to 100.
	BatchSize int

	// DryRun reports what would be migrated without writing changes.
	DryRun bool

	// Cursor allows resuming from a previous incomplete migration.
	// Empty string starts from the beginning.
	Cursor string
}

// MigrateResult reports progress from a single migration pass.
type MigrateResult struct {
	Migrated   int
	Skipped    int
	Total      int
	NextCursor string
}

// migrationState is persisted at migrationStatePath to track progress
// across restarts.
type migrationState struct {
	Cursor   string `json:"cursor"`
	Migrated int    `json:"migrated"`
	Total    int    `json:"total"`
}

// HashMigrator handles the transition from SHA-256 to BLAKE3 document
// hashes. It iterates documents stored under raw/documents/, computes
// BLAKE3 hashes, and renames files from their SHA-256-based path to a
// BLAKE3-based path. The migration is non-blocking and resumable.
type HashMigrator struct {
	store brain.Store
}

// NewHashMigrator creates a migrator bound to the given store.
func NewHashMigrator(store brain.Store) *HashMigrator {
	return &HashMigrator{store: store}
}

// Migrate processes up to BatchSize documents, re-hashing from SHA-256
// to BLAKE3. It reads migration state from the store and saves progress
// after each batch. The migration does NOT lock the store.
func (m *HashMigrator) Migrate(ctx context.Context, opts MigrateOpts) (*MigrateResult, error) {
	batchSize := opts.BatchSize
	if batchSize <= 0 {
		batchSize = defaultBatchSize
	}

	cursor := opts.Cursor
	if cursor == "" {
		state, err := m.loadState(ctx)
		if err == nil && state.Cursor != "" {
			cursor = state.Cursor
		}
	}

	entries, err := m.store.List(ctx, brain.RawDocumentsPrefix(), brain.ListOpts{
		Recursive:        true,
		IncludeGenerated: true,
	})
	if err != nil {
		return nil, fmt.Errorf("ingest: migrate list documents: %w", err)
	}

	docPaths := filterDocumentPaths(entries)
	total := len(docPaths)

	startIdx := cursorIndex(docPaths, cursor)
	endIdx := startIdx + batchSize
	if endIdx > total {
		endIdx = total
	}

	migrated := 0
	skipped := 0

	for i := startIdx; i < endIdx; i++ {
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}

		docPath := docPaths[i]
		didMigrate, err := m.migrateDocument(ctx, docPath, opts.DryRun)
		if err != nil {
			return nil, fmt.Errorf("ingest: migrate %s: %w", docPath, err)
		}
		if didMigrate {
			migrated++
		} else {
			skipped++
		}
	}

	nextCursor := ""
	if endIdx < total {
		nextCursor = string(docPaths[endIdx])
	}

	if !opts.DryRun {
		saveErr := m.saveState(ctx, migrationState{
			Cursor:   nextCursor,
			Migrated: migrated,
			Total:    total,
		})
		if saveErr != nil {
			return nil, fmt.Errorf("ingest: save migration state: %w", saveErr)
		}
	}

	return &MigrateResult{
		Migrated:   migrated,
		Skipped:    skipped,
		Total:      total,
		NextCursor: nextCursor,
	}, nil
}

// migrateDocument checks whether a document at the given path needs
// BLAKE3 re-hashing. If the filename is already a BLAKE3 slug, it
// skips. Otherwise it reads the content, computes the BLAKE3 hash,
// writes the file at the new path, and deletes the old path.
func (m *HashMigrator) migrateDocument(ctx context.Context, docPath brain.Path, dryRun bool) (bool, error) {
	slug := extractSlug(docPath)
	if slug == "" {
		return false, nil
	}

	content, err := m.store.Read(ctx, docPath)
	if err != nil {
		return false, err
	}

	blake3Slug := HashSlug(content)

	// If the slug already matches the BLAKE3 hash, no migration needed.
	if slug == blake3Slug {
		return false, nil
	}

	// Verify this was hashed with SHA-256 by checking the slug matches.
	sha256Slug := HashSlugSHA256(content)
	if slug != sha256Slug {
		// Neither SHA-256 nor BLAKE3 — could be a title-derived slug.
		// Re-hash with BLAKE3 based on the raw content for the document ID.
		if dryRun {
			return true, nil
		}
		newPath := brain.RawDocument(blake3Slug)
		return m.renameDocument(ctx, docPath, newPath, content)
	}

	if dryRun {
		return true, nil
	}

	newPath := brain.RawDocument(blake3Slug)
	return m.renameDocument(ctx, docPath, newPath, content)
}

// renameDocument writes content to newPath and deletes oldPath within
// a batch. Returns true if the migration was performed.
func (m *HashMigrator) renameDocument(ctx context.Context, oldPath, newPath brain.Path, content []byte) (bool, error) {
	if oldPath == newPath {
		return false, nil
	}

	err := m.store.Batch(ctx, brain.BatchOptions{
		Reason:  "blake3-migration",
		Message: "migrate document hash from SHA-256 to BLAKE3",
	}, func(b brain.Batch) error {
		if writeErr := b.Write(ctx, newPath, content); writeErr != nil {
			return writeErr
		}
		return b.Delete(ctx, oldPath)
	})
	if err != nil {
		return false, err
	}
	return true, nil
}

// loadState reads the migration state file. Returns zero state on
// ErrNotFound.
func (m *HashMigrator) loadState(ctx context.Context) (migrationState, error) {
	data, err := m.store.Read(ctx, brain.Path(migrationStatePath))
	if err != nil {
		if errors.Is(err, brain.ErrNotFound) {
			return migrationState{}, nil
		}
		return migrationState{}, err
	}
	var state migrationState
	if jsonErr := json.Unmarshal(data, &state); jsonErr != nil {
		return migrationState{}, fmt.Errorf("ingest: parse migration state: %w", jsonErr)
	}
	return state, nil
}

// saveState writes the migration state file.
func (m *HashMigrator) saveState(ctx context.Context, state migrationState) error {
	data, err := json.Marshal(state)
	if err != nil {
		return err
	}
	return m.store.Write(ctx, brain.Path(migrationStatePath), data)
}

// ResolveHash is the dual-read helper. It computes the BLAKE3 hash of
// content and checks whether a document exists under that hash first.
// If not found, it falls back to the SHA-256 hash. Returns the hash
// that resolved, or the BLAKE3 hash if neither exists (for new
// documents).
func ResolveHash(ctx context.Context, store brain.Store, content []byte) (string, error) {
	blake3Hash := HashSlug(content)
	blake3Path := brain.RawDocument(blake3Hash)

	exists, err := store.Exists(ctx, blake3Path)
	if err != nil {
		return "", fmt.Errorf("ingest: resolve hash blake3 check: %w", err)
	}
	if exists {
		return blake3Hash, nil
	}

	sha256Hash := HashSlugSHA256(content)
	sha256Path := brain.RawDocument(sha256Hash)

	exists, err = store.Exists(ctx, sha256Path)
	if err != nil {
		return "", fmt.Errorf("ingest: resolve hash sha256 check: %w", err)
	}
	if exists {
		return sha256Hash, nil
	}

	// Neither found — return BLAKE3 for new documents.
	return blake3Hash, nil
}

// filterDocumentPaths extracts file paths from a list result, skipping
// directories and non-.md files.
func filterDocumentPaths(entries []brain.FileInfo) []brain.Path {
	paths := make([]brain.Path, 0, len(entries))
	for _, e := range entries {
		if e.IsDir {
			continue
		}
		if !strings.HasSuffix(string(e.Path), ".md") {
			continue
		}
		paths = append(paths, e.Path)
	}
	return paths
}

// cursorIndex returns the index of the first path >= cursor. If cursor
// is empty, returns 0. Paths are sorted lexicographically by the store
// contract.
func cursorIndex(paths []brain.Path, cursor string) int {
	if cursor == "" {
		return 0
	}
	for i, p := range paths {
		if string(p) >= cursor {
			return i
		}
	}
	return len(paths)
}

// extractSlug extracts the filename stem from a raw document path like
// "raw/documents/abc123def456.md" → "abc123def456".
func extractSlug(p brain.Path) string {
	s := string(p)
	prefix := "raw/documents/"
	if !strings.HasPrefix(s, prefix) {
		return ""
	}
	name := s[len(prefix):]
	if !strings.HasSuffix(name, ".md") {
		return ""
	}
	return strings.TrimSuffix(name, ".md")
}
