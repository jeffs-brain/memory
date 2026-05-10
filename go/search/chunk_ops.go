// SPDX-License-Identifier: Apache-2.0
package search

import (
	"context"
	"fmt"
	"strings"
)

// Chunk represents a single chunk of a document for granular indexing.
// The ID uniquely identifies this chunk (typically document_hash + ordinal).
// Chunks coexist alongside file-level rows in the FTS5 index.
type Chunk struct {
	ID       string
	Path     string
	Ordinal  int
	Content  string
	Title    string
	Tags     []string
	Metadata map[string]string
}

// UpsertChunks inserts or replaces individual chunks in the FTS5 index.
// Each chunk is keyed by its ID in the path column so it coexists with
// file-level rows (which use logical brain paths). Chunks use the scope
// "chunk" and carry the parent document path in the project_slug column
// for grouping queries.
func (idx *Index) UpsertChunks(ctx context.Context, chunks []Chunk) error {
	if len(chunks) == 0 {
		return nil
	}

	for i := 0; i < len(chunks); i += insertBatchSize {
		end := min(i+insertBatchSize, len(chunks))
		batch := chunks[i:end]

		tx, err := idx.db.BeginTx(ctx, nil)
		if err != nil {
			return fmt.Errorf("search: beginning chunk upsert transaction: %w", err)
		}

		for _, c := range batch {
			if c.ID == "" {
				_ = tx.Rollback()
				return fmt.Errorf("search: chunk has empty ID")
			}
			if c.Path == "" {
				_ = tx.Rollback()
				return fmt.Errorf("search: chunk %q has empty path", c.ID)
			}

			if _, err := tx.ExecContext(ctx,
				"DELETE FROM knowledge_fts WHERE path = ?", c.ID,
			); err != nil {
				_ = tx.Rollback()
				return fmt.Errorf("search: deleting old chunk FTS entry %s: %w", c.ID, err)
			}

			if _, err := tx.ExecContext(ctx,
				`INSERT INTO knowledge_fts (path, title, summary, tags, content, scope, project_slug, session_date)
				 VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
				c.ID,
				c.Title,
				"",
				strings.Join(c.Tags, " "),
				c.Content,
				"chunk",
				c.Path,
				"",
			); err != nil {
				_ = tx.Rollback()
				return fmt.Errorf("search: inserting chunk FTS entry %s: %w", c.ID, err)
			}
		}

		if err := tx.Commit(); err != nil {
			return fmt.Errorf("search: committing chunk upsert batch: %w", err)
		}
	}
	return nil
}

// DeleteChunks removes specific chunks from the FTS index by their IDs.
func (idx *Index) DeleteChunks(ctx context.Context, chunkIDs []string) error {
	if len(chunkIDs) == 0 {
		return nil
	}

	tx, err := idx.db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("search: beginning chunk delete transaction: %w", err)
	}

	stmt, err := tx.PrepareContext(ctx, "DELETE FROM knowledge_fts WHERE path = ? AND scope = 'chunk'")
	if err != nil {
		_ = tx.Rollback()
		return fmt.Errorf("search: preparing chunk delete: %w", err)
	}
	defer stmt.Close()

	for _, id := range chunkIDs {
		if id == "" {
			_ = tx.Rollback()
			return fmt.Errorf("search: empty chunk ID in delete list")
		}
		if _, err := stmt.ExecContext(ctx, id); err != nil {
			_ = tx.Rollback()
			return fmt.Errorf("search: deleting chunk %s: %w", id, err)
		}
	}

	if err := tx.Commit(); err != nil {
		return fmt.Errorf("search: committing chunk delete: %w", err)
	}

	// Also clean up metadata for deleted chunks.
	return idx.deleteChunkMetadataBatch(ctx, chunkIDs)
}

// DeleteByDocPath removes all chunks whose parent document path matches
// the supplied path. This is used when a document is re-chunked or deleted.
func (idx *Index) DeleteByDocPath(ctx context.Context, path string) error {
	if path == "" {
		return fmt.Errorf("search: empty path in DeleteByDocPath")
	}

	// Chunks store the parent path in the project_slug column with scope = 'chunk'.
	if _, err := idx.db.ExecContext(ctx,
		"DELETE FROM knowledge_fts WHERE project_slug = ? AND scope = 'chunk'",
		path,
	); err != nil {
		return fmt.Errorf("search: deleting chunks for document %s: %w", path, err)
	}

	// Clean up chunk metadata for all chunks belonging to this document.
	if _, err := idx.db.ExecContext(ctx,
		`DELETE FROM knowledge_chunk_metadata
		 WHERE chunk_id IN (
		     SELECT chunk_id FROM knowledge_chunk_metadata
		     WHERE chunk_id LIKE ?
		 )`,
		path+"%",
	); err != nil {
		return fmt.Errorf("search: deleting chunk metadata for document %s: %w", path, err)
	}

	return nil
}

// deleteChunkMetadataBatch removes metadata rows for the given chunk IDs.
func (idx *Index) deleteChunkMetadataBatch(ctx context.Context, chunkIDs []string) error {
	if len(chunkIDs) == 0 {
		return nil
	}

	tx, err := idx.db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("search: beginning chunk metadata delete: %w", err)
	}

	stmt, err := tx.PrepareContext(ctx, "DELETE FROM knowledge_chunk_metadata WHERE chunk_id = ?")
	if err != nil {
		_ = tx.Rollback()
		return fmt.Errorf("search: preparing chunk metadata delete: %w", err)
	}
	defer stmt.Close()

	for _, id := range chunkIDs {
		if _, err := stmt.ExecContext(ctx, id); err != nil {
			_ = tx.Rollback()
			return fmt.Errorf("search: deleting metadata for chunk %s: %w", id, err)
		}
	}

	return tx.Commit()
}
