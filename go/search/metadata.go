// SPDX-License-Identifier: Apache-2.0
package search

import (
	"context"
	"fmt"
)

// chunkMetadataSchema is the DDL for the knowledge_chunk_metadata table.
// It stores extensible per-chunk key-value metadata (ontology type,
// confidence scores, embedding model, etc.). The composite primary key
// ensures at most one value per (chunk_id, key) pair.
const chunkMetadataSchema = `
CREATE TABLE IF NOT EXISTS knowledge_chunk_metadata (
    chunk_id TEXT NOT NULL,
    key      TEXT NOT NULL,
    value    TEXT NOT NULL,
    PRIMARY KEY (chunk_id, key)
);
`

// createChunkMetadataTable idempotently creates the
// knowledge_chunk_metadata table. Called during index construction.
func (idx *Index) createChunkMetadataTable(ctx context.Context) error {
	if _, err := idx.db.ExecContext(ctx, chunkMetadataSchema); err != nil {
		return fmt.Errorf("search: creating knowledge_chunk_metadata: %w", err)
	}
	return nil
}

// SetChunkMetadata upserts key-value metadata for a chunk. Empty keys
// or values are rejected. Existing entries for the same (chunk_id, key)
// pair are overwritten.
func (idx *Index) SetChunkMetadata(ctx context.Context, chunkID string, meta map[string]string) error {
	if chunkID == "" {
		return fmt.Errorf("search: SetChunkMetadata requires non-empty chunk ID")
	}
	if len(meta) == 0 {
		return nil
	}

	tx, err := idx.db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("search: beginning metadata upsert: %w", err)
	}

	stmt, err := tx.PrepareContext(ctx,
		`INSERT INTO knowledge_chunk_metadata (chunk_id, key, value)
		 VALUES (?, ?, ?)
		 ON CONFLICT(chunk_id, key) DO UPDATE SET value = excluded.value`,
	)
	if err != nil {
		_ = tx.Rollback()
		return fmt.Errorf("search: preparing metadata upsert: %w", err)
	}
	defer stmt.Close()

	for k, v := range meta {
		if k == "" {
			_ = tx.Rollback()
			return fmt.Errorf("search: metadata key must not be empty for chunk %s", chunkID)
		}
		if v == "" {
			_ = tx.Rollback()
			return fmt.Errorf("search: metadata value must not be empty for chunk %s key %s", chunkID, k)
		}
		if _, err := stmt.ExecContext(ctx, chunkID, k, v); err != nil {
			_ = tx.Rollback()
			return fmt.Errorf("search: upserting metadata for chunk %s key %s: %w", chunkID, k, err)
		}
	}

	return tx.Commit()
}

// GetChunkMetadata retrieves all key-value metadata for a chunk.
// Returns an empty map (not nil) when the chunk has no metadata.
func (idx *Index) GetChunkMetadata(ctx context.Context, chunkID string) (map[string]string, error) {
	if chunkID == "" {
		return nil, fmt.Errorf("search: GetChunkMetadata requires non-empty chunk ID")
	}

	rows, err := idx.db.QueryContext(ctx,
		"SELECT key, value FROM knowledge_chunk_metadata WHERE chunk_id = ?",
		chunkID,
	)
	if err != nil {
		return nil, fmt.Errorf("search: querying metadata for chunk %s: %w", chunkID, err)
	}
	defer rows.Close()

	out := make(map[string]string)
	for rows.Next() {
		var k, v string
		if err := rows.Scan(&k, &v); err != nil {
			return nil, fmt.Errorf("search: scanning metadata row for chunk %s: %w", chunkID, err)
		}
		out[k] = v
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("search: iterating metadata for chunk %s: %w", chunkID, err)
	}
	return out, nil
}

// QueryByMetadata returns chunk IDs where the given key matches the
// given value. Results are ordered by chunk_id for determinism. A limit
// of zero or negative defaults to 100.
func (idx *Index) QueryByMetadata(ctx context.Context, key, value string, limit int) ([]string, error) {
	if key == "" {
		return nil, fmt.Errorf("search: QueryByMetadata requires non-empty key")
	}
	if limit <= 0 {
		limit = 100
	}

	rows, err := idx.db.QueryContext(ctx,
		`SELECT chunk_id FROM knowledge_chunk_metadata
		 WHERE key = ? AND value = ?
		 ORDER BY chunk_id
		 LIMIT ?`,
		key, value, limit,
	)
	if err != nil {
		return nil, fmt.Errorf("search: querying metadata by key=%s value=%s: %w", key, value, err)
	}
	defer rows.Close()

	out := make([]string, 0)
	for rows.Next() {
		var id string
		if err := rows.Scan(&id); err != nil {
			return nil, fmt.Errorf("search: scanning metadata query result: %w", err)
		}
		out = append(out, id)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("search: iterating metadata query results: %w", err)
	}
	return out, nil
}
