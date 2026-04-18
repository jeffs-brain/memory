// SPDX-License-Identifier: Apache-2.0

package search

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"strings"
)

// IndexedRow is a snapshot of one row in the FTS index, exported for
// adapters (notably the retrieval package's IndexSource) that need the
// full payload (including body content) rather than the snippet-only
// shape returned by [Index.Search].
//
// This shape exists so the public surface of [Index] does not have to
// expose its underlying knowledge_fts column layout. We keep the export
// to a single read-only view: callers cannot mutate the index through
// it.
type IndexedRow struct {
	Path        string
	Title       string
	Summary     string
	Content     string
	Tags        string
	Scope       string
	ProjectSlug string
	SessionDate string
}

// LookupRows returns one [IndexedRow] per supplied path. Missing paths
// are silently dropped from the result. Callers needing to detect
// missing entries should compare result length against the input.
//
// Justification for the addition: the retrieval package's adapter
// (`retrieval.IndexSource`) needs to hydrate the full text body of a
// chunk when serving the trigram fallback. The existing [Index.Search]
// surface emits an FTS5 snippet (deliberately truncated and marked up
// for display) but not the indexed body, so it cannot serve that role.
// Adding LookupRows here is the smallest possible API extension: no
// existing exports are renamed, no schema changes, and the trigram
// adapter no longer has to round-trip through the brain store to fetch
// content already sitting in the FTS table.
func (idx *Index) LookupRows(ctx context.Context, paths []string) ([]IndexedRow, error) {
	if idx == nil {
		return nil, errors.New("search: LookupRows on nil index")
	}
	if len(paths) == 0 {
		return nil, nil
	}

	placeholders := make([]string, 0, len(paths))
	args := make([]any, 0, len(paths))
	for _, p := range paths {
		placeholders = append(placeholders, "?")
		args = append(args, p)
	}

	query := `SELECT path, title, summary, content, tags, scope, project_slug, session_date
	          FROM knowledge_fts
	          WHERE path IN (` + strings.Join(placeholders, ",") + `)`

	rows, err := idx.db.QueryContext(ctx, query, args...)
	if err != nil {
		return nil, fmt.Errorf("search: LookupRows query: %w", err)
	}
	defer rows.Close()

	out := make([]IndexedRow, 0, len(paths))
	for rows.Next() {
		row, err := scanIndexedRow(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, row)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("search: LookupRows iterate: %w", err)
	}
	return out, nil
}

// AllRows streams every row currently in the FTS index. Used by the
// retrieval adapter's trigram fallback to seed its slug index in one
// pass. The result preserves no particular order; callers that need
// stability must sort.
//
// As with [Index.LookupRows], this exists to keep the FTS column
// layout private while still letting one specific consumer (the
// retrieval IndexSource) read the body without re-walking the brain
// store.
func (idx *Index) AllRows(ctx context.Context) ([]IndexedRow, error) {
	if idx == nil {
		return nil, errors.New("search: AllRows on nil index")
	}

	rows, err := idx.db.QueryContext(ctx,
		`SELECT path, title, summary, content, tags, scope, project_slug, session_date
		 FROM knowledge_fts`,
	)
	if err != nil {
		return nil, fmt.Errorf("search: AllRows query: %w", err)
	}
	defer rows.Close()

	out := make([]IndexedRow, 0)
	for rows.Next() {
		row, err := scanIndexedRow(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, row)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("search: AllRows iterate: %w", err)
	}
	return out, nil
}

// rowScanner mirrors the read surface of *sql.Row and *sql.Rows so the
// scan helper can serve both the lookup and walk paths.
type rowScanner interface {
	Scan(dest ...any) error
}

func scanIndexedRow(r rowScanner) (IndexedRow, error) {
	var (
		row             IndexedRow
		sessionDateRaw  sql.NullString
	)
	if err := r.Scan(
		&row.Path,
		&row.Title,
		&row.Summary,
		&row.Content,
		&row.Tags,
		&row.Scope,
		&row.ProjectSlug,
		&sessionDateRaw,
	); err != nil {
		return IndexedRow{}, fmt.Errorf("search: scan row: %w", err)
	}
	if sessionDateRaw.Valid {
		row.SessionDate = sessionDateRaw.String
	}
	return row, nil
}
