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
	Path               string
	Title              string
	Summary            string
	Content            string
	Tags               string
	Scope              string
	ProjectSlug        string
	SessionDate        string
	SessionID          string
	ObservedOn         string
	Modified           string
	SourceRole         string
	EventDate          string
	EvidenceKind       string
	EvidenceGroup      string
	StateKey           string
	StateKind          string
	StateSubject       string
	StateValue         string
	ClaimKey           string
	ClaimStatus        string
	ValidFrom          string
	ValidTo            string
	ArtefactType       string
	ArtefactOrdinal    string
	ArtefactSection    string
	ArtefactDescriptor string
}

const indexedRowSelectColumns = `f.path, f.title, f.summary, f.content, f.tags, f.scope, f.project_slug, f.session_date,
	                 m.session_id, m.observed_on, m.modified, m.source_role, m.event_date,
	                 m.evidence_kind, m.evidence_group, m.state_key, m.state_kind, m.state_subject, m.state_value,
	                 m.claim_key, m.claim_status, m.valid_from, m.valid_to,
	                 m.artefact_type, m.artefact_ordinal, m.artefact_section, m.artefact_descriptor`

const indexedMetadataSelectColumns = `f.path, '', '', '', f.tags, f.scope, f.project_slug, f.session_date,
	                 m.session_id, m.observed_on, m.modified, m.source_role, m.event_date,
	                 m.evidence_kind, m.evidence_group, m.state_key, m.state_kind, m.state_subject, m.state_value,
	                 m.claim_key, m.claim_status, m.valid_from, m.valid_to,
	                 m.artefact_type, m.artefact_ordinal, m.artefact_section, m.artefact_descriptor`

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

	query := `SELECT ` + indexedRowSelectColumns + `
	          FROM knowledge_fts AS f
	          LEFT JOIN knowledge_index_metadata AS m ON m.path = f.path
	          WHERE f.path IN (` + strings.Join(placeholders, ",") + `)`

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

// SessionRows returns indexed rows whose metadata session_id matches
// one of the supplied ids. Missing or blank ids are ignored. Rows are
// returned in a deterministic session/date/path order so callers can
// apply stable per-session caps.
func (idx *Index) SessionRows(ctx context.Context, sessionIDs []string) ([]IndexedRow, error) {
	if idx == nil {
		return nil, errors.New("search: SessionRows on nil index")
	}
	if len(sessionIDs) == 0 {
		return nil, nil
	}
	seen := make(map[string]bool, len(sessionIDs))
	placeholders := make([]string, 0, len(sessionIDs))
	args := make([]any, 0, len(sessionIDs))
	for _, id := range sessionIDs {
		trimmed := strings.TrimSpace(id)
		if trimmed == "" || seen[trimmed] {
			continue
		}
		seen[trimmed] = true
		placeholders = append(placeholders, "?")
		args = append(args, trimmed)
	}
	if len(placeholders) == 0 {
		return nil, nil
	}

	query := `SELECT ` + indexedRowSelectColumns + `
	          FROM knowledge_fts AS f
	          JOIN knowledge_index_metadata AS m ON m.path = f.path
	          WHERE m.session_id IN (` + strings.Join(placeholders, ",") + `)
	          ORDER BY m.session_id, COALESCE(m.observed_on, ''), COALESCE(m.event_date, ''),
	                   COALESCE(m.modified, ''), f.path`

	rows, err := idx.db.QueryContext(ctx, query, args...)
	if err != nil {
		return nil, fmt.Errorf("search: SessionRows query: %w", err)
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
		return nil, fmt.Errorf("search: SessionRows iterate: %w", err)
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
		`SELECT `+indexedRowSelectColumns+`
		 FROM knowledge_fts AS f
		 LEFT JOIN knowledge_index_metadata AS m ON m.path = f.path`,
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

// AllMetadataRows streams the indexed metadata needed for filter
// checks without loading the full indexed body. Retrieval uses this
// to pre-filter vector candidates before top-k truncation.
func (idx *Index) AllMetadataRows(ctx context.Context) ([]IndexedRow, error) {
	if idx == nil {
		return nil, errors.New("search: AllMetadataRows on nil index")
	}

	rows, err := idx.db.QueryContext(ctx,
		`SELECT `+indexedMetadataSelectColumns+`
		 FROM knowledge_fts AS f
		 LEFT JOIN knowledge_index_metadata AS m ON m.path = f.path`,
	)
	if err != nil {
		return nil, fmt.Errorf("search: AllMetadataRows query: %w", err)
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
		return nil, fmt.Errorf("search: AllMetadataRows iterate: %w", err)
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
		row                   IndexedRow
		sessionDateRaw        sql.NullString
		sessionIDRaw          sql.NullString
		observedOnRaw         sql.NullString
		modifiedRaw           sql.NullString
		sourceRoleRaw         sql.NullString
		eventDateRaw          sql.NullString
		evidenceKindRaw       sql.NullString
		evidenceGroupRaw      sql.NullString
		stateKeyRaw           sql.NullString
		stateKindRaw          sql.NullString
		stateSubjectRaw       sql.NullString
		stateValueRaw         sql.NullString
		claimKeyRaw           sql.NullString
		claimStatusRaw        sql.NullString
		validFromRaw          sql.NullString
		validToRaw            sql.NullString
		artefactTypeRaw       sql.NullString
		artefactOrdinalRaw    sql.NullString
		artefactSectionRaw    sql.NullString
		artefactDescriptorRaw sql.NullString
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
		&sessionIDRaw,
		&observedOnRaw,
		&modifiedRaw,
		&sourceRoleRaw,
		&eventDateRaw,
		&evidenceKindRaw,
		&evidenceGroupRaw,
		&stateKeyRaw,
		&stateKindRaw,
		&stateSubjectRaw,
		&stateValueRaw,
		&claimKeyRaw,
		&claimStatusRaw,
		&validFromRaw,
		&validToRaw,
		&artefactTypeRaw,
		&artefactOrdinalRaw,
		&artefactSectionRaw,
		&artefactDescriptorRaw,
	); err != nil {
		return IndexedRow{}, fmt.Errorf("search: scan row: %w", err)
	}
	if sessionDateRaw.Valid {
		row.SessionDate = sessionDateRaw.String
	}
	if sessionIDRaw.Valid {
		row.SessionID = sessionIDRaw.String
	}
	if observedOnRaw.Valid {
		row.ObservedOn = observedOnRaw.String
	}
	if modifiedRaw.Valid {
		row.Modified = modifiedRaw.String
	}
	if sourceRoleRaw.Valid {
		row.SourceRole = sourceRoleRaw.String
	}
	if eventDateRaw.Valid {
		row.EventDate = eventDateRaw.String
	}
	if evidenceKindRaw.Valid {
		row.EvidenceKind = evidenceKindRaw.String
	}
	if evidenceGroupRaw.Valid {
		row.EvidenceGroup = evidenceGroupRaw.String
	}
	if stateKeyRaw.Valid {
		row.StateKey = stateKeyRaw.String
	}
	if stateKindRaw.Valid {
		row.StateKind = stateKindRaw.String
	}
	if stateSubjectRaw.Valid {
		row.StateSubject = stateSubjectRaw.String
	}
	if stateValueRaw.Valid {
		row.StateValue = stateValueRaw.String
	}
	if claimKeyRaw.Valid {
		row.ClaimKey = claimKeyRaw.String
	}
	if claimStatusRaw.Valid {
		row.ClaimStatus = claimStatusRaw.String
	}
	if validFromRaw.Valid {
		row.ValidFrom = validFromRaw.String
	}
	if validToRaw.Valid {
		row.ValidTo = validToRaw.String
	}
	if artefactTypeRaw.Valid {
		row.ArtefactType = artefactTypeRaw.String
	}
	if artefactOrdinalRaw.Valid {
		row.ArtefactOrdinal = artefactOrdinalRaw.String
	}
	if artefactSectionRaw.Valid {
		row.ArtefactSection = artefactSectionRaw.String
	}
	if artefactDescriptorRaw.Valid {
		row.ArtefactDescriptor = artefactDescriptorRaw.String
	}
	return row, nil
}
