// SPDX-License-Identifier: Apache-2.0

package retrieval

import (
	"context"
	"errors"
	"fmt"
	"strings"

	"github.com/jeffs-brain/memory/go/llm"
	"github.com/jeffs-brain/memory/go/search"
)

// IndexSource adapts a [search.Index] (and an optional companion
// [search.VectorIndex]) to the retrieval layer's [Source] contract.
//
// The Go SDK's search package indexes whole files keyed by logical
// brain path. The retrieval layer was specified against a chunk-level
// model, but for this adapter we collapse the two: every file is one
// chunk and the brain path is also the chunk identifier. Downstream
// callers that need real chunk fragmentation can layer their own
// segmentation in front and provide a different [Source].
//
// Concurrency: every method delegates to the underlying [search.Index]
// or [search.VectorIndex], both of which are safe for concurrent use.
// IndexSource holds no mutable state of its own.
type IndexSource struct {
	index    *search.Index
	vectors  *search.VectorIndex
	embedder llm.Embedder
	model    string
}

// IndexSourceOptions configures [NewIndexSource].
type IndexSourceOptions struct {
	// Vectors is the optional embedding store. When nil the
	// SearchVector leg returns no hits and the retriever degrades to
	// BM25-only ranking even in hybrid mode.
	Vectors *search.VectorIndex
	// Embedder is exposed to the IndexSource purely so the retrieval
	// layer can report back through llm.Embedder when needed; the
	// embed call itself is driven by the retriever via Config.Embedder.
	// Supplied here so callers can keep the wiring explicit.
	Embedder llm.Embedder
	// Model is the embedding model identifier passed through to
	// [search.VectorIndex.Search]. Required when Vectors is non-nil.
	Model string
}

// NewIndexSource builds an [IndexSource]. A nil index returns an
// error: BM25 is the spine of the retrieval pipeline and removing it
// leaves nothing to fall back to.
//
// Vector search is opt-in via opts.Vectors. When supplied, opts.Model
// must be set to the embedding model used to populate that index;
// otherwise the SearchVector leg cannot match stored entries.
func NewIndexSource(index *search.Index, opts IndexSourceOptions) (*IndexSource, error) {
	if index == nil {
		return nil, errors.New("retrieval: NewIndexSource requires a non-nil search.Index")
	}
	if opts.Vectors != nil && opts.Model == "" {
		return nil, errors.New("retrieval: NewIndexSource: opts.Model is required when opts.Vectors is set")
	}
	return &IndexSource{
		index:    index,
		vectors:  opts.Vectors,
		embedder: opts.Embedder,
		model:    opts.Model,
	}, nil
}

// SearchBM25 implements [Source]. The expression is forwarded
// verbatim to [search.Index.SearchRaw] (the retrieval pipeline already
// compiles the FTS5 expression via [compileToFTS]).
//
// Filters are mapped onto [search.SearchOpts]:
//   - Filters.Scope -> SearchOpts.Scope
//   - Filters.Project -> SearchOpts.ProjectSlug
//
// Tag, exact-path, and path-prefix filters are post-filtered in Go
// because the FTS5 schema does not expose them in a way we can
// compose with MATCH without rewriting the expression. The
// expectation matches the spec: filters narrow, never widen, the BM25
// candidate set.
func (s *IndexSource) SearchBM25(ctx context.Context, expr string, k int, filters Filters) ([]BM25Hit, error) {
	if s == nil || s.index == nil {
		return nil, errors.New("retrieval: IndexSource: nil index")
	}
	if expr == "" {
		return nil, nil
	}
	searchLimit := k
	if searchLimit <= 0 {
		searchLimit = 20
	}
	if filters.HasAny() {
		searchLimit = max(searchLimit*10, 200)
	}
	opts := search.SearchOpts{
		MaxResults: searchLimit,
		DateFrom:   filters.DateFrom,
		DateTo:     filters.DateTo,
		SessionIDs: normaliseStringList(filters.SessionIDs),
	}
	if strings.EqualFold(strings.TrimSpace(filters.Scope), "memory") {
		opts.Scope = "memory"
		opts.ProjectSlug = strings.TrimSpace(filters.Project)
	} else if scope, ok := exactSearchScope(filters.Scope); ok {
		opts.Scope = scope
		if scope == "project_memory" {
			opts.ProjectSlug = strings.TrimSpace(filters.Project)
		}
	} else if strings.TrimSpace(filters.Project) != "" {
		opts.Scope = "memory"
		opts.ProjectSlug = strings.TrimSpace(filters.Project)
	}
	results, err := s.index.SearchRaw(expr, opts)
	if err != nil {
		return nil, fmt.Errorf("retrieval: IndexSource SearchRaw: %w", err)
	}
	paths := make([]string, 0, len(results))
	for _, result := range results {
		paths = append(paths, result.Path)
	}
	rows, err := s.index.LookupRows(ctx, paths)
	if err != nil {
		return nil, fmt.Errorf("retrieval: IndexSource LookupRows: %w", err)
	}
	byPath := make(map[string]search.IndexedRow, len(rows))
	for _, row := range rows {
		byPath[row.Path] = row
	}

	out := make([]BM25Hit, 0, len(results))
	for _, r := range results {
		row, ok := byPath[r.Path]
		if !ok || !indexedRowPassesFilters(row, filters) {
			continue
		}
		title := r.Title
		summary := r.Summary
		content := r.Snippet
		if strings.TrimSpace(row.Title) != "" {
			title = row.Title
		}
		if strings.TrimSpace(row.Summary) != "" {
			summary = row.Summary
		}
		if strings.TrimSpace(row.Content) != "" {
			content = row.Content
		}
		out = append(out, BM25Hit{
			ID:      r.Path,
			Path:    r.Path,
			Title:   title,
			Summary: summary,
			Content: content,
			Score:   r.Score,
		})
		if k > 0 && len(out) >= k {
			break
		}
	}
	return out, nil
}

// SearchVector implements [Source]. Returns nil, nil when no vector
// index was configured so the retriever silently degrades to BM25
// without surfacing an error.
func (s *IndexSource) SearchVector(ctx context.Context, embedding []float32, k int, filters Filters) ([]VectorHit, error) {
	if s == nil || s.vectors == nil {
		return nil, nil
	}
	if len(embedding) == 0 {
		return nil, nil
	}
	searchLimit := k
	if searchLimit <= 0 {
		searchLimit = 20
	}
	var hits []search.VectorHit
	var err error
	if filters.HasAny() {
		allowed, aerr := s.allowedVectorPaths(ctx, filters)
		if aerr != nil {
			return nil, aerr
		}
		hits, err = s.vectors.SearchAllowed(ctx, embedding, s.model, searchLimit, allowed)
	} else {
		hits, err = s.vectors.Search(ctx, embedding, s.model, searchLimit)
	}
	if err != nil {
		return nil, fmt.Errorf("retrieval: IndexSource VectorIndex.Search: %w", err)
	}
	paths := make([]string, 0, len(hits))
	for _, hit := range hits {
		paths = append(paths, hit.Path)
	}
	rows, err := s.index.LookupRows(ctx, paths)
	if err != nil {
		return nil, fmt.Errorf("retrieval: IndexSource LookupRows: %w", err)
	}
	byPath := make(map[string]search.IndexedRow, len(rows))
	for _, row := range rows {
		byPath[row.Path] = row
	}

	out := make([]VectorHit, 0, len(hits))
	for _, h := range hits {
		row, ok := byPath[h.Path]
		if !ok || !indexedRowPassesFilters(row, filters) {
			continue
		}
		title := h.Title
		summary := h.Summary
		content := ""
		if strings.TrimSpace(row.Title) != "" {
			title = row.Title
		}
		if strings.TrimSpace(row.Summary) != "" {
			summary = row.Summary
		}
		if strings.TrimSpace(row.Content) != "" {
			content = row.Content
		}
		out = append(out, VectorHit{
			ID:         h.Path,
			Path:       h.Path,
			Title:      title,
			Summary:    summary,
			Content:    content,
			Similarity: float64(h.Similarity),
		})
		if k > 0 && len(out) >= k {
			break
		}
	}
	return out, nil
}

func (s *IndexSource) allowedVectorPaths(ctx context.Context, filters Filters) (map[string]struct{}, error) {
	rows, err := s.index.AllMetadataRows(ctx)
	if err != nil {
		return nil, fmt.Errorf("retrieval: IndexSource AllMetadataRows for vector filters: %w", err)
	}
	allowed := make(map[string]struct{}, len(rows))
	for _, row := range rows {
		if indexedRowPassesFilters(row, filters) {
			allowed[row.Path] = struct{}{}
		}
	}
	return allowed, nil
}

// Chunks implements [Source]. Walks every row in the FTS index so the
// retriever's trigram fallback can build a slug index. The result is
// stable across calls within a single process: the underlying SQLite
// snapshot is read in one query.
func (s *IndexSource) Chunks(ctx context.Context) ([]trigramChunk, error) {
	if s == nil || s.index == nil {
		return nil, errors.New("retrieval: IndexSource: nil index")
	}
	rows, err := s.index.AllRows(ctx)
	if err != nil {
		return nil, fmt.Errorf("retrieval: IndexSource AllRows: %w", err)
	}
	out := make([]trigramChunk, 0, len(rows))
	for _, r := range rows {
		out = append(out, trigramChunk{
			ID:        r.Path,
			Path:      r.Path,
			Title:     r.Title,
			Summary:   r.Summary,
			Content:   r.Content,
			Tags:      r.Tags,
			Scope:     r.Scope,
			Project:   r.ProjectSlug,
			Session:   r.SessionDate,
			SessionID: r.SessionID,
		})
	}
	return out, nil
}

// Lookup returns the full indexed payload for a slice of brain paths.
// Useful for adapters that have already collected chunk identifiers
// (e.g. from a cached trigram pass) and now need the body for
// rendering. Callers building [RetrievedChunk] directly may prefer to
// drive the retriever via [Retriever.Retrieve] instead.
func (s *IndexSource) Lookup(ctx context.Context, ids []string) ([]search.IndexedRow, error) {
	if s == nil || s.index == nil {
		return nil, errors.New("retrieval: IndexSource: nil index")
	}
	if len(ids) == 0 {
		return nil, nil
	}
	return s.index.LookupRows(ctx, ids)
}

// SessionNeighbours implements [SessionNeighbourSource] by looking up
// all indexed rows that share one of the seed session ids, then applying
// the same retrieval filters and caps used by the direct search legs.
func (s *IndexSource) SessionNeighbours(ctx context.Context, seeds []SessionNeighbourSeed, opts SessionNeighbourOptions) ([]search.IndexedRow, error) {
	if s == nil || s.index == nil {
		return nil, errors.New("retrieval: IndexSource: nil index")
	}
	if len(seeds) == 0 {
		return nil, nil
	}
	maxSessions := opts.MaxSessions
	if maxSessions <= 0 {
		maxSessions = 3
	}
	maxRowsPerSession := opts.MaxRowsPerSession
	if maxRowsPerSession <= 0 {
		maxRowsPerSession = 3
	}
	maxRowsTotal := opts.MaxRowsTotal
	if maxRowsTotal <= 0 {
		maxRowsTotal = maxRowsPerSession * maxSessions
	}

	sessionIDs := make([]string, 0, maxSessions)
	wantedSessions := make(map[string]bool, maxSessions)
	seedPaths := make(map[string]bool, len(seeds))
	for _, seed := range seeds {
		if strings.TrimSpace(seed.Path) != "" {
			seedPaths[seed.Path] = true
		}
		sessionID := strings.TrimSpace(seed.SessionID)
		if sessionID == "" || wantedSessions[sessionID] {
			continue
		}
		wantedSessions[sessionID] = true
		sessionIDs = append(sessionIDs, sessionID)
		if len(sessionIDs) >= maxSessions {
			break
		}
	}
	if len(sessionIDs) == 0 {
		return nil, nil
	}

	rows, err := s.index.SessionRows(ctx, sessionIDs)
	if err != nil {
		return nil, fmt.Errorf("retrieval: IndexSource SessionRows: %w", err)
	}
	out := make([]search.IndexedRow, 0, min(maxRowsTotal, len(rows)))
	perSession := make(map[string]int, len(sessionIDs))
	for _, row := range rows {
		sessionID := strings.TrimSpace(row.SessionID)
		if sessionID == "" || !wantedSessions[sessionID] {
			continue
		}
		if seedPaths[row.Path] {
			continue
		}
		if !indexedRowPassesFilters(row, opts.Filters) {
			continue
		}
		if perSession[sessionID] >= maxRowsPerSession {
			continue
		}
		out = append(out, row)
		perSession[sessionID]++
		if len(out) >= maxRowsTotal {
			break
		}
	}
	return out, nil
}

func exactSearchScope(scope string) (string, bool) {
	switch strings.ToLower(strings.TrimSpace(scope)) {
	case "global", "global_memory":
		return "global_memory", true
	case "project", "project_memory":
		return "project_memory", true
	case "wiki":
		return "wiki", true
	case "raw":
		return "raw", true
	case "raw_document":
		return "raw_document", true
	case "raw_lme":
		return "raw_lme", true
	case "sources":
		return "sources", true
	default:
		return "", false
	}
}

func indexedRowPassesFilters(row search.IndexedRow, filters Filters) bool {
	if !filters.MatchesPath(row.Path) {
		return false
	}
	if !sessionMatchesFilter(row.SessionID, filters.SessionIDs) {
		return false
	}
	if !scopeMatchesFilter(row.Scope, filters.Scope) {
		return false
	}
	if !projectMatchesFilter(row.Scope, row.ProjectSlug, filters.Project) {
		return false
	}
	if !dateMatchesFilter(row.SessionDate, filters) {
		return false
	}
	if len(filters.Tags) > 0 {
		rowTags := tagSet(row.Tags)
		for _, tag := range filters.Tags {
			want := normaliseTag(tag)
			if want == "" {
				continue
			}
			if !rowTags[want] {
				return false
			}
		}
	}
	return true
}

func sessionMatchesFilter(rowSessionID string, want []string) bool {
	sessions := normaliseStringList(want)
	if len(sessions) == 0 {
		return true
	}
	rowSessionID = strings.TrimSpace(rowSessionID)
	if rowSessionID == "" {
		return false
	}
	for _, sessionID := range sessions {
		if sessionID == rowSessionID {
			return true
		}
	}
	return false
}

func projectMatchesFilter(rowScope, rowProject, want string) bool {
	project := strings.ToLower(strings.TrimSpace(want))
	if project == "" {
		return true
	}
	rowProject = strings.ToLower(strings.TrimSpace(rowProject))
	scope := strings.ToLower(strings.TrimSpace(rowScope))
	switch scope {
	case "global_memory":
		return true
	case "project_memory":
		return rowProject == project
	case "":
		return rowProject == project
	default:
		return rowProject != "" && rowProject == project
	}
}

func dateMatchesFilter(sessionDate string, filters Filters) bool {
	if filters.DateFrom.IsZero() && filters.DateTo.IsZero() {
		return true
	}
	parsed, ok := parseCandidateTime(sessionDate)
	if !ok {
		return false
	}
	if !filters.DateFrom.IsZero() && parsed.Before(filters.DateFrom) {
		return false
	}
	if !filters.DateTo.IsZero() && parsed.After(filters.DateTo) {
		return false
	}
	return true
}

func tagSet(raw string) map[string]bool {
	out := make(map[string]bool)
	for _, field := range strings.FieldsFunc(raw, func(r rune) bool {
		switch r {
		case '[', ']', ',', '"', '\'', '\n', '\r', '\t', ' ':
			return true
		default:
			return false
		}
	}) {
		tag := normaliseTag(field)
		if tag != "" {
			out[tag] = true
		}
	}
	return out
}

func normaliseTag(tag string) string {
	return strings.ToLower(strings.Trim(strings.TrimSpace(tag), "[],'\""))
}

func scopeMatchesFilter(rowScope, want string) bool {
	trimmed := strings.ToLower(strings.TrimSpace(want))
	if trimmed == "" {
		return true
	}
	aliases := map[string]map[string]bool{
		"memory": {
			"global_memory":  true,
			"project_memory": true,
		},
		"global": {
			"global_memory": true,
		},
		"global_memory": {
			"global_memory": true,
		},
		"project": {
			"project_memory": true,
		},
		"project_memory": {
			"project_memory": true,
		},
		"raw": {
			"raw_document": true,
			"raw_lme":      true,
		},
		"raw_document": {
			"raw_document": true,
		},
		"raw_lme": {
			"raw_lme": true,
		},
		"wiki": {
			"wiki": true,
		},
		"sources": {
			"sources": true,
		},
	}
	allowed, ok := aliases[trimmed]
	if !ok {
		return strings.ToLower(strings.TrimSpace(rowScope)) == trimmed
	}
	if rowScope == "" {
		return true
	}
	return allowed[strings.ToLower(strings.TrimSpace(rowScope))]
}

// compile-time interface check.
var _ Source = (*IndexSource)(nil)
