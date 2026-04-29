// SPDX-License-Identifier: Apache-2.0

package main

import (
	"encoding/json"
	"net/http"
	"strings"
	"time"

	"github.com/jeffs-brain/memory/go/internal/httpd"
	"github.com/jeffs-brain/memory/go/retrieval"
	"github.com/jeffs-brain/memory/go/search"
)

type searchRequest struct {
	Query        string            `json:"query"`
	QuestionDate string            `json:"questionDate,omitempty"`
	TopK         int               `json:"topK,omitempty"`
	CandidateK   int               `json:"candidateK,omitempty"`
	RerankTopN   int               `json:"rerankTopN,omitempty"`
	Mode         string            `json:"mode,omitempty"`
	Filters      retrieval.Filters `json:"filters,omitempty"`
}

func (req *searchRequest) UnmarshalJSON(data []byte) error {
	type rawSearchRequest struct {
		Query           string          `json:"query"`
		QuestionDate    string          `json:"questionDate"`
		QuestionDateAlt string          `json:"question_date"`
		TopK            int             `json:"topK"`
		CandidateK      int             `json:"candidateK"`
		CandidateKAlt   int             `json:"candidate_k"`
		RerankTopN      int             `json:"rerankTopN"`
		RerankTopNAlt   int             `json:"rerank_top_n"`
		Mode            string          `json:"mode"`
		Filters         json.RawMessage `json:"filters"`
	}
	var raw rawSearchRequest
	if err := json.Unmarshal(data, &raw); err != nil {
		return err
	}
	filters, err := decodeRequestFilters(data, raw.Filters)
	if err != nil {
		return err
	}
	*req = searchRequest{
		Query:        raw.Query,
		QuestionDate: firstNonEmptyRequestValue(raw.QuestionDate, raw.QuestionDateAlt),
		TopK:         raw.TopK,
		CandidateK:   firstPositiveRequestValue(raw.CandidateK, raw.CandidateKAlt),
		RerankTopN:   firstPositiveRequestValue(raw.RerankTopN, raw.RerankTopNAlt),
		Mode:         raw.Mode,
		Filters:      filters,
	}
	return nil
}

type searchResponse struct {
	Chunks   []retrieval.RetrievedChunk `json:"chunks"`
	TookMs   int64                      `json:"tookMs"`
	Trace    *retrieval.Trace           `json:"trace,omitempty"`
	Attempts []retrieval.Attempt        `json:"attempts,omitempty"`
}

// handleSearch runs hybrid search via the bound retriever, falling back
// to BM25 via search.Index when no retriever is configured. Ingested
// raw/documents content is covered by the FTS classifier so no separate
// knowledge fallback is required.
func (d *Daemon) handleSearch(w http.ResponseWriter, r *http.Request) {
	br := d.resolveBrain(w, r)
	if br == nil {
		return
	}
	var req searchRequest
	if err := decodeJSONBody(r, &req, 256*1024); err != nil {
		httpd.ValidationError(w, err.Error())
		return
	}
	if req.Query == "" {
		httpd.ValidationError(w, "query required")
		return
	}
	if req.TopK <= 0 {
		req.TopK = 10
	}

	// Cheap after the first call: awaits the one-shot initial scan so
	// pre-seeded disk content surfaces on the first query. Subsequent
	// writes are caught by the Subscribe mechanism so no per-request
	// full index scan happens — that would serialise every concurrent
	// request on the refresh write lock.
	br.WaitReady(r.Context())

	chunks, trace, attempts, took := d.runSearchPipeline(r, br, req)

	writeJSON(w, http.StatusOK, searchResponse{
		Chunks:   chunks,
		TookMs:   took,
		Trace:    trace,
		Attempts: attempts,
	})
}

func normaliseRetrievalMode(raw string) retrieval.Mode {
	switch mode := retrieval.Mode(strings.ToLower(strings.TrimSpace(raw))); mode {
	case retrieval.ModeAuto,
		retrieval.ModeBM25,
		retrieval.ModeSemantic,
		retrieval.ModeHybrid,
		retrieval.ModeHybridRerank:
		return mode
	default:
		return retrieval.ModeAuto
	}
}

func decodeRequestFilters(data []byte, raw json.RawMessage) (retrieval.Filters, error) {
	source := raw
	if len(source) == 0 || strings.TrimSpace(string(source)) == "null" {
		source = data
	}
	var filters retrieval.Filters
	if err := json.Unmarshal(source, &filters); err != nil {
		return retrieval.Filters{}, err
	}
	return filters, nil
}

func firstNonEmptyRequestValue(values ...string) string {
	for _, value := range values {
		if value != "" {
			return value
		}
	}
	return ""
}

func firstPositiveRequestValue(values ...int) int {
	for _, value := range values {
		if value > 0 {
			return value
		}
	}
	return 0
}

func filterRetrievedChunksByPath(chunks []retrieval.RetrievedChunk, filters retrieval.Filters) []retrieval.RetrievedChunk {
	if len(chunks) == 0 {
		return nil
	}
	filtered := make([]retrieval.RetrievedChunk, 0, len(chunks))
	for _, chunk := range chunks {
		if !filters.MatchesPath(chunk.Path) {
			continue
		}
		filtered = append(filtered, chunk)
	}
	return filtered
}

func fallbackSearchOpts(topK int, filters retrieval.Filters) search.SearchOpts {
	maxResults := topK
	if maxResults <= 0 {
		maxResults = 10
	}
	if filters.HasAny() {
		maxResults = max(maxResults*10, 200)
	}
	opts := search.SearchOpts{MaxResults: maxResults}
	if scope, ok := fallbackSearchScope(filters.Scope); ok {
		opts.Scope = scope
		if scope == "project_memory" {
			opts.ProjectSlug = strings.TrimSpace(filters.Project)
		}
	}
	return opts
}

func fallbackSearchScope(raw string) (string, bool) {
	switch strings.ToLower(strings.TrimSpace(raw)) {
	case "global", "global_memory":
		return "global_memory", true
	case "project", "project_memory":
		return "project_memory", true
	case "wiki":
		return "wiki", true
	case "raw", "raw_document":
		return "raw_document", true
	case "raw_lme":
		return "raw_lme", true
	case "sources":
		return "sources", true
	default:
		return "", false
	}
}

func (d *Daemon) runSearchPipeline(r *http.Request, br *BrainResources, req searchRequest) ([]retrieval.RetrievedChunk, *retrieval.Trace, []retrieval.Attempt, int64) {
	if br.Retriever != nil {
		started := time.Now()
		mode := normaliseRetrievalMode(req.Mode)
		resp, err := br.Retriever.Retrieve(r.Context(), retrieval.Request{
			Query:        req.Query,
			QuestionDate: req.QuestionDate,
			TopK:         req.TopK,
			CandidateK:   req.CandidateK,
			RerankTopN:   req.RerankTopN,
			Mode:         mode,
			BrainID:      br.ID,
			Filters:      req.Filters,
		})
		if err == nil {
			took := int64(resp.TookMs)
			if took == 0 {
				took = time.Since(started).Milliseconds()
			}
			chunks := filterRetrievedChunksByPath(resp.Chunks, req.Filters)
			for i := range chunks {
				chunks[i] = hydrateFallbackChunk(r.Context(), br.Store, chunks[i], "")
			}
			return chunks, &resp.Trace, resp.Attempts, took
		}
	}
	if br.Search == nil {
		return nil, nil, nil, 0
	}
	started := time.Now()
	results, err := br.Search.Search(augmentRetrievalQuery(req.Query, req.QuestionDate), fallbackSearchOpts(req.TopK, req.Filters))
	if err != nil {
		return nil, nil, nil, time.Since(started).Milliseconds()
	}
	chunks := make([]retrieval.RetrievedChunk, 0, len(results))
	for _, h := range results {
		if !req.Filters.MatchesPath(h.Path) {
			continue
		}
		chunk := retrieval.RetrievedChunk{
			ChunkID:    h.Path,
			DocumentID: h.Path,
			Path:       h.Path,
			Score:      1.0 / float64(h.Score+1),
			Text:       h.Snippet,
			Title:      h.Title,
			Summary:    h.Summary,
		}
		chunks = append(chunks, hydrateFallbackChunk(r.Context(), br.Store, chunk, h.SessionDate))
	}
	return chunks, nil, nil, time.Since(started).Milliseconds()
}
