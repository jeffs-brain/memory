// SPDX-License-Identifier: Apache-2.0

package main

import (
	"net/http"
	"time"

	"github.com/jeffs-brain/memory/go/internal/httpd"
	"github.com/jeffs-brain/memory/go/retrieval"
	"github.com/jeffs-brain/memory/go/search"
)

// searchRequest is the body shape for POST /v1/brains/{brainId}/search.
type searchRequest struct {
	Query   string            `json:"query"`
	TopK    int               `json:"topK,omitempty"`
	Mode    string            `json:"mode,omitempty"`
	Filters retrieval.Filters `json:"filters,omitempty"`
}

// searchResponse mirrors the wire shape requested by the brief.
type searchResponse struct {
	Chunks   []retrieval.RetrievedChunk `json:"chunks"`
	TookMs   int64                      `json:"tookMs"`
	Trace    *retrieval.Trace           `json:"trace,omitempty"`
	Attempts []retrieval.Attempt        `json:"attempts,omitempty"`
}

// handleSearch runs a hybrid search via the bound retriever, falling
// back to BM25 via the search.Index when no retriever is configured.
// Ingested raw/documents content is now covered by the FTS classifier
// so no separate knowledge fallback is required.
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

	// Refresh the index so writes that just landed are searchable.
	if br.Search != nil {
		_ = br.Search.Update(r.Context())
	}

	chunks, trace, attempts, took := d.runSearchPipeline(r, br, req)

	writeJSON(w, http.StatusOK, searchResponse{
		Chunks:   chunks,
		TookMs:   took,
		Trace:    trace,
		Attempts: attempts,
	})
}

// runSearchPipeline runs the retriever (or BM25 fallback) and returns
// the resulting chunks. The trace and attempt log are returned only
// when the retriever produced them.
func (d *Daemon) runSearchPipeline(r *http.Request, br *BrainResources, req searchRequest) ([]retrieval.RetrievedChunk, *retrieval.Trace, []retrieval.Attempt, int64) {
	if br.Retriever != nil {
		started := time.Now()
		mode := retrieval.Mode(req.Mode)
		if mode == "" {
			mode = retrieval.ModeAuto
		}
		resp, err := br.Retriever.Retrieve(r.Context(), retrieval.Request{
			Query:   req.Query,
			TopK:    req.TopK,
			Mode:    mode,
			BrainID: br.ID,
			Filters: req.Filters,
		})
		if err == nil {
			took := int64(resp.TookMs)
			if took == 0 {
				took = time.Since(started).Milliseconds()
			}
			return resp.Chunks, &resp.Trace, resp.Attempts, took
		}
	}
	if br.Search == nil {
		return nil, nil, nil, 0
	}
	started := time.Now()
	results, err := br.Search.Search(req.Query, search.SearchOpts{MaxResults: req.TopK})
	if err != nil {
		return nil, nil, nil, time.Since(started).Milliseconds()
	}
	chunks := make([]retrieval.RetrievedChunk, 0, len(results))
	for _, h := range results {
		chunks = append(chunks, retrieval.RetrievedChunk{
			ChunkID:    h.Path,
			DocumentID: h.Path,
			Path:       h.Path,
			Score:      1.0 / float64(h.Score+1),
			Text:       h.Snippet,
			Title:      h.Title,
			Summary:    h.Summary,
		})
	}
	return chunks, nil, nil, time.Since(started).Milliseconds()
}
