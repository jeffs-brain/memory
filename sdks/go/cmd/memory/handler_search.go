// SPDX-License-Identifier: Apache-2.0

package main

import (
	"net/http"
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

func (d *Daemon) runSearchPipeline(r *http.Request, br *BrainResources, req searchRequest) ([]retrieval.RetrievedChunk, *retrieval.Trace, []retrieval.Attempt, int64) {
	if br.Retriever != nil {
		started := time.Now()
		mode := retrieval.Mode(req.Mode)
		if mode == "" {
			mode = retrieval.ModeAuto
		}
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
			return resp.Chunks, &resp.Trace, resp.Attempts, took
		}
	}
	if br.Search == nil {
		return nil, nil, nil, 0
	}
	started := time.Now()
	results, err := br.Search.Search(augmentRetrievalQuery(req.Query, req.QuestionDate), search.SearchOpts{MaxResults: req.TopK})
	if err != nil {
		return nil, nil, nil, time.Since(started).Milliseconds()
	}
	chunks := make([]retrieval.RetrievedChunk, 0, len(results))
	for _, h := range results {
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
