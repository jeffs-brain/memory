// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"encoding/base64"
	"fmt"
	"net/http"

	"github.com/jeffs-brain/memory/go/internal/httpd"
	"github.com/jeffs-brain/memory/go/knowledge"
)

// ingestFileRequest either decodes inline bytes via ContentBase64 or
// reads Path from the daemon's local filesystem.
type ingestFileRequest struct {
	Path          string   `json:"path"`
	ContentType   string   `json:"contentType,omitempty"`
	Title         string   `json:"title,omitempty"`
	Tags          []string `json:"tags,omitempty"`
	ContentBase64 string   `json:"contentBase64,omitempty"`
}

type ingestURLRequest struct {
	URL string `json:"url"`
}

func (d *Daemon) handleIngestFile(w http.ResponseWriter, r *http.Request) {
	br := d.resolveBrain(w, r)
	if br == nil {
		return
	}
	var req ingestFileRequest
	if err := decodeJSONBody(r, &req, batchBodyLimit); err != nil {
		httpd.ValidationError(w, err.Error())
		return
	}
	ireq := knowledge.IngestRequest{
		BrainID:     br.ID,
		Path:        req.Path,
		ContentType: req.ContentType,
		Title:       req.Title,
		Tags:        req.Tags,
	}
	if req.ContentBase64 != "" {
		raw, err := base64.StdEncoding.DecodeString(req.ContentBase64)
		if err != nil {
			httpd.ValidationError(w, fmt.Sprintf("invalid contentBase64: %v", err))
			return
		}
		ireq.Content = bytes.NewReader(raw)
	}
	resp, err := br.Knowledge.Ingest(r.Context(), ireq)
	if err != nil {
		httpd.InternalError(w, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, resp)
}

func (d *Daemon) handleIngestURL(w http.ResponseWriter, r *http.Request) {
	br := d.resolveBrain(w, r)
	if br == nil {
		return
	}
	var req ingestURLRequest
	if err := decodeJSONBody(r, &req, 64*1024); err != nil {
		httpd.ValidationError(w, err.Error())
		return
	}
	if req.URL == "" {
		httpd.ValidationError(w, "url required")
		return
	}
	resp, err := br.Knowledge.IngestURL(r.Context(), req.URL)
	if err != nil {
		httpd.InternalError(w, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, resp)
}
