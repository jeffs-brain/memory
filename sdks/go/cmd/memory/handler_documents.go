// SPDX-License-Identifier: Apache-2.0

package main

import (
	"encoding/base64"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"strconv"
	"time"

	"github.com/jeffs-brain/memory/go/brain"
	"github.com/jeffs-brain/memory/go/internal/httpd"
)

// Body size caps from spec/PROTOCOL.md.
const (
	docBodyLimit   = 2 * 1024 * 1024
	batchBodyLimit = 8 * 1024 * 1024
	batchOpLimit   = 1024
)

// rawFileInfo is the wire shape returned by stat / list / batch list.
type rawFileInfo struct {
	Path  string    `json:"path"`
	Size  int64     `json:"size"`
	MTime time.Time `json:"mtime"`
	IsDir bool      `json:"is_dir"`
}

func toRawFileInfo(fi brain.FileInfo) rawFileInfo {
	return rawFileInfo{
		Path:  string(fi.Path),
		Size:  fi.Size,
		MTime: fi.ModTime.UTC(),
		IsDir: fi.IsDir,
	}
}

// handleDocRead implements GET /v1/brains/{brainId}/documents/read?path=...
func (d *Daemon) handleDocRead(w http.ResponseWriter, r *http.Request) {
	br := d.resolveBrain(w, r)
	if br == nil {
		return
	}
	p := brain.Path(r.URL.Query().Get("path"))
	if err := brain.ValidatePath(p); err != nil {
		httpd.ValidationError(w, err.Error())
		return
	}
	data, err := br.Store.Read(r.Context(), p)
	if err != nil {
		if !httpd.WriteStoreError(w, err) {
			httpd.InternalError(w, err.Error())
		}
		return
	}
	w.Header().Set("Content-Type", "application/octet-stream")
	w.Header().Set("Cache-Control", "no-store")
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write(data)
}

// handleDocStat implements GET /v1/brains/{brainId}/documents/stat
func (d *Daemon) handleDocStat(w http.ResponseWriter, r *http.Request) {
	br := d.resolveBrain(w, r)
	if br == nil {
		return
	}
	p := brain.Path(r.URL.Query().Get("path"))
	if err := brain.ValidatePath(p); err != nil {
		httpd.ValidationError(w, err.Error())
		return
	}
	info, err := br.Store.Stat(r.Context(), p)
	if err != nil {
		if !httpd.WriteStoreError(w, err) {
			httpd.InternalError(w, err.Error())
		}
		return
	}
	writeJSON(w, http.StatusOK, toRawFileInfo(info))
}

// handleDocList implements GET /v1/brains/{brainId}/documents (listing)
// and HEAD /v1/brains/{brainId}/documents (existence).
func (d *Daemon) handleDocListOrHead(w http.ResponseWriter, r *http.Request) {
	if r.Method == http.MethodHead {
		d.handleDocHead(w, r)
		return
	}
	br := d.resolveBrain(w, r)
	if br == nil {
		return
	}
	q := r.URL.Query()
	dir := brain.Path(q.Get("dir"))
	recursive, _ := strconv.ParseBool(q.Get("recursive"))
	includeGenerated, _ := strconv.ParseBool(q.Get("include_generated"))
	opts := brain.ListOpts{
		Recursive:        recursive,
		Glob:             q.Get("glob"),
		IncludeGenerated: includeGenerated,
	}
	entries, err := br.Store.List(r.Context(), dir, opts)
	if err != nil {
		if !httpd.WriteStoreError(w, err) {
			httpd.InternalError(w, err.Error())
		}
		return
	}
	items := make([]rawFileInfo, 0, len(entries))
	for _, e := range entries {
		items = append(items, toRawFileInfo(e))
	}
	writeJSON(w, http.StatusOK, map[string]any{"items": items})
}

// handleDocHead implements HEAD /v1/brains/{brainId}/documents
func (d *Daemon) handleDocHead(w http.ResponseWriter, r *http.Request) {
	br := d.resolveBrain(w, r)
	if br == nil {
		return
	}
	p := brain.Path(r.URL.Query().Get("path"))
	if err := brain.ValidatePath(p); err != nil {
		httpd.ValidationError(w, err.Error())
		return
	}
	exists, err := br.Store.Exists(r.Context(), p)
	if err != nil {
		if !httpd.WriteStoreError(w, err) {
			httpd.InternalError(w, err.Error())
		}
		return
	}
	if !exists {
		w.WriteHeader(http.StatusNotFound)
		return
	}
	w.Header().Set("Cache-Control", "no-store")
	w.WriteHeader(http.StatusOK)
}

// handleDocPut implements PUT /v1/brains/{brainId}/documents
func (d *Daemon) handleDocPut(w http.ResponseWriter, r *http.Request) {
	br := d.resolveBrain(w, r)
	if br == nil {
		return
	}
	p := brain.Path(r.URL.Query().Get("path"))
	if err := brain.ValidatePath(p); err != nil {
		httpd.ValidationError(w, err.Error())
		return
	}
	body, err := readLimitedBody(r, docBodyLimit)
	if err != nil {
		if errors.Is(err, errBodyTooLarge) {
			httpd.PayloadTooLarge(w, "PUT /documents body exceeds 2 MiB")
			return
		}
		httpd.ValidationError(w, err.Error())
		return
	}
	if err := br.Store.Write(r.Context(), p, body); err != nil {
		if !httpd.WriteStoreError(w, err) {
			httpd.InternalError(w, err.Error())
		}
		return
	}
	w.WriteHeader(http.StatusNoContent)
}

// handleDocAppend implements POST /v1/brains/{brainId}/documents/append
func (d *Daemon) handleDocAppend(w http.ResponseWriter, r *http.Request) {
	br := d.resolveBrain(w, r)
	if br == nil {
		return
	}
	p := brain.Path(r.URL.Query().Get("path"))
	if err := brain.ValidatePath(p); err != nil {
		httpd.ValidationError(w, err.Error())
		return
	}
	body, err := readLimitedBody(r, docBodyLimit)
	if err != nil {
		if errors.Is(err, errBodyTooLarge) {
			httpd.PayloadTooLarge(w, "POST /documents/append body exceeds 2 MiB")
			return
		}
		httpd.ValidationError(w, err.Error())
		return
	}
	if err := br.Store.Append(r.Context(), p, body); err != nil {
		if !httpd.WriteStoreError(w, err) {
			httpd.InternalError(w, err.Error())
		}
		return
	}
	w.WriteHeader(http.StatusNoContent)
}

// handleDocDelete implements DELETE /v1/brains/{brainId}/documents
func (d *Daemon) handleDocDelete(w http.ResponseWriter, r *http.Request) {
	br := d.resolveBrain(w, r)
	if br == nil {
		return
	}
	p := brain.Path(r.URL.Query().Get("path"))
	if err := brain.ValidatePath(p); err != nil {
		httpd.ValidationError(w, err.Error())
		return
	}
	if err := br.Store.Delete(r.Context(), p); err != nil {
		if !httpd.WriteStoreError(w, err) {
			httpd.InternalError(w, err.Error())
		}
		return
	}
	w.WriteHeader(http.StatusNoContent)
}

// handleDocRename implements POST /v1/brains/{brainId}/documents/rename
func (d *Daemon) handleDocRename(w http.ResponseWriter, r *http.Request) {
	br := d.resolveBrain(w, r)
	if br == nil {
		return
	}
	var body struct {
		From string `json:"from"`
		To   string `json:"to"`
	}
	if err := decodeJSONBody(r, &body, 64*1024); err != nil {
		httpd.ValidationError(w, err.Error())
		return
	}
	src := brain.Path(body.From)
	dst := brain.Path(body.To)
	if err := brain.ValidatePath(src); err != nil {
		httpd.ValidationError(w, err.Error())
		return
	}
	if err := brain.ValidatePath(dst); err != nil {
		httpd.ValidationError(w, err.Error())
		return
	}
	if err := br.Store.Rename(r.Context(), src, dst); err != nil {
		if !httpd.WriteStoreError(w, err) {
			httpd.InternalError(w, err.Error())
		}
		return
	}
	w.WriteHeader(http.StatusNoContent)
}

// batchOp matches the wire shape of a single op in a batch request.
type batchOp struct {
	Type          string `json:"type"`
	Path          string `json:"path"`
	To            string `json:"to,omitempty"`
	ContentBase64 string `json:"content_base64,omitempty"`
}

// batchRequest matches the wire shape of POST /documents/batch-ops.
type batchRequest struct {
	Reason  string    `json:"reason,omitempty"`
	Message string    `json:"message,omitempty"`
	Author  string    `json:"author,omitempty"`
	Email   string    `json:"email,omitempty"`
	Ops     []batchOp `json:"ops"`
}

// handleDocBatch implements POST /v1/brains/{brainId}/documents/batch-ops.
func (d *Daemon) handleDocBatch(w http.ResponseWriter, r *http.Request) {
	br := d.resolveBrain(w, r)
	if br == nil {
		return
	}

	body, err := io.ReadAll(io.LimitReader(r.Body, batchBodyLimit*2))
	if err != nil {
		httpd.ValidationError(w, err.Error())
		return
	}
	var req batchRequest
	if err := json.Unmarshal(body, &req); err != nil {
		httpd.ValidationError(w, "invalid JSON: "+err.Error())
		return
	}
	if len(req.Ops) > batchOpLimit {
		httpd.PayloadTooLarge(w, "ops length exceeds 1024")
		return
	}

	// Compute decoded payload size before applying.
	decodedSize := 0
	decoded := make([][]byte, len(req.Ops))
	for i, op := range req.Ops {
		if op.ContentBase64 == "" {
			continue
		}
		raw, err := base64.StdEncoding.DecodeString(op.ContentBase64)
		if err != nil {
			httpd.ValidationError(w, "invalid base64 at op "+strconv.Itoa(i))
			return
		}
		decodedSize += len(raw)
		if decodedSize > batchBodyLimit {
			httpd.PayloadTooLarge(w, "batch payload exceeds 8 MiB after decode")
			return
		}
		decoded[i] = raw
	}

	opts := brain.BatchOptions{
		Reason:  req.Reason,
		Message: req.Message,
		Author:  req.Author,
		Email:   req.Email,
	}
	committed := 0
	err = br.Store.Batch(r.Context(), opts, func(b brain.Batch) error {
		for i, op := range req.Ops {
			p := brain.Path(op.Path)
			switch op.Type {
			case "write":
				if err := b.Write(r.Context(), p, decoded[i]); err != nil {
					return err
				}
			case "append":
				if err := b.Append(r.Context(), p, decoded[i]); err != nil {
					return err
				}
			case "delete":
				if err := b.Delete(r.Context(), p); err != nil {
					return err
				}
			case "rename":
				if err := b.Rename(r.Context(), p, brain.Path(op.To)); err != nil {
					return err
				}
			default:
				return errors.New("unknown op type: " + op.Type)
			}
			committed++
		}
		return nil
	})
	if err != nil {
		if !httpd.WriteStoreError(w, err) {
			httpd.ValidationError(w, err.Error())
		}
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"committed": committed})
}
