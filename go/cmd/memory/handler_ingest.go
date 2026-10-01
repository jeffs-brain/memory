// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"encoding/base64"
	"errors"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strings"

	"github.com/jeffs-brain/memory/go/internal/httpd"
	"github.com/jeffs-brain/memory/go/knowledge"
)

// ingestFileRequest either decodes inline bytes via ContentBase64 or,
// when the daemon has an ingest root configured, reads Path from the
// daemon's local filesystem inside that root.
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

// errPathIngestDisabled and errPathOutsideRoot are the two refusals of
// a server-side path read. Both map to 403.
var (
	errPathIngestDisabled = errors.New("server-side path ingest is disabled; send contentBase64 or start the daemon with an ingest root")
	errPathOutsideRoot    = errors.New("path resolves outside the configured ingest root")
)

// resolveIngestRoot validates the configured ingest root at startup and
// returns its absolute, symlink-free form. An empty root disables path
// ingest and returns "".
func resolveIngestRoot(root string) (string, error) {
	root = strings.TrimSpace(root)
	if root == "" {
		return "", nil
	}
	abs, err := filepath.Abs(root)
	if err != nil {
		return "", fmt.Errorf("ingest root: %w", err)
	}
	info, err := os.Stat(abs)
	if err != nil {
		return "", fmt.Errorf("ingest root: %w", err)
	}
	if !info.IsDir() {
		return "", fmt.Errorf("ingest root: %s is not a directory", abs)
	}
	// Keep the configured spelling: resolveIngestPath accepts requests
	// through it as well as through the resolved form, so a root reached
	// via a symlink (macOS /var, for one) still matches absolute paths.
	return abs, nil
}

// resolveIngestPath confines a requested server-side path to root. A
// relative path is taken relative to root. Containment is checked on
// the cleaned path first, so a path outside root is refused without
// revealing whether it exists, and again after resolving symlinks, so
// a link inside root cannot point the read elsewhere. Only regular
// files are accepted.
func resolveIngestPath(root, requested string) (string, error) {
	if root == "" {
		return "", errPathIngestDisabled
	}
	absRoot := filepath.Clean(root)
	realRoot, err := filepath.EvalSymlinks(absRoot)
	if err != nil {
		return "", fmt.Errorf("ingest root unavailable: %w", err)
	}
	candidate := requested
	if !filepath.IsAbs(candidate) {
		candidate = filepath.Join(absRoot, candidate)
	}
	candidate = filepath.Clean(candidate)
	if !withinRoot(absRoot, candidate) && !withinRoot(realRoot, candidate) {
		return "", errPathOutsideRoot
	}
	real, err := filepath.EvalSymlinks(candidate)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return "", fmt.Errorf("%w: file not found", knowledge.ErrInvalidContent)
		}
		return "", errPathOutsideRoot
	}
	if !withinRoot(realRoot, real) {
		return "", errPathOutsideRoot
	}
	info, err := os.Stat(real)
	if err != nil {
		return "", fmt.Errorf("%w: file not found", knowledge.ErrInvalidContent)
	}
	if !info.Mode().IsRegular() {
		return "", fmt.Errorf("%w: path is not a regular file", knowledge.ErrInvalidContent)
	}
	return real, nil
}

// withinRoot reports whether path is root or lies beneath it. Both
// arguments must be absolute and cleaned.
func withinRoot(root, path string) bool {
	rel, err := filepath.Rel(root, path)
	if err != nil || filepath.IsAbs(rel) {
		return false
	}
	return rel != ".." && !strings.HasPrefix(rel, ".."+string(filepath.Separator))
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
	switch {
	case req.ContentBase64 != "":
		raw, err := base64.StdEncoding.DecodeString(req.ContentBase64)
		if err != nil {
			httpd.ValidationError(w, fmt.Sprintf("invalid contentBase64: %v", err))
			return
		}
		ireq.Content = bytes.NewReader(raw)
	case strings.TrimSpace(req.Path) == "":
		httpd.ValidationError(w, "path or contentBase64 required")
		return
	default:
		resolved, err := resolveIngestPath(d.IngestRoot, req.Path)
		if err != nil {
			writeIngestError(w, err)
			return
		}
		ireq.Path = resolved
	}
	resp, err := br.Knowledge.Ingest(r.Context(), ireq)
	if err != nil {
		writeIngestError(w, err)
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
		writeIngestError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, resp)
}

// writeIngestError maps ingest failures onto the status codes documented
// in spec/PROTOCOL.md: refused paths are 403, bad input and blocked URLs
// are 400, upstream fetch failures are 502, and anything else is a
// generic 500.
func writeIngestError(w http.ResponseWriter, err error) {
	var fetchErr *knowledge.FetchError
	switch {
	case errors.Is(err, errPathIngestDisabled), errors.Is(err, errPathOutsideRoot):
		httpd.Forbidden(w, err.Error())
	case errors.Is(err, knowledge.ErrInvalidURL),
		errors.Is(err, knowledge.ErrBlockedAddress),
		errors.Is(err, knowledge.ErrInvalidContent):
		httpd.ValidationError(w, err.Error())
	case errors.As(err, &fetchErr):
		httpd.BadGateway(w, fetchErr.Error())
	default:
		if !httpd.WriteStoreError(w, err) {
			httpd.InternalError(w, err.Error())
		}
	}
}
