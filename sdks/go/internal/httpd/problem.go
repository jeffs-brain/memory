// SPDX-License-Identifier: Apache-2.0

// Package httpd contains internal HTTP daemon helpers shared by cmd/memory
// subcommands. It defines the Problem+JSON encoder used across all
// handlers and the route registrar pattern lifted from the jeff server.
package httpd

import (
	"encoding/json"
	"errors"
	"net/http"

	"github.com/jeffs-brain/memory/go/brain"
)

// Problem is an RFC 7807-adjacent error body. Fields match the shape
// documented in spec/PROTOCOL.md.
type Problem struct {
	Status int    `json:"status"`
	Title  string `json:"title,omitempty"`
	Detail string `json:"detail,omitempty"`
	Code   string `json:"code,omitempty"`
}

// WriteProblem encodes p as application/problem+json and writes it.
func WriteProblem(w http.ResponseWriter, p Problem) {
	w.Header().Set("Content-Type", "application/problem+json")
	w.WriteHeader(p.Status)
	_ = json.NewEncoder(w).Encode(p)
}

// NotImplemented writes a 501 Problem+JSON body indicating the handler is
// not yet wired. Used by stub handlers that have not been ported.
func NotImplemented(w http.ResponseWriter) {
	WriteProblem(w, Problem{
		Status: http.StatusNotImplemented,
		Title:  "Not Implemented",
		Detail: "handler not wired",
		Code:   "not_implemented",
	})
}

// NotFound writes a 404 Problem+JSON.
func NotFound(w http.ResponseWriter, detail string) {
	WriteProblem(w, Problem{
		Status: http.StatusNotFound,
		Title:  "Not Found",
		Detail: detail,
		Code:   "not_found",
	})
}

// Conflict writes a 409 Problem+JSON.
func Conflict(w http.ResponseWriter, detail string) {
	WriteProblem(w, Problem{
		Status: http.StatusConflict,
		Title:  "Conflict",
		Detail: detail,
		Code:   "conflict",
	})
}

// ValidationError writes a 400 Problem+JSON with the validation_error
// code; used for body parsing failures and bad path shapes.
func ValidationError(w http.ResponseWriter, detail string) {
	WriteProblem(w, Problem{
		Status: http.StatusBadRequest,
		Title:  "Bad Request",
		Detail: detail,
		Code:   "validation_error",
	})
}

// PayloadTooLarge writes a 413 Problem+JSON.
func PayloadTooLarge(w http.ResponseWriter, detail string) {
	WriteProblem(w, Problem{
		Status: http.StatusRequestEntityTooLarge,
		Title:  "Payload Too Large",
		Detail: detail,
		Code:   "payload_too_large",
	})
}

// UnsupportedMediaType writes a 415 Problem+JSON.
func UnsupportedMediaType(w http.ResponseWriter, detail string) {
	WriteProblem(w, Problem{
		Status: http.StatusUnsupportedMediaType,
		Title:  "Unsupported Media Type",
		Detail: detail,
		Code:   "unsupported_media_type",
	})
}

// Unauthorized writes a 401 Problem+JSON.
func Unauthorized(w http.ResponseWriter, detail string) {
	WriteProblem(w, Problem{
		Status: http.StatusUnauthorized,
		Title:  "Unauthorized",
		Detail: detail,
		Code:   "unauthorized",
	})
}

// Forbidden writes a 403 Problem+JSON.
func Forbidden(w http.ResponseWriter, detail string) {
	WriteProblem(w, Problem{
		Status: http.StatusForbidden,
		Title:  "Forbidden",
		Detail: detail,
		Code:   "forbidden",
	})
}

// InternalError writes a 500 Problem+JSON with a generic title.
func InternalError(w http.ResponseWriter, detail string) {
	WriteProblem(w, Problem{
		Status: http.StatusInternalServerError,
		Title:  "Internal Server Error",
		Detail: detail,
		Code:   "internal_error",
	})
}

// WriteStoreError maps a brain.Store error to a Problem+JSON response
// using the canonical sentinel-to-code mapping. Returns true when err
// matched a known sentinel and the response has been written; callers
// should fall back to [InternalError] otherwise.
func WriteStoreError(w http.ResponseWriter, err error) bool {
	if err == nil {
		return false
	}
	switch {
	case errors.Is(err, brain.ErrNotFound):
		NotFound(w, err.Error())
	case errors.Is(err, brain.ErrConflict):
		Conflict(w, err.Error())
	case errors.Is(err, brain.ErrInvalidPath), errors.Is(err, brain.ErrValidation):
		ValidationError(w, err.Error())
	case errors.Is(err, brain.ErrPayloadTooLarge):
		PayloadTooLarge(w, err.Error())
	case errors.Is(err, brain.ErrUnsupportedMedia):
		UnsupportedMediaType(w, err.Error())
	case errors.Is(err, brain.ErrUnauthorized):
		Unauthorized(w, err.Error())
	case errors.Is(err, brain.ErrForbidden):
		Forbidden(w, err.Error())
	case errors.Is(err, brain.ErrReadOnly):
		WriteProblem(w, Problem{Status: http.StatusServiceUnavailable, Title: "Read Only", Detail: err.Error(), Code: "read_only"})
	default:
		return false
	}
	return true
}
