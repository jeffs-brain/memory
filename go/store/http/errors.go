// SPDX-License-Identifier: Apache-2.0

package http

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"

	"github.com/jeffs-brain/memory/go/brain"
)

// problem is the Problem+JSON payload emitted by the reference server. All
// fields are optional per RFC 7807; an empty body is valid.
type problem struct {
	Status int    `json:"status,omitempty"`
	Title  string `json:"title,omitempty"`
	Detail string `json:"detail,omitempty"`
	Code   string `json:"code,omitempty"`
}

// codeToError maps the 11-code Problem+JSON vocabulary from spec/PROTOCOL.md
// to the brain package sentinels. Unknown codes fall back to status-based
// dispatch handled by mapStatus.
var codeToError = map[string]error{
	"not_found":              brain.ErrNotFound,
	"conflict":               brain.ErrConflict,
	"unauthorized":           brain.ErrUnauthorized,
	"forbidden":              brain.ErrForbidden,
	"validation_error":       brain.ErrValidation,
	"payload_too_large":      brain.ErrPayloadTooLarge,
	"unsupported_media_type": brain.ErrUnsupportedMedia,
	"rate_limited":           brain.ErrRateLimited,
	"internal_error":         brain.ErrInternal,
	"bad_gateway":            brain.ErrBadGateway,
	"timeout":                brain.ErrTimeout,
}

// mapStatus returns the sentinel that matches an HTTP status when the
// Problem+JSON body carries no recognised code. Codes that are truly novel
// surface as ErrStore wrapping the raw status.
func mapStatus(status int) error {
	switch status {
	case http.StatusNotFound:
		return brain.ErrNotFound
	case http.StatusBadRequest:
		return brain.ErrInvalidPath
	case http.StatusConflict:
		return brain.ErrConflict
	case http.StatusUnauthorized:
		return brain.ErrUnauthorized
	case http.StatusForbidden:
		return brain.ErrForbidden
	case http.StatusRequestEntityTooLarge:
		return brain.ErrPayloadTooLarge
	case http.StatusUnsupportedMediaType:
		return brain.ErrUnsupportedMedia
	case http.StatusTooManyRequests:
		return brain.ErrRateLimited
	case http.StatusInternalServerError:
		return brain.ErrInternal
	case http.StatusBadGateway:
		return brain.ErrBadGateway
	case http.StatusGatewayTimeout:
		return brain.ErrTimeout
	}
	return ErrStore
}

// ErrStore is the catch-all sentinel wrapped by responseError when no more
// specific sentinel applies. Callers may test with errors.Is(err, ErrStore)
// for generic HTTP failures.
var ErrStore = errors.New("store/http: remote error")

// StoreError enriches a sentinel with the parsed Problem+JSON payload so
// callers that care about title/detail/code can drill in via errors.As.
type StoreError struct {
	Status  int
	Title   string
	Detail  string
	Code    string
	Wrapped error
}

// Error formats the status and title for human consumption.
func (e *StoreError) Error() string {
	if e == nil {
		return "<nil>"
	}
	base := fmt.Sprintf("store/http: %d %s", e.Status, e.Title)
	if e.Detail != "" {
		return base + ": " + e.Detail
	}
	return base
}

// Unwrap exposes the sentinel so errors.Is works.
func (e *StoreError) Unwrap() error { return e.Wrapped }

// parseProblem reads the response body as Problem+JSON, tolerating empty
// or malformed payloads. The returned problem always carries at least the
// HTTP status.
func parseProblem(resp *http.Response) problem {
	p := problem{Status: resp.StatusCode, Title: http.StatusText(resp.StatusCode)}
	if resp.Body == nil {
		return p
	}
	body, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil || len(body) == 0 {
		return p
	}
	var parsed problem
	if err := json.Unmarshal(body, &parsed); err != nil {
		return p
	}
	if parsed.Status != 0 {
		p.Status = parsed.Status
	}
	if parsed.Title != "" {
		p.Title = parsed.Title
	}
	if parsed.Detail != "" {
		p.Detail = parsed.Detail
	}
	if parsed.Code != "" {
		p.Code = parsed.Code
	}
	return p
}

// responseError builds the typed error for a non-2xx response. Code-based
// dispatch wins when the server emits a recognised identifier; otherwise
// status is the fallback. The returned error always unwraps to a brain
// sentinel so callers can use errors.Is.
func responseError(resp *http.Response) error {
	prob := parseProblem(resp)
	var wrapped error
	if prob.Code != "" {
		if s, ok := codeToError[prob.Code]; ok {
			wrapped = s
		}
	}
	if wrapped == nil {
		wrapped = mapStatus(prob.Status)
	}
	return &StoreError{
		Status:  prob.Status,
		Title:   prob.Title,
		Detail:  prob.Detail,
		Code:    prob.Code,
		Wrapped: wrapped,
	}
}
