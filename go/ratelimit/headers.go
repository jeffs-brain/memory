// SPDX-License-Identifier: Apache-2.0

package ratelimit

import (
	"net/http"
	"strconv"
	"strings"
	"time"
)

// headerNames maps the canonical header names checked when parsing
// rate-limit responses. Providers use varying casing; http.Header
// canonicalises on lookup.
var headerNames = struct {
	remaining  []string
	limit      []string
	resetAt    []string
	retryAfter []string
}{
	remaining:  []string{"X-Ratelimit-Remaining", "X-RateLimit-Remaining", "Ratelimit-Remaining"},
	limit:      []string{"X-Ratelimit-Limit", "X-RateLimit-Limit", "Ratelimit-Limit"},
	resetAt:    []string{"X-Ratelimit-Reset", "X-RateLimit-Reset", "Ratelimit-Reset"},
	retryAfter: []string{"Retry-After"},
}

// ParseHeaders extracts rate-limit metadata from an HTTP response.
// Missing or unparseable headers are silently ignored (zero values).
func ParseHeaders(resp *http.Response) Headers {
	if resp == nil {
		return Headers{}
	}
	return ParseHeaderMap(resp.Header)
}

// ParseHeaderMap extracts rate-limit metadata from a raw header map.
// Useful when callers have headers but not a full *http.Response.
func ParseHeaderMap(h http.Header) Headers {
	if h == nil {
		return Headers{}
	}
	return Headers{
		Remaining:  firstInt(h, headerNames.remaining),
		Limit:      firstInt(h, headerNames.limit),
		ResetAt:    firstTime(h, headerNames.resetAt),
		RetryAfter: firstDuration(h, headerNames.retryAfter),
	}
}

// firstInt returns the integer value of the first header that parses
// successfully from the candidate list.
func firstInt(h http.Header, candidates []string) int {
	for _, name := range candidates {
		v := h.Get(name)
		if v == "" {
			continue
		}
		n, err := strconv.Atoi(strings.TrimSpace(v))
		if err == nil {
			return n
		}
	}
	return 0
}

// firstTime returns the time value of the first header that parses
// successfully. Supports both Unix epoch seconds and RFC 1123 format.
func firstTime(h http.Header, candidates []string) time.Time {
	for _, name := range candidates {
		v := strings.TrimSpace(h.Get(name))
		if v == "" {
			continue
		}
		// Try Unix epoch seconds first (most common).
		if epoch, err := strconv.ParseInt(v, 10, 64); err == nil {
			return time.Unix(epoch, 0)
		}
		// Try RFC 1123 (used by some CDN providers).
		if t, err := time.Parse(time.RFC1123, v); err == nil {
			return t
		}
	}
	return time.Time{}
}

// firstDuration parses a Retry-After header as either seconds (integer)
// or an HTTP-date (RFC 1123). Returns zero on failure.
func firstDuration(h http.Header, candidates []string) time.Duration {
	for _, name := range candidates {
		v := strings.TrimSpace(h.Get(name))
		if v == "" {
			continue
		}
		// Integer seconds.
		if secs, err := strconv.Atoi(v); err == nil {
			return time.Duration(secs) * time.Second
		}
		// HTTP-date: compute duration from now.
		if t, err := time.Parse(time.RFC1123, v); err == nil {
			d := time.Until(t)
			if d > 0 {
				return d
			}
		}
	}
	return 0
}
