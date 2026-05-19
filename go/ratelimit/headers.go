// SPDX-License-Identifier: Apache-2.0

package ratelimit

import (
	"net/http"
	"strconv"
	"strings"
	"time"
)

// HeaderNames configures which HTTP header names to check when parsing
// rate-limit responses. The zero value uses DefaultHeaderNames.
type HeaderNames struct {
	Remaining  []string
	Limit      []string
	ResetAt    []string
	RetryAfter []string
}

// DefaultHeaderNames returns the standard header names checked when
// parsing rate-limit responses, covering OpenAI, Anthropic, and other
// common LLM provider conventions.
func DefaultHeaderNames() HeaderNames {
	return HeaderNames{
		Remaining: []string{
			"X-Ratelimit-Remaining",
			"X-RateLimit-Remaining",
			"Ratelimit-Remaining",
			"X-RateLimit-Remaining-Requests",
			"X-RateLimit-Remaining-Tokens",
		},
		Limit: []string{
			"X-Ratelimit-Limit",
			"X-RateLimit-Limit",
			"Ratelimit-Limit",
			"X-RateLimit-Limit-Requests",
			"X-RateLimit-Limit-Tokens",
		},
		ResetAt: []string{
			"X-Ratelimit-Reset",
			"X-RateLimit-Reset",
			"Ratelimit-Reset",
			"X-RateLimit-Reset-Requests",
			"X-RateLimit-Reset-Tokens",
		},
		RetryAfter: []string{
			"Retry-After",
		},
	}
}

// defaultNames is the package-level default used when no custom names
// are provided.
var defaultNames = DefaultHeaderNames()

// resolve returns the provided names if non-empty, or the defaults.
func (h HeaderNames) resolve() HeaderNames {
	if len(h.Remaining) == 0 {
		h.Remaining = defaultNames.Remaining
	}
	if len(h.Limit) == 0 {
		h.Limit = defaultNames.Limit
	}
	if len(h.ResetAt) == 0 {
		h.ResetAt = defaultNames.ResetAt
	}
	if len(h.RetryAfter) == 0 {
		h.RetryAfter = defaultNames.RetryAfter
	}
	return h
}

// ParseHeaders extracts rate-limit metadata from an HTTP response.
// Missing or unparseable headers are silently ignored (zero values).
func ParseHeaders(resp *http.Response) Headers {
	if resp == nil {
		return Headers{}
	}
	return ParseHeaderMap(resp.Header)
}

// ParseHeaderMap extracts rate-limit metadata from a raw header map
// using the default header names.
func ParseHeaderMap(h http.Header) Headers {
	return ParseHeaderMapWith(h, HeaderNames{})
}

// ParseHeaderMapWith extracts rate-limit metadata from a raw header map
// using custom header names. Zero-value fields in names fall back to
// the defaults.
func ParseHeaderMapWith(h http.Header, names HeaderNames) Headers {
	if h == nil {
		return Headers{}
	}
	n := names.resolve()
	return Headers{
		Remaining:  firstInt(h, n.Remaining),
		Limit:      firstInt(h, n.Limit),
		ResetAt:    firstTime(h, n.ResetAt),
		RetryAfter: firstDuration(h, n.RetryAfter),
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
