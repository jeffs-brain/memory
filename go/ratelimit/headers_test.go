// SPDX-License-Identifier: Apache-2.0

package ratelimit

import (
	"net/http"
	"testing"
	"time"
)

func TestParseHeaderMap_Remaining(t *testing.T) {
	cases := []struct {
		name   string
		header string
		value  string
		want   int
	}{
		{"standard casing", "X-Ratelimit-Remaining", "42", 42},
		{"mixed casing", "X-RateLimit-Remaining", "100", 100},
		{"no prefix", "Ratelimit-Remaining", "7", 7},
		{"remaining-requests variant", "X-RateLimit-Remaining-Requests", "33", 33},
		{"remaining-tokens variant", "X-RateLimit-Remaining-Tokens", "55", 55},
		{"empty value", "X-Ratelimit-Remaining", "", 0},
		{"non-numeric", "X-Ratelimit-Remaining", "abc", 0},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			h := http.Header{}
			h.Set(tc.header, tc.value)
			got := ParseHeaderMap(h)
			if got.Remaining != tc.want {
				t.Fatalf("Remaining = %d, want %d", got.Remaining, tc.want)
			}
		})
	}
}

func TestParseHeaderMap_Limit(t *testing.T) {
	h := http.Header{}
	h.Set("X-Ratelimit-Limit", "1000")
	got := ParseHeaderMap(h)
	if got.Limit != 1000 {
		t.Fatalf("Limit = %d, want 1000", got.Limit)
	}
}

func TestParseHeaderMap_LimitVariants(t *testing.T) {
	cases := []struct {
		name   string
		header string
		value  string
		want   int
	}{
		{"limit-requests", "X-RateLimit-Limit-Requests", "200", 200},
		{"limit-tokens", "X-RateLimit-Limit-Tokens", "500", 500},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			h := http.Header{}
			h.Set(tc.header, tc.value)
			got := ParseHeaderMap(h)
			if got.Limit != tc.want {
				t.Fatalf("Limit = %d, want %d", got.Limit, tc.want)
			}
		})
	}
}

func TestParseHeaderMap_ResetAt_Unix(t *testing.T) {
	h := http.Header{}
	h.Set("X-Ratelimit-Reset", "1700000000")
	got := ParseHeaderMap(h)
	want := time.Unix(1700000000, 0)
	if !got.ResetAt.Equal(want) {
		t.Fatalf("ResetAt = %v, want %v", got.ResetAt, want)
	}
}

func TestParseHeaderMap_ResetAt_RFC1123(t *testing.T) {
	h := http.Header{}
	ts := "Mon, 02 Jan 2006 15:04:05 MST"
	h.Set("X-Ratelimit-Reset", ts)
	got := ParseHeaderMap(h)
	want, _ := time.Parse(time.RFC1123, ts)
	if !got.ResetAt.Equal(want) {
		t.Fatalf("ResetAt = %v, want %v", got.ResetAt, want)
	}
}

func TestParseHeaderMap_RetryAfter_Seconds(t *testing.T) {
	h := http.Header{}
	h.Set("Retry-After", "30")
	got := ParseHeaderMap(h)
	if got.RetryAfter != 30*time.Second {
		t.Fatalf("RetryAfter = %v, want 30s", got.RetryAfter)
	}
}

func TestParseHeaderMap_NilHeader(t *testing.T) {
	got := ParseHeaderMap(nil)
	if got.Remaining != 0 || got.Limit != 0 || got.RetryAfter != 0 {
		t.Fatalf("expected zero values for nil header, got %+v", got)
	}
}

func TestParseHeaders_NilResponse(t *testing.T) {
	got := ParseHeaders(nil)
	if got.Remaining != 0 {
		t.Fatalf("expected zero for nil response, got %+v", got)
	}
}

func TestParseHeaders_FromResponse(t *testing.T) {
	resp := &http.Response{
		Header: http.Header{},
	}
	resp.Header.Set("X-Ratelimit-Remaining", "55")
	resp.Header.Set("Retry-After", "10")

	got := ParseHeaders(resp)
	if got.Remaining != 55 {
		t.Fatalf("Remaining = %d, want 55", got.Remaining)
	}
	if got.RetryAfter != 10*time.Second {
		t.Fatalf("RetryAfter = %v, want 10s", got.RetryAfter)
	}
}

func TestParseHeaderMap_WhitespaceHandling(t *testing.T) {
	h := http.Header{}
	h.Set("X-Ratelimit-Remaining", "  25  ")
	h.Set("Retry-After", " 5 ")

	got := ParseHeaderMap(h)
	if got.Remaining != 25 {
		t.Fatalf("Remaining = %d, want 25", got.Remaining)
	}
	if got.RetryAfter != 5*time.Second {
		t.Fatalf("RetryAfter = %v, want 5s", got.RetryAfter)
	}
}

func TestParseHeaderMapWith_CustomNames(t *testing.T) {
	h := http.Header{}
	h.Set("My-Custom-Remaining", "42")
	h.Set("My-Custom-Limit", "100")

	names := HeaderNames{
		Remaining: []string{"My-Custom-Remaining"},
		Limit:     []string{"My-Custom-Limit"},
	}
	got := ParseHeaderMapWith(h, names)
	if got.Remaining != 42 {
		t.Fatalf("Remaining = %d, want 42", got.Remaining)
	}
	if got.Limit != 100 {
		t.Fatalf("Limit = %d, want 100", got.Limit)
	}
}
