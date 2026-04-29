// SPDX-License-Identifier: Apache-2.0

package search

import (
	"testing"
	"time"
)

func TestParseSortMode(t *testing.T) {
	cases := []struct {
		in   string
		want SortMode
	}{
		{"", SortRelevance},
		{"relevance", SortRelevance},
		{"recency", SortRecency},
		{"relevance_then_recency", SortRelevanceThenRecency},
		{"garbage", SortRelevance},
		{"  RECENCY  ", SortRecency},
	}
	for _, c := range cases {
		if got := ParseSortMode(c.in); got != c.want {
			t.Errorf("ParseSortMode(%q) = %v, want %v", c.in, got, c.want)
		}
	}
}

func TestApplySort_RelevanceTruncates(t *testing.T) {
	in := []SearchResult{
		{Path: "a"}, {Path: "b"}, {Path: "c"}, {Path: "d"},
	}
	out := applySort(in, SortRelevance, 2)
	if len(out) != 2 || out[0].Path != "a" || out[1].Path != "b" {
		t.Errorf("relevance truncate = %v", searchSortPaths(out))
	}
}

func TestApplySort_RecencyNewestFirst(t *testing.T) {
	t2023 := time.Date(2023, 5, 1, 0, 0, 0, 0, time.UTC)
	t2024 := time.Date(2024, 5, 1, 0, 0, 0, 0, time.UTC)
	t2025 := time.Date(2025, 5, 1, 0, 0, 0, 0, time.UTC)
	in := []SearchResult{
		{Path: "older", Modified: t2023},
		{Path: "newest", Modified: t2025},
		{Path: "middle", Modified: t2024},
	}
	out := applySort(in, SortRecency, 10)
	want := []string{"newest", "middle", "older"}
	if !searchEqualStrings(searchSortPaths(out), want) {
		t.Errorf("recency order = %v, want %v", searchSortPaths(out), want)
	}
}

func TestApplySort_UndatedFallToEnd(t *testing.T) {
	t2024 := time.Date(2024, 5, 1, 0, 0, 0, 0, time.UTC)
	in := []SearchResult{
		{Path: "undated"},
		{Path: "dated", Modified: t2024},
	}
	out := applySort(in, SortRecency, 10)
	if out[0].Path != "dated" || out[1].Path != "undated" {
		t.Errorf("undated not at end: %v", searchSortPaths(out))
	}
}

func TestApplySort_StableOnEqualDates(t *testing.T) {
	t1 := time.Date(2024, 5, 1, 0, 0, 0, 0, time.UTC)
	in := []SearchResult{
		{Path: "first", Modified: t1},
		{Path: "second", Modified: t1},
		{Path: "third", Modified: t1},
	}
	out := applySort(in, SortRecency, 10)
	want := []string{"first", "second", "third"}
	if !searchEqualStrings(searchSortPaths(out), want) {
		t.Errorf("stable sort violated: %v, want %v", searchSortPaths(out), want)
	}
}

func TestParseModifiedString(t *testing.T) {
	cases := []struct {
		in     string
		wantOK bool
		want   time.Time
	}{
		{"", false, time.Time{}},
		{"2024-04-15", true, time.Date(2024, 4, 15, 0, 0, 0, 0, time.UTC)},
		{"2024-04-15T10:30:00Z", true, time.Date(2024, 4, 15, 10, 30, 0, 0, time.UTC)},
		{"2024/04/15", true, time.Date(2024, 4, 15, 0, 0, 0, 0, time.UTC)},
		{"2024/04/15 (Mon) 10:30", true, time.Date(2024, 4, 15, 10, 30, 0, 0, time.UTC)},
		{"nope", false, time.Time{}},
	}
	for _, c := range cases {
		got, ok := parseModifiedString(c.in)
		if ok != c.wantOK {
			t.Errorf("parseModifiedString(%q) ok = %v, want %v", c.in, ok, c.wantOK)
		}
		if ok && !got.Equal(c.want) {
			t.Errorf("parseModifiedString(%q) time = %v, want %v", c.in, got, c.want)
		}
	}
}

func searchSortPaths(hits []SearchResult) []string {
	out := make([]string, len(hits))
	for i, h := range hits {
		out[i] = h.Path
	}
	return out
}

func searchEqualStrings(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
