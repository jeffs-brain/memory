// SPDX-License-Identifier: Apache-2.0

package search

import (
	"reflect"
	"sort"
	"testing"
)

func TestTrigrams(t *testing.T) {
	cases := []struct {
		name string
		in   string
		want []string
	}{
		{
			name: "empty",
			in:   "",
			want: nil,
		},
		{
			name: "single word",
			in:   "bosch",
			want: []string{"$bo", "bos", "osc", "sch", "ch$"},
		},
		{
			name: "multi word",
			in:   "oude reimer",
			want: []string{
				"$ou", "oud", "ude", "de$",
				"$re", "rei", "eim", "ime", "mer", "er$",
			},
		},
		{
			name: "punctuation replaced with spaces",
			in:   "oude-reimer.md",
			want: []string{
				"$ou", "oud", "ude", "de$",
				"$re", "rei", "eim", "ime", "mer", "er$",
				"$md", "md$",
			},
		},
		{
			name: "case folded",
			in:   "BOSCH",
			want: []string{"$bo", "bos", "osc", "sch", "ch$"},
		},
		{
			name: "short word keeps boundary grams",
			in:   "ai",
			want: []string{"$ai", "ai$"},
		},
		{
			name: "digits preserved",
			in:   "v2 plan",
			want: []string{
				"$v2", "v2$",
				"$pl", "pla", "lan", "an$",
			},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := trigrams(tc.in)
			gotList := keys(got)
			sort.Strings(gotList)
			wantList := append([]string(nil), tc.want...)
			sort.Strings(wantList)
			if tc.want == nil {
				if len(gotList) != 0 {
					t.Errorf("trigrams(%q) = %v, want empty", tc.in, gotList)
				}
				return
			}
			if !reflect.DeepEqual(gotList, wantList) {
				t.Errorf("trigrams(%q)\n got  = %v\n want = %v", tc.in, gotList, wantList)
			}
		})
	}
}

func TestBuildTrigramIndex(t *testing.T) {
	paths := []string{
		"clients/oude-reimer.md",
		"clients/bosch.md",
		"projects/a-ware.md",
	}
	idx := BuildTrigramIndex(paths)

	if idx == nil {
		t.Fatal("BuildTrigramIndex returned nil")
	}
	if got := idx.Paths(); len(got) != 3 {
		t.Errorf("Paths() len = %d, want 3", len(got))
	}

	slugs, ok := idx.index["oud"]
	if !ok {
		t.Fatal(`index["oud"] missing`)
	}
	if !containsString(slugs, "clients/oude-reimer.md") {
		t.Errorf(`index["oud"] = %v, missing oude-reimer`, slugs)
	}

	bosSlugs, ok := idx.index["bos"]
	if !ok {
		t.Fatal(`index["bos"] missing`)
	}
	if !containsString(bosSlugs, "clients/bosch.md") {
		t.Errorf(`index["bos"] = %v, missing bosch`, bosSlugs)
	}

	dup := BuildTrigramIndex([]string{"clients/bosch.md", "clients/bosch.md"})
	if len(dup.Paths()) != 1 {
		t.Errorf("dup Paths() len = %d, want 1", len(dup.Paths()))
	}
}

func TestFuzzySearch_ExactMatch(t *testing.T) {
	paths := []string{
		"clients/oude-reimer.md",
		"clients/bosch.md",
	}
	idx := BuildTrigramIndex(paths)

	hits := idx.FuzzySearch("oude", 5)
	if len(hits) == 0 {
		t.Fatal("FuzzySearch('oude') returned no hits")
	}
	if hits[0].Path != "clients/oude-reimer.md" {
		t.Errorf("top hit = %q, want clients/oude-reimer.md", hits[0].Path)
	}
	if hits[0].Similarity <= 0 {
		t.Errorf("similarity = %v, want > 0", hits[0].Similarity)
	}
}

func TestFuzzySearch_Typo(t *testing.T) {
	paths := []string{
		"clients/oude-reimer.md",
		"clients/bosch.md",
		"projects/royal-aware.md",
	}
	idx := BuildTrigramIndex(paths)

	hits := idx.FuzzySearch("dude reimer", 5)
	if len(hits) == 0 {
		t.Fatal("FuzzySearch('dude reimer') returned no hits for the typo query")
	}
	if hits[0].Path != "clients/oude-reimer.md" {
		t.Errorf("top hit = %q, want clients/oude-reimer.md", hits[0].Path)
	}
	if hits[0].Similarity <= 0 {
		t.Errorf("similarity = %v, want > 0", hits[0].Similarity)
	}
	if hits[0].Similarity >= 1.0 {
		t.Errorf("similarity = %v, want < 1 for typo match", hits[0].Similarity)
	}
}

func TestFuzzySearch_NoMatch(t *testing.T) {
	paths := []string{
		"clients/oude-reimer.md",
		"clients/bosch.md",
	}
	idx := BuildTrigramIndex(paths)

	hits := idx.FuzzySearch("kubernetes", 5)
	if len(hits) != 0 {
		t.Errorf("FuzzySearch('kubernetes') = %+v, want empty", hits)
	}
}

func TestFuzzySearch_NilIndex(t *testing.T) {
	var idx *TrigramIndex
	if hits := idx.FuzzySearch("anything", 5); hits != nil {
		t.Errorf("nil-receiver FuzzySearch = %+v, want nil", hits)
	}
}

func keys(m map[string]struct{}) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	return out
}

func containsString(xs []string, want string) bool {
	for _, x := range xs {
		if x == want {
			return true
		}
	}
	return false
}
