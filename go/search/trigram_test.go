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
			in:   "zenco",
			want: []string{"$ze", "zen", "enc", "nco", "co$"},
		},
		{
			name: "multi word",
			in:   "mill brook",
			want: []string{
				"$mi", "mil", "ill", "ll$",
				"$br", "bro", "roo", "ook", "ok$",
			},
		},
		{
			name: "punctuation replaced with spaces",
			in:   "mill-brook.md",
			want: []string{
				"$mi", "mil", "ill", "ll$",
				"$br", "bro", "roo", "ook", "ok$",
				"$md", "md$",
			},
		},
		{
			name: "case folded",
			in:   "ZENCO",
			want: []string{"$ze", "zen", "enc", "nco", "co$"},
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
		"clients/mill-brook.md",
		"clients/zenco.md",
		"projects/e-volt.md",
	}
	idx := BuildTrigramIndex(paths)

	if idx == nil {
		t.Fatal("BuildTrigramIndex returned nil")
	}
	if got := idx.Paths(); len(got) != 3 {
		t.Errorf("Paths() len = %d, want 3", len(got))
	}

	slugs, ok := idx.index["mil"]
	if !ok {
		t.Fatal(`index["mil"] missing`)
	}
	if !containsString(slugs, "clients/mill-brook.md") {
		t.Errorf(`index["mil"] = %v, missing mill-brook`, slugs)
	}

	zenSlugs, ok := idx.index["zen"]
	if !ok {
		t.Fatal(`index["zen"] missing`)
	}
	if !containsString(zenSlugs, "clients/zenco.md") {
		t.Errorf(`index["zen"] = %v, missing zenco`, zenSlugs)
	}

	dup := BuildTrigramIndex([]string{"clients/zenco.md", "clients/zenco.md"})
	if len(dup.Paths()) != 1 {
		t.Errorf("dup Paths() len = %d, want 1", len(dup.Paths()))
	}
}

func TestFuzzySearch_ExactMatch(t *testing.T) {
	paths := []string{
		"clients/mill-brook.md",
		"clients/zenco.md",
	}
	idx := BuildTrigramIndex(paths)

	hits := idx.FuzzySearch("mill", 5)
	if len(hits) == 0 {
		t.Fatal("FuzzySearch('mill') returned no hits")
	}
	if hits[0].Path != "clients/mill-brook.md" {
		t.Errorf("top hit = %q, want clients/mill-brook.md", hits[0].Path)
	}
	if hits[0].Similarity <= 0 {
		t.Errorf("similarity = %v, want > 0", hits[0].Similarity)
	}
}

func TestFuzzySearch_Typo(t *testing.T) {
	paths := []string{
		"clients/mill-brook.md",
		"clients/zenco.md",
		"projects/nova-evolt.md",
	}
	idx := BuildTrigramIndex(paths)

	hits := idx.FuzzySearch("hill brook", 5)
	if len(hits) == 0 {
		t.Fatal("FuzzySearch('hill brook') returned no hits for the typo query")
	}
	if hits[0].Path != "clients/mill-brook.md" {
		t.Errorf("top hit = %q, want clients/mill-brook.md", hits[0].Path)
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
		"clients/mill-brook.md",
		"clients/zenco.md",
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
