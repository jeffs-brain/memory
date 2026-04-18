// SPDX-License-Identifier: Apache-2.0

package search

import (
	"sort"
	"strings"
	"unicode"
)

// TrigramIndex is a lazy in-memory index mapping trigrams to article
// slugs. Built from the set of wiki paths the first time a fuzzy
// lookup is requested. Cheap because we only store slugs, not content.
type TrigramIndex struct {
	// index maps a trigram to the list of slugs containing it.
	index map[string][]string
	// paths is the full slug set the index was built from, kept for
	// result shaping and introspection.
	paths []string
	// pathTrigrams caches the trigram set for each slug so we compute
	// it once at build time and reuse on every query.
	pathTrigrams map[string]map[string]struct{}
}

// TrigramHit is a single fuzzy match: the slug that matched plus the
// Jaccard similarity against the query text.
type TrigramHit struct {
	Path       string
	Similarity float64 // 0-1, Jaccard over trigram sets
}

// defaultFuzzyThreshold is the minimum Jaccard similarity a slug must
// clear before FuzzySearch will return it. 0.3 keeps single-character
// typos (Dude to Oude) inside the net while rejecting unrelated slugs
// that share only one or two trigrams.
const defaultFuzzyThreshold = 0.3

// BuildTrigramIndex walks every supplied path and builds the trigram
// map. "bosch" produces {"$bo", "bos", "osc", "sch", "ch$"} where $ is
// a boundary marker that biases matches toward word start and end.
func BuildTrigramIndex(paths []string) *TrigramIndex {
	idx := &TrigramIndex{
		index:        make(map[string][]string),
		paths:        make([]string, 0, len(paths)),
		pathTrigrams: make(map[string]map[string]struct{}, len(paths)),
	}
	seenPath := make(map[string]bool, len(paths))
	for _, p := range paths {
		if p == "" || seenPath[p] {
			continue
		}
		seenPath[p] = true
		idx.paths = append(idx.paths, p)

		tri := trigrams(slugText(p))
		idx.pathTrigrams[p] = tri
		for gram := range tri {
			idx.index[gram] = append(idx.index[gram], p)
		}
	}
	return idx
}

// Paths returns the slug set the index was built from.
func (t *TrigramIndex) Paths() []string {
	if t == nil {
		return nil
	}
	out := make([]string, len(t.paths))
	copy(out, t.paths)
	return out
}

// FuzzySearch returns the top-k slugs ranked by trigram overlap with
// query, filtered to a minimum similarity threshold. Uses Jaccard
// similarity: |Q intersect D| / |Q union D|. The threshold defaults
// to 0.3; slugs scoring below it are discarded.
func (t *TrigramIndex) FuzzySearch(query string, k int) []TrigramHit {
	return t.fuzzySearchWithThreshold(query, k, defaultFuzzyThreshold)
}

// fuzzySearchWithThreshold is the underlying implementation so tests
// can exercise alternative thresholds without re-implementing the
// ranking logic.
func (t *TrigramIndex) fuzzySearchWithThreshold(query string, k int, threshold float64) []TrigramHit {
	if t == nil || len(t.index) == 0 {
		return nil
	}
	if k <= 0 {
		k = 10
	}

	queryGrams := trigrams(query)
	if len(queryGrams) == 0 {
		return nil
	}

	// Collect candidate slugs that share at least one trigram with the
	// query. We only score candidates so the full corpus is not walked
	// for every fuzzy lookup.
	candidates := make(map[string]struct{})
	for gram := range queryGrams {
		for _, p := range t.index[gram] {
			candidates[p] = struct{}{}
		}
	}
	if len(candidates) == 0 {
		return nil
	}

	hits := make([]TrigramHit, 0, len(candidates))
	for p := range candidates {
		pathGrams := t.pathTrigrams[p]
		if len(pathGrams) == 0 {
			continue
		}
		sim := jaccard(queryGrams, pathGrams)
		if sim < threshold {
			continue
		}
		hits = append(hits, TrigramHit{Path: p, Similarity: sim})
	}

	sort.SliceStable(hits, func(i, j int) bool {
		if hits[i].Similarity != hits[j].Similarity {
			return hits[i].Similarity > hits[j].Similarity
		}
		return hits[i].Path < hits[j].Path
	})

	if len(hits) > k {
		hits = hits[:k]
	}
	return hits
}

// jaccard computes |A intersect B| / |A union B| for two trigram
// sets.
func jaccard(a, b map[string]struct{}) float64 {
	if len(a) == 0 || len(b) == 0 {
		return 0
	}
	var intersection int
	smaller, larger := a, b
	if len(b) < len(a) {
		smaller, larger = b, a
	}
	for g := range smaller {
		if _, ok := larger[g]; ok {
			intersection++
		}
	}
	union := len(a) + len(b) - intersection
	if union == 0 {
		return 0
	}
	return float64(intersection) / float64(union)
}

// trigrams returns the set of boundary-padded 3-grams for text,
// lowercased, with non-alphanumeric characters replaced by spaces.
// Each whitespace-separated word is padded with a single boundary
// marker ($) at the start and end so matches inside short slugs keep
// their word-boundary signal.
func trigrams(text string) map[string]struct{} {
	out := make(map[string]struct{})
	if text == "" {
		return out
	}

	var b strings.Builder
	for _, r := range strings.ToLower(text) {
		switch {
		case unicode.IsLetter(r), unicode.IsDigit(r):
			b.WriteRune(r)
		default:
			b.WriteRune(' ')
		}
	}
	cleaned := b.String()

	for _, word := range strings.Fields(cleaned) {
		padded := "$" + word + "$"
		runes := []rune(padded)
		if len(runes) < 3 {
			continue
		}
		for i := 0; i+3 <= len(runes); i++ {
			out[string(runes[i:i+3])] = struct{}{}
		}
	}

	return out
}

// slugText converts a wiki-relative path to a whitespace-separated
// token string suitable for trigram generation. Only the filename
// stem is considered so short queries that match the slug itself are
// not drowned out by parent-directory noise:
// "clients/oude-reimer.md" becomes "oude reimer".
func slugText(p string) string {
	s := strings.ToLower(p)
	if i := strings.LastIndex(s, "/"); i >= 0 {
		s = s[i+1:]
	}
	s = strings.TrimSuffix(s, ".md")
	var b strings.Builder
	for _, r := range s {
		switch {
		case unicode.IsLetter(r), unicode.IsDigit(r):
			b.WriteRune(r)
		default:
			b.WriteRune(' ')
		}
	}
	return strings.Join(strings.Fields(b.String()), " ")
}
