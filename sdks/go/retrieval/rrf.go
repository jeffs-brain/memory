// SPDX-License-Identifier: Apache-2.0

package retrieval

import "sort"

// RRFDefaultK is the canonical constant in the Reciprocal Rank Fusion
// formula: rrf_score(doc) = sum over lists of 1 / (k + rank(doc)).
// Cormack, Clarke, Buettcher (SIGIR 2009) tested k in [10, 1000] and
// found k = 60 robust across TREC tracks; the spec pins it at 60.
const RRFDefaultK = 60

// rrfCandidate is the shape the fusion step accepts. It mirrors the
// TypeScript RRFCandidate so callers that port tests from the TS
// reference can keep the same field layout.
type rrfCandidate struct {
	id               string
	path             string
	title            string
	summary          string
	content          string
	bm25Rank         int
	haveBM25Rank     bool
	vectorSimilarity float64
	haveVectorSim    bool
}

// reciprocalRankFusion fuses an arbitrary number of ranked lists into
// a single ranking. Order of the input lists does not change the
// fused score; it only controls which list seeds the metadata for a
// candidate seen in multiple lists. Later lists fill in blanks but
// never overwrite non-empty fields (the one-way merge rule from the
// spec).
//
// Ties on score are broken by path ascending so the output is stable
// across runs with identical inputs.
func reciprocalRankFusion(lists [][]rrfCandidate, k int) []RetrievedChunk {
	safeK := k
	if safeK <= 0 {
		safeK = RRFDefaultK
	}

	type bucket struct {
		id               string
		path             string
		title            string
		summary          string
		content          string
		bm25Rank         int
		haveBM25Rank     bool
		vectorSimilarity float64
		haveVectorSim    bool
		score            float64
	}

	buckets := make(map[string]*bucket)
	order := make([]string, 0)

	for _, list := range lists {
		for rank, c := range list {
			if c.id == "" {
				continue
			}
			contribution := 1.0 / float64(safeK+rank+1)
			existing, ok := buckets[c.id]
			if !ok {
				b := &bucket{
					id:      c.id,
					path:    c.path,
					title:   c.title,
					summary: c.summary,
					content: c.content,
					score:   contribution,
				}
				if c.haveBM25Rank {
					b.bm25Rank = c.bm25Rank
					b.haveBM25Rank = true
				}
				if c.haveVectorSim {
					b.vectorSimilarity = c.vectorSimilarity
					b.haveVectorSim = true
				}
				buckets[c.id] = b
				order = append(order, c.id)
				continue
			}

			// One-way fill: later lists seed only when the first
			// list left the field empty. BM25 tends to carry full
			// title + summary; vector hits can arrive bare.
			if existing.title == "" && c.title != "" {
				existing.title = c.title
			}
			if existing.summary == "" && c.summary != "" {
				existing.summary = c.summary
			}
			if existing.content == "" && c.content != "" {
				existing.content = c.content
			}
			if !existing.haveBM25Rank && c.haveBM25Rank {
				existing.bm25Rank = c.bm25Rank
				existing.haveBM25Rank = true
			}
			if !existing.haveVectorSim && c.haveVectorSim {
				existing.vectorSimilarity = c.vectorSimilarity
				existing.haveVectorSim = true
			}
			existing.score += contribution
		}
	}

	out := make([]RetrievedChunk, 0, len(order))
	for _, id := range order {
		b := buckets[id]
		chunk := RetrievedChunk{
			ChunkID:    b.id,
			DocumentID: b.id,
			Path:       b.path,
			Score:      b.score,
			Text:       b.content,
			Title:      b.title,
			Summary:    b.summary,
		}
		if b.haveBM25Rank {
			chunk.BM25Rank = b.bm25Rank
		}
		if b.haveVectorSim {
			chunk.VectorSimilarity = b.vectorSimilarity
		}
		out = append(out, chunk)
	}

	sort.SliceStable(out, func(i, j int) bool {
		if out[i].Score != out[j].Score {
			return out[i].Score > out[j].Score
		}
		return out[i].Path < out[j].Path
	})
	return out
}
