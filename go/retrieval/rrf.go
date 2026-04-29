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
	candidates := reciprocalRankFusionCandidates(lists, k)
	out := make([]RetrievedChunk, 0, len(candidates))
	for _, c := range candidates {
		chunk := RetrievedChunk{
			ChunkID:    c.id,
			DocumentID: c.id,
			Path:       c.path,
			Score:      c.score,
			Text:       c.content,
			Title:      c.title,
			Summary:    c.summary,
		}
		if c.haveBM25Rank {
			chunk.BM25Rank = c.bm25Rank
		}
		if c.haveVectorSim {
			chunk.VectorSimilarity = c.vectorSimilarity
		}
		out = append(out, chunk)
	}
	return out
}

type fusedCandidate struct {
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

func reciprocalRankFusionCandidates(lists [][]rrfCandidate, k int) []fusedCandidate {
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

	out := make([]fusedCandidate, 0, len(order))
	for _, id := range order {
		b := buckets[id]
		out = append(out, fusedCandidate{
			id:               b.id,
			path:             b.path,
			title:            b.title,
			summary:          b.summary,
			content:          b.content,
			bm25Rank:         b.bm25Rank,
			haveBM25Rank:     b.haveBM25Rank,
			vectorSimilarity: b.vectorSimilarity,
			haveVectorSim:    b.haveVectorSim,
			score:            b.score,
		})
	}

	sort.SliceStable(out, func(i, j int) bool {
		if out[i].score != out[j].score {
			return out[i].score > out[j].score
		}
		return out[i].path < out[j].path
	})
	return out
}
