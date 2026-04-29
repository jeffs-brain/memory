// SPDX-License-Identifier: Apache-2.0

package retrieval

import (
	"math"
	"testing"
)

func TestReciprocalRankFusion_SingleListMatchesFormula(t *testing.T) {
	t.Parallel()
	list := []rrfCandidate{
		{id: "a", path: "a.md"},
		{id: "b", path: "b.md"},
		{id: "c", path: "c.md"},
	}
	out := reciprocalRankFusion([][]rrfCandidate{list}, RRFDefaultK)
	if len(out) != 3 {
		t.Fatalf("want 3, got %d", len(out))
	}
	wantScores := []float64{
		1.0 / float64(RRFDefaultK+1),
		1.0 / float64(RRFDefaultK+2),
		1.0 / float64(RRFDefaultK+3),
	}
	for i, r := range out {
		if math.Abs(r.Score-wantScores[i]) > 1e-12 {
			t.Fatalf("position %d: score %v, want %v", i, r.Score, wantScores[i])
		}
	}
}

func TestReciprocalRankFusion_TwoListsSumContributions(t *testing.T) {
	t.Parallel()
	// Candidate "a" appears top of list 1 and second of list 2.
	// Its score should be 1/(60+1) + 1/(60+2).
	list1 := []rrfCandidate{
		{id: "a", path: "a.md", title: "A", haveBM25Rank: true, bm25Rank: 0},
		{id: "b", path: "b.md", haveBM25Rank: true, bm25Rank: 1},
	}
	list2 := []rrfCandidate{
		{id: "c", path: "c.md", haveVectorSim: true, vectorSimilarity: 0.9},
		{id: "a", path: "a.md", haveVectorSim: true, vectorSimilarity: 0.8},
	}
	out := reciprocalRankFusion([][]rrfCandidate{list1, list2}, RRFDefaultK)
	if len(out) != 3 {
		t.Fatalf("want 3, got %d", len(out))
	}
	byID := indexByID(out)
	wantA := 1.0/float64(RRFDefaultK+1) + 1.0/float64(RRFDefaultK+2)
	if math.Abs(byID["a"].Score-wantA) > 1e-12 {
		t.Fatalf("a score %v, want %v", byID["a"].Score, wantA)
	}
	// Candidate "a" should rank first because its fused score
	// exceeds either single list position.
	if out[0].ChunkID != "a" {
		t.Fatalf("top result %q, want a", out[0].ChunkID)
	}
	// Carry-through: both bm25Rank and vectorSimilarity should
	// survive the merge.
	if byID["a"].BM25Rank != 0 {
		t.Fatalf("a BM25Rank %d, want 0", byID["a"].BM25Rank)
	}
	if byID["a"].VectorSimilarity == 0 {
		t.Fatalf("a lost vector similarity in merge")
	}
}

func TestReciprocalRankFusion_TieBreakByPathAsc(t *testing.T) {
	t.Parallel()
	// Two candidates tied at rank 0 in separate lists -> identical
	// scores -> expect path ascending.
	list1 := []rrfCandidate{
		{id: "zebra", path: "z.md"},
	}
	list2 := []rrfCandidate{
		{id: "alpha", path: "a.md"},
	}
	out := reciprocalRankFusion([][]rrfCandidate{list1, list2}, RRFDefaultK)
	if len(out) != 2 {
		t.Fatalf("want 2, got %d", len(out))
	}
	if out[0].Path != "a.md" {
		t.Fatalf("tie-break failed: first path %q, want a.md", out[0].Path)
	}
}

func TestReciprocalRankFusion_MetadataFillFromLaterLists(t *testing.T) {
	t.Parallel()
	list1 := []rrfCandidate{
		{id: "a", path: "a.md"},
	}
	list2 := []rrfCandidate{
		{id: "a", path: "a.md", title: "Hydrated", summary: "Sum", content: "body"},
	}
	out := reciprocalRankFusion([][]rrfCandidate{list1, list2}, RRFDefaultK)
	if len(out) != 1 {
		t.Fatalf("want 1, got %d", len(out))
	}
	if out[0].Title != "Hydrated" {
		t.Fatalf("title not filled: %q", out[0].Title)
	}
	if out[0].Summary != "Sum" {
		t.Fatalf("summary not filled: %q", out[0].Summary)
	}
	if out[0].Text != "body" {
		t.Fatalf("text not filled: %q", out[0].Text)
	}
}

func TestReciprocalRankFusion_NoOverwriteOfEarlyMetadata(t *testing.T) {
	t.Parallel()
	list1 := []rrfCandidate{
		{id: "a", path: "a.md", title: "First", summary: "FirstSum"},
	}
	list2 := []rrfCandidate{
		{id: "a", path: "a.md", title: "Second", summary: "SecondSum"},
	}
	out := reciprocalRankFusion([][]rrfCandidate{list1, list2}, RRFDefaultK)
	if out[0].Title != "First" {
		t.Fatalf("early title overwritten: %q", out[0].Title)
	}
	if out[0].Summary != "FirstSum" {
		t.Fatalf("early summary overwritten: %q", out[0].Summary)
	}
}

func TestReciprocalRankFusion_ZeroKFallsBack(t *testing.T) {
	t.Parallel()
	list := []rrfCandidate{{id: "a", path: "a.md"}}
	out := reciprocalRankFusion([][]rrfCandidate{list}, 0)
	want := 1.0 / float64(RRFDefaultK+1)
	if math.Abs(out[0].Score-want) > 1e-12 {
		t.Fatalf("score %v, want %v (fallback to RRFDefaultK)", out[0].Score, want)
	}
}

func indexByID(rs []RetrievedChunk) map[string]RetrievedChunk {
	out := make(map[string]RetrievedChunk, len(rs))
	for _, r := range rs {
		out[r.ChunkID] = r
	}
	return out
}
