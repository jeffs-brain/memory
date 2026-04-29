// SPDX-License-Identifier: Apache-2.0

package retrieval

import "testing"

func TestReweightTemporalRanking_FiltersFutureBodyEventDatesForTotals(t *testing.T) {
	t.Parallel()
	results := []RetrievedChunk{
		{
			Path:  "memory/global/past.md",
			Score: 0.9,
			Text:  "On March 12, the user raised $250 for a food bank.",
		},
		{
			Path:  "memory/global/future.md",
			Score: 0.8,
			Text:  "On April 17th, the user helped with registration and raised $250 in donations.",
		},
	}

	out := reweightTemporalRanking("How much money did I raise for charity in total?", "2023/03/20 (Mon) 23:59", results)
	if len(out) != 1 {
		t.Fatalf("results = %d, want 1", len(out))
	}
	if out[0].Path != "memory/global/past.md" {
		t.Fatalf("path = %q, want past event", out[0].Path)
	}
}

func TestReweightTemporalRanking_KeepsFutureBodyEventDatesForPlanning(t *testing.T) {
	t.Parallel()
	results := []RetrievedChunk{
		{
			Path:  "memory/global/future.md",
			Score: 0.8,
			Text:  "On April 17th, the user plans to help with registration at a charity walk.",
		},
	}

	out := reweightTemporalRanking("What upcoming charity event am I planning for?", "2023/03/20 (Mon) 23:59", results)
	if len(out) != 1 {
		t.Fatalf("results = %d, want 1", len(out))
	}
	if out[0].Path != "memory/global/future.md" {
		t.Fatalf("path = %q, want future planning event", out[0].Path)
	}
}
