// SPDX-License-Identifier: Apache-2.0

package retrieval

import "testing"

func TestPromoteCurrentStateEvidenceUsesStateMetadata(t *testing.T) {
	chunks := []RetrievedChunk{
		{
			Path: "memory/global/old.md",
			Text: "The user previously used the old venue.",
		},
		{
			Path: "memory/global/current.md",
			Text: "The user currently attends the weekly class at Riverside Studio.",
			Metadata: map[string]any{
				"state_key":     "state.location.weekly.class",
				"state_subject": "weekly class venue",
				"valid_from":    "2024-03-10",
				"claim_status":  "asserted",
			},
		},
	}

	got, trace := promoteCurrentStateEvidence("Where do I currently take the weekly class?", "2024-03-20", chunks)
	if !trace.Intent {
		t.Fatal("expected state intent")
	}
	if trace.Promotions != 1 {
		t.Fatalf("promotions = %d, want 1", trace.Promotions)
	}
	if got[0].Path != "memory/global/current.md" {
		t.Fatalf("top path = %q, want current state", got[0].Path)
	}
}

func TestPromoteCurrentStateEvidenceDoesNotPromoteFutureState(t *testing.T) {
	chunks := []RetrievedChunk{
		{
			Path: "memory/global/current.md",
			Text: "The user currently attends the weekly advanced pottery class at Riverside Studio and uses the Riverside Studio locker.",
			Metadata: map[string]any{
				"state_key":     "state.location.weekly.class",
				"state_subject": "weekly class venue",
				"valid_from":    "2024-04-10",
			},
		},
		{
			Path: "memory/global/older.md",
			Text: "The user attends the weekly class at North Studio.",
			Metadata: map[string]any{
				"state_key":     "state.location.weekly.class",
				"state_subject": "weekly class venue",
				"valid_from":    "2024-03-01",
			},
		},
	}

	got, _ := promoteCurrentStateEvidence("Where do I currently take the weekly class?", "2024-03-20", chunks)
	if got[0].Path != "memory/global/older.md" {
		t.Fatalf("top path = %q, want non-future state", got[0].Path)
	}
}

func TestPromoteCurrentStateEvidenceSkipsOrdinaryQueries(t *testing.T) {
	chunks := []RetrievedChunk{
		{Path: "a.md", Text: "The user currently owns a commuter bike."},
		{Path: "b.md", Text: "A historical bike note."},
	}
	got, trace := promoteCurrentStateEvidence("When did I mention the commuter bike?", "2024-03-20", chunks)
	if trace.Intent {
		t.Fatal("ordinary historical query should not be state intent")
	}
	if got[0].Path != "a.md" || got[1].Path != "b.md" {
		t.Fatalf("chunks reordered for ordinary query: %#v", chunkPaths(got))
	}
}
