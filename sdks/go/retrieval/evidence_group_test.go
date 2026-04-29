// SPDX-License-Identifier: Apache-2.0

package retrieval

import "testing"

func TestClassifyAggregateEvidence(t *testing.T) {
	tests := []struct {
		name string
		text string
		want evidenceKind
	}{
		{
			name: "atomic gift spend",
			text: "We bought a birthday gift for Sam for £35 and donated £20 to the school fair.",
			want: evidenceKindAtomic,
		},
		{
			name: "gift rollup",
			text: "Gift and donation rollup: in total we spent £55 across presents and school donations.",
			want: evidenceKindRollup,
		},
		{
			name: "gift plan",
			text: "Gift plan: next steps are choosing a card and checking whether another donation is needed.",
			want: evidenceKindPlan,
		},
		{
			name: "fundraising recap",
			text: "Fundraising event recap: cake stall raised £120 and raffle raised £80.",
			want: evidenceKindRecap,
		},
		{
			name: "unknown fundraising note",
			text: "The fundraising colour palette was green.",
			want: evidenceKindUnknown,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := classifyAggregateEvidence(RetrievedChunk{Text: tt.text})
			if got.kind != tt.want {
				t.Fatalf("kind = %q, want %q", got.kind, tt.want)
			}
			if tt.want != evidenceKindUnknown && got.group == "" {
				t.Fatalf("group should be populated for %q", tt.want)
			}
		})
	}
}

func TestGroupAggregateEvidenceSuppressesRollupCoveredByAtomicFacts(t *testing.T) {
	chunks := []RetrievedChunk{
		{
			Path: "notes/gifts/card.md",
			Text: "We paid £4 for the card.",
		},
		{
			Path: "notes/gifts/wrapping.md",
			Text: "We bought wrapping paper for £3.",
		},
		{
			Path: "notes/gifts/rollup.md",
			Text: "Gift rollup: in total we spent £7 on the card and wrapping paper.",
		},
		{
			Path: "notes/gifts/colour.md",
			Text: "The gift colour was blue.",
		},
	}

	grouped, trace := groupAggregateEvidence("How much did we spend on gifts?", chunks)
	if trace.Suppressed != 1 {
		t.Fatalf("suppressed = %d, want 1", trace.Suppressed)
	}
	if len(grouped) != 3 {
		t.Fatalf("len(grouped) = %d, want 3", len(grouped))
	}
	if grouped[0].Path != "notes/gifts/card.md" || grouped[1].Path != "notes/gifts/wrapping.md" || grouped[2].Path != "notes/gifts/colour.md" {
		t.Fatalf("paths = %#v, want atomic facts and unknown note in original order", chunkPaths(grouped))
	}
	if got := grouped[0].Metadata["evidence_kind"]; got != string(evidenceKindAtomic) {
		t.Fatalf("evidence_kind = %q, want atomic", got)
	}
	if got := grouped[0].Metadata["evidence_group"]; got == "" {
		t.Fatal("evidence_group should be populated")
	}
	if got, ok := grouped[2].Metadata["evidence_kind"]; ok {
		t.Fatalf("unknown evidence should not be annotated, got %q", got)
	}
}

func TestGroupAggregateEvidenceKeepsRollupWhenAtomicFactsAreMissing(t *testing.T) {
	chunks := []RetrievedChunk{
		{
			Path: "notes/fundraising/rollup.md",
			Text: "Fundraising total so far is £120.",
		},
	}

	grouped, trace := groupAggregateEvidence("What is the fundraising total?", chunks)
	if trace.Suppressed != 0 {
		t.Fatalf("suppressed = %d, want 0", trace.Suppressed)
	}
	if len(grouped) != 1 || grouped[0].Path != "notes/fundraising/rollup.md" {
		t.Fatalf("grouped = %#v, want original rollup", grouped)
	}
}

func TestGroupAggregateEvidenceDoesNotSuppressUnrelatedSameAmount(t *testing.T) {
	chunks := []RetrievedChunk{
		{
			Path: "notes/travel/taxi.md",
			Text: "We paid £20 for the taxi after the concert.",
		},
		{
			Path: "notes/food/dinner.md",
			Text: "Dinner rollup: in total we paid £20 for dinner.",
		},
	}

	grouped, trace := groupAggregateEvidence("How much did we pay?", chunks)
	if trace.Suppressed != 0 {
		t.Fatalf("suppressed = %d, want 0", trace.Suppressed)
	}
	if len(grouped) != 2 {
		t.Fatalf("len(grouped) = %d, want 2", len(grouped))
	}
}

func TestGroupAggregateEvidenceDoesNotRunForOrdinaryQueries(t *testing.T) {
	chunks := []RetrievedChunk{{Path: "notes/fundraising/rollup.md", Text: "Fundraising total so far is £120."}}
	grouped, trace := groupAggregateEvidence("What colour was the fundraising poster?", chunks)
	if trace != (aggregateEvidenceTrace{}) {
		t.Fatalf("trace = %#v, want zero trace", trace)
	}
	if len(grouped) != 1 || grouped[0].Metadata != nil {
		t.Fatalf("ordinary query should preserve chunks without annotations: %#v", grouped)
	}
}
