// SPDX-License-Identifier: Apache-2.0

package memory

import "testing"

func TestEnrichDerivedMemoryMetadataAddsClaimAndState(t *testing.T) {
	memories := []ExtractedMemory{{
		Content:     "The user currently owns a commuter bike and a road bike.",
		Type:        "user",
		Scope:       "global",
		SourceRole:  "user",
		SessionDate: "2024-03-20",
	}}

	got := enrichDerivedMemoryMetadata(memories)
	if got[0].ClaimKey == "" {
		t.Fatal("claim key should be derived")
	}
	if got[0].ClaimStatus != "asserted" {
		t.Fatalf("claim status = %q, want asserted", got[0].ClaimStatus)
	}
	if got[0].ValidFrom != "2024-03-20" {
		t.Fatalf("valid from = %q, want session date", got[0].ValidFrom)
	}
	if got[0].StateKey == "" {
		t.Fatal("state key should be derived")
	}
	if got[0].StateKind != "owned_item_set" {
		t.Fatalf("state kind = %q, want owned_item_set", got[0].StateKind)
	}
}

func TestEnrichDerivedMemoryMetadataSkipsAssistantRecommendationState(t *testing.T) {
	memories := []ExtractedMemory{{
		Content:    "Currently, this restaurant is a good option for dinner.",
		Type:       "reference",
		Scope:      "project",
		SourceRole: "assistant",
	}}

	got := enrichDerivedMemoryMetadata(memories)
	if got[0].StateKey != "" {
		t.Fatalf("assistant recommendation should not become user state: %q", got[0].StateKey)
	}
}
