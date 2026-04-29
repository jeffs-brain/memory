// SPDX-License-Identifier: Apache-2.0

package lme

import "testing"

func TestDataset_SampleFillsRequestedSize(t *testing.T) {
	ds := &Dataset{
		Questions: []Question{
			{ID: "a1", Category: "alpha"},
			{ID: "a2", Category: "alpha"},
			{ID: "a3", Category: "alpha"},
			{ID: "b1", Category: "beta"},
			{ID: "b2", Category: "beta"},
			{ID: "b3", Category: "beta"},
			{ID: "g1", Category: "gamma"},
			{ID: "g2", Category: "gamma"},
			{ID: "d1", Category: "delta"},
			{ID: "d2", Category: "delta"},
		},
	}

	sampled := ds.Sample(5, 42)
	if len(sampled) != 5 {
		t.Fatalf("Sample(5): got %d, want 5", len(sampled))
	}

	seen := make(map[string]struct{}, len(sampled))
	for _, q := range sampled {
		if _, ok := seen[q.ID]; ok {
			t.Fatalf("duplicate question selected: %s", q.ID)
		}
		seen[q.ID] = struct{}{}
	}
}
