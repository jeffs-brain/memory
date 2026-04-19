// SPDX-License-Identifier: Apache-2.0

package lme

import (
	"testing"

	"github.com/jeffs-brain/memory/go/retrieval"
)

func TestBuildRunManifest_IncludesActorSettings(t *testing.T) {
	manifest := BuildRunManifest(
		&LMEResult{
			DatasetSHA: "dataset-sha",
			IngestMode: "none",
		},
		RunConfig{
			Seed:               7,
			SampleSize:         5,
			ActorEndpoint:      "http://127.0.0.1:18850",
			ActorEndpointStyle: " retrieve-only ",
			ActorTopK:          0,
			ActorCandidateK:    120,
			ActorRerankTopN:    0,
			ActorFilters: retrieval.Filters{
				Scope:      " project ",
				Project:    " eval-lme ",
				PathPrefix: " memory/project/eval-lme/ ",
			},
		},
		"judge-m",
	)

	if manifest.ActorEndpointStyle != "retrieve-only" {
		t.Fatalf("ActorEndpointStyle = %q, want retrieve-only", manifest.ActorEndpointStyle)
	}
	if manifest.ActorBrain != "eval-lme" {
		t.Fatalf("ActorBrain = %q, want eval-lme", manifest.ActorBrain)
	}
	if manifest.ActorTopK == nil || *manifest.ActorTopK != 20 {
		t.Fatalf("ActorTopK = %v, want 20", manifest.ActorTopK)
	}
	if manifest.ActorCandidateK == nil || *manifest.ActorCandidateK != 120 {
		t.Fatalf("ActorCandidateK = %v, want 120", manifest.ActorCandidateK)
	}
	if manifest.ActorRerankTopN == nil || *manifest.ActorRerankTopN != 0 {
		t.Fatalf("ActorRerankTopN = %v, want 0", manifest.ActorRerankTopN)
	}
	if manifest.ActorScope != "project" {
		t.Fatalf("ActorScope = %q, want project", manifest.ActorScope)
	}
	if manifest.ActorProject != "eval-lme" {
		t.Fatalf("ActorProject = %q, want eval-lme", manifest.ActorProject)
	}
	if manifest.ActorPathPrefix != "memory/project/eval-lme/" {
		t.Fatalf("ActorPathPrefix = %q, want memory/project/eval-lme/", manifest.ActorPathPrefix)
	}
}

func TestRunManifest_IsComparableIncludesActorSettings(t *testing.T) {
	base := RunManifest{
		DatasetSHA:         "dataset-sha",
		JudgeModel:         "judge-m",
		JudgePromptVersion: 6,
		ActorEndpointStyle: "retrieve-only",
		ActorBrain:         "eval-lme",
		ActorTopK:          intPtr(20),
		ActorCandidateK:    intPtr(100),
		ActorRerankTopN:    intPtr(60),
		ActorScope:         "project",
		ActorProject:       "eval-lme",
	}
	if !base.IsComparable(base) {
		t.Fatal("expected identical manifests to be comparable")
	}

	changed := base
	changed.ActorCandidateK = intPtr(120)
	if base.IsComparable(changed) {
		t.Fatal("expected changed actor settings to break comparability")
	}
}
