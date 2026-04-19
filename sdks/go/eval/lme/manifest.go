// SPDX-License-Identifier: Apache-2.0

package lme

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

// RunManifest records the exact configuration of an LME run so
// reproducibility can be verified. Actor-backed runs also persist the
// daemon retrieval settings that materially change cross-SDK outcomes.
type RunManifest struct {
	DatasetSHA         string `json:"dataset_sha"`
	JudgeModel         string `json:"judge_model,omitempty"`
	JudgeModelSHA      string `json:"judge_model_sha,omitempty"`
	JudgePromptVersion int    `json:"judge_prompt_version"`
	JudgeBackend       string `json:"judge_backend,omitempty"`
	RunSeed            int64  `json:"run_seed"`
	SampleSize         int    `json:"sample_size"`
	IngestMode         string `json:"ingest_mode"`
	ActorEndpointStyle string `json:"actor_endpoint_style,omitempty"`
	ActorBrain         string `json:"actor_brain,omitempty"`
	ActorTopK          *int   `json:"actor_topk,omitempty"`
	ActorCandidateK    *int   `json:"actor_candidatek,omitempty"`
	ActorRerankTopN    *int   `json:"actor_rerank_topn,omitempty"`
	ActorScope         string `json:"actor_scope,omitempty"`
	ActorProject       string `json:"actor_project,omitempty"`
	ActorPathPrefix    string `json:"actor_path_prefix,omitempty"`
}

// BuildRunManifest derives the persisted manifest from the completed run and
// the runner configuration.
func BuildRunManifest(result *LMEResult, cfg RunConfig, judgeModel string) RunManifest {
	manifest := RunManifest{
		JudgeModel:         judgeModel,
		JudgePromptVersion: JudgePromptVersion,
		RunSeed:            cfg.Seed,
		SampleSize:         cfg.SampleSize,
	}
	if result != nil {
		manifest.DatasetSHA = result.DatasetSHA
		manifest.IngestMode = result.IngestMode
	}

	if strings.TrimSpace(cfg.ActorEndpoint) == "" {
		return manifest
	}

	style := strings.ToLower(strings.TrimSpace(cfg.ActorEndpointStyle))
	if style == "" {
		style = "full"
	}
	brainID := strings.TrimSpace(cfg.ActorBrainID)
	if brainID == "" {
		brainID = "eval-lme"
	}
	topK := cfg.ActorTopK
	if topK <= 0 {
		topK = 20
	}
	candidateK := cfg.ActorCandidateK
	rerankTopN := cfg.ActorRerankTopN

	manifest.ActorEndpointStyle = style
	manifest.ActorBrain = brainID
	manifest.ActorTopK = intPtr(topK)
	manifest.ActorCandidateK = intPtr(candidateK)
	manifest.ActorRerankTopN = intPtr(rerankTopN)
	manifest.ActorScope = strings.TrimSpace(cfg.ActorFilters.Scope)
	manifest.ActorProject = strings.TrimSpace(cfg.ActorFilters.Project)
	manifest.ActorPathPrefix = strings.TrimSpace(cfg.ActorFilters.PathPrefix)
	return manifest
}

// SaveManifest writes the manifest to a JSON file.
func SaveManifest(m RunManifest, path string) error {
	data, err := json.MarshalIndent(m, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal manifest: %w", err)
	}
	if err := os.WriteFile(path, data, 0o644); err != nil {
		return fmt.Errorf("write manifest: %w", err)
	}
	return nil
}

// LoadManifest reads a manifest from a JSON file.
func LoadManifest(path string) (RunManifest, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return RunManifest{}, fmt.Errorf("read manifest: %w", err)
	}
	var m RunManifest
	if err := json.Unmarshal(data, &m); err != nil {
		return RunManifest{}, fmt.Errorf("parse manifest: %w", err)
	}
	return m, nil
}

// IsComparable returns true if two manifests share the same key fields
// needed for a meaningful score comparison.
func (m RunManifest) IsComparable(other RunManifest) bool {
	return m.DatasetSHA == other.DatasetSHA &&
		m.JudgeModel == other.JudgeModel &&
		m.JudgeModelSHA == other.JudgeModelSHA &&
		m.JudgePromptVersion == other.JudgePromptVersion &&
		m.ActorEndpointStyle == other.ActorEndpointStyle &&
		m.ActorBrain == other.ActorBrain &&
		intPtrEqual(m.ActorTopK, other.ActorTopK) &&
		intPtrEqual(m.ActorCandidateK, other.ActorCandidateK) &&
		intPtrEqual(m.ActorRerankTopN, other.ActorRerankTopN) &&
		m.ActorScope == other.ActorScope &&
		m.ActorProject == other.ActorProject &&
		m.ActorPathPrefix == other.ActorPathPrefix
}

func intPtr(v int) *int {
	return &v
}

func intPtrEqual(left, right *int) bool {
	switch {
	case left == nil || right == nil:
		return left == right
	default:
		return *left == *right
	}
}
