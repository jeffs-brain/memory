// SPDX-License-Identifier: Apache-2.0

package lme

import (
	"encoding/json"
	"fmt"
	"os"
)

// RunManifest records the exact configuration of an LME run so
// reproducibility can be verified. Runs are only comparable when all four
// key fields match.
type RunManifest struct {
	DatasetSHA         string `json:"dataset_sha"`
	JudgeModel         string `json:"judge_model,omitempty"`
	JudgeModelSHA      string `json:"judge_model_sha,omitempty"`
	JudgePromptVersion int    `json:"judge_prompt_version"`
	JudgeBackend       string `json:"judge_backend,omitempty"`
	RunSeed            int64  `json:"run_seed"`
	SampleSize         int    `json:"sample_size"`
	IngestMode         string `json:"ingest_mode"`
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
		m.JudgePromptVersion == other.JudgePromptVersion
}
