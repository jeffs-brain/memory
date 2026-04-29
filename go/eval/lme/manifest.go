// SPDX-License-Identifier: Apache-2.0

package lme

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"strings"

	"github.com/jeffs-brain/memory/go/retrieval"
)

// RunManifest records the exact configuration of an LME run so
// reproducibility can be verified. Actor-backed runs also persist the
// daemon retrieval settings that materially change cross-SDK outcomes.
type RunManifest struct {
	DatasetSHA            string                `json:"dataset_sha"`
	SampleSignature       string                `json:"sample_signature,omitempty"`
	SampleIDs             []string              `json:"sample_ids,omitempty"`
	JudgeModel            string                `json:"judge_model,omitempty"`
	JudgeModelSHA         string                `json:"judge_model_sha,omitempty"`
	JudgePromptVersion    int                   `json:"judge_prompt_version"`
	JudgeBackend          string                `json:"judge_backend,omitempty"`
	JudgeFailureMode      string                `json:"judge_failure_mode,omitempty"`
	ReaderFailureMode     string                `json:"reader_failure_mode,omitempty"`
	ReaderModel           string                `json:"reader_model,omitempty"`
	ExtractModel          string                `json:"extract_model,omitempty"`
	ExtractHeuristics     string                `json:"extract_heuristics,omitempty"`
	ExtractionPipeline    int                   `json:"extraction_pipeline,omitempty"`
	ReplayConcurrency     int                   `json:"replay_concurrency,omitempty"`
	Contextualise         bool                  `json:"contextualise"`
	ExtractOnly           bool                  `json:"extract_only"`
	CacheSignature        string                `json:"cache_signature,omitempty"`
	CacheSignatureInputs  *CacheSignatureInputs `json:"cache_signature_inputs,omitempty"`
	RunSeed               int64                 `json:"run_seed"`
	SampleSize            int                   `json:"sample_size"`
	IngestMode            string                `json:"ingest_mode"`
	BenchmarkMode         string                `json:"benchmark_mode"`
	ContextSource         string                `json:"context_source"`
	ActorEndpointStyle    string                `json:"actor_endpoint_style,omitempty"`
	ActorBrain            string                `json:"actor_brain,omitempty"`
	ActorRetrievalMode    string                `json:"actor_retrieval_mode,omitempty"`
	ActorTopK             *int                  `json:"actor_topk,omitempty"`
	ActorCandidateK       *int                  `json:"actor_candidatek,omitempty"`
	ActorRerankTopN       *int                  `json:"actor_rerank_topn,omitempty"`
	ActorScope            string                `json:"actor_scope,omitempty"`
	ActorProject          string                `json:"actor_project,omitempty"`
	ActorPathPrefix       string                `json:"actor_path_prefix,omitempty"`
	ActorQuestionSessions bool                  `json:"actor_question_sessions,omitempty"`
}

// CacheSignatureInputs records the fields used to validate that a populated
// brain cache matches the extraction run a script is about to reuse.
type CacheSignatureInputs struct {
	DatasetSHA         string   `json:"dataset_sha"`
	SampleSignature    string   `json:"sample_signature"`
	SampleIDs          []string `json:"sample_ids"`
	IngestMode         string   `json:"ingest_mode"`
	ExtractModel       string   `json:"extract_model,omitempty"`
	ExtractHeuristics  string   `json:"extract_heuristics,omitempty"`
	ExtractionPipeline int      `json:"extraction_pipeline,omitempty"`
	Contextualise      bool     `json:"contextualise"`
}

// BuildRunManifest derives the persisted manifest from the completed run and
// the runner configuration.
func BuildRunManifest(result *LMEResult, cfg RunConfig, judgeModel string) RunManifest {
	spec := inferManifestBenchmarkSpec(cfg)
	if cfg.ExtractOnly {
		spec = benchmarkSpec{
			Mode:          BenchmarkModeExtractPrep,
			ContextSource: ContextSourceExtractPrep,
		}
	}

	readerModel := ""
	if cfg.Reader != nil {
		readerModel = strings.TrimSpace(cfg.Reader.Model)
	}
	extractModel := strings.TrimSpace(cfg.ReplayExtractModel)
	if extractModel == "" && cfg.IngestMode == "replay" {
		extractModel = DefaultReplayExtractModel
	}
	replayConcurrency := 0
	if cfg.IngestMode == "replay" {
		replayConcurrency = normaliseReplayConcurrencyForManifest(cfg.ReplayConcurrency)
	}
	manifest := RunManifest{
		JudgeModel:         judgeModel,
		JudgePromptVersion: JudgePromptVersion,
		JudgeFailureMode:   "question-error",
		ReaderFailureMode:  "question-error",
		ReaderModel:        readerModel,
		ExtractModel:       extractModel,
		ExtractHeuristics:  normaliseExtractHeuristicsEnv(os.Getenv("JB_EXTRACT_HEURISTICS")),
		ExtractionPipeline: ReplayExtractionPipelineVersion,
		ReplayConcurrency:  replayConcurrency,
		Contextualise:      cfg.Contextualiser != nil,
		ExtractOnly:        cfg.ExtractOnly,
		RunSeed:            cfg.Seed,
		SampleSize:         cfg.SampleSize,
		BenchmarkMode:      spec.Mode,
		ContextSource:      spec.ContextSource,
	}
	if result != nil {
		manifest.DatasetSHA = result.DatasetSHA
		manifest.IngestMode = result.IngestMode
		if manifest.IngestMode == "replay" && manifest.ReplayConcurrency == 0 {
			manifest.ReplayConcurrency = normaliseReplayConcurrencyForManifest(cfg.ReplayConcurrency)
		}
		manifest.SampleIDs = append([]string(nil), result.SampleIDs...)
		manifest.SampleSignature = buildSampleSignatureFromOutcomes(result.Questions)
	}
	if len(manifest.SampleIDs) == 0 {
		manifest.SampleIDs = append([]string(nil), cfg.SampleIDs...)
	}
	if manifest.SampleSignature == "" {
		manifest.SampleSignature = buildSampleSignatureFromIDs(manifest.SampleIDs)
	}
	if manifest.IngestMode == "replay" || cfg.ExtractOnly {
		manifest.CacheSignatureInputs = buildCacheSignatureInputs(manifest)
		manifest.CacheSignature = buildCacheSignature(manifest.CacheSignatureInputs)
	}

	if strings.TrimSpace(cfg.ActorEndpoint) == "" {
		return manifest
	}

	style := spec.ActorEndpointStyle
	if style == "" {
		style = actorEndpointStyleFull
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
	if style == actorEndpointStyleRetrieve {
		manifest.ActorRetrievalMode = string(normaliseActorRetrievalMode(cfg.ActorRetrievalMode))
		manifest.ActorTopK = intPtr(topK)
		manifest.ActorCandidateK = intPtr(candidateK)
		manifest.ActorRerankTopN = intPtr(rerankTopN)
	} else {
		manifest.ActorRetrievalMode = string(retrieval.ModeHybridRerank)
		manifest.ActorTopK = intPtr(5)
	}
	manifest.ActorScope = strings.TrimSpace(cfg.ActorFilters.Scope)
	manifest.ActorProject = strings.TrimSpace(cfg.ActorFilters.Project)
	manifest.ActorPathPrefix = strings.TrimSpace(cfg.ActorFilters.PathPrefix)
	manifest.ActorQuestionSessions = cfg.ActorFilterQuestionSessions
	return manifest
}

func inferManifestBenchmarkSpec(cfg RunConfig) benchmarkSpec {
	var ds *Dataset
	if strings.TrimSpace(cfg.DatasetPath) != "" && strings.TrimSpace(cfg.ActorEndpoint) == "" && !cfg.AgenticMode {
		loaded, err := LoadDataset(cfg.DatasetPath)
		if err == nil {
			ds = loaded
		}
	}
	spec, err := resolveBenchmarkSpec(ds, cfg)
	if err == nil {
		return spec
	}
	spec, err = inferBenchmarkSpec(ds, cfg)
	if err == nil {
		return spec
	}
	return benchmarkSpec{
		Mode:          BenchmarkModeOracle,
		ContextSource: ContextSourceDatasetOracle,
	}
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
		m.SampleSignature == other.SampleSignature &&
		m.JudgeModel == other.JudgeModel &&
		m.JudgeModelSHA == other.JudgeModelSHA &&
		m.JudgePromptVersion == other.JudgePromptVersion &&
		m.JudgeFailureMode == other.JudgeFailureMode &&
		m.ReaderFailureMode == other.ReaderFailureMode &&
		m.ReaderModel == other.ReaderModel &&
		m.ExtractModel == other.ExtractModel &&
		m.ExtractHeuristics == other.ExtractHeuristics &&
		m.ExtractionPipeline == other.ExtractionPipeline &&
		m.Contextualise == other.Contextualise &&
		m.ExtractOnly == other.ExtractOnly &&
		m.CacheSignature == other.CacheSignature &&
		m.BenchmarkMode == other.BenchmarkMode &&
		m.ContextSource == other.ContextSource &&
		m.ActorEndpointStyle == other.ActorEndpointStyle &&
		m.ActorBrain == other.ActorBrain &&
		m.ActorRetrievalMode == other.ActorRetrievalMode &&
		intPtrEqual(m.ActorTopK, other.ActorTopK) &&
		intPtrEqual(m.ActorCandidateK, other.ActorCandidateK) &&
		intPtrEqual(m.ActorRerankTopN, other.ActorRerankTopN) &&
		m.ActorScope == other.ActorScope &&
		m.ActorProject == other.ActorProject &&
		m.ActorPathPrefix == other.ActorPathPrefix &&
		m.ActorQuestionSessions == other.ActorQuestionSessions
}

func buildCacheSignatureInputs(m RunManifest) *CacheSignatureInputs {
	if m.DatasetSHA == "" || m.SampleSignature == "" || m.IngestMode == "" {
		return nil
	}
	return &CacheSignatureInputs{
		DatasetSHA:         m.DatasetSHA,
		SampleSignature:    m.SampleSignature,
		SampleIDs:          append([]string(nil), m.SampleIDs...),
		IngestMode:         m.IngestMode,
		ExtractModel:       m.ExtractModel,
		ExtractHeuristics:  m.ExtractHeuristics,
		ExtractionPipeline: m.ExtractionPipeline,
		Contextualise:      m.Contextualise,
	}
}

func normaliseExtractHeuristicsEnv(raw string) string {
	mode := strings.ToLower(strings.TrimSpace(raw))
	if mode == "" {
		return "default"
	}
	return mode
}

func buildCacheSignature(inputs *CacheSignatureInputs) string {
	if inputs == nil {
		return ""
	}
	data, err := json.Marshal(inputs)
	if err != nil {
		return ""
	}
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:])
}

func normaliseReplayConcurrencyForManifest(value int) int {
	if value <= 0 {
		return defaultReplayConcurrency
	}
	if value > maxReplayConcurrency {
		return maxReplayConcurrency
	}
	return value
}

func buildSampleSignatureFromOutcomes(outcomes []QuestionOutcome) string {
	if len(outcomes) == 0 {
		return ""
	}
	ids := make([]string, 0, len(outcomes))
	for _, outcome := range outcomes {
		if strings.TrimSpace(outcome.ID) != "" {
			ids = append(ids, outcome.ID)
		}
	}
	return buildSampleSignatureFromIDs(ids)
}

func buildSampleSignatureFromIDs(ids []string) string {
	if len(ids) == 0 {
		return ""
	}
	sum := sha256.Sum256([]byte(strings.Join(ids, "\n")))
	return hex.EncodeToString(sum[:])
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
