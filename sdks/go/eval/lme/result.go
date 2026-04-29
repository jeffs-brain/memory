// SPDX-License-Identifier: Apache-2.0

package lme

// LMEResult holds the outcome of a LongMemEval benchmark run.
type LMEResult struct {
	DatasetSHA               string                `json:"dataset_sha"`
	IngestMode               string                `json:"ingest_mode"`
	SampleIDs                []string              `json:"sample_ids,omitempty"`
	JudgeModel               string                `json:"judge_model,omitempty"`
	RunSeed                  int64                 `json:"run_seed"`
	QuestionsRun             int                   `json:"questions_run"`
	OverallScore             float64               `json:"overall_score"`
	TaskAvgScore             float64               `json:"task_avg_score"`
	AbstentionScore          float64               `json:"abstention_score"`
	OverallScoreCI           [2]float64            `json:"overall_score_ci"`
	ExactMatchScore          float64               `json:"exact_match_score"`
	ByCategory               map[string]Category   `json:"by_category"`
	PerCategoryCI            map[string][2]float64 `json:"per_category_ci,omitempty"`
	JudgeAgreement           float64               `json:"judge_agreement,omitempty"`
	ArbitratedScore          float64               `json:"arbitrated_score,omitempty"`
	ArbitratedCategoryScores map[string]float64    `json:"arbitrated_category_scores,omitempty"`
	ArbitratedReviewed       int                   `json:"arbitrated_reviewed,omitempty"`
	Toggles                  FeatureToggles        `json:"toggles"`
	CostAccounting           CostAccounting        `json:"cost_accounting"`
	ExtractionSummary        *ExtractionSummary    `json:"extraction_summary,omitempty"`
	LatencyP50Ms             int                   `json:"latency_p50_ms"`
	LatencyP95Ms             int                   `json:"latency_p95_ms"`
	Questions                []QuestionOutcome     `json:"questions"`
}

// Category aggregates per-category LME scores.
type Category struct {
	Run       int     `json:"run"`
	Correct   int     `json:"correct"`
	Partial   int     `json:"partial"`
	Incorrect int     `json:"incorrect"`
	Score     float64 `json:"score"`
}

// FeatureToggles records which workstream features were active during the
// run, enabling A/B attribution in later phases.
type FeatureToggles struct {
	L0Buffer        bool `json:"l0_buffer"`
	QueryDistill    bool `json:"query_distill"`
	HookPlugin      bool `json:"hook_plugin"`
	SkillProcedural bool `json:"skill_procedural"`
	FeedbackSignal  bool `json:"feedback_signal"`
}

// CostAccounting breaks down the run cost by stage so expensive components
// are visible per-run rather than hidden in a single total.
type CostAccounting struct {
	IngestUSD float64 `json:"ingest_usd"`
	AgentUSD  float64 `json:"agent_usd"`
	JudgeUSD  float64 `json:"judge_usd"`
	TotalUSD  float64 `json:"total_usd"`
}

// ExtractionSummary records replay ingest counts without storing extracted
// memory content in the result JSON.
type ExtractionSummary struct {
	IngestMode         string         `json:"ingest_mode"`
	ExtractModel       string         `json:"extract_model,omitempty"`
	ExtractHeuristics  string         `json:"extract_heuristics,omitempty"`
	ExtractionPipeline int            `json:"extraction_pipeline,omitempty"`
	ReplayConcurrency  int            `json:"replay_concurrency,omitempty"`
	Contextualise      bool           `json:"contextualise"`
	SessionsProcessed  int            `json:"sessions_processed,omitempty"`
	FactsExtracted     int            `json:"facts_extracted,omitempty"`
	FactsWritten       int            `json:"facts_written,omitempty"`
	FailedSessions     int            `json:"failed_sessions,omitempty"`
	FallbackSessions   int            `json:"fallback_sessions,omitempty"`
	EmptySessions      int            `json:"empty_sessions,omitempty"`
	DuplicatePaths     int            `json:"duplicate_paths,omitempty"`
	WarningCounts      map[string]int `json:"warning_counts,omitempty"`
	WarningCount       int            `json:"warning_count,omitempty"`
	WarningPreviews    []string       `json:"warning_previews,omitempty"`
}

// QuestionOutcome records the full trace for a single LME question.
type QuestionOutcome struct {
	ID                   string                `json:"id"`
	Category             string                `json:"category"`
	Question             string                `json:"question"`
	QuestionDate         string                `json:"question_date,omitempty"`
	GroundTruth          string                `json:"ground_truth"`
	AgentAnswer          string                `json:"agent_answer"`
	JudgeVerdict         string                `json:"judge_verdict,omitempty"`
	JudgeRationale       string                `json:"judge_rationale,omitempty"`
	HumanVerdict         string                `json:"human_verdict,omitempty"`
	RecalledMemories     []string              `json:"recalled_memories,omitempty"`
	RetrievedArticles    []string              `json:"retrieved_articles,omitempty"`
	RetrievalDiagnostics *RetrievalDiagnostics `json:"retrieval_diagnostics,omitempty"`
	ToolCalls            []string              `json:"tool_calls,omitempty"`
	LatencyMs            int                   `json:"latency_ms"`
	InputTokens          int                   `json:"input_tokens"`
	OutputTokens         int                   `json:"output_tokens"`
	Error                string                `json:"error,omitempty"`
}

// RetrievalDiagnostics captures benchmark-agnostic retrieval evidence for a
// single retrieve-only actor call. It stores compact previews and hashes, not
// full retrieved content.
type RetrievalDiagnostics struct {
	Request  RetrievalRequestDiagnostics  `json:"request"`
	Response RetrievalResponseDiagnostics `json:"response"`
	Evidence RetrievalEvidenceSummary     `json:"evidence"`
	Returned []RetrievedPassageDiagnostic `json:"returned,omitempty"`
	Trace    *RetrievalTraceDiagnostics   `json:"trace,omitempty"`
	Attempts []RetrievalAttemptDiagnostic `json:"attempts,omitempty"`
	Error    string                       `json:"error,omitempty"`
}

type RetrievalRequestDiagnostics struct {
	EndpointStyle string                     `json:"endpoint_style"`
	BrainID       string                     `json:"brain_id"`
	Mode          string                     `json:"mode"`
	TopK          int                        `json:"top_k"`
	CandidateK    int                        `json:"candidate_k,omitempty"`
	RerankTopN    int                        `json:"rerank_top_n,omitempty"`
	QuestionDate  string                     `json:"question_date,omitempty"`
	Filters       RetrievalFilterDiagnostics `json:"filters,omitempty"`
	QueryHash     string                     `json:"query_hash,omitempty"`
	QueryPreview  string                     `json:"query_preview,omitempty"`
}

type RetrievalFilterDiagnostics struct {
	Scope      string   `json:"scope,omitempty"`
	Project    string   `json:"project,omitempty"`
	PathPrefix string   `json:"path_prefix,omitempty"`
	Paths      []string `json:"paths,omitempty"`
	Tags       []string `json:"tags,omitempty"`
	SessionIDs []string `json:"session_ids,omitempty"`
}

type RetrievalResponseDiagnostics struct {
	HTTPStatus int `json:"http_status,omitempty"`
	TookMs     int `json:"took_ms,omitempty"`
}

type RetrievalEvidenceSummary struct {
	ReturnedCount    int     `json:"returned_count"`
	RenderedBytes    int     `json:"rendered_bytes"`
	RenderedRunes    int     `json:"rendered_runes"`
	ApproxTokens     int     `json:"approx_tokens"`
	UniquePaths      int     `json:"unique_paths"`
	UniqueSessionIDs int     `json:"unique_session_ids"`
	MinScore         float64 `json:"min_score,omitempty"`
	MaxScore         float64 `json:"max_score,omitempty"`
}

type RetrievedPassageDiagnostic struct {
	Rank             int      `json:"rank"`
	Path             string   `json:"path,omitempty"`
	ChunkID          string   `json:"chunk_id,omitempty"`
	DocumentID       string   `json:"document_id,omitempty"`
	SessionID        string   `json:"session_id,omitempty"`
	Date             string   `json:"date,omitempty"`
	Score            float64  `json:"score,omitempty"`
	BM25Rank         int      `json:"bm25_rank,omitempty"`
	VectorSimilarity float64  `json:"vector_similarity,omitempty"`
	RerankScore      float64  `json:"rerank_score,omitempty"`
	TextBytes        int      `json:"text_bytes"`
	TextRunes        int      `json:"text_runes"`
	ApproxTokens     int      `json:"approx_tokens"`
	TextSHA256       string   `json:"text_sha256,omitempty"`
	Preview          string   `json:"preview,omitempty"`
	MetadataKeys     []string `json:"metadata_keys,omitempty"`
}

type RetrievalTraceDiagnostics struct {
	RequestedMode               string `json:"requested_mode,omitempty"`
	EffectiveMode               string `json:"effective_mode,omitempty"`
	Intent                      string `json:"intent,omitempty"`
	UsedRetry                   bool   `json:"used_retry"`
	RRFK                        int    `json:"rrf_k,omitempty"`
	CandidateK                  int    `json:"candidate_k,omitempty"`
	RerankTopN                  int    `json:"rerank_top_n,omitempty"`
	FellBackToBM25              bool   `json:"fell_back_to_bm25"`
	EmbedderUsed                bool   `json:"embedder_used"`
	Reranked                    bool   `json:"reranked"`
	RerankProvider              string `json:"rerank_provider,omitempty"`
	RerankSkipReason            string `json:"rerank_skip_reason,omitempty"`
	VectorSkipReason            string `json:"vector_skip_reason,omitempty"`
	BM25Hits                    int    `json:"bm25_hits,omitempty"`
	VectorHits                  int    `json:"vector_hits,omitempty"`
	FusedHits                   int    `json:"fused_hits,omitempty"`
	SessionExpansions           int    `json:"session_expansions,omitempty"`
	EpisodicRecall              bool   `json:"episodic_recall,omitempty"`
	EpisodicRecallHits          int    `json:"episodic_recall_hits,omitempty"`
	EpisodicRecallReason        string `json:"episodic_recall_reason,omitempty"`
	AggregateEvidenceGroups     int    `json:"aggregate_evidence_groups,omitempty"`
	AggregateEvidenceSuppressed int    `json:"aggregate_evidence_suppressed,omitempty"`
	StateIntent                 bool   `json:"state_intent,omitempty"`
	StatePromotions             int    `json:"state_promotions,omitempty"`
	Agreements                  int    `json:"agreements,omitempty"`
	UnanimitySkipped            bool   `json:"unanimity_skipped"`
}

type RetrievalAttemptDiagnostic struct {
	Rung         int    `json:"rung"`
	Mode         string `json:"mode"`
	TopK         int    `json:"top_k"`
	Reason       string `json:"reason,omitempty"`
	Chunks       int    `json:"chunks"`
	QueryHash    string `json:"query_hash,omitempty"`
	QueryPreview string `json:"query_preview,omitempty"`
}
