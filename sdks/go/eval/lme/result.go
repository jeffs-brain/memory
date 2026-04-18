// SPDX-License-Identifier: Apache-2.0

package lme

// LMEResult holds the outcome of a LongMemEval benchmark run.
type LMEResult struct {
	DatasetSHA               string                `json:"dataset_sha"`
	IngestMode               string                `json:"ingest_mode"`
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

// QuestionOutcome records the full trace for a single LME question.
type QuestionOutcome struct {
	ID                string   `json:"id"`
	Category          string   `json:"category"`
	Question          string   `json:"question"`
	QuestionDate      string   `json:"question_date,omitempty"`
	GroundTruth       string   `json:"ground_truth"`
	AgentAnswer       string   `json:"agent_answer"`
	JudgeVerdict      string   `json:"judge_verdict,omitempty"`
	JudgeRationale    string   `json:"judge_rationale,omitempty"`
	HumanVerdict      string   `json:"human_verdict,omitempty"`
	RecalledMemories  []string `json:"recalled_memories,omitempty"`
	RetrievedArticles []string `json:"retrieved_articles,omitempty"`
	ToolCalls         []string `json:"tool_calls,omitempty"`
	LatencyMs         int      `json:"latency_ms"`
	InputTokens       int      `json:"input_tokens"`
	OutputTokens      int      `json:"output_tokens"`
	Error             string   `json:"error,omitempty"`
}
