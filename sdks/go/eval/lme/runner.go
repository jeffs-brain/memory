// SPDX-License-Identifier: Apache-2.0

package lme

import (
	"context"
	"fmt"
	"log/slog"
	"os"
	"sort"
	"strings"
	"time"

	"github.com/jeffs-brain/memory/go/brain"
	"github.com/jeffs-brain/memory/go/llm"
	"github.com/jeffs-brain/memory/go/memory"
	"github.com/jeffs-brain/memory/go/query"
	"github.com/jeffs-brain/memory/go/store/mem"
)

// RunConfig holds all configuration for an LME benchmark run.
type RunConfig struct {
	DatasetPath string
	SampleSize  int
	Seed        int64
	IngestMode  string
	ExpectedSHA string
	MaxCostUSD  float64

	// Judge configures the LLM judge scorer. When nil, only exact-match
	// scoring is used.
	Judge *JudgeConfig

	// Provider is the LLM provider used for replay extraction. Required
	// when IngestMode is "replay".
	Provider llm.Provider

	// ReplayConcurrency caps the number of in-flight extraction LLM
	// calls during replay ingest.
	ReplayConcurrency int

	// ReplayExtractModel overrides the extraction model used by replay
	// ingest. Empty selects [DefaultReplayExtractModel]. Useful for
	// pinning extraction on a cheaper SKU than the actor / judge.
	ReplayExtractModel string

	// Contextualiser, when non-nil, is threaded into the replay ingest
	// so extracted facts carry a situating prefix.
	Contextualiser *memory.Contextualiser

	// BrainCache, when non-empty, turns the eval brain into a persistent
	// filesystem store. Unimplemented in this port; see the runner's
	// replay branch for the TODO.
	BrainCache string

	// Reader configures the LLM reader that generates answers from
	// retrieved content. When nil, raw session content is passed
	// directly to the judge.
	Reader *ReaderConfig

	// Reranker configures the cross-encoder reranker.
	Reranker *RerankerConfig

	// AgenticMode enables the full agentic-loop run mode. Requires
	// AgentFactory or a provider that can drive the loop via the actor
	// model configured on Reader.
	AgenticMode bool

	// AgentFactory constructs the per-question agent resources. When
	// nil a minimal factory is used that wires the store and the
	// in-memory memory facade; callers can plug their own retriever.
	AgentFactory AgentFactory

	// MaxIterations caps the agent loop per question. Zero selects the
	// package default.
	MaxIterations int

	// QuestionTimeout bounds a single agentic question run. Zero
	// selects the package default.
	QuestionTimeout time.Duration

	// Concurrency caps the number of questions processed in parallel.
	Concurrency int

	// Store, when non-nil, replaces the default in-memory eval store so
	// callers can persist the ingested corpus between runs. Primarily
	// useful for debugging.
	Store brain.Store
}

// Concurrency bounds and defaults.
const (
	defaultConcurrency = 8
	minConcurrency     = 1
	maxConcurrency     = 64
)

// clampConcurrency normalises a user-supplied worker count.
func clampConcurrency(n int) int {
	if n <= 0 {
		return defaultConcurrency
	}
	if n < minConcurrency {
		return minConcurrency
	}
	if n > maxConcurrency {
		return maxConcurrency
	}
	return n
}

// Run executes a full LME benchmark: load dataset, ingest into an
// isolated brain, query each question, and score results.
//
// This port implements the bulk-ingest path end-to-end. Replay mode and
// agentic mode still need the replay ingest pipeline and the agent App
// scaffolding ported from jeff; toggling them returns a NotImplemented
// error until that work lands.
func Run(ctx context.Context, cfg RunConfig) (*LMEResult, error) {
	start := time.Now()
	_ = start

	ds, err := LoadDataset(cfg.DatasetPath)
	if err != nil {
		return nil, fmt.Errorf("lme run: %w", err)
	}

	if cfg.ExpectedSHA != "" {
		if err := ds.VerifySHA(cfg.ExpectedSHA); err != nil {
			return nil, fmt.Errorf("lme run: %w", err)
		}
	}

	questions := ds.Questions
	if cfg.SampleSize > 0 && cfg.SampleSize < len(questions) {
		questions = ds.Sample(cfg.SampleSize, cfg.Seed)
	}

	var evalStore brain.Store = cfg.Store
	if evalStore == nil {
		evalStore = mem.New()
	}

	ingestMode := cfg.IngestMode
	if ingestMode == "" {
		ingestMode = "bulk"
	}

	costs := &CostAccumulator{}

	// The bulk path writes haystack sessions to raw/lme/. The replay
	// path instead streams every session through the extraction LLM
	// and writes the resulting facts into the project memory prefix.
	// The agent still benefits from having the raw transcripts on hand
	// when it falls back to naive keyword search, so when replay is
	// selected we also run a bulk pass so both surfaces are populated.
	switch ingestMode {
	case "replay":
		if cfg.Provider == nil {
			return nil, fmt.Errorf("lme run: replay ingest requires cfg.Provider")
		}
		if err := ingestBulk(ctx, evalStore, &Dataset{Questions: questions}); err != nil {
			return nil, fmt.Errorf("lme run: bulk seed before replay: %w", err)
		}
		replayRes, err := IngestReplay(ctx, evalStore, &Dataset{Questions: questions}, cfg.Provider, ReplayOpts{
			Concurrency:    cfg.ReplayConcurrency,
			ExtractModel:   cfg.ReplayExtractModel,
			Contextualiser: cfg.Contextualiser,
		})
		if err != nil {
			return nil, fmt.Errorf("lme run: replay ingest: %w", err)
		}
		extractModel := cfg.ReplayExtractModel
		if extractModel == "" {
			extractModel = DefaultReplayExtractModel
		}
		if replayRes != nil && replayRes.FactsExtracted > 0 {
			// Rough cost attribution: assume 1.5k input + 0.3k output
			// tokens per extraction call (extraction prompts are hefty
			// but the response is short). The per-fact split is
			// conservative and only used for the running USD estimate
			// logged to stderr; actual bills will come from the
			// provider's usage report.
			approxCalls := replayRes.SessionsProcessed
			estUsage := Usage{
				InputTokens:  approxCalls * 1500,
				OutputTokens: approxCalls * 300,
			}
			costs.AddIngest(EstimateUSD(extractModel, estUsage))
			fmt.Fprintf(os.Stderr, "[replay] est ingest cost $%.4f (%d sessions, model=%s)\n",
				EstimateUSD(extractModel, estUsage), approxCalls, extractModel)
		}
	case "none":
		// Deliberately empty: the caller wants the runner to score
		// against a brain that has already been populated out-of-band
		// (e.g. via a prior replay run). Useful for A/B retrieval
		// changes that should not pay for extraction twice.
	default: // "bulk"
		if err := ingestBulk(ctx, evalStore, &Dataset{Questions: questions}); err != nil {
			return nil, fmt.Errorf("lme run: ingest: %w", err)
		}
	}

	var outcomes []QuestionOutcome
	readerModel := ""
	if cfg.Reader != nil && cfg.Reader.Provider != nil {
		readerModel = cfg.Reader.Model
	}
	workers := clampConcurrency(cfg.Concurrency)

	if cfg.AgenticMode {
		if cfg.Reader == nil || cfg.Reader.Provider == nil {
			return nil, fmt.Errorf("lme run: agentic mode requires cfg.Reader.Provider for the actor model")
		}
		factory := cfg.AgentFactory
		if factory == nil {
			factory = defaultAgentFactory
		}
		actorModel := cfg.Reader.Model
		outcomes = RunQuestionsAgentic(
			ctx,
			factory,
			evalStore,
			cfg.Reader.Provider,
			actorModel,
			questions,
			AgenticOpts{
				MaxIterations:   cfg.MaxIterations,
				QuestionTimeout: cfg.QuestionTimeout,
			},
			costs,
			workers,
		)
	} else {
		outcomes = runQuestions(ctx, evalStore, questions, cfg.Reader, cfg.Reranker, readerModel, costs, workers)
	}

	// Score: LLM judge when configured, exact-match fallback.
	var result *LMEResult
	if cfg.Judge != nil && cfg.Judge.Provider != nil {
		judgeResult, _, judgeUsage, err := ScoreWithJudge(ctx, *cfg.Judge, outcomes)
		if err != nil {
			return nil, fmt.Errorf("lme run: judge scoring: %w", err)
		}
		costs.AddJudge(EstimateUSD(cfg.Judge.Model, judgeUsage))
		result = judgeResult
	} else {
		result = ScoreExactMatch(outcomes)
	}

	result.DatasetSHA = ds.SHA256
	result.IngestMode = ingestMode
	result.RunSeed = cfg.Seed
	result.CostAccounting = costs.Snapshot()

	populateStats(result, cfg.Seed)

	return result, nil
}

// populateStats fills in latency percentiles and bootstrap confidence
// intervals derived from the collected question outcomes.
func populateStats(result *LMEResult, runSeed int64) {
	if result == nil || len(result.Questions) == 0 {
		return
	}

	lats := make([]int, 0, len(result.Questions))
	for _, q := range result.Questions {
		lats = append(lats, q.LatencyMs)
	}
	result.LatencyP50Ms = LatencyPercentile(lats, 50)
	result.LatencyP95Ms = LatencyPercentile(lats, 95)

	seed := runSeed
	if seed == 0 {
		seed = time.Now().UnixNano()
	}

	overallOutcomes := make([]bool, len(result.Questions))
	for i, q := range result.Questions {
		overallOutcomes[i] = isCorrectVerdict(q.JudgeVerdict)
	}
	result.OverallScoreCI = BootstrapCI(overallOutcomes, seed, 0)

	result.PerCategoryCI = BootstrapCategoryCI(
		result.Questions,
		func(q QuestionOutcome) string { return q.Category },
		func(q QuestionOutcome) bool { return isCorrectVerdict(q.JudgeVerdict) },
		seed,
		0,
	)
}

// isCorrectVerdict mirrors the aggregation in score_judge.go: both
// "correct" and "abstain_correct" are scored as hits.
func isCorrectVerdict(verdict string) bool {
	switch verdict {
	case "correct", "abstain_correct":
		return true
	default:
		return false
	}
}

// ingestBulk writes each question's haystack sessions as a flat set of
// markdown files under raw/lme/. Each file carries a YAML frontmatter
// stanza that records the session id so [searchForAnswer] can recover
// them later.
func ingestBulk(ctx context.Context, store brain.Store, ds *Dataset) error {
	seen := make(map[string]bool)
	return store.Batch(ctx, brain.BatchOptions{Reason: "lme bulk ingest"}, func(b brain.Batch) error {
		for _, q := range ds.Questions {
			for i, sess := range q.HaystackSessions {
				sid := ""
				if i < len(q.SessionIDs) {
					sid = q.SessionIDs[i]
				}
				if sid == "" {
					sid = fmt.Sprintf("%s-s%d", q.ID, i)
				}
				if seen[sid] {
					continue
				}
				seen[sid] = true

				sessionDate := ""
				if i < len(q.HaystackDates) {
					sessionDate = q.HaystackDates[i]
				}

				var body strings.Builder
				fmt.Fprintln(&body, "---")
				fmt.Fprintf(&body, "session_id: %s\n", sid)
				if sessionDate != "" {
					fmt.Fprintf(&body, "session_date: %s\n", sessionDate)
				}
				fmt.Fprintln(&body, "---")
				body.WriteByte('\n')
				for _, msg := range sess {
					fmt.Fprintf(&body, "[%s]: %s\n\n", msg.Role, msg.Content)
				}

				p := brain.Path(fmt.Sprintf("raw/lme/%s.md", sanitiseSessionID(sid)))
				if err := b.Write(ctx, p, []byte(body.String())); err != nil {
					return fmt.Errorf("write %s: %w", p, err)
				}
			}
		}
		return nil
	})
}

// sanitiseSessionID maps a raw session ID into a path-safe slug.
func sanitiseSessionID(s string) string {
	var b strings.Builder
	b.Grow(len(s))
	for _, r := range s {
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9', r == '-', r == '_':
			b.WriteRune(r)
		default:
			b.WriteByte('_')
		}
	}
	out := b.String()
	if out == "" {
		return "session"
	}
	return out
}

// runQuestions evaluates each question against the eval brain store.
// Applies the full direct-search pipeline with optional reader LLM.
func runQuestions(ctx context.Context, store brain.Store, questions []Question, reader *ReaderConfig, reranker *RerankerConfig, readerModel string, costs *CostAccumulator, workers int) []QuestionOutcome {
	return runQuestionsConcurrent(ctx, questions, workers, func(ctx context.Context, _ int, q Question) QuestionOutcome {
		return processQuestionStore(ctx, store, q, reader, reranker, readerModel, costs)
	})
}

// processQuestionStore runs the direct-search pipeline for a single
// question.
func processQuestionStore(
	ctx context.Context,
	store brain.Store,
	q Question,
	reader *ReaderConfig,
	reranker *RerankerConfig,
	readerModel string,
	costs *CostAccumulator,
) QuestionOutcome {
	if err := ctx.Err(); err != nil {
		return QuestionOutcome{
			ID:           q.ID,
			Category:     q.Category,
			Question:     q.Question,
			QuestionDate: q.QuestionDate,
			GroundTruth:  q.Answer,
			Error:        "context cancelled",
		}
	}

	qStart := time.Now()

	answer := searchForAnswer(ctx, store, q)

	if answer != "" {
		answer = processSessionContextForQuestion(answer, q.Question)
	}

	if q.QuestionDate != "" && answer != "" {
		expansion := query.ExpandTemporal(q.Question, q.QuestionDate)
		if expansion.Resolved && len(expansion.DateHints) > 0 {
			dateHints := fmt.Sprintf("[Resolved temporal references: %s]", strings.Join(expansion.DateHints, ", "))
			answer = dateHints + "\n\n" + answer
		}
	}

	_ = reranker // reranker wiring lives in the FTS path; the direct path keeps parity.

	var inputTokens, outputTokens int
	if reader != nil && answer != "" {
		readAnswer, usage, readErr := ReadAnswer(ctx, *reader, q.Question, q.QuestionDate, answer)
		answer = readAnswer
		if costs != nil {
			costs.AddAgent(EstimateUSD(readerModel, usage))
		}
		if readErr != nil {
			slog.Warn("lme reader: call failed, falling back to raw retrieval",
				"question", q.ID, "err", readErr)
		}
		inputTokens = usage.InputTokens
		outputTokens = usage.OutputTokens
	}

	return QuestionOutcome{
		ID:           q.ID,
		Category:     q.Category,
		Question:     q.Question,
		QuestionDate: q.QuestionDate,
		GroundTruth:  q.Answer,
		AgentAnswer:  answer,
		LatencyMs:    int(time.Since(qStart).Milliseconds()),
		InputTokens:  inputTokens,
		OutputTokens: outputTokens,
	}
}

// searchForAnswer collects content from ALL sessions referenced by the
// question. Multi-session questions reference 2-6 sessions whose combined
// content contains the ground-truth answer.
func searchForAnswer(ctx context.Context, store brain.Store, q Question) string {
	files, err := store.List(ctx, brain.Path("raw/lme"), brain.ListOpts{
		Recursive:        true,
		IncludeGenerated: true,
	})
	if err != nil {
		return ""
	}

	wanted := make(map[string]bool, len(q.SessionIDs))
	for _, sid := range q.SessionIDs {
		wanted[sid] = true
	}

	// Sort files for deterministic output even when the store's List
	// happens to return entries in another order.
	sort.Slice(files, func(i, j int) bool {
		return files[i].Path < files[j].Path
	})

	var parts []string
	for _, f := range files {
		if f.IsDir {
			continue
		}
		content, err := store.Read(ctx, f.Path)
		if err != nil {
			continue
		}
		contentStr := string(content)
		for sid := range wanted {
			if containsSessionID(contentStr, sid) {
				parts = append(parts, contentStr)
				break
			}
		}
	}

	return strings.Join(parts, "\n\n")
}

func containsSessionID(content, sessionID string) bool {
	return len(sessionID) > 0 && strings.Contains(content, "session_id: "+sessionID)
}

// Shutdown is a helper the CLI calls at the end of a run. Currently a
// no-op but kept so future resource teardown stays a caller concern.
func Shutdown() {
	_ = os.Stderr.Sync()
}
