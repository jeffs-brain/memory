// SPDX-License-Identifier: Apache-2.0

package lme

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"net/url"
	"os"
	"sort"
	"strings"
	"time"

	"github.com/jeffs-brain/memory/go/brain"
	"github.com/jeffs-brain/memory/go/llm"
	"github.com/jeffs-brain/memory/go/memory"
	"github.com/jeffs-brain/memory/go/query"
	"github.com/jeffs-brain/memory/go/store/mem"
	"github.com/jeffs-brain/memory/go/store/pt"
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
	// filesystem store rooted at the supplied directory. Required when
	// [ExtractOnly] is set so downstream daemons can attach to the same
	// populated brain.
	BrainCache string

	// ExtractOnly toggles the extract-once mode. When true the runner
	// loads the dataset, seeds bulk sessions, runs the replay extraction
	// pipeline if [IngestMode] selects it, writes a manifest, and exits
	// before the question-answer / judge phase. Requires [BrainCache].
	ExtractOnly bool

	// ActorEndpoint, when set, swaps the in-process retrieval + reader
	// step for an HTTP POST to {endpoint}/v1/brains/{ActorBrainID}/ask.
	// The daemon responds over SSE; the runner stitches answer_delta
	// frames into the agent answer and feeds it straight to the judge.
	ActorEndpoint string

	// ActorBrainID identifies the brain the ActorEndpoint should query.
	// Empty defers to the caller-configured brain id (e.g. "eval-lme").
	ActorBrainID string

	// ActorEndpointStyle selects how the runner uses [ActorEndpoint]:
	//   - "" or "full" (default): POST to /v1/brains/{id}/ask, stream
	//     answer_delta frames, judge in-process. Daemon owns retrieval
	//     AND reading, so prompt/reader variations leak across SDKs.
	//   - "retrieve-only": POST to /v1/brains/{id}/search, then apply
	//     the in-process augmented CoT reader + judge locally. Daemon
	//     acts as a pure retrieval substrate so every SDK is scored
	//     against the same reader prompt and judge config.
	ActorEndpointStyle string

	// ActorTopK overrides the default retrieval breadth per question.
	// Zero selects 20, which keeps multi-session recall healthy while
	// staying comfortably under reader budget. LongMemEval's multi-
	// session questions reference 2-6 sessions so top-5 BM25 rankings
	// frequently miss the supporting evidence.
	ActorTopK int

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

const (
	defaultConcurrency = 8
	minConcurrency     = 1
	maxConcurrency     = 256
)

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

	if cfg.ExtractOnly && cfg.BrainCache == "" {
		return nil, fmt.Errorf("lme run: extract-only requires a persistent --brain-cache path")
	}
	if cfg.ActorEndpoint != "" {
		if _, perr := url.Parse(cfg.ActorEndpoint); perr != nil {
			return nil, fmt.Errorf("lme run: invalid --actor-endpoint %q: %w", cfg.ActorEndpoint, perr)
		}
	}

	var evalStore brain.Store = cfg.Store
	if evalStore == nil {
		if cfg.BrainCache != "" {
			// Use the passthrough store so every logical path lands at
			// root/path on disk with no remapping. The HTTP daemon and
			// every SDK reads the brain root the same way, so a
			// populated cache is directly consumable by downstream
			// `memory serve` processes.
			ptStore, ferr := pt.New(cfg.BrainCache)
			if ferr != nil {
				return nil, fmt.Errorf("lme run: open brain cache %s: %w", cfg.BrainCache, ferr)
			}
			evalStore = ptStore
		} else {
			evalStore = mem.New()
		}
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

	// Extract-only mode short-circuits before the answer + judge phase so
	// callers can feed the populated brain to a long-lived daemon. The
	// returned LMEResult carries the extraction-side bookkeeping only.
	if cfg.ExtractOnly {
		result := &LMEResult{
			QuestionsRun:   len(questions),
			Questions:      nil,
			ByCategory:     map[string]Category{},
			DatasetSHA:     ds.SHA256,
			IngestMode:     ingestMode,
			RunSeed:        cfg.Seed,
			CostAccounting: costs.Snapshot(),
		}
		fmt.Fprintf(os.Stderr,
			"[extract-only] done: sessions=unique questions=%d ingest_mode=%s wall=%s cost=$%.4f brain=%s\n",
			len(questions), ingestMode, time.Since(start).Truncate(time.Second),
			result.CostAccounting.TotalUSD, cfg.BrainCache)
		return result, nil
	}

	var outcomes []QuestionOutcome
	readerModel := ""
	if cfg.Reader != nil && cfg.Reader.Provider != nil {
		readerModel = cfg.Reader.Model
	}
	workers := clampConcurrency(cfg.Concurrency)

	switch {
	case cfg.ActorEndpoint != "":
		brainID := cfg.ActorBrainID
		if brainID == "" {
			brainID = "eval-lme"
		}
		style := strings.ToLower(strings.TrimSpace(cfg.ActorEndpointStyle))
		if style == "" {
			style = "full"
		}
		fmt.Fprintf(os.Stderr, "[actor] endpoint=%s brain=%s style=%s workers=%d\n", cfg.ActorEndpoint, brainID, style, workers)
		topK := cfg.ActorTopK
		if topK <= 0 {
			topK = 20
		}
		switch style {
		case "retrieve-only":
			outcomes = runQuestionsActorRetrieveOnly(ctx, cfg.ActorEndpoint, brainID, questions, cfg.Reader, readerModel, costs, workers, topK)
		case "full":
			outcomes = runQuestionsActor(ctx, cfg.ActorEndpoint, brainID, questions, workers)
		default:
			return nil, fmt.Errorf("lme run: unknown --actor-endpoint-style %q (want 'full' or 'retrieve-only')", cfg.ActorEndpointStyle)
		}
	case cfg.AgenticMode:
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
	default:
		outcomes = runQuestions(ctx, evalStore, questions, cfg.Reader, cfg.Reranker, readerModel, costs, workers)
	}

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

func runQuestions(ctx context.Context, store brain.Store, questions []Question, reader *ReaderConfig, reranker *RerankerConfig, readerModel string, costs *CostAccumulator, workers int) []QuestionOutcome {
	return runQuestionsConcurrent(ctx, questions, workers, func(ctx context.Context, _ int, q Question) QuestionOutcome {
		return processQuestionStore(ctx, store, q, reader, reranker, readerModel, costs)
	})
}

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

// actorHTTPClient's long timeout covers the SSE streaming window;
// individual requests honour the caller's context.
var actorHTTPClient = &http.Client{Timeout: 5 * time.Minute}

// runQuestionsActor answers every question by POSTing to the actor
// daemon's /v1/brains/{id}/ask endpoint. Retrieval and reading live in
// the daemon, so this path collapses the in-process pipeline to a single
// HTTP + SSE round trip per question. The judge still runs in-process so
// judge config stays comparable across SDKs.
func runQuestionsActor(
	ctx context.Context,
	endpoint, brainID string,
	questions []Question,
	workers int,
) []QuestionOutcome {
	return runQuestionsConcurrent(ctx, questions, workers, func(ctx context.Context, _ int, q Question) QuestionOutcome {
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
		answer, usage, err := callActorEndpoint(ctx, endpoint, brainID, q.Question, q.QuestionDate)
		outcome := QuestionOutcome{
			ID:           q.ID,
			Category:     q.Category,
			Question:     q.Question,
			QuestionDate: q.QuestionDate,
			GroundTruth:  q.Answer,
			AgentAnswer:  answer,
			LatencyMs:    int(time.Since(qStart).Milliseconds()),
			InputTokens:  usage.InputTokens,
			OutputTokens: usage.OutputTokens,
		}
		if err != nil {
			outcome.Error = err.Error()
			slog.Warn("lme actor: call failed", "question", q.ID, "err", err)
		}
		return outcome
	})
}

// runQuestionsActorRetrieveOnly treats the actor daemon as a pure
// retrieval substrate. It POSTs to {endpoint}/v1/brains/{id}/search for
// every question, then runs the augmented CoT reader + judge in-process
// so the per-question prompt is identical across SDKs. This isolates
// retrieval quality as the only variable that changes between runs.
func runQuestionsActorRetrieveOnly(
	ctx context.Context,
	endpoint, brainID string,
	questions []Question,
	reader *ReaderConfig,
	readerModel string,
	costs *CostAccumulator,
	workers int,
	topK int,
) []QuestionOutcome {
	return runQuestionsConcurrent(ctx, questions, workers, func(ctx context.Context, _ int, q Question) QuestionOutcome {
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
		content, searchErr := callActorRetrieve(ctx, endpoint, brainID, q.Question, topK)
		if searchErr != nil {
			slog.Warn("lme actor retrieve: call failed", "question", q.ID, "err", searchErr)
			return QuestionOutcome{
				ID:           q.ID,
				Category:     q.Category,
				Question:     q.Question,
				QuestionDate: q.QuestionDate,
				GroundTruth:  q.Answer,
				Error:        searchErr.Error(),
				LatencyMs:    int(time.Since(qStart).Milliseconds()),
			}
		}

		if content != "" {
			content = processSessionContextForQuestion(content, q.Question)
		}
		if q.QuestionDate != "" && content != "" {
			expansion := query.ExpandTemporal(q.Question, q.QuestionDate)
			if expansion.Resolved && len(expansion.DateHints) > 0 {
				dateHints := fmt.Sprintf("[Resolved temporal references: %s]", strings.Join(expansion.DateHints, ", "))
				content = dateHints + "\n\n" + content
			}
		}

		var inputTokens, outputTokens int
		answer := content
		if reader != nil && content != "" {
			readAnswer, usage, readErr := ReadAnswer(ctx, *reader, q.Question, q.QuestionDate, content)
			answer = readAnswer
			if costs != nil {
				costs.AddAgent(EstimateUSD(readerModel, usage))
			}
			if readErr != nil {
				slog.Warn("lme reader (retrieve-only): call failed, falling back to raw retrieval",
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
	})
}

// callActorRetrieve posts a single query to the actor daemon's /search
// endpoint and returns the retrieved chunks concatenated into a reader
// content string. Chunk text is joined with blank-line separators so
// the downstream session-block parser can split them back out when a
// session id header is present.
func callActorRetrieve(ctx context.Context, endpoint, brainID, question string, topK int) (string, error) {
	if endpoint == "" {
		return "", fmt.Errorf("lme: actor endpoint is empty")
	}
	if topK <= 0 {
		topK = 5
	}
	base := strings.TrimRight(endpoint, "/")
	reqBody := map[string]any{
		"query": question,
		"topK":  topK,
		"mode":  "hybrid",
	}
	body, err := json.Marshal(reqBody)
	if err != nil {
		return "", fmt.Errorf("lme actor retrieve: marshal: %w", err)
	}
	searchURL := fmt.Sprintf("%s/v1/brains/%s/search", base, url.PathEscape(brainID))
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, searchURL, bytes.NewReader(body))
	if err != nil {
		return "", fmt.Errorf("lme actor retrieve: build request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	resp, err := actorHTTPClient.Do(httpReq)
	if err != nil {
		return "", fmt.Errorf("lme actor retrieve: post: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 400 {
		buf := make([]byte, 512)
		n, _ := resp.Body.Read(buf)
		return "", fmt.Errorf("lme actor retrieve: HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(buf[:n])))
	}

	var decoded struct {
		Chunks []struct {
			Path    string `json:"path"`
			Text    string `json:"text"`
			Title   string `json:"title"`
			Summary string `json:"summary"`
			Score   float64 `json:"score"`
		} `json:"chunks"`
	}
	dec := json.NewDecoder(resp.Body)
	dec.UseNumber()
	if err := dec.Decode(&decoded); err != nil {
		return "", fmt.Errorf("lme actor retrieve: decode: %w", err)
	}
	if len(decoded.Chunks) == 0 {
		return "", nil
	}
	var b strings.Builder
	for i, c := range decoded.Chunks {
		if i > 0 {
			b.WriteString("\n\n")
		}
		body := c.Text
		if body == "" {
			body = c.Summary
		}
		b.WriteString(body)
	}
	return b.String(), nil
}

// callActorEndpoint posts a single question to an HTTP actor daemon and
// consumes the SSE stream, stitching answer_delta frames into the final
// answer. The daemon's ask endpoint is spec-compliant so this works
// identically against the TS, Go, and Python SDKs.
func callActorEndpoint(ctx context.Context, endpoint, brainID, question, questionDate string) (string, Usage, error) {
	if endpoint == "" {
		return "", Usage{}, fmt.Errorf("lme: actor endpoint is empty")
	}
	base := strings.TrimRight(endpoint, "/")
	// Opt the daemon into the LME CoT reader prompt so full-mode scores
	// reflect the same reading strategy the retrieve-only path uses and
	// the paper specifies. Daemons that don't yet understand readerMode
	// ignore it, preserving backward compatibility.
	reqBody := map[string]any{
		"question":     question,
		"topK":         5,
		"mode":         "hybrid",
		"readerMode":   "augmented",
		"questionDate": questionDate,
	}
	body, err := json.Marshal(reqBody)
	if err != nil {
		return "", Usage{}, fmt.Errorf("lme actor: marshal: %w", err)
	}
	askURL := fmt.Sprintf("%s/v1/brains/%s/ask", base, url.PathEscape(brainID))
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, askURL, bytes.NewReader(body))
	if err != nil {
		return "", Usage{}, fmt.Errorf("lme actor: build request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Accept", "text/event-stream")
	resp, err := actorHTTPClient.Do(httpReq)
	if err != nil {
		return "", Usage{}, fmt.Errorf("lme actor: post: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 400 {
		buf := make([]byte, 512)
		n, _ := resp.Body.Read(buf)
		return "", Usage{}, fmt.Errorf("lme actor: HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(buf[:n])))
	}
	var answer strings.Builder
	var usage Usage
	scanner := bufio.NewScanner(resp.Body)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	var currentEvent string
	for scanner.Scan() {
		line := scanner.Text()
		if line == "" {
			currentEvent = ""
			continue
		}
		if strings.HasPrefix(line, "event:") {
			currentEvent = strings.TrimSpace(strings.TrimPrefix(line, "event:"))
			continue
		}
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		payload := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if payload == "" {
			continue
		}
		switch currentEvent {
		case "answer_delta":
			var delta struct {
				Text string `json:"text"`
			}
			if err := json.Unmarshal([]byte(payload), &delta); err == nil {
				answer.WriteString(delta.Text)
			}
		case "done":
			var doneFrame struct {
				OK    bool `json:"ok"`
				Usage struct {
					InputTokens  int `json:"input_tokens"`
					OutputTokens int `json:"output_tokens"`
				} `json:"usage"`
			}
			if err := json.Unmarshal([]byte(payload), &doneFrame); err == nil {
				usage.InputTokens += doneFrame.Usage.InputTokens
				usage.OutputTokens += doneFrame.Usage.OutputTokens
			}
		case "error":
			var errFrame struct {
				Message string `json:"message"`
			}
			if err := json.Unmarshal([]byte(payload), &errFrame); err == nil && errFrame.Message != "" {
				return answer.String(), usage, fmt.Errorf("lme actor: stream error: %s", errFrame.Message)
			}
		}
	}
	if err := scanner.Err(); err != nil {
		return answer.String(), usage, fmt.Errorf("lme actor: stream read: %w", err)
	}
	return answer.String(), usage, nil
}
