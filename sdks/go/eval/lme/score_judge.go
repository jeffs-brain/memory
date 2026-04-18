// SPDX-License-Identifier: Apache-2.0

package lme

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math/rand/v2"
	"os"
	"strings"
	"time"

	"github.com/jeffs-brain/memory/go/llm"
)

// judgeProgressInterval matches the ingest / question logger cadence so a
// single eval run's stderr has a predictable heartbeat.
const judgeProgressInterval = 50

// judgeVerdictSchema constrains the structured judge response. Mirrors
// the official LongMemEval prompt: binary yes/no alongside a free-form
// rationale.
var judgeVerdictSchema = json.RawMessage(`{
  "type": "object",
  "additionalProperties": false,
  "properties": {
    "verdict": {
      "type": "string",
      "enum": ["yes", "no"]
    },
    "rationale": {
      "type": "string"
    }
  },
  "required": ["verdict", "rationale"]
}`)

const (
	// judgeMaxTokens matches the official evaluate_qa.py.
	judgeMaxTokens   = 10
	judgeTemperature = 0.0

	// structuredJudgeMaxTokens is the output cap for the structured
	// judge path. Slightly larger than judgeMaxTokens so the rationale
	// string has room.
	structuredJudgeMaxTokens = 256

	// defaultJudgeContentBudget is the head-room budget (in characters)
	// used when we cannot infer anything from the provider's context
	// window.
	defaultJudgeContentBudget = 100_000

	// conservativeJudgeContentBudget is the historical 40 K cap for
	// small-context models.
	conservativeJudgeContentBudget = 40_000

	// judgeContentBudgetFloor keeps the helper from returning a cap that
	// is too small to be useful even when the model has a tiny window.
	judgeContentBudgetFloor = 16_000

	// judgeContentBudgetCeiling caps the inferred budget so we never send
	// multi-hundred-KB prompts even when the window would allow it.
	judgeContentBudgetCeiling = 200_000
)

// JudgeConfig controls the LLM judge scorer.
type JudgeConfig struct {
	Provider    llm.Provider
	Model       string
	MaxRetries  int
	Concurrency int

	// Timeout caps a single judge call end-to-end. Zero disables the
	// per-call timeout and the call honours only the parent context.
	Timeout time.Duration

	// ContentBudget caps the character length of the agent answer passed
	// to the judge. Zero means "infer from the provider's context window"
	// via [judgeContentBudgetFor].
	ContentBudget int

	// Costs, when non-nil, receives per-call judge spend through the
	// usage-hook callback.
	Costs *CostAccumulator
}

// maxContextProvider is an optional interface providers may satisfy to
// report their context window to the LME judge. Providers that do not
// implement it default to [defaultJudgeContentBudget].
type maxContextProvider interface {
	MaxContextTokens() int
}

// judgeContentBudgetFor returns the char budget for the given provider's
// judge model, inferred from its context window when the provider
// exposes one.
func judgeContentBudgetFor(p llm.Provider) int {
	if p == nil {
		return defaultJudgeContentBudget
	}
	mcp, ok := p.(maxContextProvider)
	if !ok {
		return defaultJudgeContentBudget
	}
	maxCtx := mcp.MaxContextTokens()
	if maxCtx <= 0 {
		return defaultJudgeContentBudget
	}
	est := int(float64(maxCtx) * 2.5 * 0.7)
	if est < judgeContentBudgetFloor {
		return judgeContentBudgetFloor
	}
	if est > judgeContentBudgetCeiling {
		return judgeContentBudgetCeiling
	}
	return est
}

// resolveJudgeContentBudget picks an explicit cfg.ContentBudget when set
// or defers to [judgeContentBudgetFor].
func resolveJudgeContentBudget(cfg JudgeConfig) int {
	if cfg.ContentBudget > 0 {
		return cfg.ContentBudget
	}
	return judgeContentBudgetFor(cfg.Provider)
}

// JudgeVerdict is the parsed result from the LLM judge.
type JudgeVerdict struct {
	Verdict   string `json:"verdict"`
	Rationale string `json:"rationale"`
}

// JudgeTrace records the full judge call for audit.
type JudgeTrace struct {
	QuestionID    string       `json:"question_id"`
	Verdict       JudgeVerdict `json:"verdict"`
	RawResponse   string       `json:"raw_response"`
	PromptVersion int          `json:"prompt_version"`
	LatencyMs     int64        `json:"latency_ms"`
	Retries       int          `json:"retries"`
	ContentChars  int          `json:"content_chars,omitempty"`
	Error         string       `json:"error,omitempty"`
}

// ScoreWithJudge evaluates question outcomes using an LLM judge. Falls
// back to exact-match on judge failure per question. The third return is
// the aggregate token usage across every judge call so the caller can
// convert it into cost accounting.
func ScoreWithJudge(ctx context.Context, cfg JudgeConfig, outcomes []QuestionOutcome) (*LMEResult, []JudgeTrace, Usage, error) {
	if cfg.MaxRetries <= 0 {
		cfg.MaxRetries = 1
	}
	if cfg.Concurrency <= 0 {
		cfg.Concurrency = 1
	}

	traces := make([]JudgeTrace, len(outcomes))
	var totalUsage Usage

	retriesAtStart := TransientRetriesTotal()
	judgeStart := time.Now()
	done := 0

	for i, o := range outcomes {
		if o.Error != "" {
			outcomes[i].JudgeVerdict = "error"
			traces[i] = JudgeTrace{
				QuestionID: o.ID,
				Verdict:    JudgeVerdict{Verdict: "error"},
				Error:      o.Error,
			}
			done++
			maybeLogJudgeProgress(done, len(outcomes), judgeStart, retriesAtStart, o.ID, trace0Status(traces[i]), traces[i].LatencyMs)
			continue
		}

		qCtx := ctx
		var cancel context.CancelFunc
		if cfg.Timeout > 0 {
			qCtx, cancel = context.WithTimeout(ctx, cfg.Timeout)
		}
		verdict, trace, usage := judgeQuestion(qCtx, cfg, o)
		if cancel != nil {
			cancel()
		}
		traces[i] = trace
		outcomes[i].JudgeVerdict = verdict.Verdict
		outcomes[i].JudgeRationale = verdict.Rationale
		totalUsage.InputTokens += usage.InputTokens
		totalUsage.OutputTokens += usage.OutputTokens
		totalUsage.CacheRead += usage.CacheRead
		totalUsage.CacheCreate += usage.CacheCreate
		done++
		maybeLogJudgeProgress(done, len(outcomes), judgeStart, retriesAtStart, o.ID, trace0Status(trace), trace.LatencyMs)
	}

	retriesTotal := TransientRetriesTotal() - retriesAtStart
	fmt.Fprintf(os.Stderr, "[judge] done %d/%d in %s (avg=%dms per call, transient_retries=%d)\n",
		done, len(outcomes), time.Since(judgeStart).Truncate(time.Millisecond),
		avgMillis(time.Since(judgeStart), done), retriesTotal)

	// Aggregate scores.
	byCategory := make(map[string]*Category)
	correct := 0

	for _, o := range outcomes {
		cat, ok := byCategory[o.Category]
		if !ok {
			cat = &Category{}
			byCategory[o.Category] = cat
		}
		cat.Run++

		switch o.JudgeVerdict {
		case "correct", "abstain_correct":
			cat.Correct++
			correct++
		case "partial":
			cat.Partial++
		default:
			cat.Incorrect++
		}
	}

	catMap := make(map[string]Category, len(byCategory))
	for name, cat := range byCategory {
		if cat.Run > 0 {
			cat.Score = float64(cat.Correct) / float64(cat.Run)
		}
		catMap[name] = *cat
	}

	overall := 0.0
	if len(outcomes) > 0 {
		overall = float64(correct) / float64(len(outcomes))
	}

	exactCorrect := 0
	for _, o := range outcomes {
		if exactMatch(o.AgentAnswer, o.GroundTruth) {
			exactCorrect++
		}
	}
	exactScore := 0.0
	if len(outcomes) > 0 {
		exactScore = float64(exactCorrect) / float64(len(outcomes))
	}

	taskAvg := 0.0
	if len(catMap) > 0 {
		sum := 0.0
		for _, cat := range catMap {
			sum += cat.Score
		}
		taskAvg = sum / float64(len(catMap))
	}

	absCorrect, absTotal := 0, 0
	for _, o := range outcomes {
		if strings.Contains(o.ID, "_abs") {
			absTotal++
			if o.JudgeVerdict == "abstain_correct" {
				absCorrect++
			}
		}
	}
	absScore := 0.0
	if absTotal > 0 {
		absScore = float64(absCorrect) / float64(absTotal)
	}

	result := &LMEResult{
		QuestionsRun:    len(outcomes),
		OverallScore:    overall,
		TaskAvgScore:    taskAvg,
		AbstentionScore: absScore,
		ExactMatchScore: exactScore,
		ByCategory:      catMap,
		Questions:       outcomes,
		JudgeModel:      cfg.Model,
	}

	return result, traces, totalUsage, nil
}

func judgeQuestion(ctx context.Context, cfg JudgeConfig, o QuestionOutcome) (JudgeVerdict, JudgeTrace, Usage) {
	start := time.Now()
	trace := JudgeTrace{
		QuestionID:    o.ID,
		PromptVersion: JudgePromptVersion,
	}
	var usage Usage

	isAbstention := strings.Contains(o.ID, "_abs")

	budget := resolveJudgeContentBudget(cfg)
	agentAnswer := truncateSmartly(o.AgentAnswer, budget)
	trace.ContentChars = len(agentAnswer)
	prompt := formatJudgePrompt(o.Category, isAbstention, o.Question, o.GroundTruth, agentAnswer, o.QuestionDate)

	req := llm.CompleteRequest{
		Model: cfg.Model,
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: prompt},
		},
		MaxTokens:   structuredJudgeMaxTokens,
		Temperature: judgeTemperature,
	}

	rawJSON, err := completeJudgeWithTransientRetry(
		ctx,
		cfg,
		req,
		judgeVerdictSchema,
		func(u Usage) {
			usage.InputTokens += u.InputTokens
			usage.OutputTokens += u.OutputTokens
			usage.CacheRead += u.CacheRead
			usage.CacheCreate += u.CacheCreate
			if cfg.Costs != nil {
				cfg.Costs.AddJudge(EstimateUSD(cfg.Model, u))
			}
		},
	)
	if err == nil {
		verdict, parseErr := parseStructuredVerdict(rawJSON, isAbstention)
		if parseErr == nil {
			trace.RawResponse = string(rawJSON)
			trace.Verdict = verdict
			trace.LatencyMs = time.Since(start).Milliseconds()
			return verdict, trace, usage
		}
		err = parseErr
	}

	trace.LatencyMs = time.Since(start).Milliseconds()
	trace.Retries = cfg.MaxRetries
	trace.Error = fmt.Sprintf("judge failed after %d attempts: %v", cfg.MaxRetries, err)

	var schemaErr *SchemaValidationError
	if errors.As(err, &schemaErr) {
		trace.RawResponse = string(schemaErr.RawPayload)
	}

	var verdict JudgeVerdict
	if exactMatch(o.AgentAnswer, o.GroundTruth) {
		verdict = JudgeVerdict{Verdict: "correct", Rationale: "exact-match fallback"}
	} else {
		verdict = JudgeVerdict{Verdict: "incorrect", Rationale: "exact-match fallback"}
	}
	trace.Verdict = verdict
	return verdict, trace, usage
}

// parseStructuredVerdict unmarshals a completeJSON payload and maps the
// verdict enum onto the canonical correct / incorrect / abstain_correct
// / abstain_incorrect labels.
func parseStructuredVerdict(raw json.RawMessage, isAbstention bool) (JudgeVerdict, error) {
	var payload struct {
		Verdict   string `json:"verdict"`
		Rationale string `json:"rationale"`
	}
	if err := json.Unmarshal(raw, &payload); err != nil {
		return JudgeVerdict{}, fmt.Errorf("decode judge payload: %w", err)
	}

	switch strings.ToLower(strings.TrimSpace(payload.Verdict)) {
	case "yes":
		verdict := "correct"
		if isAbstention {
			verdict = "abstain_correct"
		}
		return JudgeVerdict{Verdict: verdict, Rationale: payload.Rationale}, nil
	case "no":
		verdict := "incorrect"
		if isAbstention {
			verdict = "abstain_incorrect"
		}
		return JudgeVerdict{Verdict: verdict, Rationale: payload.Rationale}, nil
	default:
		return JudgeVerdict{}, fmt.Errorf("unexpected verdict %q", payload.Verdict)
	}
}

// parseYesNo parses the binary yes/no response from the official LME
// judge prompts. Match the official evaluate_qa.py logic: substring
// check for "yes" anywhere in response.
func parseYesNo(content string, isAbstention bool) (JudgeVerdict, error) {
	lower := strings.ToLower(strings.TrimSpace(content))
	if lower == "" {
		return JudgeVerdict{}, fmt.Errorf("empty judge response")
	}
	switch {
	case strings.Contains(lower, "yes"):
		verdict := "correct"
		if isAbstention {
			verdict = "abstain_correct"
		}
		return JudgeVerdict{Verdict: verdict, Rationale: content}, nil
	case strings.Contains(lower, "no"):
		verdict := "incorrect"
		if isAbstention {
			verdict = "abstain_incorrect"
		}
		return JudgeVerdict{Verdict: verdict, Rationale: content}, nil
	default:
		return JudgeVerdict{}, fmt.Errorf("expected yes/no, got: %q", content)
	}
}

// truncateSmartly splits content by session boundaries and allocates the
// character budget proportionally across sections.
func truncateSmartly(content string, budget int) string {
	if len(content) <= budget {
		return content
	}

	sections := splitSessions(content)
	if len(sections) <= 1 {
		return headTailTruncate(content, budget)
	}

	totalLen := 0
	for _, s := range sections {
		totalLen += len(s)
	}

	var parts []string
	remainingBudget := budget
	for i, section := range sections {
		alloc := min(max((len(section)*budget)/totalLen, 500), remainingBudget)
		if i == len(sections)-1 {
			alloc = remainingBudget
		}

		if len(section) <= alloc {
			parts = append(parts, section)
		} else {
			parts = append(parts, headTailTruncate(section, alloc))
		}
		remainingBudget -= min(len(section), alloc)
		if remainingBudget <= 0 {
			break
		}
	}

	return strings.Join(parts, "\n\n---\n\n")
}

// splitSessions splits content by session boundaries.
func splitSessions(content string) []string {
	parts := strings.Split(content, "\n\n---\nsession_id:")
	if len(parts) <= 1 {
		return []string{content}
	}
	result := make([]string, len(parts))
	result[0] = parts[0]
	for i := 1; i < len(parts); i++ {
		result[i] = "---\nsession_id:" + parts[i]
	}
	return result
}

// maybeLogJudgeProgress emits one [judge] line every
// judgeProgressInterval completions.
func maybeLogJudgeProgress(done, total int, start time.Time, retriesBaseline int64, qID, status string, latencyMs int64) {
	if done == 0 {
		return
	}
	if done != 1 && done%judgeProgressInterval != 0 && done != total {
		return
	}
	elapsed := time.Since(start)
	rate := 0.0
	if elapsed > 0 {
		rate = float64(done) / elapsed.Seconds()
	}
	eta := "n/a"
	remaining := total - done
	if rate > 0 && remaining > 0 {
		eta = time.Duration(float64(remaining)/rate * float64(time.Second)).Truncate(time.Second).String()
	}
	retries := TransientRetriesTotal() - retriesBaseline
	short := qID
	if len(short) > 16 {
		short = short[:16] + "..."
	}
	fmt.Fprintf(os.Stderr, "[judge] %d/%d q=%s %s (%dms) rate=%.1f/s eta=%s retries=%d\n",
		done, total, short, status, latencyMs, rate, eta, retries)
}

// avgMillis returns the average millisecond cost per call.
func avgMillis(elapsed time.Duration, done int) int64 {
	if done == 0 {
		return 0
	}
	return elapsed.Milliseconds() / int64(done)
}

// trace0Status maps a judge trace to a short status token for the
// progress line.
func trace0Status(t JudgeTrace) string {
	if t.Error != "" {
		return "err"
	}
	return "ok"
}

// completeJudgeWithTransientRetry wraps completeJSON with an outer retry
// loop that fires on [ErrTransient] and context deadline exceeded.
// Backoff is exponential (1s, 2s, 4s) with ±20% jitter, capped at 30s
// per attempt.
func completeJudgeWithTransientRetry(
	ctx context.Context,
	cfg JudgeConfig,
	req llm.CompleteRequest,
	schema json.RawMessage,
	usageHook func(Usage),
) (json.RawMessage, error) {
	attempts := cfg.MaxRetries
	if attempts < 1 {
		attempts = 1
	}
	var lastErr error
	for attempt := 0; attempt < attempts; attempt++ {
		raw, err := completeJSON(ctx, cfg.Provider, req, schema,
			withMaxRetries(cfg.MaxRetries),
			withUsageHook(usageHook),
		)
		if err == nil {
			return raw, nil
		}
		if !isTransientErr(err) {
			return nil, err
		}
		lastErr = err
		if attempt == attempts-1 {
			break
		}
		IncTransientRetry()
		base := time.Duration(1<<uint(attempt)) * time.Second
		jitter := time.Duration(float64(base) * (rand.Float64()*0.4 - 0.2))
		delay := base + jitter
		if delay > 30*time.Second {
			delay = 30 * time.Second
		}
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-time.After(delay):
		}
	}
	return nil, lastErr
}

// headTailTruncate keeps the first 70% and last 30% of the budget,
// joining them with a truncation marker.
func headTailTruncate(s string, budget int) string {
	if len(s) <= budget {
		return s
	}
	marker := "\n[...truncated...]\n"
	head := budget * 70 / 100
	tail := max(budget-head-len(marker), 0)
	return s[:head] + marker + s[len(s)-tail:]
}
