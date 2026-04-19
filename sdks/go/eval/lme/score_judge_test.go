// SPDX-License-Identifier: Apache-2.0

package lme

import (
	"context"
	"fmt"
	"strings"
	"testing"

	"github.com/jeffs-brain/memory/go/llm"
)

func TestJudgeVerdict_Correct(t *testing.T) {
	fp := &scriptedProvider{
		responses: []llm.CompleteResponse{
			{Text: `{"verdict": "yes", "rationale": "answer matches"}`},
		},
	}

	outcomes := []QuestionOutcome{
		{ID: "q1", Category: "single-session", Question: "What colour?", GroundTruth: "red", AgentAnswer: "The answer is red."},
	}

	result, traces, _, err := ScoreWithJudge(context.Background(), JudgeConfig{
		Provider:   fp,
		Model:      "fake-judge",
		MaxRetries: 1,
	}, outcomes)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if result.OverallScore != 1.0 {
		t.Errorf("OverallScore = %v, want 1.0", result.OverallScore)
	}
	if traces[0].Verdict.Verdict != "correct" {
		t.Errorf("trace verdict = %q, want correct", traces[0].Verdict.Verdict)
	}
	if result.Questions[0].JudgeVerdict != "correct" {
		t.Errorf("outcome JudgeVerdict = %q, want correct", result.Questions[0].JudgeVerdict)
	}
}

func TestJudgeVerdict_Partial(t *testing.T) {
	fp := &scriptedProvider{
		responses: []llm.CompleteResponse{
			{Text: `{"verdict": "no", "rationale": "missing fact C"}`},
		},
	}

	outcomes := []QuestionOutcome{
		{ID: "q1", Category: "multi-session", Question: "List all facts", GroundTruth: "A, B, C", AgentAnswer: "A and B"},
	}

	result, traces, _, err := ScoreWithJudge(context.Background(), JudgeConfig{
		Provider:   fp,
		Model:      "fake-judge",
		MaxRetries: 1,
	}, outcomes)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if result.OverallScore != 0.0 {
		t.Errorf("OverallScore = %v, want 0.0", result.OverallScore)
	}
	if traces[0].Verdict.Verdict != "incorrect" {
		t.Errorf("trace verdict = %q, want incorrect", traces[0].Verdict.Verdict)
	}
	cat := result.ByCategory["multi-session"]
	if cat.Incorrect != 1 {
		t.Errorf("category incorrect = %d, want 1", cat.Incorrect)
	}
}

func TestJudgeVerdict_Incorrect(t *testing.T) {
	fp := &scriptedProvider{
		responses: []llm.CompleteResponse{
			{Text: `{"verdict": "no", "rationale": "wrong year"}`},
		},
	}

	outcomes := []QuestionOutcome{
		{ID: "q1", Category: "temporal", Question: "When?", GroundTruth: "2024", AgentAnswer: "2023"},
	}

	result, _, _, err := ScoreWithJudge(context.Background(), JudgeConfig{
		Provider:   fp,
		Model:      "fake-judge",
		MaxRetries: 1,
	}, outcomes)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if result.OverallScore != 0.0 {
		t.Errorf("OverallScore = %v, want 0.0", result.OverallScore)
	}
	cat := result.ByCategory["temporal"]
	if cat.Incorrect != 1 {
		t.Errorf("category incorrect = %d, want 1", cat.Incorrect)
	}
}

func TestJudgeVerdict_AbstainCorrect(t *testing.T) {
	fp := &scriptedProvider{
		responses: []llm.CompleteResponse{
			{Text: `{"verdict": "yes", "rationale": "model abstained correctly"}`},
		},
	}

	outcomes := []QuestionOutcome{
		{ID: "q1_abs", Category: "single-session-user", Question: "What is my hamster's name?", GroundTruth: "You did not mention this information.", AgentAnswer: "I don't have that information."},
	}

	result, _, _, err := ScoreWithJudge(context.Background(), JudgeConfig{
		Provider:   fp,
		Model:      "fake-judge",
		MaxRetries: 1,
	}, outcomes)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if result.OverallScore != 1.0 {
		t.Errorf("OverallScore = %v, want 1.0 (abstain_correct counts as correct)", result.OverallScore)
	}
}

func TestJudgeVerdict_AbstainIncorrect(t *testing.T) {
	fp := &scriptedProvider{
		responses: []llm.CompleteResponse{
			{Text: `{"verdict": "no", "rationale": "model invented an answer"}`},
		},
	}

	outcomes := []QuestionOutcome{
		{ID: "q1_abs", Category: "single-session-user", Question: "What is my hamster's name?", GroundTruth: "You did not mention this information.", AgentAnswer: "I think it is Fluffy."},
	}

	result, _, _, err := ScoreWithJudge(context.Background(), JudgeConfig{
		Provider:   fp,
		Model:      "fake-judge",
		MaxRetries: 1,
	}, outcomes)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if result.OverallScore != 0.0 {
		t.Errorf("OverallScore = %v, want 0.0 (abstain_incorrect is not correct)", result.OverallScore)
	}
}

func TestJudgeVerdict_DeterministicAbstentionMismatchSkipsJudge(t *testing.T) {
	fp := &scriptedProvider{
		responses: []llm.CompleteResponse{
			{Text: `{"verdict": "yes", "rationale": "judge noise"}`},
		},
	}

	outcomes := []QuestionOutcome{
		{
			ID:          "q1",
			Category:    "single-session",
			Question:    "How much cashback did I earn?",
			GroundTruth: "$0.75",
			AgentAnswer: "The information provided is not enough to answer the question.",
		},
	}

	result, traces, _, err := ScoreWithJudge(context.Background(), JudgeConfig{
		Provider:   fp,
		Model:      "fake-judge",
		MaxRetries: 1,
	}, outcomes)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if result.Questions[0].JudgeVerdict != "incorrect" {
		t.Fatalf("JudgeVerdict = %q, want incorrect", result.Questions[0].JudgeVerdict)
	}
	if traces[0].Verdict.Verdict != "incorrect" {
		t.Fatalf("trace verdict = %q, want incorrect", traces[0].Verdict.Verdict)
	}
	if traces[0].Verdict.Rationale != "deterministic abstention mismatch" {
		t.Fatalf("trace rationale = %q", traces[0].Verdict.Rationale)
	}
	if fp.callIdx != 0 {
		t.Fatalf("provider called %d times, want 0", fp.callIdx)
	}
}

func TestJudgeVerdict_DeterministicAbstentionCorrectSkipsJudge(t *testing.T) {
	fp := &scriptedProvider{
		responses: []llm.CompleteResponse{
			{Text: `{"verdict": "no", "rationale": "judge noise"}`},
		},
	}

	outcomes := []QuestionOutcome{
		{
			ID:          "q1_abs",
			Category:    "single-session-user",
			Question:    "What is my hamster's name?",
			GroundTruth: "You did not mention this information.",
			AgentAnswer: "I do not have that information.",
		},
	}

	result, traces, _, err := ScoreWithJudge(context.Background(), JudgeConfig{
		Provider:   fp,
		Model:      "fake-judge",
		MaxRetries: 1,
	}, outcomes)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if result.Questions[0].JudgeVerdict != "abstain_correct" {
		t.Fatalf("JudgeVerdict = %q, want abstain_correct", result.Questions[0].JudgeVerdict)
	}
	if traces[0].Verdict.Verdict != "abstain_correct" {
		t.Fatalf("trace verdict = %q, want abstain_correct", traces[0].Verdict.Verdict)
	}
	if traces[0].Verdict.Rationale != "deterministic abstention scoring" {
		t.Fatalf("trace rationale = %q", traces[0].Verdict.Rationale)
	}
	if fp.callIdx != 0 {
		t.Fatalf("provider called %d times, want 0", fp.callIdx)
	}
}

func TestJudgeVerdict_RetryThenSuccess(t *testing.T) {
	fp := &scriptedProvider{
		responses: []llm.CompleteResponse{
			// Schema-violating: "maybe" is not in the verdict enum.
			{Text: `{"verdict": "maybe", "rationale": "unsure"}`},
			{Text: `{"verdict": "yes", "rationale": "second attempt"}`},
		},
		errors: []error{nil, nil},
	}

	outcomes := []QuestionOutcome{
		{ID: "q1", Category: "single-session", Question: "Q?", GroundTruth: "A", AgentAnswer: "A"},
	}

	result, _, _, err := ScoreWithJudge(context.Background(), JudgeConfig{
		Provider:   fp,
		Model:      "fake-judge",
		MaxRetries: 3,
	}, outcomes)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if result.OverallScore != 1.0 {
		t.Errorf("OverallScore = %v, want 1.0", result.OverallScore)
	}
	if fp.callIdx != 2 {
		t.Errorf("expected provider called twice (retry once), got %d", fp.callIdx)
	}
}

func TestJudgeVerdict_AllRetriesFail_RecordsError(t *testing.T) {
	fp := &scriptedProvider{
		errors: []error{
			fmt.Errorf("network error"),
			fmt.Errorf("network error"),
		},
	}

	outcomes := []QuestionOutcome{
		{ID: "q1", Category: "single-session", Question: "Q?", GroundTruth: "red", AgentAnswer: "The answer is red"},
		{ID: "q2", Category: "single-session", Question: "Q?", GroundTruth: "blue", AgentAnswer: "green"},
	}

	result, traces, _, err := ScoreWithJudge(context.Background(), JudgeConfig{
		Provider:   fp,
		Model:      "fake-judge",
		MaxRetries: 1,
	}, outcomes)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if result.Questions[0].JudgeVerdict != "error" {
		t.Errorf("q1 verdict = %q, want error", result.Questions[0].JudgeVerdict)
	}
	if result.Questions[0].JudgeRationale != "judge failure" {
		t.Errorf("q1 rationale = %q, want 'judge failure'", result.Questions[0].JudgeRationale)
	}
	if result.Questions[1].JudgeVerdict != "error" {
		t.Errorf("q2 verdict = %q, want error", result.Questions[1].JudgeVerdict)
	}

	if result.OverallScore != 0 {
		t.Errorf("OverallScore = %v, want 0", result.OverallScore)
	}
	if traces[0].Error == "" {
		t.Error("expected non-empty error in trace")
	}
	if traces[1].Error == "" {
		t.Error("expected non-empty error in trace")
	}
}

func TestJudgeVerdict_PerCategoryAggregation(t *testing.T) {
	fp := &scriptedProvider{
		responses: []llm.CompleteResponse{
			{Text: `{"verdict": "yes", "rationale": "match"}`},
			{Text: `{"verdict": "no", "rationale": "mismatch"}`},
			{Text: `{"verdict": "yes", "rationale": "match"}`},
			{Text: `{"verdict": "no", "rationale": "mismatch"}`},
		},
	}

	outcomes := []QuestionOutcome{
		{ID: "q1", Category: "single-session", Question: "Q1", GroundTruth: "A", AgentAnswer: "A"},
		{ID: "q2", Category: "single-session", Question: "Q2", GroundTruth: "B", AgentAnswer: "C"},
		{ID: "q3", Category: "temporal", Question: "Q3", GroundTruth: "2024", AgentAnswer: "2024"},
		{ID: "q4", Category: "temporal", Question: "Q4", GroundTruth: "Jan", AgentAnswer: "February-ish"},
	}

	result, _, _, err := ScoreWithJudge(context.Background(), JudgeConfig{
		Provider:   fp,
		Model:      "fake-judge",
		MaxRetries: 1,
	}, outcomes)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if result.OverallScore != 0.5 {
		t.Errorf("OverallScore = %v, want 0.5", result.OverallScore)
	}

	ss := result.ByCategory["single-session"]
	if ss.Run != 2 || ss.Correct != 1 || ss.Incorrect != 1 {
		t.Errorf("single-session = %+v, want Run=2 Correct=1 Incorrect=1", ss)
	}
	if ss.Score != 0.5 {
		t.Errorf("single-session Score = %v, want 0.5", ss.Score)
	}

	tmp := result.ByCategory["temporal"]
	if tmp.Run != 2 || tmp.Correct != 1 || tmp.Incorrect != 1 {
		t.Errorf("temporal = %+v, want Run=2 Correct=1 Incorrect=1", tmp)
	}
	if tmp.Score != 0.5 {
		t.Errorf("temporal Score = %v, want 0.5", tmp.Score)
	}
}

func TestJudgeVerdict_ExactMatchAlongsideJudge(t *testing.T) {
	fp := &scriptedProvider{
		responses: []llm.CompleteResponse{
			{Text: `{"verdict": "yes", "rationale": "exact"}`},
			{Text: `{"verdict": "yes", "rationale": "paraphrase still correct"}`},
		},
	}

	outcomes := []QuestionOutcome{
		{ID: "q1", Category: "single-session", Question: "Q?", GroundTruth: "red", AgentAnswer: "red"},
		{ID: "q2", Category: "single-session", Question: "Q?", GroundTruth: "the capital of France", AgentAnswer: "Paris"},
	}

	result, _, _, err := ScoreWithJudge(context.Background(), JudgeConfig{
		Provider:   fp,
		Model:      "fake-judge",
		MaxRetries: 1,
	}, outcomes)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if result.OverallScore != 1.0 {
		t.Errorf("OverallScore (judge) = %v, want 1.0", result.OverallScore)
	}
	if result.ExactMatchScore != 0.5 {
		t.Errorf("ExactMatchScore = %v, want 0.5", result.ExactMatchScore)
	}
}

func TestJudgeVerdict_ErrorOutcomeSkipped(t *testing.T) {
	fp := &scriptedProvider{
		responses: []llm.CompleteResponse{
			{Text: `{"verdict": "yes", "rationale": "match"}`},
		},
	}

	outcomes := []QuestionOutcome{
		{ID: "q1", Category: "single-session", Question: "Q?", GroundTruth: "A", AgentAnswer: "", Error: "timeout"},
		{ID: "q2", Category: "single-session", Question: "Q?", GroundTruth: "A", AgentAnswer: "A"},
	}

	result, traces, _, err := ScoreWithJudge(context.Background(), JudgeConfig{
		Provider:   fp,
		Model:      "fake-judge",
		MaxRetries: 1,
	}, outcomes)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if result.Questions[0].JudgeVerdict != "error" {
		t.Errorf("q1 verdict = %q, want error", result.Questions[0].JudgeVerdict)
	}
	if traces[0].Error != "timeout" {
		t.Errorf("q1 trace error = %q, want 'timeout'", traces[0].Error)
	}

	if result.Questions[1].JudgeVerdict != "correct" {
		t.Errorf("q2 verdict = %q, want correct", result.Questions[1].JudgeVerdict)
	}

	if fp.callIdx != 1 {
		t.Errorf("provider called %d times, want 1 (error outcome skipped)", fp.callIdx)
	}
}

func TestJudgeVerdict_JudgeModelRecorded(t *testing.T) {
	fp := &scriptedProvider{
		responses: []llm.CompleteResponse{
			{Text: `{"verdict": "yes", "rationale": "match"}`},
		},
	}

	outcomes := []QuestionOutcome{
		{ID: "q1", Category: "single-session", Question: "Q?", GroundTruth: "A", AgentAnswer: "A"},
	}

	result, _, _, err := ScoreWithJudge(context.Background(), JudgeConfig{
		Provider:   fp,
		Model:      "gemma4-31b-it",
		MaxRetries: 1,
	}, outcomes)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if result.JudgeModel != "gemma4-31b-it" {
		t.Errorf("JudgeModel = %q, want 'gemma4-31b-it'", result.JudgeModel)
	}
}

func TestFormatJudgePrompt_IncludesQuestionDateWhenSet(t *testing.T) {
	prompt := formatJudgePrompt(
		"temporal-reasoning",
		false,
		"When did the meeting happen?",
		"14 February 2024",
		"The meeting happened on 2024-02-14.",
		"2024-01-15",
	)

	if !strings.Contains(prompt, "Question date: 2024-01-15") {
		t.Errorf("prompt missing question-date anchor, got:\n%s", prompt)
	}
	if !strings.Contains(prompt, "When did the meeting happen?") {
		t.Errorf("prompt missing question body, got:\n%s", prompt)
	}
	if !strings.HasPrefix(prompt, "Question date: 2024-01-15\n\n") {
		t.Errorf("question-date anchor should precede the prompt body, got:\n%s", prompt)
	}
}

func TestFormatJudgePrompt_OmitsQuestionDateWhenEmpty(t *testing.T) {
	prompt := formatJudgePrompt(
		"single-session",
		false,
		"What colour was the car?",
		"red",
		"The car was red.",
		"",
	)

	if strings.Contains(prompt, "Question date:") {
		t.Errorf("prompt should not contain a Question date line when empty, got:\n%s", prompt)
	}
	if !strings.Contains(prompt, "Correct Answer: red") {
		t.Errorf("prompt missing correct-answer field, got:\n%s", prompt)
	}
}

func TestJudgeVerdict_ThreadsQuestionDate(t *testing.T) {
	fp := &scriptedProvider{
		responses: []llm.CompleteResponse{{Text: `{"verdict": "yes", "rationale": "match"}`}},
	}

	outcomes := []QuestionOutcome{
		{
			ID:           "q1",
			Category:     "temporal-reasoning",
			Question:     "When did the meeting happen?",
			QuestionDate: "2024-01-15",
			GroundTruth:  "Tuesday",
			AgentAnswer:  "It happened on Tuesday.",
		},
	}

	_, _, _, err := ScoreWithJudge(context.Background(), JudgeConfig{
		Provider:   fp,
		Model:      "fake-judge",
		MaxRetries: 1,
	}, outcomes)
	if err != nil {
		t.Fatalf("ScoreWithJudge: %v", err)
	}

	if len(fp.lastReqs) != 1 {
		t.Fatalf("expected 1 judge call, got %d", len(fp.lastReqs))
	}
	// The system-instruction message is injected first by completeJSON, then
	// the user prompt. Look at the last user message for the date anchor.
	last := fp.lastReqs[0].Messages[len(fp.lastReqs[0].Messages)-1]
	if !strings.Contains(last.Content, "Question date: 2024-01-15") {
		t.Errorf("judge prompt missing question-date anchor, got:\n%s", last.Content)
	}
}

func TestScoreWithJudge_AggregatesUsage(t *testing.T) {
	fp := &scriptedProvider{
		responses: []llm.CompleteResponse{
			{Text: `{"verdict": "yes", "rationale": "ok"}`, TokensIn: 120, TokensOut: 3},
			{Text: `{"verdict": "no", "rationale": "miss"}`, TokensIn: 140, TokensOut: 4},
			{Text: `{"verdict": "yes", "rationale": "ok"}`, TokensIn: 100, TokensOut: 2},
		},
	}

	outcomes := []QuestionOutcome{
		{ID: "q1", Category: "single-session", Question: "Q1?", GroundTruth: "A1", AgentAnswer: "A1"},
		{ID: "q2", Category: "single-session", Question: "Q2?", GroundTruth: "A2", AgentAnswer: "nope"},
		{ID: "q3", Category: "temporal", Question: "Q3?", GroundTruth: "A3", AgentAnswer: "A3"},
	}

	_, _, usage, err := ScoreWithJudge(context.Background(), JudgeConfig{
		Provider:   fp,
		Model:      "fake-judge",
		MaxRetries: 1,
	}, outcomes)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if usage.InputTokens != 360 {
		t.Errorf("InputTokens = %d, want 360", usage.InputTokens)
	}
	if usage.OutputTokens != 9 {
		t.Errorf("OutputTokens = %d, want 9", usage.OutputTokens)
	}
}

func TestScoreWithJudge_SkipsUsageForErrorOutcomes(t *testing.T) {
	fp := &scriptedProvider{
		responses: []llm.CompleteResponse{
			{Text: `{"verdict": "yes", "rationale": "ok"}`, TokensIn: 50, TokensOut: 2},
		},
	}

	outcomes := []QuestionOutcome{
		{ID: "q1", Category: "single-session", Question: "Q?", GroundTruth: "A", AgentAnswer: "A"},
		{ID: "q2", Category: "single-session", Question: "Q?", GroundTruth: "A", Error: "context cancelled"},
	}

	_, _, usage, err := ScoreWithJudge(context.Background(), JudgeConfig{
		Provider:   fp,
		Model:      "fake-judge",
		MaxRetries: 1,
	}, outcomes)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if usage.InputTokens != 50 || usage.OutputTokens != 2 {
		t.Errorf("expected only successful call counted, got %+v", usage)
	}
}

func TestScoreWithJudge_ReusesCachedVerdict(t *testing.T) {
	fp := &scriptedProvider{
		responses: []llm.CompleteResponse{
			{Text: `{"verdict": "yes", "rationale": "cached ok"}`, TokensIn: 80, TokensOut: 4},
		},
	}

	outcomes := []QuestionOutcome{
		{ID: "q1", Category: "single-session", Question: "Q?", GroundTruth: "A", AgentAnswer: "A"},
	}
	cacheDir := t.TempDir()
	cfg := JudgeConfig{
		Provider:   fp,
		Model:      "fake-judge",
		MaxRetries: 1,
		CacheDir:   cacheDir,
	}

	first, firstTrace, firstUsage, err := ScoreWithJudge(context.Background(), cfg, append([]QuestionOutcome(nil), outcomes...))
	if err != nil {
		t.Fatalf("first ScoreWithJudge: %v", err)
	}
	if fp.callIdx != 1 {
		t.Fatalf("provider call count after first run = %d, want 1", fp.callIdx)
	}
	if first.OverallScore != 1.0 {
		t.Fatalf("first OverallScore = %v, want 1.0", first.OverallScore)
	}
	if firstUsage.InputTokens != 80 || firstUsage.OutputTokens != 4 {
		t.Fatalf("first usage = %+v, want input=80 output=4", firstUsage)
	}
	if firstTrace[0].RawResponse == "" {
		t.Fatal("first trace raw response = empty, want cached payload recorded")
	}

	second, secondTrace, secondUsage, err := ScoreWithJudge(context.Background(), cfg, append([]QuestionOutcome(nil), outcomes...))
	if err != nil {
		t.Fatalf("second ScoreWithJudge: %v", err)
	}
	if fp.callIdx != 1 {
		t.Fatalf("provider call count after second run = %d, want cache hit without new call", fp.callIdx)
	}
	if second.OverallScore != 1.0 {
		t.Fatalf("second OverallScore = %v, want 1.0", second.OverallScore)
	}
	if secondUsage.InputTokens != 0 || secondUsage.OutputTokens != 0 {
		t.Fatalf("second usage = %+v, want zero on cache hit", secondUsage)
	}
	if secondTrace[0].RawResponse == "" {
		t.Fatal("second trace raw response = empty, want cached payload recorded")
	}
	if secondTrace[0].Verdict.Verdict != "correct" {
		t.Fatalf("second trace verdict = %q, want correct", secondTrace[0].Verdict.Verdict)
	}
}

func TestParseStructuredVerdict(t *testing.T) {
	cases := []struct {
		name      string
		payload   string
		abstain   bool
		want      string
		wantError bool
	}{
		{"yes_std", `{"verdict":"yes","rationale":"ok"}`, false, "correct", false},
		{"no_std", `{"verdict":"no","rationale":"ok"}`, false, "incorrect", false},
		{"yes_abs", `{"verdict":"yes","rationale":"ok"}`, true, "abstain_correct", false},
		{"no_abs", `{"verdict":"no","rationale":"ok"}`, true, "abstain_incorrect", false},
		{"maybe", `{"verdict":"maybe","rationale":"ok"}`, false, "", true},
		{"malformed", `{not json`, false, "", true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			v, err := parseStructuredVerdict([]byte(tc.payload), tc.abstain)
			if tc.wantError {
				if err == nil {
					t.Fatalf("expected error")
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if v.Verdict != tc.want {
				t.Errorf("Verdict = %q, want %q", v.Verdict, tc.want)
			}
		})
	}
}

func TestParseYesNo(t *testing.T) {
	cases := []struct {
		in      string
		abstain bool
		want    string
		err     bool
	}{
		{"Yes", false, "correct", false},
		{"No", false, "incorrect", false},
		{"yes, clearly", false, "correct", false},
		{"no, not quite", false, "incorrect", false},
		{"Yes", true, "abstain_correct", false},
		{"No", true, "abstain_incorrect", false},
		{"", false, "", true},
		{"maybe", false, "", true},
	}
	for _, tc := range cases {
		got, err := parseYesNo(tc.in, tc.abstain)
		if tc.err {
			if err == nil {
				t.Errorf("parseYesNo(%q) expected error", tc.in)
			}
			continue
		}
		if err != nil {
			t.Errorf("parseYesNo(%q): unexpected error %v", tc.in, err)
			continue
		}
		if got.Verdict != tc.want {
			t.Errorf("parseYesNo(%q) verdict = %q, want %q", tc.in, got.Verdict, tc.want)
		}
	}
}
