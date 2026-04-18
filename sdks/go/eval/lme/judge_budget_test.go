// SPDX-License-Identifier: Apache-2.0

package lme

import (
	"context"
	"fmt"
	"strings"
	"testing"

	"github.com/jeffs-brain/memory/go/llm"
)

func TestJudgeContentBudgetFor_Table(t *testing.T) {
	tests := []struct {
		name   string
		maxCtx int
		wantLo int
		wantHi int
	}{
		{
			name:   "nil_provider_defaults_to_100k",
			maxCtx: -1,
			wantLo: defaultJudgeContentBudget,
			wantHi: defaultJudgeContentBudget,
		},
		{
			name:   "gemma4_31b_32k_yields_roughly_57k_clamped_within_floor",
			maxCtx: 32_767,
			wantLo: 55_000,
			wantHi: 60_000,
		},
		{
			name:   "gpt_4o_128k_clamped_to_200k_ceiling",
			maxCtx: 128_000,
			wantLo: judgeContentBudgetCeiling,
			wantHi: judgeContentBudgetCeiling,
		},
		{
			name:   "small_8k_model_hits_16k_floor",
			maxCtx: 8_000,
			wantLo: judgeContentBudgetFloor,
			wantHi: judgeContentBudgetFloor,
		},
		{
			name:   "tiny_4k_model_still_floors_at_16k",
			maxCtx: 4_096,
			wantLo: judgeContentBudgetFloor,
			wantHi: judgeContentBudgetFloor,
		},
		{
			name:   "zero_ctx_falls_back_to_default",
			maxCtx: 0,
			wantLo: defaultJudgeContentBudget,
			wantHi: defaultJudgeContentBudget,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var got int
			if tc.maxCtx == -1 {
				got = judgeContentBudgetFor(nil)
			} else {
				got = judgeContentBudgetFor(&maxContextOnlyProvider{maxCtx: tc.maxCtx})
			}
			if got < tc.wantLo || got > tc.wantHi {
				t.Errorf("judgeContentBudgetFor(ctx=%d) = %d, want in [%d, %d]", tc.maxCtx, got, tc.wantLo, tc.wantHi)
			}
		})
	}
}

func TestJudgeContentBudget_Gemma4Safety(t *testing.T) {
	p := &maxContextOnlyProvider{maxCtx: 32_767}
	got := judgeContentBudgetFor(p)
	if got > 60_000 {
		t.Errorf("gemma4-safe budget = %d, want <= 60000 to stay within the 32K context window", got)
	}
	if got < judgeContentBudgetFloor {
		t.Errorf("gemma4-safe budget = %d, want >= floor %d", got, judgeContentBudgetFloor)
	}
}

func TestResolveJudgeContentBudget_ExplicitOverridesInferred(t *testing.T) {
	p := &maxContextOnlyProvider{maxCtx: 128_000}

	inferred := judgeContentBudgetFor(p)
	if inferred == 12_345 {
		t.Fatalf("test sentinel collides with inferred value")
	}

	got := resolveJudgeContentBudget(JudgeConfig{Provider: p, ContentBudget: 12_345})
	if got != 12_345 {
		t.Errorf("explicit ContentBudget override not honoured: got %d, want 12345", got)
	}
}

func TestResolveReaderContentBudget_ExplicitOverridesInferred(t *testing.T) {
	p := &maxContextOnlyProvider{maxCtx: 128_000}

	got := resolveReaderContentBudget(ReaderConfig{Provider: p, ContentBudget: 54_321})
	if got != 54_321 {
		t.Errorf("explicit ContentBudget override not honoured: got %d, want 54321", got)
	}

	// The Go SDK port's reader budget falls back to the static default
	// rather than the per-provider inference used by jeff, because the
	// baseline [llm.Provider] interface does not expose MaxContextTokens.
	gotInferred := resolveReaderContentBudget(ReaderConfig{Provider: p})
	if gotInferred != defaultJudgeContentBudget {
		t.Errorf("inferred budget mismatch: got %d, want %d", gotInferred, defaultJudgeContentBudget)
	}
}

// largeMultiSessionContent builds an answer composed of ten
// session-delimited blocks whose total length is at least totalLen.
func largeMultiSessionContent(totalLen int) string {
	const sessions = 10
	sectionLen := totalLen / sessions
	var b strings.Builder
	for i := 0; i < sessions; i++ {
		if i == 0 {
			fmt.Fprintf(&b, "---\nsession_id: s%d\n", i)
		} else {
			fmt.Fprintf(&b, "\n\n---\nsession_id: s%d\n", i)
		}
		unit := fmt.Sprintf("line-%d-filler ", i)
		reps := sectionLen/len(unit) + 1
		b.WriteString(strings.Repeat(unit, reps))
	}
	return b.String()
}

func TestJudge_BudgetControlsTruncation(t *testing.T) {
	content := largeMultiSessionContent(60_000)
	if len(content) < 58_000 {
		t.Fatalf("test fixture too short: %d chars", len(content))
	}

	cases := []struct {
		name             string
		budget           int
		wantTruncated    bool
		wantMinPromptLen int
	}{
		{
			name:             "budget_100k_preserves_content",
			budget:           100_000,
			wantTruncated:    false,
			wantMinPromptLen: len(content),
		},
		{
			name:             "budget_40k_truncates",
			budget:           40_000,
			wantTruncated:    true,
			wantMinPromptLen: 35_000,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			fp := &scriptedProvider{
				responses: []llm.CompleteResponse{
					{Text: `{"verdict": "yes", "rationale": "ok"}`, TokensIn: 100, TokensOut: 1},
				},
				maxCtx: 128_000,
			}

			outcomes := []QuestionOutcome{
				{
					ID:          "q-large",
					Category:    "multi-session",
					Question:    "What happened across sessions?",
					GroundTruth: "Many things.",
					AgentAnswer: content,
				},
			}

			_, traces, _, err := ScoreWithJudge(context.Background(), JudgeConfig{
				Provider:      fp,
				Model:         "fake-judge",
				MaxRetries:    1,
				ContentBudget: tc.budget,
			}, outcomes)
			if err != nil {
				t.Fatalf("ScoreWithJudge: %v", err)
			}
			if len(traces) != 1 {
				t.Fatalf("expected 1 trace, got %d", len(traces))
			}
			got := traces[0].ContentChars
			if got <= 0 {
				t.Fatalf("trace ContentChars not populated: %d", got)
			}
			if tc.wantTruncated {
				if got >= len(content) {
					t.Errorf("expected truncation at budget %d: got %d chars, full content %d", tc.budget, got, len(content))
				}
				maxAllowed := tc.budget + tc.budget/20
				if got > maxAllowed {
					t.Errorf("content %d chars exceeds budget %d (allowance %d)", got, tc.budget, maxAllowed)
				}
			} else {
				if got != len(content) {
					t.Errorf("expected no truncation at budget %d: got %d chars, want %d", tc.budget, got, len(content))
				}
			}
		})
	}
}

func TestJudge_TraceContentCharsPopulatedByDefault(t *testing.T) {
	fp := &scriptedProvider{
		responses: []llm.CompleteResponse{
			{Text: `{"verdict":"yes","rationale":"ok"}`, TokensIn: 100, TokensOut: 1},
		},
		maxCtx: 32_767,
	}

	short := "The answer is blue."
	outcomes := []QuestionOutcome{{
		ID:          "q1",
		Category:    "single-session",
		Question:    "What colour?",
		GroundTruth: "blue",
		AgentAnswer: short,
	}}

	_, traces, _, err := ScoreWithJudge(context.Background(), JudgeConfig{
		Provider:   fp,
		Model:      "fake-judge",
		MaxRetries: 1,
	}, outcomes)
	if err != nil {
		t.Fatalf("ScoreWithJudge: %v", err)
	}
	if traces[0].ContentChars != len(short) {
		t.Errorf("ContentChars = %d, want %d", traces[0].ContentChars, len(short))
	}
}
