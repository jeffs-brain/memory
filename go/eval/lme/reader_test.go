// SPDX-License-Identifier: Apache-2.0

package lme

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/jeffs-brain/memory/go/llm"
)

func TestReadAnswer_NilProvider_ReturnsRawContent(t *testing.T) {
	raw := "---\nsession_id: s1\nThe car was red."
	got, usage, err := ReadAnswer(context.Background(), ReaderConfig{}, "What colour was the car?", "2024-01-15", raw)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got != raw {
		t.Errorf("expected raw content returned unchanged, got %q", got)
	}
	if usage != (Usage{}) {
		t.Errorf("expected zero usage when provider is nil, got %+v", usage)
	}
}

func TestReadAnswer_GeneratesConciseAnswer(t *testing.T) {
	fp := &scriptedProvider{
		responses: []llm.CompleteResponse{
			{Text: "The car was red.", TokensIn: 42, TokensOut: 7},
		},
	}

	raw := "---\nsession_id: s1\nuser: I saw a red car outside the building yesterday."
	got, usage, err := ReadAnswer(context.Background(), ReaderConfig{
		Provider: fp,
		Model:    "fake-reader",
	}, "What colour was the car?", "2024-01-15", raw)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got != "The car was red." {
		t.Errorf("expected concise answer, got %q", got)
	}
	if usage.InputTokens != 42 || usage.OutputTokens != 7 {
		t.Errorf("expected usage {42,7}, got %+v", usage)
	}

	if fp.callIdx != 1 {
		t.Errorf("provider called %d times, want 1", fp.callIdx)
	}
}

func TestReadAnswer_ReturnsErrorOnProviderFailure(t *testing.T) {
	fp := &scriptedProvider{
		errors: []error{fmt.Errorf("connection refused")},
	}

	raw := "---\nsession_id: s1\nSome session content."
	got, usage, err := ReadAnswer(context.Background(), ReaderConfig{
		Provider: fp,
		Model:    "fake-reader",
	}, "What happened?", "2024-01-15", raw)
	if err == nil {
		t.Fatal("expected provider error to be surfaced")
	}
	if !strings.Contains(err.Error(), "connection refused") {
		t.Errorf("unexpected error: %v", err)
	}
	if got != "" {
		t.Errorf("expected empty answer on provider error, got %q", got)
	}
	if usage != (Usage{}) {
		t.Errorf("expected zero usage on provider error, got %+v", usage)
	}
}

func TestReadAnswer_RetriesTransientProviderFailure(t *testing.T) {
	ResetTransientRetries()
	fp := &scriptedProvider{
		errors: []error{fmt.Errorf("llm: openai 502: error code: 502")},
		responses: []llm.CompleteResponse{
			{},
			{Text: "Recovered answer.", TokensIn: 12, TokensOut: 3},
		},
	}

	raw := "---\nsession_id: s1\nSome session content."
	got, usage, err := ReadAnswer(context.Background(), ReaderConfig{
		Provider: fp,
		Model:    "fake-reader",
	}, "What happened?", "2024-01-15", raw)
	if err != nil {
		t.Fatalf("unexpected error after transient retry: %v", err)
	}
	if got != "Recovered answer." {
		t.Fatalf("answer = %q, want recovered answer", got)
	}
	if usage.InputTokens != 12 || usage.OutputTokens != 3 {
		t.Fatalf("usage = %+v, want input=12 output=3", usage)
	}
	if fp.callIdx != 2 {
		t.Fatalf("provider called %d times, want 2", fp.callIdx)
	}
	if gotRetries := TransientRetriesTotal(); gotRetries != 1 {
		t.Fatalf("TransientRetriesTotal = %d, want 1", gotRetries)
	}
}

func TestReadAnswer_EmptyRetrievedContent(t *testing.T) {
	fp := &scriptedProvider{
		responses: []llm.CompleteResponse{
			{Text: "should not be called"},
		},
	}

	got, usage, err := ReadAnswer(context.Background(), ReaderConfig{
		Provider: fp,
		Model:    "fake-reader",
	}, "What happened?", "2024-01-15", "")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got != "" {
		t.Errorf("expected empty string for empty content, got %q", got)
	}
	if usage != (Usage{}) {
		t.Errorf("expected zero usage for empty content, got %+v", usage)
	}

	if fp.callIdx != 0 {
		t.Errorf("provider called %d times, want 0 (empty content)", fp.callIdx)
	}
}

func TestReadAnswer_EmptyResponseReturnsEmptyAnswer(t *testing.T) {
	fp := &scriptedProvider{
		responses: []llm.CompleteResponse{
			{Text: "   ", TokensIn: 10, TokensOut: 1},
		},
	}

	raw := "---\nsession_id: s1\nSome content here."
	got, usage, err := ReadAnswer(context.Background(), ReaderConfig{
		Provider: fp,
		Model:    "fake-reader",
	}, "Question?", "2024-01-15", raw)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got != "" {
		t.Errorf("expected empty answer when response is whitespace-only, got %q", got)
	}
	if usage.InputTokens != 10 || usage.OutputTokens != 1 {
		t.Errorf("expected usage reported even on whitespace response, got %+v", usage)
	}
}

func TestReaderPrompt_TodayAnchor(t *testing.T) {
	cp := &scriptedProvider{
		responses: []llm.CompleteResponse{{Text: "ok", TokensIn: 1, TokensOut: 1}},
	}
	_, _, err := ReadAnswer(context.Background(), ReaderConfig{
		Provider: cp,
		Model:    "fake-reader",
	}, "What did I do?", "2023/05/26 (Fri) 02:28", "Retrieved facts (1):\n 1. [2023-05-20] [s1] did a thing")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(cp.lastReqs) != 1 {
		t.Fatalf("expected 1 request, got %d", len(cp.lastReqs))
	}
	msgs := cp.lastReqs[0].Messages
	if len(msgs) != 1 {
		t.Fatalf("expected single user message, got %d", len(msgs))
	}
	body := msgs[0].Content
	if !strings.Contains(body, "2023-05-26") {
		t.Errorf("expected ISO today anchor 2023-05-26 in reader prompt, got:\n%s", body)
	}
	if !strings.Contains(body, "Friday") {
		t.Errorf("expected weekday name in reader prompt for temporal grounding, got:\n%s", body)
	}
	if !strings.Contains(strings.ToLower(body), "today is ") {
		t.Errorf("expected 'Today is ' anchor phrase in reader prompt, got:\n%s", body)
	}
}

func TestReaderPrompt_EnumerationClause(t *testing.T) {
	lower := strings.ToLower(readerUserTemplate)
	for _, kw := range []string{"list", "count", "enumerat", "total"} {
		if !strings.Contains(lower, kw) {
			t.Errorf("readerUserTemplate should cover enumeration keyword %q", kw)
		}
	}
	if !strings.Contains(lower, "one per line") {
		t.Errorf("readerUserTemplate should instruct 'one per line' for list questions")
	}
	for _, snippet := range []string{
		"avoid double counting the roll-up",
		"include all confirmed historical amounts for the same subject across sessions",
		"choose the single best-matching fact for that singular item",
		"do not combine multiple different handbags",
		"if any named part is missing or lacks an amount, do not return a partial total",
		"count it once",
		"prefer direct transactional facts over plans, budgets, broad summaries",
	} {
		if !strings.Contains(lower, snippet) {
			t.Errorf("readerUserTemplate should include named-item total guidance %q", snippet)
		}
	}
}

func TestReaderPrompt_AbstentionClause(t *testing.T) {
	lower := strings.ToLower(readerUserTemplate)
	for _, snippet := range []string{
		"if the retrieved facts do not directly answer the question, state that clearly in the first sentence",
		"do not narrate your search process",
		"do not pad the answer with near-miss facts about a different city, person, product, or date",
		"the information provided is not enough to answer the question",
	} {
		if !strings.Contains(lower, snippet) {
			t.Errorf("readerUserTemplate should include abstention guidance %q", snippet)
		}
	}
}

func TestReaderPrompt_ConflictResolutionClause(t *testing.T) {
	lower := strings.ToLower(readerUserTemplate)
	for _, snippet := range []string{
		"never use a fact dated after the current date",
		"for habit and routine questions",
		"a \"30-minute morning commute\" note does not replace a direct statement of a \"45-minute daily commute to work\"",
		"combine them if the connection is explicit in the retrieved facts",
	} {
		if !strings.Contains(lower, strings.ToLower(snippet)) {
			t.Errorf("readerUserTemplate should include conflict-resolution guidance %q", snippet)
		}
	}
}

func TestReaderPrompt_PreferenceClause(t *testing.T) {
	lower := strings.ToLower(readerUserTemplate)
	for _, snippet := range []string{
		"infer durable preferences from concrete desired features",
		"ignore unrelated hostel, budget, or solo-travel examples",
	} {
		if !strings.Contains(lower, snippet) {
			t.Errorf("readerUserTemplate should include preference guidance %q", snippet)
		}
	}
}

func TestReadAnswer_CacheHitSkipsProvider(t *testing.T) {
	fp := &scriptedProvider{
		responses: []llm.CompleteResponse{
			{Text: "The discussion ended with a shortlist.", TokensIn: 18, TokensOut: 7},
		},
	}

	raw := "---\nsession_id: s1\n[user]: We ended with a shortlist of options."
	cfg := ReaderConfig{
		Provider: fp,
		Model:    "fake-reader",
		CacheDir: t.TempDir(),
	}

	got1, usage1, err := ReadAnswer(context.Background(), cfg, "What happened in the discussion?", "2024-01-15", raw)
	if err != nil {
		t.Fatalf("unexpected error on first read: %v", err)
	}
	got2, usage2, err := ReadAnswer(context.Background(), cfg, "What happened in the discussion?", "2024-01-15", raw)
	if err != nil {
		t.Fatalf("unexpected error on second read: %v", err)
	}
	if got1 != "The discussion ended with a shortlist." || got2 != got1 {
		t.Fatalf("unexpected cached answers: first=%q second=%q", got1, got2)
	}
	if usage1.InputTokens != 18 || usage1.OutputTokens != 7 {
		t.Fatalf("first usage = %+v, want input=18 output=7", usage1)
	}
	if usage2 != (Usage{}) {
		t.Fatalf("second usage = %+v, want zero usage on cache hit", usage2)
	}
	if fp.callIdx != 1 {
		t.Fatalf("provider called %d times, want 1", fp.callIdx)
	}
}

func TestReadAnswer_WaitsForSharedCacheWriter(t *testing.T) {
	raw := "---\nsession_id: s1\n[user]: We finalised the shortlist."
	cacheDir := t.TempDir()
	cfg := ReaderConfig{
		Provider: &scriptedProvider{
			errors: []error{fmt.Errorf("provider should not be called while waiting on shared cache")},
		},
		Model:    "fake-reader",
		CacheDir: cacheDir,
	}
	content := truncateReaderContent(raw, resolveReaderContentBudget(cfg), "What happened in the discussion?")
	prompt := BuildReaderPrompt("What happened in the discussion?", "2024-01-15", content)
	cachePath := readerCachePath(cacheDir, cfg.Model, prompt)
	lockPath := cachePath + ".lock"
	if err := os.MkdirAll(filepath.Dir(cachePath), 0o755); err != nil {
		t.Fatalf("mkdir cache dir: %v", err)
	}
	if err := os.WriteFile(lockPath, []byte("locked\n"), 0o644); err != nil {
		t.Fatalf("seed lock file: %v", err)
	}

	go func() {
		time.Sleep(200 * time.Millisecond)
		_ = storeReaderCache(cachePath, readerCacheRecord{
			Answer: "The discussion ended with a shortlist.",
			Usage:  Usage{InputTokens: 5, OutputTokens: 3},
		})
		_ = os.Remove(lockPath)
	}()

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	got, usage, err := ReadAnswer(ctx, cfg, "What happened in the discussion?", "2024-01-15", raw)
	if err != nil {
		t.Fatalf("unexpected error waiting for shared cache: %v", err)
	}
	if got != "The discussion ended with a shortlist." {
		t.Fatalf("cached answer = %q, want shared cached value", got)
	}
	if usage != (Usage{}) {
		t.Fatalf("usage = %+v, want zero usage on shared cache hit", usage)
	}
	if cfg.Provider.(*scriptedProvider).callIdx != 0 {
		t.Fatalf("provider called %d times, want 0", cfg.Provider.(*scriptedProvider).callIdx)
	}
}

func TestTruncateReaderContent_PreservesRelevantMidSessionMatch(t *testing.T) {
	section := "---\nsession_id: answer_9282283d_2\nsession_date: 2023/11/03 (Fri) 19:56\n---\n\n"
	for i := 0; i < 80; i++ {
		section += fmt.Sprintf("[user]: filler line %d about work and errands.\n", i)
	}
	section += "[user]: And by the way, speaking of boundaries, I see Dr. Smith every week, and she's been helping me work on this stuff.\n"
	for i := 80; i < 160; i++ {
		section += fmt.Sprintf("[user]: more filler line %d about unrelated tasks.\n", i)
	}

	got := truncateReaderContent(section, 500, "How often do I see my therapist, Dr. Smith?")
	if !strings.Contains(got, "I see Dr. Smith every week") {
		t.Fatalf("truncateReaderContent dropped the relevant mid-session line:\n%s", got)
	}
}

func TestTruncateSmartly_SplitsRenderedSessionBlocks(t *testing.T) {
	content := strings.Join([]string{
		"=== Session Date: 2023/11/03 (Fri) 19:56 ===\n---\nsession_id: s1\nsession_date: 2023/11/03 (Fri) 19:56\n---\n[user]: alpha " + strings.Repeat("alpha ", 200),
		"=== Session Date: 2023/11/02 (Thu) 09:15 ===\n---\nsession_id: s2\nsession_date: 2023/11/02 (Thu) 09:15\n---\n[user]: bravo " + strings.Repeat("bravo ", 200),
	}, "\n\n---\n\n")

	truncated := truncateSmartly(content, 1200)
	if !strings.Contains(truncated, "session_id: s1") {
		t.Fatalf("truncateSmartly dropped the first rendered block:\n%s", truncated)
	}
	if !strings.Contains(truncated, "session_id: s2") {
		t.Fatalf("truncateSmartly dropped the second rendered block:\n%s", truncated)
	}
}

func TestReaderTodayAnchor_AcceptsLMEFormat(t *testing.T) {
	tests := []struct {
		in   string
		want string
	}{
		{"2023/05/26 (Fri) 02:28", "2023-05-26 (Friday)"},
		{"2023/05/26 02:28", "2023-05-26 (Friday)"},
		{"2023/05/26", "2023-05-26 (Friday)"},
		{"2023-05-26", "2023-05-26 (Friday)"},
		{"", "unknown"},
	}
	for _, tc := range tests {
		got := readerTodayAnchor(tc.in)
		if got != tc.want {
			t.Errorf("readerTodayAnchor(%q) = %q, want %q", tc.in, got, tc.want)
		}
	}
}
