// SPDX-License-Identifier: Apache-2.0

package lme

import (
	"context"
	"fmt"
	"strings"
	"testing"

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

func TestReadAnswer_FallsBackOnProviderError(t *testing.T) {
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
	if got != raw {
		t.Errorf("expected fallback to raw content, got %q", got)
	}
	if usage != (Usage{}) {
		t.Errorf("expected zero usage on provider error, got %+v", usage)
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

func TestReadAnswer_EmptyResponseFallsBack(t *testing.T) {
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
	if got != raw {
		t.Errorf("expected fallback to raw content when response is whitespace-only, got %q", got)
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
