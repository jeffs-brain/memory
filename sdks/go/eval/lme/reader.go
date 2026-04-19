// SPDX-License-Identifier: Apache-2.0

package lme

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/jeffs-brain/memory/go/llm"
)

// readerTodayAnchor renders a "YYYY-MM-DD (Weekday)" string for the
// reader's temporal grounding line, falling back to the raw input when
// nothing parses.
func readerTodayAnchor(questionDate string) string {
	s := strings.TrimSpace(questionDate)
	if s == "" {
		return "unknown"
	}
	for _, layout := range []string{
		"2006/01/02 (Mon) 15:04",
		"2006/01/02 15:04",
		"2006/01/02",
		"2006-01-02 15:04",
		"2006-01-02",
		time.RFC3339,
	} {
		if t, err := time.Parse(layout, s); err == nil {
			return fmt.Sprintf("%s (%s)", t.Format("2006-01-02"), t.Weekday().String())
		}
	}
	return s
}

// readerMaxTokens matches the official run_generation.py CoT setting.
const readerMaxTokens = 800
const readerTemperature = 0.0

// ReaderMaxTokens and ReaderTemperature are exported so the HTTP /ask
// handler can run the same generation parameters as the in-process LME
// reader when a caller opts in to the augmented prompt.
const (
	ReaderMaxTokens   = readerMaxTokens
	ReaderTemperature = readerTemperature
)

// readerUserTemplate is the LME CoT prompt augmented with recency and
// enumeration guidance so multi-session list/count and knowledge-update
// questions score correctly.
const readerUserTemplate = `I will give you several history chats between you and a user. Please answer the question based on the relevant chat history. Answer the question step by step: first extract all the relevant information, and then reason over the information to get the answer.

Resolving conflicting information:
- Each fact is tagged with a date. When the same topic appears with different values on different dates, prefer the value from the most recent session date.
- Treat explicit supersession phrases as hard overrides regardless of how often the old value appears: "now", "currently", "most recently", "actually", "correction", "I updated", "I changed", "no longer".
- Do not vote by frequency. One later correction outweighs any number of earlier mentions.

Enumeration and counting:
- When the question asks to list, count, enumerate, or total ("how many", "list", "which", "what are all", "total", "in total"), return every matching item you find across the retrieved facts, one per line, each tagged with its session date. Then state the count or total explicitly at the end.
- Do not summarise into a single sentence when the question demands a list.
- Add numeric values across sessions when the question asks for a total (hours, days, money, items). Show the arithmetic.

Temporal reasoning:
- Today is %s (this is the current date). Resolve relative references ("recently", "last week", "a few days ago", "this month") against this anchor.
- For date-arithmetic questions ("how many days between X and Y"), first extract each event's ISO date from the fact tags, then compute the difference in days.

History Chats:

%s

Current Date: %s
Question: %s
Answer (step by step):`

// BuildReaderPrompt renders the augmented LME CoT user prompt that the
// in-process reader sends to the LLM. Exposed so the HTTP /ask handler
// can opt in to the same prompt shape (recency, enumeration, temporal
// guidance, today anchor, step-by-step CoT) used by the eval harness.
//
// questionDate accepts the LME "YYYY/MM/DD (Mon) HH:MM" form and a few
// siblings; an empty string falls through to "unknown".
func BuildReaderPrompt(question, questionDate, retrievedContent string) string {
	date := questionDate
	if date == "" {
		date = "unknown"
	}
	return fmt.Sprintf(
		readerUserTemplate,
		readerTodayAnchor(questionDate),
		retrievedContent,
		date,
		question,
	)
}

// ProcessSessionContextForQuestion is the exported entry point to the
// session-block aware preprocessor. The HTTP /ask handler calls this
// when retrieved chunks carry session_id frontmatter so the augmented
// reader sees the same chronologically-sorted, assistant-filtered shape
// as the in-process pipeline.
func ProcessSessionContextForQuestion(raw, question string) string {
	return processSessionContextForQuestion(raw, question)
}

// ReaderConfig controls the reading/answering step.
type ReaderConfig struct {
	Provider llm.Provider
	Model    string

	// ContentBudget caps the character length of retrieved content
	// passed to the reader. Zero uses defaultJudgeContentBudget.
	ContentBudget int
}

func resolveReaderContentBudget(cfg ReaderConfig) int {
	if cfg.ContentBudget > 0 {
		return cfg.ContentBudget
	}
	return defaultJudgeContentBudget
}

// ReadAnswer generates a concise answer from retrieved session content by
// passing it through an LLM reader. Uses the official LME CoT prompt
// format. Returns the usage reported by the provider so callers can
// account for the spend.
func ReadAnswer(ctx context.Context, cfg ReaderConfig, question, questionDate, retrievedContent string) (string, Usage, error) {
	if cfg.Provider == nil {
		return retrievedContent, Usage{}, nil
	}

	if retrievedContent == "" {
		return "", Usage{}, nil
	}

	content := truncateSmartly(retrievedContent, resolveReaderContentBudget(cfg))

	prompt := BuildReaderPrompt(question, questionDate, content)

	resp, err := cfg.Provider.Complete(ctx, llm.CompleteRequest{
		Model: cfg.Model,
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: prompt},
		},
		MaxTokens:   readerMaxTokens,
		Temperature: readerTemperature,
	})
	if err != nil {
		return retrievedContent, Usage{}, err
	}

	answer := strings.TrimSpace(resp.Text)
	usage := usageFromResponse(resp)
	if answer == "" {
		return retrievedContent, usage, nil
	}
	return answer, usage, nil
}
