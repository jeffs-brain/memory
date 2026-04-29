// SPDX-License-Identifier: Apache-2.0

package query

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/jeffs-brain/memory/go/llm"
)

const promptVersion = 1

// TODO(next): this prompt was authored for jeff and mentions concrete
// personas and noise patterns ("error paste", etc.). Review for jeff-
// flavoured phrasing before shipping to external callers.
const distillSystemPrompt = `You are a search query distiller. Given a raw user message (which may be a huge error paste, a vague question, or a multi-part request), produce structured search queries that will retrieve the most relevant information from a knowledge base.

Respond with ONLY a JSON array of query objects:
[{"text": "concise search query", "domain": "optional domain hint", "entities": ["extracted entities"], "recency_bias": "recent|historical|", "confidence": 0.0-1.0}]

Rules:
- Extract the actual question from noise (error logs, pasted code, etc.)
- Split multi-intent queries into separate query objects
- Expand abbreviations and jargon where possible
- Resolve anaphoric references ("it", "that") using context if available
- Maximum 3 queries per input
- Each query text should be 5-30 words, focused and searchable
- Set confidence to 0.0-1.0 based on how certain you are the rewrite captures the intent`

const maxDistillTokens = 512
const distillTemperature = 0.1

// callDistillLLM issues a distillation completion against the supplied
// provider. The system prompt is prepended as a [llm.RoleSystem] message
// because the SDK's [llm.CompleteRequest] does not carry a dedicated
// System field.
func callDistillLLM(ctx context.Context, provider llm.Provider, raw string, history []llm.Message, maxQueries int) ([]Query, error) {
	userContent := buildDistillPrompt(raw, history, maxQueries)

	resp, err := provider.Complete(ctx, llm.CompleteRequest{
		Messages: []llm.Message{
			{Role: llm.RoleSystem, Content: distillSystemPrompt},
			{Role: llm.RoleUser, Content: userContent},
		},
		MaxTokens:   maxDistillTokens,
		Temperature: distillTemperature,
	})
	if err != nil {
		return nil, fmt.Errorf("distill: %w", err)
	}

	return parseDistillResponse(resp.Text, maxQueries)
}

func buildDistillPrompt(raw string, history []llm.Message, maxQueries int) string {
	var b strings.Builder

	// Include last 2-3 user turns for anaphora resolution.
	if len(history) > 0 {
		b.WriteString("Recent conversation context:\n")
		count := 0
		for i := len(history) - 1; i >= 0 && count < 3; i-- {
			if history[i].Role == llm.RoleUser {
				content := history[i].Content
				if len(content) > 500 {
					content = content[:500] + "..."
				}
				fmt.Fprintf(&b, "[previous user message]: %s\n", content)
				count++
			}
		}
		b.WriteByte('\n')
	}

	fmt.Fprintf(&b, "Raw user input to distil (produce up to %d queries):\n%s", maxQueries, raw)
	return b.String()
}

func parseDistillResponse(content string, maxQueries int) ([]Query, error) {
	content = strings.TrimSpace(content)

	// Find the JSON array in the response.
	start := strings.Index(content, "[")
	end := strings.LastIndex(content, "]")
	if start < 0 || end <= start {
		return nil, fmt.Errorf("no JSON array found in response")
	}
	content = content[start : end+1]

	var queries []Query
	if err := json.Unmarshal([]byte(content), &queries); err != nil {
		return nil, fmt.Errorf("parse distill response: %w", err)
	}

	// Enforce max queries.
	if len(queries) > maxQueries {
		queries = queries[:maxQueries]
	}

	// Validate: drop empty text entries.
	valid := make([]Query, 0, len(queries))
	for _, q := range queries {
		q.Text = strings.TrimSpace(q.Text)
		if q.Text != "" {
			valid = append(valid, q)
		}
	}

	return valid, nil
}
