// SPDX-License-Identifier: Apache-2.0

package memory

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/jeffs-brain/memory/go/llm"
)

// Episode thresholds.
const (
	episodeMaxTokens   = 1024
	episodeTemperature = 0.2
	episodeMinMessages = 8
)

// writeToolNames lists tool names that indicate a write/edit
// operation.
var writeToolNames = map[string]bool{
	"write": true,
	"edit":  true,
}

// Episode records a durable summary of a completed session.
type Episode struct {
	ProjectPath string
	SessionID   string
	Summary     string
	Outcome     string
	Heuristics  []string
	Tags        []string
}

// EpisodeStore is the persistence boundary for episodic memory. The
// port leaves the concrete implementation to the caller because the
// upstream jeff harness wires this up to its session database, which
// is not part of the memory layer.
//
// TODO(integration): provide a default in-memory or file-backed
// implementation once the SDK settles on a session model.
type EpisodeStore interface {
	CreateEpisode(Episode) error
}

// EpisodeRecorder evaluates whether a completed session produced a
// significant episode worth recording, and if so, summarises it.
type EpisodeRecorder struct{}

// NewEpisodeRecorder creates a new EpisodeRecorder.
func NewEpisodeRecorder() *EpisodeRecorder {
	return &EpisodeRecorder{}
}

// episodeSystemPrompt is the system prompt for episode summarisation.
// Ported verbatim from jeff.
const episodeSystemPrompt = `You are summarising a coding session for episodic memory.
Produce a JSON object:
{
  "significant": true/false,
  "summary": "one paragraph of what was attempted and what happened",
  "outcome": "success|partial|failure",
  "heuristics": ["generalised learning 1", "learning 2"],
  "tags": ["tag1", "tag2"]
}
If the session was routine (simple Q&A, single file read), set significant=false.
Respond with ONLY valid JSON, no other text.`

type episodeResult struct {
	Significant bool     `json:"significant"`
	Summary     string   `json:"summary"`
	Outcome     string   `json:"outcome"`
	Heuristics  []string `json:"heuristics"`
	Tags        []string `json:"tags"`
}

// MaybeRecord evaluates the session messages and decides whether to
// record an episode.
func (r *EpisodeRecorder) MaybeRecord(
	ctx context.Context,
	provider llm.Provider,
	model string,
	store EpisodeStore,
	projectPath string,
	sessionID string,
	messages []Message,
) error {
	if !shouldRecordEpisode(messages) {
		return nil
	}

	userPrompt := buildEpisodePrompt(messages)

	resp, err := provider.Complete(ctx, llm.CompleteRequest{
		Model: model,
		Messages: []llm.Message{
			{Role: RoleSystem, Content: episodeSystemPrompt},
			{Role: RoleUser, Content: userPrompt},
		},
		MaxTokens:   episodeMaxTokens,
		Temperature: episodeTemperature,
	})
	if err != nil {
		return fmt.Errorf("episode summarisation: %w", err)
	}

	result := parseEpisodeResult(resp.Text)
	if !result.Significant {
		return nil
	}

	ep := Episode{
		ProjectPath: projectPath,
		SessionID:   sessionID,
		Summary:     result.Summary,
		Outcome:     result.Outcome,
		Heuristics:  result.Heuristics,
		Tags:        result.Tags,
	}

	if store == nil {
		return nil
	}
	if err := store.CreateEpisode(ep); err != nil {
		return fmt.Errorf("storing episode: %w", err)
	}

	return nil
}

// shouldRecordEpisode checks whether the session is worth evaluating.
func shouldRecordEpisode(messages []Message) bool {
	if len(messages) < episodeMinMessages {
		return false
	}

	for _, m := range messages {
		if m.Role != RoleAssistant {
			continue
		}
		for _, tc := range m.ToolCalls {
			if writeToolNames[tc.Name] {
				return true
			}
		}
	}

	return false
}

// buildEpisodePrompt constructs the user message for the episode
// summariser.
func buildEpisodePrompt(messages []Message) string {
	var b strings.Builder

	b.WriteString("## Session transcript\n\n")

	for _, m := range messages {
		switch m.Role {
		case RoleUser:
			content := m.Content
			if len(content) > 1000 {
				content = content[:1000] + "\n[...truncated]"
			}
			b.WriteString(fmt.Sprintf("[user]: %s\n\n", content))

		case RoleAssistant:
			content := m.Content
			if len(content) > 1000 {
				content = content[:1000] + "\n[...truncated]"
			}
			if content != "" {
				b.WriteString(fmt.Sprintf("[assistant]: %s\n\n", content))
			}
			for _, tc := range m.ToolCalls {
				args := string(tc.Arguments)
				if len(args) > 200 {
					args = args[:200] + "..."
				}
				b.WriteString(fmt.Sprintf("[tool_call %s]: %s\n\n", tc.Name, args))
			}

		case RoleTool:
			content := m.Content
			if len(content) > 300 {
				content = content[:300] + "..."
			}
			b.WriteString(fmt.Sprintf("[tool_result %s]: %s\n\n", m.Name, content))
		}
	}

	return b.String()
}

// parseEpisodeResult extracts the episode data from the model's JSON
// response.
func parseEpisodeResult(content string) episodeResult {
	content = strings.TrimSpace(content)

	if idx := strings.Index(content, "{"); idx >= 0 {
		if end := strings.LastIndex(content, "}"); end > idx {
			content = content[idx : end+1]
		}
	}

	var result episodeResult
	if err := json.Unmarshal([]byte(content), &result); err != nil {
		return episodeResult{}
	}

	return result
}
