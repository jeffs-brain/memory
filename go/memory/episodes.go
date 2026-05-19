// SPDX-License-Identifier: Apache-2.0

package memory

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/jeffs-brain/memory/go/brain"
	"github.com/jeffs-brain/memory/go/llm"
)

// Episode thresholds.
const (
	episodeMaxTokens   = 1024
	episodeTemperature = 0.2
	episodeMinMessages = 8

	episodesPrefix    = "episodes"
	defaultListLimit  = 50
	defaultQueryLimit = 20
	maxLimit          = 200
)

// EpisodeOutcome enumerates the possible episode result states.
type EpisodeOutcome string

const (
	EpisodeOutcomeSuccess EpisodeOutcome = "success"
	EpisodeOutcomePartial EpisodeOutcome = "partial"
	EpisodeOutcomeFailure EpisodeOutcome = "failure"
	EpisodeOutcomeUnknown EpisodeOutcome = "unknown"
)

// EpisodeScope constrains the scope of an episode.
type EpisodeScope string

const (
	EpisodeScopeGlobal  EpisodeScope = "global"
	EpisodeScopeProject EpisodeScope = "project"
	EpisodeScopeAgent   EpisodeScope = "agent"
)

// EpisodeSignals captures message-level signals for gating decisions.
type EpisodeSignals struct {
	MessageCount            int  `json:"message_count"`
	SubstantiveMessageCount int  `json:"substantive_message_count"`
	UserMessageCount        int  `json:"user_message_count"`
	AssistantMessageCount   int  `json:"assistant_message_count"`
	ToolMessageCount        int  `json:"tool_message_count"`
	ToolCallCount           int  `json:"tool_call_count"`
	WriteSignal             bool `json:"write_signal"`
	EditSignal              bool `json:"edit_signal"`
	ToolSignal              bool `json:"tool_signal"`
}

// EpisodeHeuristic stores a heuristic extracted from an episode.
type EpisodeHeuristic struct {
	Rule        string `json:"rule"`
	Context     string `json:"context"`
	Confidence  string `json:"confidence"`
	Category    string `json:"category"`
	Scope       string `json:"scope"`
	AntiPattern bool   `json:"anti_pattern"`
}

// EpisodeRecord is the canonical episode data model, matching the TS
// implementation's EpisodeRecord type.
type EpisodeRecord struct {
	Path                brain.Path         `json:"path"`
	SessionID           string             `json:"session_id"`
	ActorID             string             `json:"actor_id"`
	Scope               EpisodeScope       `json:"scope"`
	Name                string             `json:"name"`
	Summary             string             `json:"summary"`
	Outcome             EpisodeOutcome     `json:"outcome"`
	RetryFeedback       string             `json:"retry_feedback"`
	ShouldRecordEpisode bool               `json:"should_record_episode"`
	OpenQuestions       []string           `json:"open_questions"`
	Heuristics          []EpisodeHeuristic `json:"heuristics"`
	Tags                []string           `json:"tags"`
	Created             string             `json:"created,omitempty"`
	Modified            string             `json:"modified,omitempty"`
	StartedAt           string             `json:"started_at,omitempty"`
	EndedAt             string             `json:"ended_at,omitempty"`
	Signals             EpisodeSignals     `json:"signals"`
}

// EpisodeListOptions controls filtering for episode listing.
type EpisodeListOptions struct {
	ActorID   string
	Scope     EpisodeScope
	Outcome   EpisodeOutcome
	SessionID string
	Tags      []string
	From      *time.Time
	To        *time.Time
	Limit     int
}

// EpisodeQueryOptions extends list options with a free-text query.
type EpisodeQueryOptions struct {
	EpisodeListOptions
	Query string
}

// EpisodeQueryHit is an episode with a relevance score.
type EpisodeQueryHit struct {
	EpisodeRecord
	Score int `json:"score"`
}

// writeToolNames lists tool names that indicate a write/edit operation.
var writeToolNames = map[string]bool{
	"write": true,
	"edit":  true,
}

// EpisodeStore is the persistence boundary for episodic memory.
type EpisodeStore interface {
	CreateEpisode(ctx context.Context, ep EpisodeRecord) error
	GetEpisode(ctx context.Context, sessionID string) (EpisodeRecord, error)
	ListEpisodes(ctx context.Context, opts EpisodeListOptions) ([]EpisodeRecord, error)
	UpdateEpisode(ctx context.Context, ep EpisodeRecord) error
	DeleteEpisode(ctx context.Context, sessionID string) error
	QueryEpisodes(ctx context.Context, opts EpisodeQueryOptions) ([]EpisodeQueryHit, error)
}

// BrainEpisodeStore implements EpisodeStore backed by a brain.Store,
// persisting episodes as markdown files with YAML frontmatter under
// the "episodes/" prefix.
type BrainEpisodeStore struct {
	store brain.Store
}

// NewBrainEpisodeStore creates a new EpisodeStore backed by the given
// brain.Store.
func NewBrainEpisodeStore(store brain.Store) *BrainEpisodeStore {
	return &BrainEpisodeStore{store: store}
}

// EpisodePath returns the brain.Path for an episode with the given sessionID.
func EpisodePath(sessionID string) brain.Path {
	name := sessionID
	if !strings.HasSuffix(name, ".md") {
		name += ".md"
	}
	return brain.Path(episodesPrefix + "/" + name)
}

// CreateEpisode writes a new episode record to the store.
func (s *BrainEpisodeStore) CreateEpisode(ctx context.Context, ep EpisodeRecord) error {
	path := EpisodePath(ep.SessionID)
	exists, err := s.store.Exists(ctx, path)
	if err != nil {
		return fmt.Errorf("checking episode existence: %w", err)
	}
	if exists {
		return fmt.Errorf("episode %q already exists", ep.SessionID)
	}

	now := time.Now().UTC().Format(time.RFC3339)
	if ep.Created == "" {
		ep.Created = now
	}
	if ep.Modified == "" {
		ep.Modified = now
	}
	ep.Path = path
	if ep.Name == "" {
		ep.Name = "Episode " + ep.SessionID
	}
	normaliseEpisodeDefaults(&ep)

	content := buildEpisodeFileContent(ep)
	return s.store.Write(ctx, path, []byte(content))
}

// GetEpisode retrieves an episode by sessionID.
func (s *BrainEpisodeStore) GetEpisode(ctx context.Context, sessionID string) (EpisodeRecord, error) {
	path := EpisodePath(sessionID)
	data, err := s.store.Read(ctx, path)
	if err != nil {
		return EpisodeRecord{}, fmt.Errorf("reading episode %q: %w", sessionID, err)
	}
	record, err := parseEpisodeFileContent(path, string(data))
	if err != nil {
		return EpisodeRecord{}, fmt.Errorf("parsing episode %q: %w", sessionID, err)
	}
	return record, nil
}

// ListEpisodes returns all episodes matching the given options.
func (s *BrainEpisodeStore) ListEpisodes(ctx context.Context, opts EpisodeListOptions) ([]EpisodeRecord, error) {
	entries, err := s.store.List(ctx, brain.Path(episodesPrefix), brain.ListOpts{
		Recursive:        true,
		IncludeGenerated: true,
	})
	if err != nil {
		return []EpisodeRecord{}, nil
	}

	var records []EpisodeRecord
	for _, entry := range entries {
		if entry.IsDir || !strings.HasSuffix(string(entry.Path), ".md") {
			continue
		}
		data, readErr := s.store.Read(ctx, entry.Path)
		if readErr != nil {
			continue
		}
		record, parseErr := parseEpisodeFileContent(entry.Path, string(data))
		if parseErr != nil {
			continue
		}
		if matchesEpisodeFilters(record, opts) {
			records = append(records, record)
		}
	}

	sortEpisodesNewestFirst(records)

	limit := normaliseEpisodeLimit(opts.Limit, defaultListLimit)
	if len(records) > limit {
		records = records[:limit]
	}

	if records == nil {
		records = []EpisodeRecord{}
	}
	return records, nil
}

// UpdateEpisode overwrites an existing episode.
func (s *BrainEpisodeStore) UpdateEpisode(ctx context.Context, ep EpisodeRecord) error {
	path := EpisodePath(ep.SessionID)
	exists, err := s.store.Exists(ctx, path)
	if err != nil {
		return fmt.Errorf("checking episode existence: %w", err)
	}
	if !exists {
		return fmt.Errorf("episode %q not found: %w", ep.SessionID, brain.ErrNotFound)
	}

	ep.Modified = time.Now().UTC().Format(time.RFC3339)
	ep.Path = path
	if ep.Name == "" {
		ep.Name = "Episode " + ep.SessionID
	}
	normaliseEpisodeDefaults(&ep)

	content := buildEpisodeFileContent(ep)
	return s.store.Write(ctx, path, []byte(content))
}

// DeleteEpisode removes an episode by sessionID.
func (s *BrainEpisodeStore) DeleteEpisode(ctx context.Context, sessionID string) error {
	return s.store.Delete(ctx, EpisodePath(sessionID))
}

// QueryEpisodes performs a text query with relevance scoring.
func (s *BrainEpisodeStore) QueryEpisodes(ctx context.Context, opts EpisodeQueryOptions) ([]EpisodeQueryHit, error) {
	baseOpts := opts.EpisodeListOptions
	baseOpts.Limit = maxLimit

	records, err := s.ListEpisodes(ctx, baseOpts)
	if err != nil {
		return []EpisodeQueryHit{}, err
	}

	queryTokens := tokeniseText(strings.ToLower(opts.Query), 32)
	trimmedQuery := strings.ToLower(strings.TrimSpace(opts.Query))
	if len(queryTokens) == 0 && trimmedQuery == "" {
		return []EpisodeQueryHit{}, nil
	}

	var hits []EpisodeQueryHit
	for i := range records {
		score := scoreEpisode(records[i], trimmedQuery, queryTokens)
		if score > 0 {
			hits = append(hits, EpisodeQueryHit{
				EpisodeRecord: records[i],
				Score:         score,
			})
		}
	}

	sortQueryHits(hits)

	limit := normaliseEpisodeLimit(opts.Limit, defaultQueryLimit)
	if len(hits) > limit {
		hits = hits[:limit]
	}

	if hits == nil {
		hits = []EpisodeQueryHit{}
	}
	return hits, nil
}

// normaliseEpisodeDefaults ensures slice fields are never nil.
func normaliseEpisodeDefaults(ep *EpisodeRecord) {
	if ep.Tags == nil {
		ep.Tags = []string{}
	}
	if ep.OpenQuestions == nil {
		ep.OpenQuestions = []string{}
	}
	if ep.Heuristics == nil {
		ep.Heuristics = []EpisodeHeuristic{}
	}
}

// normaliseEpisodeLimit clamps a limit value.
func normaliseEpisodeLimit(value, fallback int) int {
	if value <= 0 {
		return fallback
	}
	if value > maxLimit {
		return maxLimit
	}
	return value
}

// EpisodeRecorder evaluates whether a completed session produced a
// significant episode worth recording, and if so, summarises it.
type EpisodeRecorder struct{}

// NewEpisodeRecorder creates a new EpisodeRecorder.
func NewEpisodeRecorder() *EpisodeRecorder {
	return &EpisodeRecorder{}
}

// episodeSystemPrompt is the system prompt for episode summarisation.
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
// record an episode. This is the legacy interface; new callers should
// prefer BrainEpisodeStore directly.
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

	ep := EpisodeRecord{
		SessionID:           sessionID,
		ActorID:             projectPath,
		Scope:               EpisodeScopeProject,
		Summary:             result.Summary,
		Outcome:             normaliseOutcome(result.Outcome),
		ShouldRecordEpisode: true,
		OpenQuestions:       []string{},
		Heuristics:          heuristicsFromStrings(result.Heuristics),
		Tags:                result.Tags,
		Signals:             detectSignals(messages),
	}

	if store == nil {
		return nil
	}
	if err := store.CreateEpisode(ctx, ep); err != nil {
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

// detectSignals analyses session messages and returns signal metrics.
func detectSignals(messages []Message) EpisodeSignals {
	var signals EpisodeSignals
	signals.MessageCount = len(messages)

	for _, m := range messages {
		if m.Role != RoleSystem {
			signals.SubstantiveMessageCount++
		}
		switch m.Role {
		case RoleUser:
			signals.UserMessageCount++
		case RoleAssistant:
			signals.AssistantMessageCount++
			for _, tc := range m.ToolCalls {
				signals.ToolCallCount++
				signals.ToolSignal = true
				name := strings.ToLower(strings.TrimSpace(tc.Name))
				if name == "write" {
					signals.WriteSignal = true
				}
				if name == "edit" {
					signals.EditSignal = true
				}
			}
		case RoleTool:
			signals.ToolMessageCount++
			signals.ToolSignal = true
		}
	}

	return signals
}

// heuristicsFromStrings converts plain-string heuristics from the
// legacy LLM output into EpisodeHeuristic structs.
func heuristicsFromStrings(raw []string) []EpisodeHeuristic {
	out := make([]EpisodeHeuristic, 0, len(raw))
	for _, r := range raw {
		if strings.TrimSpace(r) == "" {
			continue
		}
		out = append(out, EpisodeHeuristic{
			Rule:       r,
			Context:    "",
			Confidence: "low",
			Category:   "general",
			Scope:      "project",
		})
	}
	return out
}
