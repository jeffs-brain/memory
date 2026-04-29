// SPDX-License-Identifier: Apache-2.0

package llm

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
)

// AnthropicConfig configures [NewAnthropic].
type AnthropicConfig struct {
	APIKey     string
	BaseURL    string
	Version    string
	// Model is the default completion model used when CompleteRequest.Model is
	// empty. If unset here too, requests without a model will fail at the API.
	Model      string
	HTTPClient *http.Client
}

const (
	anthropicDefaultBase    = "https://api.anthropic.com"
	anthropicDefaultVersion = "2023-06-01"
)

// NewAnthropic returns a [Provider] backed by the Anthropic Messages API.
func NewAnthropic(cfg AnthropicConfig) Provider {
	if cfg.BaseURL == "" {
		cfg.BaseURL = anthropicDefaultBase
	}
	if cfg.Version == "" {
		cfg.Version = anthropicDefaultVersion
	}
	if cfg.HTTPClient == nil {
		cfg.HTTPClient = http.DefaultClient
	}
	return &anthropicProvider{cfg: cfg}
}

type anthropicProvider struct {
	cfg AnthropicConfig
}

type anthropicRequest struct {
	Model       string             `json:"model"`
	Messages    []anthropicMessage `json:"messages"`
	System      string             `json:"system,omitempty"`
	Temperature float64            `json:"temperature,omitempty"`
	MaxTokens   int                `json:"max_tokens"`
	StopSeqs    []string           `json:"stop_sequences,omitempty"`
	Stream      bool               `json:"stream,omitempty"`
	Tools       []anthropicTool    `json:"tools,omitempty"`
}

type anthropicMessage struct {
	Role    string             `json:"role"`
	Content []anthropicContent `json:"content"`
}

type anthropicContent struct {
	Type      string          `json:"type"`
	Text      string          `json:"text,omitempty"`
	ID        string          `json:"id,omitempty"`
	Name      string          `json:"name,omitempty"`
	Input     json.RawMessage `json:"input,omitempty"`
	ToolUseID string          `json:"tool_use_id,omitempty"`
	Content   string          `json:"content,omitempty"`
}

type anthropicTool struct {
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	InputSchema map[string]any `json:"input_schema,omitempty"`
}

type anthropicResponse struct {
	ID           string             `json:"id"`
	Type         string             `json:"type"`
	Role         string             `json:"role"`
	Content      []anthropicContent `json:"content"`
	StopReason   string             `json:"stop_reason"`
	StopSequence string             `json:"stop_sequence"`
	Usage        anthropicUsage     `json:"usage"`
	Error        *anthropicError    `json:"error,omitempty"`
}

type anthropicUsage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}

type anthropicError struct {
	Type    string `json:"type"`
	Message string `json:"message"`
}

type anthropicEvent struct {
	Type         string             `json:"type"`
	Index        int                `json:"index"`
	Delta        anthropicDelta     `json:"delta"`
	ContentBlock anthropicContent   `json:"content_block"`
	Message      anthropicResponse  `json:"message"`
	Usage        *anthropicUsage    `json:"usage"`
}

type anthropicDelta struct {
	Type         string `json:"type"`
	Text         string `json:"text,omitempty"`
	StopReason   string `json:"stop_reason,omitempty"`
	StopSequence string `json:"stop_sequence,omitempty"`
	PartialJSON  string `json:"partial_json,omitempty"`
}

func (p *anthropicProvider) Complete(ctx context.Context, req CompleteRequest) (CompleteResponse, error) {
	if len(req.Messages) == 0 {
		return CompleteResponse{}, ErrEmpty
	}
	if req.Model == "" {
		req.Model = p.cfg.Model
	}
	body, err := json.Marshal(anthropicBuildRequest(req, false))
	if err != nil {
		return CompleteResponse{}, fmt.Errorf("llm: marshal anthropic request: %w", err)
	}
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, p.cfg.BaseURL+"/v1/messages", bytes.NewReader(body))
	if err != nil {
		return CompleteResponse{}, err
	}
	p.setHeaders(httpReq)
	resp, err := p.cfg.HTTPClient.Do(httpReq)
	if err != nil {
		return CompleteResponse{}, fmt.Errorf("llm: anthropic request: %w", err)
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return CompleteResponse{}, err
	}
	if resp.StatusCode >= 400 {
		return CompleteResponse{}, anthropicParseError(resp.StatusCode, raw)
	}
	var parsed anthropicResponse
	if err := json.Unmarshal(raw, &parsed); err != nil {
		return CompleteResponse{}, fmt.Errorf("llm: decode anthropic response: %w", err)
	}
	if parsed.Error != nil {
		return CompleteResponse{}, fmt.Errorf("llm: anthropic: %s", parsed.Error.Message)
	}
	out := CompleteResponse{
		Stop:      mapAnthropicStop(parsed.StopReason),
		TokensIn:  parsed.Usage.InputTokens,
		TokensOut: parsed.Usage.OutputTokens,
	}
	var textBuf strings.Builder
	for _, block := range parsed.Content {
		switch block.Type {
		case "text":
			textBuf.WriteString(block.Text)
		case "tool_use":
			out.ToolCalls = append(out.ToolCalls, ToolCall{
				ID:        block.ID,
				Name:      block.Name,
				Arguments: json.RawMessage(block.Input),
			})
		}
	}
	out.Text = textBuf.String()
	return out, nil
}

func (p *anthropicProvider) CompleteStream(ctx context.Context, req CompleteRequest) (<-chan StreamChunk, error) {
	if len(req.Messages) == 0 {
		return nil, ErrEmpty
	}
	if req.Model == "" {
		req.Model = p.cfg.Model
	}
	body, err := json.Marshal(anthropicBuildRequest(req, true))
	if err != nil {
		return nil, fmt.Errorf("llm: marshal anthropic request: %w", err)
	}
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, p.cfg.BaseURL+"/v1/messages", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	p.setHeaders(httpReq)
	httpReq.Header.Set("Accept", "text/event-stream")
	resp, err := p.cfg.HTTPClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("llm: anthropic stream request: %w", err)
	}
	if resp.StatusCode >= 400 {
		raw, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		return nil, anthropicParseError(resp.StatusCode, raw)
	}
	out := make(chan StreamChunk, 16)
	go func() {
		defer resp.Body.Close()
		defer close(out)
		scanner := bufio.NewScanner(resp.Body)
		scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)
		for scanner.Scan() {
			line := scanner.Text()
			if !strings.HasPrefix(line, "data: ") {
				continue
			}
			payload := strings.TrimPrefix(line, "data: ")
			if payload == "" {
				continue
			}
			var ev anthropicEvent
			if err := json.Unmarshal([]byte(payload), &ev); err != nil {
				continue
			}
			switch ev.Type {
			case "content_block_delta":
				if ev.Delta.Text != "" {
					select {
					case <-ctx.Done():
						return
					case out <- StreamChunk{DeltaText: ev.Delta.Text}:
					}
				}
			case "message_delta":
				if ev.Delta.StopReason != "" {
					select {
					case <-ctx.Done():
					case out <- StreamChunk{Stop: mapAnthropicStop(ev.Delta.StopReason)}:
					}
				}
			case "message_stop":
				return
			}
		}
	}()
	return out, nil
}

func (p *anthropicProvider) Close() error { return nil }

func (p *anthropicProvider) setHeaders(req *http.Request) {
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-api-key", p.cfg.APIKey)
	req.Header.Set("anthropic-version", p.cfg.Version)
}

func anthropicBuildRequest(req CompleteRequest, stream bool) anthropicRequest {
	out := anthropicRequest{
		Model:       req.Model,
		Temperature: req.Temperature,
		MaxTokens:   req.MaxTokens,
		StopSeqs:    req.Stop,
		Stream:      stream,
	}
	if out.MaxTokens == 0 {
		// Anthropic requires max_tokens; pick a reasonable default if
		// the caller left it zero.
		out.MaxTokens = 1024
	}
	var system strings.Builder
	for _, m := range req.Messages {
		if m.Role == RoleSystem {
			if system.Len() > 0 {
				system.WriteString("\n\n")
			}
			system.WriteString(m.Content)
			continue
		}
		role := "user"
		switch m.Role {
		case RoleAssistant:
			role = "assistant"
		case RoleTool:
			role = "user"
		}
		out.Messages = append(out.Messages, anthropicMessage{
			Role: role,
			Content: []anthropicContent{{
				Type: "text",
				Text: m.Content,
			}},
		})
	}
	out.System = system.String()
	for _, t := range req.Tools {
		out.Tools = append(out.Tools, anthropicTool{
			Name:        t.Name,
			Description: t.Description,
			InputSchema: t.Schema,
		})
	}
	return out
}

func mapAnthropicStop(s string) StopReason {
	switch s {
	case "end_turn":
		return StopEndTurn
	case "max_tokens":
		return StopMaxTokens
	case "tool_use":
		return StopToolUse
	case "stop_sequence":
		return StopStop
	default:
		return StopEndTurn
	}
}

func anthropicParseError(status int, body []byte) error {
	var env struct {
		Error *anthropicError `json:"error"`
	}
	if err := json.Unmarshal(body, &env); err == nil && env.Error != nil {
		return fmt.Errorf("llm: anthropic %d: %s", status, env.Error.Message)
	}
	msg := strings.TrimSpace(string(body))
	if msg == "" {
		msg = http.StatusText(status)
	}
	return fmt.Errorf("llm: anthropic %d: %s", status, msg)
}
