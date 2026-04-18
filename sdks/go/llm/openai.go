// SPDX-License-Identifier: Apache-2.0

package llm

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
)

// OpenAIConfig configures [NewOpenAI].
type OpenAIConfig struct {
	APIKey     string
	BaseURL    string
	// Model is the default completion model used when CompleteRequest.Model is
	// empty. If unset here too, requests without a model will fail at the API.
	Model      string
	HTTPClient *http.Client
}

// OpenAIEmbedConfig configures [NewOpenAIEmbedder].
type OpenAIEmbedConfig struct {
	APIKey     string
	BaseURL    string
	Model      string
	Dimensions int
	HTTPClient *http.Client
}

const (
	openAIDefaultBase       = "https://api.openai.com/v1"
	openAIDefaultEmbedModel = "text-embedding-3-small"
	openAIDefaultEmbedDims  = 1536
)

// NewOpenAI returns a [Provider] backed by the OpenAI Chat Completions API.
func NewOpenAI(cfg OpenAIConfig) Provider {
	if cfg.BaseURL == "" {
		cfg.BaseURL = openAIDefaultBase
	}
	if cfg.HTTPClient == nil {
		cfg.HTTPClient = http.DefaultClient
	}
	return &openAIProvider{cfg: cfg}
}

// NewOpenAIEmbedder returns an [Embedder] for OpenAI embedding models.
func NewOpenAIEmbedder(cfg OpenAIEmbedConfig) Embedder {
	if cfg.BaseURL == "" {
		cfg.BaseURL = openAIDefaultBase
	}
	if cfg.Model == "" {
		cfg.Model = openAIDefaultEmbedModel
	}
	if cfg.Dimensions == 0 {
		cfg.Dimensions = openAIDefaultEmbedDims
	}
	if cfg.HTTPClient == nil {
		cfg.HTTPClient = http.DefaultClient
	}
	return &openAIEmbedder{cfg: cfg}
}

type openAIProvider struct {
	cfg OpenAIConfig
}

type openAIChatRequest struct {
	Model       string              `json:"model"`
	Messages    []openAIChatMessage `json:"messages"`
	Temperature float64             `json:"temperature,omitempty"`
	MaxTokens   int                 `json:"max_tokens,omitempty"`
	Stop        []string            `json:"stop,omitempty"`
	Stream      bool                `json:"stream,omitempty"`
	Tools       []openAIToolDef     `json:"tools,omitempty"`
}

type openAIChatMessage struct {
	Role       string           `json:"role"`
	Content    string           `json:"content,omitempty"`
	ToolCalls  []openAIToolCall `json:"tool_calls,omitempty"`
	ToolCallID string           `json:"tool_call_id,omitempty"`
	Name       string           `json:"name,omitempty"`
}

type openAIToolDef struct {
	Type     string                 `json:"type"`
	Function openAIToolDefFunction  `json:"function"`
}

type openAIToolDefFunction struct {
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	Parameters  map[string]any `json:"parameters,omitempty"`
}

type openAIToolCall struct {
	ID       string                  `json:"id"`
	Type     string                  `json:"type"`
	Function openAIToolCallFunction  `json:"function"`
}

type openAIToolCallFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type openAIChatResponse struct {
	Choices []openAIChoice `json:"choices"`
	Usage   openAIUsage    `json:"usage"`
	Error   *openAIError   `json:"error,omitempty"`
}

type openAIChoice struct {
	Index        int               `json:"index"`
	Message      openAIChatMessage `json:"message"`
	FinishReason string            `json:"finish_reason"`
}

type openAIUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

type openAIError struct {
	Message string `json:"message"`
	Type    string `json:"type"`
	Code    string `json:"code"`
}

type openAIStreamChunk struct {
	Choices []openAIStreamChoice `json:"choices"`
	Error   *openAIError         `json:"error,omitempty"`
}

type openAIStreamChoice struct {
	Index        int                `json:"index"`
	Delta        openAIStreamDelta  `json:"delta"`
	FinishReason string             `json:"finish_reason"`
}

type openAIStreamDelta struct {
	Role      string                  `json:"role,omitempty"`
	Content   string                  `json:"content,omitempty"`
	ToolCalls []openAIStreamToolCall  `json:"tool_calls,omitempty"`
}

type openAIStreamToolCall struct {
	Index    int                          `json:"index"`
	ID       string                       `json:"id,omitempty"`
	Type     string                       `json:"type,omitempty"`
	Function openAIStreamToolCallFunction `json:"function"`
}

type openAIStreamToolCallFunction struct {
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments,omitempty"`
}

func (p *openAIProvider) Complete(ctx context.Context, req CompleteRequest) (CompleteResponse, error) {
	if len(req.Messages) == 0 {
		return CompleteResponse{}, ErrEmpty
	}
	if req.Model == "" {
		req.Model = p.cfg.Model
	}
	body, err := json.Marshal(openAIBuildRequest(req, false))
	if err != nil {
		return CompleteResponse{}, fmt.Errorf("llm: marshal openai request: %w", err)
	}
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, p.cfg.BaseURL+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return CompleteResponse{}, err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+p.cfg.APIKey)
	resp, err := p.cfg.HTTPClient.Do(httpReq)
	if err != nil {
		return CompleteResponse{}, fmt.Errorf("llm: openai request: %w", err)
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return CompleteResponse{}, err
	}
	if resp.StatusCode >= 400 {
		return CompleteResponse{}, openAIParseError(resp.StatusCode, raw)
	}
	var parsed openAIChatResponse
	if err := json.Unmarshal(raw, &parsed); err != nil {
		return CompleteResponse{}, fmt.Errorf("llm: decode openai response: %w", err)
	}
	if parsed.Error != nil {
		return CompleteResponse{}, fmt.Errorf("llm: openai: %s", parsed.Error.Message)
	}
	if len(parsed.Choices) == 0 {
		return CompleteResponse{}, errors.New("llm: openai returned no choices")
	}
	choice := parsed.Choices[0]
	out := CompleteResponse{
		Text:      choice.Message.Content,
		Stop:      mapOpenAIFinish(choice.FinishReason),
		TokensIn:  parsed.Usage.PromptTokens,
		TokensOut: parsed.Usage.CompletionTokens,
	}
	for _, tc := range choice.Message.ToolCalls {
		out.ToolCalls = append(out.ToolCalls, ToolCall{
			ID:        tc.ID,
			Name:      tc.Function.Name,
			Arguments: json.RawMessage(tc.Function.Arguments),
		})
	}
	return out, nil
}

func (p *openAIProvider) CompleteStream(ctx context.Context, req CompleteRequest) (<-chan StreamChunk, error) {
	if len(req.Messages) == 0 {
		return nil, ErrEmpty
	}
	if req.Model == "" {
		req.Model = p.cfg.Model
	}
	body, err := json.Marshal(openAIBuildRequest(req, true))
	if err != nil {
		return nil, fmt.Errorf("llm: marshal openai request: %w", err)
	}
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, p.cfg.BaseURL+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+p.cfg.APIKey)
	httpReq.Header.Set("Accept", "text/event-stream")
	resp, err := p.cfg.HTTPClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("llm: openai stream request: %w", err)
	}
	if resp.StatusCode >= 400 {
		raw, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		return nil, openAIParseError(resp.StatusCode, raw)
	}
	out := make(chan StreamChunk, 16)
	go func() {
		defer resp.Body.Close()
		defer close(out)
		scanner := bufio.NewScanner(resp.Body)
		scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)
		toolBuffers := map[int]*openAIToolBuffer{}
		for scanner.Scan() {
			line := scanner.Text()
			if !strings.HasPrefix(line, "data: ") {
				continue
			}
			payload := strings.TrimPrefix(line, "data: ")
			if payload == "[DONE]" {
				return
			}
			var chunk openAIStreamChunk
			if err := json.Unmarshal([]byte(payload), &chunk); err != nil {
				continue
			}
			if chunk.Error != nil {
				return
			}
			for _, c := range chunk.Choices {
				if c.Delta.Content != "" {
					select {
					case <-ctx.Done():
						return
					case out <- StreamChunk{DeltaText: c.Delta.Content}:
					}
				}
				for _, tc := range c.Delta.ToolCalls {
					buf, ok := toolBuffers[tc.Index]
					if !ok {
						buf = &openAIToolBuffer{}
						toolBuffers[tc.Index] = buf
					}
					if tc.ID != "" {
						buf.ID = tc.ID
					}
					if tc.Function.Name != "" {
						buf.Name = tc.Function.Name
					}
					if tc.Function.Arguments != "" {
						buf.Args.WriteString(tc.Function.Arguments)
					}
				}
				if c.FinishReason != "" {
					for _, idx := range sortedToolIndexes(toolBuffers) {
						buf := toolBuffers[idx]
						select {
						case <-ctx.Done():
							return
						case out <- StreamChunk{ToolCall: &ToolCall{
							ID:        buf.ID,
							Name:      buf.Name,
							Arguments: json.RawMessage(buf.Args.String()),
						}}:
						}
					}
					select {
					case <-ctx.Done():
					case out <- StreamChunk{Stop: mapOpenAIFinish(c.FinishReason)}:
					}
				}
			}
		}
	}()
	return out, nil
}

func (p *openAIProvider) Close() error { return nil }

type openAIToolBuffer struct {
	ID   string
	Name string
	Args strings.Builder
}

func sortedToolIndexes(buffers map[int]*openAIToolBuffer) []int {
	if len(buffers) == 0 {
		return nil
	}
	idx := make([]int, 0, len(buffers))
	for i := range buffers {
		idx = append(idx, i)
	}
	// Stable small sort; tool calls are always few.
	for i := 1; i < len(idx); i++ {
		for j := i; j > 0 && idx[j-1] > idx[j]; j-- {
			idx[j-1], idx[j] = idx[j], idx[j-1]
		}
	}
	return idx
}

func openAIBuildRequest(req CompleteRequest, stream bool) openAIChatRequest {
	out := openAIChatRequest{
		Model:       req.Model,
		Temperature: req.Temperature,
		MaxTokens:   req.MaxTokens,
		Stop:        req.Stop,
		Stream:      stream,
	}
	for _, m := range req.Messages {
		msg := openAIChatMessage{
			Role:    string(m.Role),
			Content: m.Content,
		}
		out.Messages = append(out.Messages, msg)
	}
	for _, t := range req.Tools {
		out.Tools = append(out.Tools, openAIToolDef{
			Type: "function",
			Function: openAIToolDefFunction{
				Name:        t.Name,
				Description: t.Description,
				Parameters:  t.Schema,
			},
		})
	}
	return out
}

func mapOpenAIFinish(s string) StopReason {
	switch s {
	case "stop":
		return StopEndTurn
	case "length":
		return StopMaxTokens
	case "tool_calls", "function_call":
		return StopToolUse
	case "content_filter":
		return StopEndTurn
	default:
		return StopEndTurn
	}
}

func openAIParseError(status int, body []byte) error {
	var env struct {
		Error *openAIError `json:"error"`
	}
	if err := json.Unmarshal(body, &env); err == nil && env.Error != nil {
		return fmt.Errorf("llm: openai %d: %s", status, env.Error.Message)
	}
	msg := strings.TrimSpace(string(body))
	if msg == "" {
		msg = http.StatusText(status)
	}
	return fmt.Errorf("llm: openai %d: %s", status, msg)
}

type openAIEmbedder struct {
	cfg OpenAIEmbedConfig
}

type openAIEmbedRequest struct {
	Input      []string `json:"input"`
	Model      string   `json:"model"`
	Dimensions int      `json:"dimensions,omitempty"`
}

type openAIEmbedResponse struct {
	Data  []openAIEmbedDatum `json:"data"`
	Error *openAIError       `json:"error,omitempty"`
}

type openAIEmbedDatum struct {
	Index     int       `json:"index"`
	Embedding []float32 `json:"embedding"`
}

func (e *openAIEmbedder) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}
	body, err := json.Marshal(openAIEmbedRequest{
		Input: texts,
		Model: e.cfg.Model,
	})
	if err != nil {
		return nil, fmt.Errorf("llm: marshal openai embed request: %w", err)
	}
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, e.cfg.BaseURL+"/embeddings", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+e.cfg.APIKey)
	resp, err := e.cfg.HTTPClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("llm: openai embed: %w", err)
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode >= 400 {
		return nil, openAIParseError(resp.StatusCode, raw)
	}
	var parsed openAIEmbedResponse
	if err := json.Unmarshal(raw, &parsed); err != nil {
		return nil, fmt.Errorf("llm: decode openai embed response: %w", err)
	}
	if parsed.Error != nil {
		return nil, fmt.Errorf("llm: openai embed: %s", parsed.Error.Message)
	}
	out := make([][]float32, len(parsed.Data))
	for _, d := range parsed.Data {
		if d.Index < 0 || d.Index >= len(out) {
			continue
		}
		out[d.Index] = d.Embedding
	}
	return out, nil
}

func (e *openAIEmbedder) Dimensions() int { return e.cfg.Dimensions }

func (e *openAIEmbedder) Close() error { return nil }
