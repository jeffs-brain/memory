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

// OllamaConfig configures [NewOllama].
type OllamaConfig struct {
	BaseURL    string
	Model      string
	HTTPClient *http.Client
}

// OllamaEmbedConfig configures [NewOllamaEmbedder].
type OllamaEmbedConfig struct {
	BaseURL    string
	Model      string
	Dimensions int
	HTTPClient *http.Client
}

const (
	ollamaDefaultBase       = "http://localhost:11434"
	ollamaDefaultEmbedModel = "bge-m3"
	ollamaDefaultEmbedDims  = 1024
)

// NewOllama returns a [Provider] backed by a local Ollama instance using
// its /api/chat endpoint.
func NewOllama(cfg OllamaConfig) Provider {
	if cfg.BaseURL == "" {
		cfg.BaseURL = ollamaDefaultBase
	}
	if cfg.HTTPClient == nil {
		cfg.HTTPClient = http.DefaultClient
	}
	return &ollamaProvider{cfg: cfg}
}

// NewOllamaEmbedder returns an [Embedder] backed by Ollama's /api/embed
// endpoint.
func NewOllamaEmbedder(cfg OllamaEmbedConfig) Embedder {
	if cfg.BaseURL == "" {
		cfg.BaseURL = ollamaDefaultBase
	}
	if cfg.Model == "" {
		cfg.Model = ollamaDefaultEmbedModel
	}
	if cfg.Dimensions == 0 {
		cfg.Dimensions = ollamaDefaultEmbedDims
	}
	if cfg.HTTPClient == nil {
		cfg.HTTPClient = http.DefaultClient
	}
	return &ollamaEmbedder{cfg: cfg}
}

type ollamaProvider struct {
	cfg OllamaConfig
}

type ollamaChatRequest struct {
	Model    string          `json:"model"`
	Messages []ollamaMessage `json:"messages"`
	Stream   bool            `json:"stream"`
	Options  *ollamaOptions  `json:"options,omitempty"`
}

type ollamaMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type ollamaOptions struct {
	Temperature float64  `json:"temperature,omitempty"`
	NumPredict  int      `json:"num_predict,omitempty"`
	Stop        []string `json:"stop,omitempty"`
}

type ollamaChatResponse struct {
	Model         string        `json:"model"`
	Message       ollamaMessage `json:"message"`
	Done          bool          `json:"done"`
	DoneReason    string        `json:"done_reason"`
	PromptEval    int           `json:"prompt_eval_count"`
	EvalCount     int           `json:"eval_count"`
	Error         string        `json:"error,omitempty"`
}

func (p *ollamaProvider) Complete(ctx context.Context, req CompleteRequest) (CompleteResponse, error) {
	if len(req.Messages) == 0 {
		return CompleteResponse{}, ErrEmpty
	}
	if req.Model == "" {
		req.Model = p.cfg.Model
	}
	body, err := json.Marshal(ollamaBuildRequest(req, false))
	if err != nil {
		return CompleteResponse{}, fmt.Errorf("llm: marshal ollama request: %w", err)
	}
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, p.cfg.BaseURL+"/api/chat", bytes.NewReader(body))
	if err != nil {
		return CompleteResponse{}, err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	resp, err := p.cfg.HTTPClient.Do(httpReq)
	if err != nil {
		return CompleteResponse{}, fmt.Errorf("llm: ollama request: %w", err)
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return CompleteResponse{}, err
	}
	if resp.StatusCode >= 400 {
		return CompleteResponse{}, fmt.Errorf("llm: ollama %d: %s", resp.StatusCode, strings.TrimSpace(string(raw)))
	}
	var parsed ollamaChatResponse
	if err := json.Unmarshal(raw, &parsed); err != nil {
		return CompleteResponse{}, fmt.Errorf("llm: decode ollama response: %w", err)
	}
	if parsed.Error != "" {
		return CompleteResponse{}, fmt.Errorf("llm: ollama: %s", parsed.Error)
	}
	return CompleteResponse{
		Text:      parsed.Message.Content,
		Stop:      mapOllamaDone(parsed.DoneReason),
		TokensIn:  parsed.PromptEval,
		TokensOut: parsed.EvalCount,
	}, nil
}

func (p *ollamaProvider) CompleteStream(ctx context.Context, req CompleteRequest) (<-chan StreamChunk, error) {
	if len(req.Messages) == 0 {
		return nil, ErrEmpty
	}
	if req.Model == "" {
		req.Model = p.cfg.Model
	}
	body, err := json.Marshal(ollamaBuildRequest(req, true))
	if err != nil {
		return nil, fmt.Errorf("llm: marshal ollama request: %w", err)
	}
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, p.cfg.BaseURL+"/api/chat", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	resp, err := p.cfg.HTTPClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("llm: ollama stream: %w", err)
	}
	if resp.StatusCode >= 400 {
		raw, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		return nil, fmt.Errorf("llm: ollama %d: %s", resp.StatusCode, strings.TrimSpace(string(raw)))
	}
	out := make(chan StreamChunk, 16)
	go func() {
		defer resp.Body.Close()
		defer close(out)
		scanner := bufio.NewScanner(resp.Body)
		scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)
		for scanner.Scan() {
			line := strings.TrimSpace(scanner.Text())
			if line == "" {
				continue
			}
			var evt ollamaChatResponse
			if err := json.Unmarshal([]byte(line), &evt); err != nil {
				continue
			}
			if evt.Message.Content != "" {
				select {
				case <-ctx.Done():
					return
				case out <- StreamChunk{DeltaText: evt.Message.Content}:
				}
			}
			if evt.Done {
				select {
				case <-ctx.Done():
				case out <- StreamChunk{Stop: mapOllamaDone(evt.DoneReason)}:
				}
				return
			}
		}
	}()
	return out, nil
}

func (p *ollamaProvider) Close() error { return nil }

func ollamaBuildRequest(req CompleteRequest, stream bool) ollamaChatRequest {
	out := ollamaChatRequest{
		Model:  req.Model,
		Stream: stream,
	}
	for _, m := range req.Messages {
		role := string(m.Role)
		switch m.Role {
		case RoleTool:
			role = "tool"
		}
		out.Messages = append(out.Messages, ollamaMessage{
			Role:    role,
			Content: m.Content,
		})
	}
	if req.Temperature != 0 || req.MaxTokens != 0 || len(req.Stop) > 0 {
		out.Options = &ollamaOptions{
			Temperature: req.Temperature,
			NumPredict:  req.MaxTokens,
			Stop:        req.Stop,
		}
	}
	return out
}

func mapOllamaDone(s string) StopReason {
	switch s {
	case "stop":
		return StopEndTurn
	case "length":
		return StopMaxTokens
	default:
		return StopEndTurn
	}
}

type ollamaEmbedder struct {
	cfg OllamaEmbedConfig
}

type ollamaEmbedRequest struct {
	Model string   `json:"model"`
	Input []string `json:"input"`
}

type ollamaEmbedResponse struct {
	Embeddings [][]float32 `json:"embeddings"`
	Error      string      `json:"error,omitempty"`
}

func (e *ollamaEmbedder) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}
	body, err := json.Marshal(ollamaEmbedRequest{
		Model: e.cfg.Model,
		Input: texts,
	})
	if err != nil {
		return nil, fmt.Errorf("llm: marshal ollama embed request: %w", err)
	}
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, e.cfg.BaseURL+"/api/embed", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	resp, err := e.cfg.HTTPClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("llm: ollama embed: %w", err)
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode >= 400 {
		return nil, fmt.Errorf("llm: ollama embed %d: %s", resp.StatusCode, strings.TrimSpace(string(raw)))
	}
	var parsed ollamaEmbedResponse
	if err := json.Unmarshal(raw, &parsed); err != nil {
		return nil, fmt.Errorf("llm: decode ollama embed response: %w", err)
	}
	if parsed.Error != "" {
		return nil, fmt.Errorf("llm: ollama embed: %s", parsed.Error)
	}
	return parsed.Embeddings, nil
}

func (e *ollamaEmbedder) Dimensions() int { return e.cfg.Dimensions }

func (e *ollamaEmbedder) Close() error { return nil }
