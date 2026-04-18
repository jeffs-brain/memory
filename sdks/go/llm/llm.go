// SPDX-License-Identifier: Apache-2.0

package llm

import (
	"context"
	"encoding/json"
	"errors"
)

// Role is the speaker of a [Message].
type Role string

const (
	// RoleSystem is a system instruction.
	RoleSystem Role = "system"
	// RoleUser is an end-user turn.
	RoleUser Role = "user"
	// RoleAssistant is a model turn.
	RoleAssistant Role = "assistant"
	// RoleTool is a tool-result turn.
	RoleTool Role = "tool"
)

// Message is a single conversation turn.
//
// Attachments and tool calls beyond plain text are intentionally omitted
// from the initial shape; they will be added when a downstream consumer
// actually needs them.
type Message struct {
	Role    Role
	Content string
}

// CompleteRequest is the common request every [Provider] accepts.
type CompleteRequest struct {
	Model       string
	Messages    []Message
	Temperature float64
	MaxTokens   int
	Stop        []string
	Stream      bool
	Tools       []ToolDef
}

// CompleteResponse is returned by [Provider.Complete].
type CompleteResponse struct {
	Text      string
	Stop      StopReason
	TokensIn  int
	TokensOut int
	ToolCalls []ToolCall
}

// StreamChunk is one unit emitted on a streaming channel.
//
// Consumers should treat DeltaText as incremental; Stop is non-empty on the
// final chunk. ToolCall is set when a provider streams tool calls inline.
type StreamChunk struct {
	DeltaText string
	ToolCall  *ToolCall
	Stop      StopReason
}

// StopReason describes why generation halted.
type StopReason string

const (
	// StopEndTurn is a normal completion.
	StopEndTurn StopReason = "end_turn"
	// StopMaxTokens is the token budget exhausted.
	StopMaxTokens StopReason = "max_tokens"
	// StopToolUse is a tool-call handoff.
	StopToolUse StopReason = "tool_use"
	// StopStop is a user-supplied stop sequence hit.
	StopStop StopReason = "stop_sequence"
)

// ToolDef describes a callable tool a model may invoke.
type ToolDef struct {
	Name        string
	Description string
	Schema      map[string]any
}

// ToolCall is a model's request to invoke a tool.
type ToolCall struct {
	Name      string
	Arguments json.RawMessage
	ID        string
}

// Provider is the chat completion surface every backend implements.
type Provider interface {
	// Complete runs a non-streaming completion.
	Complete(ctx context.Context, req CompleteRequest) (CompleteResponse, error)
	// CompleteStream runs a streaming completion. The returned channel
	// is closed once generation is done or the context is cancelled.
	CompleteStream(ctx context.Context, req CompleteRequest) (<-chan StreamChunk, error)
	// Close releases any held resources.
	Close() error
}

// Embedder is a sibling interface for embedding models. It is exposed
// separately from [Provider] because embed endpoints and chat endpoints
// rarely share lifecycle in practice.
type Embedder interface {
	Embed(ctx context.Context, texts []string) ([][]float32, error)
	Dimensions() int
	Close() error
}

// ErrEmpty is returned when a provider receives no messages to complete on.
var ErrEmpty = errors.New("llm: empty messages")

// ErrNoProvider is returned by [ProviderFromEnv] when the environment does
// not select a provider and no auto-detected backend is reachable.
var ErrNoProvider = errors.New("llm: no provider configured")
