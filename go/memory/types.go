// SPDX-License-Identifier: Apache-2.0

package memory

import (
	"encoding/json"

	"github.com/jeffs-brain/memory/go/llm"
)

// Role re-exports [llm.Role] so callers do not need two imports.
type Role = llm.Role

// Role aliases, re-exported for convenience.
const (
	RoleSystem    = llm.RoleSystem
	RoleUser      = llm.RoleUser
	RoleAssistant = llm.RoleAssistant
	RoleTool      = llm.RoleTool
)

// ToolCall mirrors a tool invocation requested by the model. It is kept
// inside the memory package because the SDK's llm.Provider type carries
// a simpler surface; memory extraction and reflection need access to the
// tool-call stream to analyse sessions.
type ToolCall struct {
	ID        string
	Name      string
	Arguments json.RawMessage
}

// ToolResultBlock represents a tool-result attachment on a message.
type ToolResultBlock struct {
	ToolCallID string
	Content    string
	IsError    bool
}

// ContentBlock is a structured content fragment attached to a [Message].
// Only the subset memory analysis uses is exposed.
type ContentBlock struct {
	Type       string
	Text       string
	ToolUse    *ToolCall
	ToolResult *ToolResultBlock
}

// Message is the memory package's conversation-turn type. It carries the
// tool-call metadata needed by extract / reflect / buffer analysis. A
// [Message] is converted to an [llm.Message] at the provider call site
// via [Message.AsLLM].
type Message struct {
	Role       Role
	Content    string
	ToolCalls  []ToolCall
	ToolCallID string
	Name       string
	Blocks     []ContentBlock
}

// AsLLM returns the [llm.Message] equivalent for passing to a provider.
// Tool calls and blocks are dropped; providers see only role + text.
func (m Message) AsLLM() llm.Message {
	return llm.Message{Role: m.Role, Content: m.Content}
}

// MessagesAsLLM converts a message slice for provider calls.
func MessagesAsLLM(msgs []Message) []llm.Message {
	out := make([]llm.Message, len(msgs))
	for i, m := range msgs {
		out[i] = m.AsLLM()
	}
	return out
}
