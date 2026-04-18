// SPDX-License-Identifier: Apache-2.0

// Package tools defines the MCP tool contracts exposed by the
// memory-mcp wrapper. Each tool lives in its own file and delegates to
// the [MemoryClient] implementation supplied at registration time.
package tools

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/modelcontextprotocol/go-sdk/mcp"
)

// MemoryClient is the dispatch surface the tools speak to. Defined as
// an interface locally so the tools package can be imported by the
// main package without a cycle; the real client implementation lives
// in the parent package.
type MemoryClient interface {
	Remember(ctx context.Context, args RememberArgs) (map[string]any, error)
	Search(ctx context.Context, args SearchArgs) (map[string]any, error)
	Recall(ctx context.Context, args RecallArgs) (map[string]any, error)
	Ask(ctx context.Context, args AskArgs, progress ProgressEmitter) (map[string]any, error)
	IngestFile(ctx context.Context, args IngestFileArgs, progress ProgressEmitter) (map[string]any, error)
	IngestURL(ctx context.Context, args IngestURLArgs, progress ProgressEmitter) (map[string]any, error)
	Extract(ctx context.Context, args ExtractArgs, progress ProgressEmitter) (map[string]any, error)
	Reflect(ctx context.Context, args ReflectArgs, progress ProgressEmitter) (map[string]any, error)
	Consolidate(ctx context.Context, args ConsolidateArgs, progress ProgressEmitter) (map[string]any, error)
	CreateBrain(ctx context.Context, args CreateBrainArgs) (map[string]any, error)
	ListBrains(ctx context.Context) (map[string]any, error)
}

// ProgressEmitter mirrors the MemoryClient-side helper signature so
// tool handlers can forward progress from the MCP request context.
type ProgressEmitter func(progress float64, message string)

// RememberArgs is the input shape for memory_remember.
type RememberArgs struct {
	Content string   `json:"content"`
	Title   string   `json:"title,omitempty"`
	Brain   string   `json:"brain,omitempty"`
	Tags    []string `json:"tags,omitempty"`
	Path    string   `json:"path,omitempty"`
}

// SearchArgs is the input shape for memory_search.
type SearchArgs struct {
	Query string `json:"query"`
	Brain string `json:"brain,omitempty"`
	TopK  int    `json:"top_k,omitempty"`
	Scope string `json:"scope,omitempty"`
	Sort  string `json:"sort,omitempty"`
}

// RecallArgs is the input shape for memory_recall.
type RecallArgs struct {
	Query     string `json:"query"`
	Brain     string `json:"brain,omitempty"`
	Scope     string `json:"scope,omitempty"`
	SessionID string `json:"session_id,omitempty"`
	TopK      int    `json:"top_k,omitempty"`
}

// AskArgs is the input shape for memory_ask.
type AskArgs struct {
	Query string `json:"query"`
	Brain string `json:"brain,omitempty"`
	TopK  int    `json:"top_k,omitempty"`
}

// IngestFileArgs is the input shape for memory_ingest_file.
type IngestFileArgs struct {
	Path  string `json:"path"`
	Brain string `json:"brain,omitempty"`
	As    string `json:"as,omitempty"`
}

// IngestURLArgs is the input shape for memory_ingest_url.
type IngestURLArgs struct {
	URL   string `json:"url"`
	Brain string `json:"brain,omitempty"`
}

// ExtractMessage is one turn supplied to memory_extract.
type ExtractMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// ExtractArgs is the input shape for memory_extract.
type ExtractArgs struct {
	Messages  []ExtractMessage `json:"messages"`
	Brain     string           `json:"brain,omitempty"`
	ActorID   string           `json:"actor_id,omitempty"`
	SessionID string           `json:"session_id,omitempty"`
}

// ReflectArgs is the input shape for memory_reflect.
type ReflectArgs struct {
	SessionID string `json:"session_id"`
	Brain     string `json:"brain,omitempty"`
}

// ConsolidateArgs is the input shape for memory_consolidate.
type ConsolidateArgs struct {
	Brain string `json:"brain,omitempty"`
}

// CreateBrainArgs is the input shape for memory_create_brain.
type CreateBrainArgs struct {
	Name       string `json:"name"`
	Slug       string `json:"slug,omitempty"`
	Visibility string `json:"visibility,omitempty"`
}

// ListBrainsArgs is the empty input shape for memory_list_brains.
type ListBrainsArgs struct{}

// Register wires every tool into the supplied server. Tools share a
// closure around client so handlers can dispatch without plumbing it
// through per call.
func Register(server *mcp.Server, client MemoryClient) {
	registerRemember(server, client)
	registerRecall(server, client)
	registerSearch(server, client)
	registerAsk(server, client)
	registerIngestFile(server, client)
	registerIngestURL(server, client)
	registerExtract(server, client)
	registerReflect(server, client)
	registerConsolidate(server, client)
	registerCreateBrain(server, client)
	registerListBrains(server, client)
}

// structuredResult wraps a map payload in the [mcp.CallToolResult]
// expected by the SDK. The JSON rendering of the same payload is
// surfaced as text content so MCP clients that do not parse
// structured content still see the raw data.
func structuredResult(payload map[string]any) (*mcp.CallToolResult, any, error) {
	body, err := json.MarshalIndent(payload, "", "  ")
	if err != nil {
		return nil, nil, fmt.Errorf("memory-mcp: marshalling result: %w", err)
	}
	return &mcp.CallToolResult{
		Content: []mcp.Content{&mcp.TextContent{Text: string(body)}},
		StructuredContent: payload,
	}, payload, nil
}

// progressFromRequest returns a [ProgressEmitter] that maps onto the
// MCP request's progress token. A nil token yields a nil emitter so
// handlers can cheaply skip progress work.
func progressFromRequest(ctx context.Context, req *mcp.CallToolRequest) ProgressEmitter {
	token := req.Params.GetProgressToken()
	if token == nil {
		return nil
	}
	return func(progress float64, message string) {
		params := &mcp.ProgressNotificationParams{
			ProgressToken: token,
			Progress:      progress,
			Message:       message,
		}
		_ = req.Session.NotifyProgress(ctx, params)
	}
}
