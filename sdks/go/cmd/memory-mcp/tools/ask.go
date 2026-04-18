// SPDX-License-Identifier: Apache-2.0

package tools

import (
	"context"

	"github.com/google/jsonschema-go/jsonschema"
	"github.com/modelcontextprotocol/go-sdk/mcp"
)

func registerAsk(server *mcp.Server, client MemoryClient) {
	schema := &jsonschema.Schema{
		Type: "object",
		Properties: map[string]*jsonschema.Schema{
			"query": {Type: "string", MinLength: ptrInt(1), MaxLength: ptrInt(8192)},
			"brain": {Type: "string"},
			"top_k": {Type: "integer", Minimum: ptrFloat(1), Maximum: ptrFloat(50)},
		},
		Required: []string{"query"},
	}
	// TODO(next-pass): swap to a streaming completion path that emits one
	// notifications/progress frame per answer_delta chunk. The Go SDK supports
	// NotifyProgress but the memory llm.Provider.CompleteStream plumbing is
	// not wired into the hosted mode yet.
	mcp.AddTool(server, &mcp.Tool{
		Name:        "memory_ask",
		Description: "Ask a question grounded in the brain. Streams answer tokens as MCP progress notifications and returns the final answer with citations.",
		InputSchema: schema,
	}, func(ctx context.Context, req *mcp.CallToolRequest, args AskArgs) (*mcp.CallToolResult, any, error) {
		progress := progressFromRequest(ctx, req)
		result, err := client.Ask(ctx, args, progress)
		if err != nil {
			return nil, nil, err
		}
		return structuredResult(result)
	})
}
