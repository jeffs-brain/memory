// SPDX-License-Identifier: Apache-2.0

package tools

import (
	"context"

	"github.com/google/jsonschema-go/jsonschema"
	"github.com/modelcontextprotocol/go-sdk/mcp"
)

func registerRecall(server *mcp.Server, client MemoryClient) {
	schema := &jsonschema.Schema{
		Type: "object",
		Properties: map[string]*jsonschema.Schema{
			"query":      {Type: "string", MinLength: ptrInt(1), MaxLength: ptrInt(4096)},
			"brain":      {Type: "string"},
			"scope":      {Type: "string", Enum: []any{"global", "project", "agent"}},
			"session_id": {Type: "string"},
			"top_k":      {Type: "integer", Minimum: ptrFloat(1), Maximum: ptrFloat(50)},
		},
		Required: []string{"query"},
	}
	mcp.AddTool(server, &mcp.Tool{
		Name:        "memory_recall",
		Description: "Recall memories for a query. Pass session_id to weight recent session context; otherwise uses the dedicated memory-search surface. `scope` selects the memory namespace rather than a generic metadata filter.",
		InputSchema: schema,
	}, func(ctx context.Context, req *mcp.CallToolRequest, args RecallArgs) (*mcp.CallToolResult, any, error) {
		result, err := client.Recall(ctx, args)
		if err != nil {
			return nil, nil, err
		}
		return structuredResult(result)
	})
}
