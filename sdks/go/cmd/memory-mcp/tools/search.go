// SPDX-License-Identifier: Apache-2.0

package tools

import (
	"context"

	"github.com/google/jsonschema-go/jsonschema"
	"github.com/modelcontextprotocol/go-sdk/mcp"
)

func registerSearch(server *mcp.Server, client MemoryClient) {
	schema := &jsonschema.Schema{
		Type: "object",
		Properties: map[string]*jsonschema.Schema{
			"query": {Type: "string", MinLength: ptrInt(1), MaxLength: ptrInt(4096)},
			"brain": {Type: "string"},
			"top_k": {Type: "integer", Minimum: ptrFloat(1), Maximum: ptrFloat(100)},
			"scope": {Type: "string", Enum: []any{"all", "global", "project", "agent"}},
			"sort":  {Type: "string", Enum: []any{"relevance", "recency", "relevance_then_recency"}},
		},
		Required: []string{"query"},
	}
	mcp.AddTool(server, &mcp.Tool{
		Name:        "memory_search",
		Description: "Search memory notes in a brain and return matching note content with citations. `scope` selects the memory namespace, and `sort` controls whether relevance or recency wins.",
		InputSchema: schema,
	}, func(ctx context.Context, req *mcp.CallToolRequest, args SearchArgs) (*mcp.CallToolResult, any, error) {
		result, err := client.Search(ctx, args)
		if err != nil {
			return nil, nil, err
		}
		return structuredResult(result)
	})
}
