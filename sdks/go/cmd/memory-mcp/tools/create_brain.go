// SPDX-License-Identifier: Apache-2.0

package tools

import (
	"context"

	"github.com/google/jsonschema-go/jsonschema"
	"github.com/modelcontextprotocol/go-sdk/mcp"
)

func registerCreateBrain(server *mcp.Server, client MemoryClient) {
	schema := &jsonschema.Schema{
		Type: "object",
		Properties: map[string]*jsonschema.Schema{
			"name":       {Type: "string", MinLength: ptrInt(1), MaxLength: ptrInt(128)},
			"slug":       {Type: "string", MinLength: ptrInt(1), MaxLength: ptrInt(64), Pattern: "^[a-z0-9][a-z0-9-]*$"},
			"visibility": {Type: "string", Enum: []any{"private", "tenant", "public"}},
		},
		Required: []string{"name"},
	}
	mcp.AddTool(server, &mcp.Tool{
		Name:        "memory_create_brain",
		Description: "Create a new brain. Generates a slug from the name if one is not provided.",
		InputSchema: schema,
	}, func(ctx context.Context, req *mcp.CallToolRequest, args CreateBrainArgs) (*mcp.CallToolResult, any, error) {
		result, err := client.CreateBrain(ctx, args)
		if err != nil {
			return nil, nil, err
		}
		return structuredResult(result)
	})
}
