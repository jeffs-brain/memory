// SPDX-License-Identifier: Apache-2.0

package tools

import (
	"context"

	"github.com/google/jsonschema-go/jsonschema"
	"github.com/modelcontextprotocol/go-sdk/mcp"
)

func registerReflect(server *mcp.Server, client MemoryClient) {
	schema := &jsonschema.Schema{
		Type: "object",
		Properties: map[string]*jsonschema.Schema{
			"session_id": {Type: "string", MinLength: ptrInt(1)},
			"brain":      {Type: "string"},
		},
		Required: []string{"session_id"},
	}
	mcp.AddTool(server, &mcp.Tool{
		Name:        "memory_reflect",
		Description: "Close a session and trigger server-side reflection over its messages.",
		InputSchema: schema,
	}, func(ctx context.Context, req *mcp.CallToolRequest, args ReflectArgs) (*mcp.CallToolResult, any, error) {
		progress := progressFromRequest(ctx, req)
		result, err := client.Reflect(ctx, args, progress)
		if err != nil {
			return nil, nil, err
		}
		return structuredResult(result)
	})
}
