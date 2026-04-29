// SPDX-License-Identifier: Apache-2.0

package tools

import (
	"context"

	"github.com/google/jsonschema-go/jsonschema"
	"github.com/modelcontextprotocol/go-sdk/mcp"
)

// registerRemember wires memory_remember into the server.
func registerRemember(server *mcp.Server, client MemoryClient) {
	schema := &jsonschema.Schema{
		Type: "object",
		Properties: map[string]*jsonschema.Schema{
			"content": {Type: "string", MinLength: ptrInt(1), MaxLength: ptrInt(5_000_000), Description: "Markdown body of the new memory."},
			"title":   {Type: "string", MinLength: ptrInt(1), MaxLength: ptrInt(512)},
			"brain":   {Type: "string"},
			"tags":    {Type: "array", Items: &jsonschema.Schema{Type: "string", MinLength: ptrInt(1), MaxLength: ptrInt(64)}, MaxItems: ptrInt(64)},
			"path":    {Type: "string", MinLength: ptrInt(1), MaxLength: ptrInt(1024)},
		},
		Required: []string{"content"},
	}
	mcp.AddTool(server, &mcp.Tool{
		Name:        "memory_remember",
		Description: "Store a new memory (markdown document) in the brain. Returns the created document id and path.",
		InputSchema: schema,
	}, func(ctx context.Context, req *mcp.CallToolRequest, args RememberArgs) (*mcp.CallToolResult, any, error) {
		result, err := client.Remember(ctx, args)
		if err != nil {
			return nil, nil, err
		}
		return structuredResult(result)
	})
}
