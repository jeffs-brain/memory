// SPDX-License-Identifier: Apache-2.0

package tools

import (
	"context"

	"github.com/google/jsonschema-go/jsonschema"
	"github.com/modelcontextprotocol/go-sdk/mcp"
)

func registerExtract(server *mcp.Server, client MemoryClient) {
	messageSchema := &jsonschema.Schema{
		Type: "object",
		Properties: map[string]*jsonschema.Schema{
			"role":    {Type: "string", Enum: []any{"system", "user", "assistant", "tool"}},
			"content": {Type: "string", MinLength: ptrInt(1)},
		},
		Required: []string{"role", "content"},
	}
	schema := &jsonschema.Schema{
		Type: "object",
		Properties: map[string]*jsonschema.Schema{
			"messages":   {Type: "array", Items: messageSchema, MinItems: ptrInt(1), MaxItems: ptrInt(500)},
			"brain":      {Type: "string"},
			"actor_id":   {Type: "string"},
			"session_id": {Type: "string"},
		},
		Required: []string{"messages"},
	}
	mcp.AddTool(server, &mcp.Tool{
		Name:        "memory_extract",
		Description: "Submit a conversation transcript so the server can asynchronously extract memorable facts. If session_id is provided the messages are appended to that session; otherwise a transcript document is created.",
		InputSchema: schema,
	}, func(ctx context.Context, req *mcp.CallToolRequest, args ExtractArgs) (*mcp.CallToolResult, any, error) {
		progress := progressFromRequest(ctx, req)
		result, err := client.Extract(ctx, args, progress)
		if err != nil {
			return nil, nil, err
		}
		return structuredResult(result)
	})
}
