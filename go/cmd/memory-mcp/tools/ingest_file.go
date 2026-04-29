// SPDX-License-Identifier: Apache-2.0

package tools

import (
	"context"

	"github.com/google/jsonschema-go/jsonschema"
	"github.com/modelcontextprotocol/go-sdk/mcp"
)

func registerIngestFile(server *mcp.Server, client MemoryClient) {
	schema := &jsonschema.Schema{
		Type: "object",
		Properties: map[string]*jsonschema.Schema{
			"path":  {Type: "string", MinLength: ptrInt(1)},
			"brain": {Type: "string"},
			"as":    {Type: "string", Enum: []any{"markdown", "text", "pdf", "json"}},
		},
		Required: []string{"path"},
	}
	mcp.AddTool(server, &mcp.Tool{
		Name:        "memory_ingest_file",
		Description: "Ingest a local file (<= 25 MB) into the brain. Returns the ingest result.",
		InputSchema: schema,
	}, func(ctx context.Context, req *mcp.CallToolRequest, args IngestFileArgs) (*mcp.CallToolResult, any, error) {
		progress := progressFromRequest(ctx, req)
		result, err := client.IngestFile(ctx, args, progress)
		if err != nil {
			return nil, nil, err
		}
		return structuredResult(result)
	})
}
