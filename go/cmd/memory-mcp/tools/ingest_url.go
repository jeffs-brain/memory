// SPDX-License-Identifier: Apache-2.0

package tools

import (
	"context"

	"github.com/google/jsonschema-go/jsonschema"
	"github.com/modelcontextprotocol/go-sdk/mcp"
)

func registerIngestURL(server *mcp.Server, client MemoryClient) {
	schema := &jsonschema.Schema{
		Type: "object",
		Properties: map[string]*jsonschema.Schema{
			"url":   {Type: "string", Format: "uri"},
			"brain": {Type: "string"},
		},
		Required: []string{"url"},
	}
	mcp.AddTool(server, &mcp.Tool{
		Name:        "memory_ingest_url",
		Description: "Fetch a URL and ingest its contents into the brain. Uses the server-side /ingest/url endpoint when available; otherwise fetches locally and creates a document.",
		InputSchema: schema,
	}, func(ctx context.Context, req *mcp.CallToolRequest, args IngestURLArgs) (*mcp.CallToolResult, any, error) {
		progress := progressFromRequest(ctx, req)
		result, err := client.IngestURL(ctx, args, progress)
		if err != nil {
			return nil, nil, err
		}
		return structuredResult(result)
	})
}
