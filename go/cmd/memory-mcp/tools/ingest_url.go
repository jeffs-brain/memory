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
			"url":     {Type: "string", Format: "uri"},
			"brain":   {Type: "string"},
			"extract": {Type: "boolean"},
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

		if !args.Extract {
			// Strip internal field before returning to the caller.
			delete(result, "_document_content")
			return structuredResult(result)
		}

		// Read the stored content from the ingest result (populated by
		// the local client from the brain store). No URL re-fetch needed.
		content, _ := result["_document_content"].(string)
		delete(result, "_document_content")

		extractionResult := map[string]any{
			"factsExtracted": 0,
			"memories":       []any{},
		}
		if content != "" {
			extraction, extractErr := client.ExtractAfterIngest(ctx, ExtractAfterIngestArgs{
				Content:        content,
				DocumentSource: args.URL,
				Brain:          args.Brain,
			})
			if extractErr == nil {
				extractionResult = extraction
			}
		}

		combined := map[string]any{
			"ingest":     result,
			"extraction": extractionResult,
		}
		return structuredResult(combined)
	})
}
