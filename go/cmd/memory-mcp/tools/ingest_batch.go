// SPDX-License-Identifier: Apache-2.0

package tools

import (
	"context"
	"fmt"

	"github.com/google/jsonschema-go/jsonschema"
	"github.com/modelcontextprotocol/go-sdk/mcp"
)

const maxBatchSize = 50

// IngestBatchArgs is the input shape for memory_ingest_batch.
type IngestBatchArgs struct {
	Files []IngestBatchFileArg `json:"files"`
	Brain string               `json:"brain,omitempty"`
}

// IngestBatchFileArg describes a single file entry in a batch ingest request.
type IngestBatchFileArg struct {
	Path string `json:"path"`
	As   string `json:"as,omitempty"`
}

func registerIngestBatch(server *mcp.Server, client MemoryClient) {
	fileSchema := &jsonschema.Schema{
		Type: "object",
		Properties: map[string]*jsonschema.Schema{
			"path": {Type: "string", MinLength: ptrInt(1)},
			"as":   {Type: "string", Enum: []any{"markdown", "text", "pdf", "json"}},
		},
		Required: []string{"path"},
	}

	schema := &jsonschema.Schema{
		Type: "object",
		Properties: map[string]*jsonschema.Schema{
			"files": {
				Type:     "array",
				Items:    fileSchema,
				MinItems: ptrInt(1),
				MaxItems: ptrInt(maxBatchSize),
			},
			"brain": {Type: "string"},
		},
		Required: []string{"files"},
	}

	mcp.AddTool(server, &mcp.Tool{
		Name:        "memory_ingest_batch",
		Description: "Ingest up to 50 local files in a single call. Returns per-file results.",
		InputSchema: schema,
	}, func(ctx context.Context, req *mcp.CallToolRequest, args IngestBatchArgs) (*mcp.CallToolResult, any, error) {
		if len(args.Files) == 0 {
			return nil, nil, fmt.Errorf("memory_ingest_batch: files array must contain at least 1 entry")
		}
		if len(args.Files) > maxBatchSize {
			return nil, nil, fmt.Errorf("memory_ingest_batch: files array exceeds maximum of %d entries", maxBatchSize)
		}

		progress := progressFromRequest(ctx, req)
		total := len(args.Files)
		succeeded := 0
		failed := 0
		results := make([]map[string]any, 0, total)

		for i, file := range args.Files {
			ingestArgs := IngestFileArgs{
				Path:  file.Path,
				Brain: args.Brain,
				As:    file.As,
			}

			result, err := client.IngestFile(ctx, ingestArgs, progress)
			if err != nil {
				results = append(results, map[string]any{
					"path":   file.Path,
					"status": "error",
					"error":  err.Error(),
				})
				failed++
			} else {
				entry := map[string]any{
					"path":   file.Path,
					"status": "success",
				}
				if docID, ok := result["document_id"]; ok {
					entry["documentId"] = docID
				}
				if hash, ok := result["hash"]; ok {
					entry["hash"] = hash
				}
				if byteSize, ok := result["byte_size"]; ok {
					entry["bytes"] = byteSize
				}
				results = append(results, entry)
				succeeded++
			}

			if progress != nil {
				progress(float64(i+1), fmt.Sprintf("%d/%d %s", i+1, total, file.Path))
			}
		}

		payload := map[string]any{
			"total":     total,
			"succeeded": succeeded,
			"failed":    failed,
			"results":   results,
		}
		return structuredResult(payload)
	})
}
