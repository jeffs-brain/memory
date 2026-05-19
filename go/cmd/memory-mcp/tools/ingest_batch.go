// SPDX-License-Identifier: Apache-2.0

package tools

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"

	"github.com/google/jsonschema-go/jsonschema"
	"github.com/modelcontextprotocol/go-sdk/mcp"
	"golang.org/x/sync/errgroup"
)

const (
	maxBatchSize   = 50
	maxConcurrency = 5
)

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
		var succeededCount atomic.Int64
		var failedCount atomic.Int64
		var completedCount atomic.Int64
		results := make([]batchFileResult, total)
		var mu sync.Mutex

		g, gctx := errgroup.WithContext(ctx)
		g.SetLimit(maxConcurrency)

		for i, file := range args.Files {
			g.Go(func() error {
				ingestArgs := IngestFileArgs{
					Path:  file.Path,
					Brain: args.Brain,
					As:    file.As,
				}

				result, ingestErr := client.IngestFile(gctx, ingestArgs, progress)
				if ingestErr != nil {
					results[i] = batchFileResult{
						Path:   file.Path,
						Status: "error",
						Error:  ingestErr.Error(),
					}
					failedCount.Add(1)
				} else {
					results[i] = batchFileResult{
						Path:       file.Path,
						Status:     "success",
						DocumentID: result.DocumentID,
						Hash:       result.Hash,
					}
					succeededCount.Add(1)
				}

				done := completedCount.Add(1)
				if progress != nil {
					mu.Lock()
					progress(float64(done), fmt.Sprintf("%d/%d %s", done, total, file.Path))
					mu.Unlock()
				}

				return nil
			})
		}

		_ = g.Wait()

		payload := batchResult{
			Total:     total,
			Succeeded: succeededCount.Load(),
			Failed:    failedCount.Load(),
			Results:   results,
		}
		return structuredResult(payload)
	})
}
