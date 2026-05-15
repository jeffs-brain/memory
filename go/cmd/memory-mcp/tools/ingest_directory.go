// SPDX-License-Identifier: Apache-2.0

package tools

import (
	"context"
	"fmt"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/google/jsonschema-go/jsonschema"
	"github.com/google/uuid"
	"github.com/jeffs-brain/memory/go/ingest"
	"github.com/jeffs-brain/memory/go/ingest/trigger"
	"github.com/modelcontextprotocol/go-sdk/mcp"
	"golang.org/x/sync/errgroup"
)

const (
	maxDirectoryFiles       = 500
	maxDirectoryConcurrency = 5
)

// IngestDirectoryArgs is the input shape for memory_ingest_directory.
type IngestDirectoryArgs struct {
	Directory string `json:"directory"`
	Glob      string `json:"glob,omitempty"`
	Brain     string `json:"brain,omitempty"`
	Recursive *bool  `json:"recursive,omitempty"`
	MaxFiles  int    `json:"maxFiles,omitempty"`
}

// DirectoryIngestOpts carries optional dependencies for directory ingest.
// When TriggerBus is non-nil, directory ingest operates in async mode:
// files are dispatched to the bus and the tool returns immediately.
type DirectoryIngestOpts struct {
	TriggerBus trigger.Bus
}

func registerIngestDirectory(server *mcp.Server, client MemoryClient) {
	registerIngestDirectoryWithOpts(server, client, DirectoryIngestOpts{})
}

func registerIngestDirectoryWithOpts(server *mcp.Server, client MemoryClient, opts DirectoryIngestOpts) {
	schema := &jsonschema.Schema{
		Type: "object",
		Properties: map[string]*jsonschema.Schema{
			"directory": {Type: "string", MinLength: ptrInt(1)},
			"glob":      {Type: "string"},
			"brain":     {Type: "string"},
			"recursive": {Type: "boolean"},
			"maxFiles":  {Type: "integer", Minimum: ptrFloat(1), Maximum: ptrFloat(float64(maxDirectoryFiles))},
		},
		Required: []string{"directory"},
	}

	mcp.AddTool(server, &mcp.Tool{
		Name:        "memory_ingest_directory",
		Description: "Ingest files from a directory. Walks recursively, respects .gitignore, and supports glob filtering. Returns a jobGroupId for async tracking when an event bus is configured.",
		InputSchema: schema,
	}, func(ctx context.Context, req *mcp.CallToolRequest, args IngestDirectoryArgs) (*mcp.CallToolResult, any, error) {
		// Sanitise directory path to prevent traversal.
		cleanedDir := filepath.Clean(args.Directory)
		if !filepath.IsAbs(cleanedDir) {
			return nil, nil, fmt.Errorf("memory_ingest_directory: directory must be an absolute path")
		}
		if strings.Contains(cleanedDir, "..") {
			return nil, nil, fmt.Errorf("memory_ingest_directory: directory path must not contain '..'")
		}

		recursive := true
		if args.Recursive != nil {
			recursive = *args.Recursive
		}
		maxFiles := 100
		if args.MaxFiles > 0 {
			if args.MaxFiles > maxDirectoryFiles {
				return nil, nil, fmt.Errorf("memory_ingest_directory: maxFiles exceeds maximum of %d", maxDirectoryFiles)
			}
			maxFiles = args.MaxFiles
		}

		enumerated, skipped, err := ingest.EnumerateFiles(ctx, ingest.EnumerateOptions{
			Directory: cleanedDir,
			Glob:      args.Glob,
			Recursive: recursive,
			MaxFiles:  maxFiles,
		})
		if err != nil {
			return nil, nil, fmt.Errorf("memory_ingest_directory: enumerate: %w", err)
		}

		jobGroupId := uuid.New().String()
		total := len(enumerated)

		brainId := args.Brain
		if brainId == "" {
			brainId = "default"
		}

		// Async mode: publish events to the trigger bus and return immediately.
		if opts.TriggerBus != nil {
			for _, file := range enumerated {
				evt := trigger.IngestTriggerEvent{
					ID:      uuid.New().String(),
					BrainID: brainId,
					Source:  trigger.SourceEventBus,
					Payload: trigger.TriggerPayload{
						Kind: trigger.PayloadFile,
						Path: file.Path,
					},
					Metadata:  map[string]any{"jobGroupId": jobGroupId},
					Timestamp: time.Now(),
				}
				if pubErr := opts.TriggerBus.Publish(evt); pubErr != nil {
					// Log but continue — partial dispatch is better than none.
					skipped = append(skipped, fmt.Sprintf("%s: bus publish failed: %s", file.Path, pubErr.Error()))
				}
			}

			payload := map[string]any{
				"jobGroupId":     jobGroupId,
				"filesQueued":    total,
				"filesSkipped":   len(skipped),
				"skippedReasons": skipped,
				"async":          true,
			}
			return structuredResult(payload)
		}

		// Sync fallback: process files directly.
		progress := progressFromRequest(ctx, req)
		var succeededCount atomic.Int64
		var failedCount atomic.Int64
		var completedCount atomic.Int64
		results := make([]map[string]any, total)
		var mu sync.Mutex

		g, gctx := errgroup.WithContext(ctx)
		g.SetLimit(maxDirectoryConcurrency)

		for i, file := range enumerated {
			g.Go(func() error {
				ingestArgs := IngestFileArgs{
					Path:  file.Path,
					Brain: args.Brain,
				}

				result, ingestErr := client.IngestFile(gctx, ingestArgs, progress)
				if ingestErr != nil {
					results[i] = map[string]any{
						"path":   file.Path,
						"status": "error",
						"error":  ingestErr.Error(),
					}
					failedCount.Add(1)
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
					results[i] = entry
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

		payload := map[string]any{
			"jobGroupId":     jobGroupId,
			"filesQueued":    total,
			"filesSkipped":   len(skipped),
			"skippedReasons": skipped,
			"async":          false,
			"total":          total,
			"succeeded":      succeededCount.Load(),
			"failed":         failedCount.Load(),
			"skipped":        len(skipped),
			"results":        results,
		}
		return structuredResult(payload)
	})
}
