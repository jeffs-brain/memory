// SPDX-License-Identifier: Apache-2.0

package tools

import (
	"context"
	"fmt"

	"github.com/google/jsonschema-go/jsonschema"
	"github.com/jeffs-brain/memory/go/ingest"
	"github.com/modelcontextprotocol/go-sdk/mcp"
)

const maxDirectoryFiles = 500

// IngestDirectoryArgs is the input shape for memory_ingest_directory.
type IngestDirectoryArgs struct {
	Directory string `json:"directory"`
	Glob      string `json:"glob,omitempty"`
	Brain     string `json:"brain,omitempty"`
	Recursive *bool  `json:"recursive,omitempty"`
	MaxFiles  int    `json:"maxFiles,omitempty"`
}

func registerIngestDirectory(server *mcp.Server, client MemoryClient) {
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
		Description: "Ingest files from a directory. Walks recursively, respects .gitignore, and supports glob filtering.",
		InputSchema: schema,
	}, func(ctx context.Context, req *mcp.CallToolRequest, args IngestDirectoryArgs) (*mcp.CallToolResult, any, error) {
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
			Directory: args.Directory,
			Glob:      args.Glob,
			Recursive: recursive,
			MaxFiles:  maxFiles,
		})
		if err != nil {
			return nil, nil, fmt.Errorf("memory_ingest_directory: enumerate: %w", err)
		}

		progress := progressFromRequest(ctx, req)
		total := len(enumerated)
		succeeded := 0
		failed := 0
		results := make([]map[string]any, 0, total)

		for i, file := range enumerated {
			ingestArgs := IngestFileArgs{
				Path:  file.Path,
				Brain: args.Brain,
			}

			result, ingestErr := client.IngestFile(ctx, ingestArgs, progress)
			if ingestErr != nil {
				results = append(results, map[string]any{
					"path":   file.Path,
					"status": "error",
					"error":  ingestErr.Error(),
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
			"total":          total,
			"succeeded":      succeeded,
			"failed":         failed,
			"skipped":        len(skipped),
			"skippedReasons": skipped,
			"results":        results,
		}
		return structuredResult(payload)
	})
}
