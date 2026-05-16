// SPDX-License-Identifier: Apache-2.0

package tools

import (
	"context"
	"os"
	"path/filepath"
	"strings"

	"github.com/google/jsonschema-go/jsonschema"
	"github.com/modelcontextprotocol/go-sdk/mcp"
)

// textFormatExtensions are file extensions whose raw bytes are valid text
// content suitable for extraction. Binary formats (PDF, etc.) need
// normalisation before extraction and should be skipped here.
var textFormatExtensions = map[string]bool{
	".md": true, ".markdown": true, ".txt": true, ".text": true,
	".json": true, ".html": true, ".htm": true, ".csv": true,
	".tsv": true, ".xml": true, ".yaml": true, ".yml": true,
	".toml": true, ".rst": true, ".adoc": true, ".org": true,
	".tex": true,
}

// isTextFormat reports whether the file content is suitable for direct
// extraction. When the explicit "as" format is one of the known text
// formats, or the file extension is a known text extension, extraction
// is safe. Binary formats like PDF require prior normalisation and should
// not be passed raw to the extractor.
func isTextFormat(asFormat, ext string) bool {
	switch asFormat {
	case "markdown", "text", "json":
		return true
	case "pdf":
		return false
	}
	return textFormatExtensions[ext]
}

func registerIngestFile(server *mcp.Server, client MemoryClient) {
	schema := &jsonschema.Schema{
		Type: "object",
		Properties: map[string]*jsonschema.Schema{
			"path":    {Type: "string", MinLength: ptrInt(1)},
			"brain":   {Type: "string"},
			"as":      {Type: "string", Enum: []any{"markdown", "text", "pdf", "json"}},
			"extract": {Type: "boolean"},
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

		if !args.Extract {
			return structuredResult(result)
		}

		// Skip extraction for non-text formats where raw bytes are meaningless.
		ext := strings.ToLower(filepath.Ext(args.Path))
		extraction := &ExtractAfterIngestResult{
			FactsExtracted: 0,
			Memories:       []ExtractedMemory{},
		}
		if isTextFormat(args.As, ext) {
			absPath := args.Path
			if !filepath.IsAbs(absPath) {
				absPath, _ = filepath.Abs(absPath)
			}
			raw, readErr := os.ReadFile(absPath)
			if readErr == nil && len(raw) > 0 {
				extracted, extractErr := client.ExtractAfterIngest(ctx, ExtractAfterIngestArgs{
					Content:        string(raw),
					DocumentSource: args.Path,
					Brain:          args.Brain,
				})
				if extractErr == nil {
					extraction = extracted
				}
			}
		}

		combined := ingestWithExtractionResult{
			Ingest:     result,
			Extraction: extraction,
		}
		return structuredResult(combined)
	})
}
