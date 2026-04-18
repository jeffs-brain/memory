// SPDX-License-Identifier: Apache-2.0

package tools

import (
	"context"

	"github.com/google/jsonschema-go/jsonschema"
	"github.com/modelcontextprotocol/go-sdk/mcp"
)

func registerConsolidate(server *mcp.Server, client MemoryClient) {
	schema := &jsonschema.Schema{
		Type: "object",
		Properties: map[string]*jsonschema.Schema{
			"brain": {Type: "string"},
		},
	}
	mcp.AddTool(server, &mcp.Tool{
		Name:        "memory_consolidate",
		Description: "Trigger a consolidation pass on the brain (compile summaries, promote stable notes, prune stale episodic memory).",
		InputSchema: schema,
	}, func(ctx context.Context, req *mcp.CallToolRequest, args ConsolidateArgs) (*mcp.CallToolResult, any, error) {
		progress := progressFromRequest(ctx, req)
		result, err := client.Consolidate(ctx, args, progress)
		if err != nil {
			return nil, nil, err
		}
		return structuredResult(result)
	})
}
