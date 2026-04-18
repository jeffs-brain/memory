// SPDX-License-Identifier: Apache-2.0

package tools

import (
	"context"

	"github.com/google/jsonschema-go/jsonschema"
	"github.com/modelcontextprotocol/go-sdk/mcp"
)

func registerListBrains(server *mcp.Server, client MemoryClient) {
	schema := &jsonschema.Schema{Type: "object", Properties: map[string]*jsonschema.Schema{}}
	mcp.AddTool(server, &mcp.Tool{
		Name:        "memory_list_brains",
		Description: "List all brains the caller has access to.",
		InputSchema: schema,
	}, func(ctx context.Context, req *mcp.CallToolRequest, _ ListBrainsArgs) (*mcp.CallToolResult, any, error) {
		result, err := client.ListBrains(ctx)
		if err != nil {
			return nil, nil, err
		}
		return structuredResult(result)
	})
}
