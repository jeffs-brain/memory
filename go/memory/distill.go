// SPDX-License-Identifier: Apache-2.0

package memory

import (
	"context"

	"github.com/jeffs-brain/memory/go/brain"
	"github.com/jeffs-brain/memory/go/llm"
	"github.com/jeffs-brain/memory/go/query"
)

// DistilledRecall performs query distillation on the raw user input
// before passing it to the standard Recall path. If distillation is
// disabled (nil distiller) or fails, the raw input is used unchanged.
func (m *Memory) DistilledRecall(
	ctx context.Context,
	provider llm.Provider,
	model string,
	distiller query.Distiller,
	projectPath string,
	userQuery string,
	history []llm.Message,
	surfaced map[brain.Path]bool,
	weights RecallWeights,
) ([]SurfacedMemory, *query.Trace, error) {
	searchQuery := userQuery
	var trace *query.Trace

	if distiller != nil {
		result, err := distiller.Distill(ctx, userQuery, history, query.Options{
			Scope:         "recall",
			CloudProvider: provider,
		})
		trace = &result.Trace
		if err == nil && len(result.Queries) > 0 {
			searchQuery = result.Queries[0].Text
		}
	}

	memories, err := m.Recall(ctx, provider, model, projectPath, searchQuery, surfaced, weights)
	return memories, trace, err
}
