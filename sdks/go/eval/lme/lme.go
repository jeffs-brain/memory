// SPDX-License-Identifier: Apache-2.0

// Package lme is a placeholder for the long-memory-eval harness port.
//
// The upstream runner at jeff/apps/jeff/internal/knowledge/eval/lme is
// heavily coupled to the internal llm package and a full port is deferred
// until the SDK scaffolds its own LLM abstraction.
package lme

import (
	"context"
	"time"
)

// RunConfig holds configuration for an LME benchmark run.
type RunConfig struct {
	DatasetPath string
	SampleSize  int
	Seed        int64
	Concurrency int
	Timeout     time.Duration
}

// RunSummary is the aggregated outcome of a run.
type RunSummary struct {
	Total   int
	Passed  int
	Failed  int
	Errors  int
	Elapsed time.Duration
}

// Runner executes an LME benchmark.
type Runner interface {
	Run(ctx context.Context, cfg RunConfig) (RunSummary, error)
}
