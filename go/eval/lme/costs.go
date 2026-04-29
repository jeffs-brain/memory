// SPDX-License-Identifier: Apache-2.0

package lme

import (
	"sync"

	"github.com/jeffs-brain/memory/go/llm"
)

// Pricing holds per-million-token input and output prices in USD.
type Pricing struct {
	InputPerMTok  float64 `json:"input_per_mtok"`
	OutputPerMTok float64 `json:"output_per_mtok"`
}

// DefaultPricing is the hard-coded price table for models the SDK
// routinely invokes. Prices are USD per million tokens and reflect
// public list prices.
var DefaultPricing = map[string]Pricing{
	"gpt-4o":            {InputPerMTok: 2.50, OutputPerMTok: 10.00},
	"gpt-4o-2024-08-06": {InputPerMTok: 2.50, OutputPerMTok: 10.00},
	"gpt-4o-mini":       {InputPerMTok: 0.15, OutputPerMTok: 0.60},
	"claude-sonnet-4":   {InputPerMTok: 3.00, OutputPerMTok: 15.00},
	"claude-sonnet-4-6": {InputPerMTok: 3.00, OutputPerMTok: 15.00},
	"claude-sonnet-4-7": {InputPerMTok: 3.00, OutputPerMTok: 15.00},
	"claude-opus-4":     {InputPerMTok: 15.00, OutputPerMTok: 75.00},
	"claude-opus-4-6":   {InputPerMTok: 15.00, OutputPerMTok: 75.00},
	"claude-opus-4-7":   {InputPerMTok: 15.00, OutputPerMTok: 75.00},
	"claude-haiku-4-5":  {InputPerMTok: 1.00, OutputPerMTok: 5.00},
	"gemma-4-31B-it":    {InputPerMTok: 0.0, OutputPerMTok: 0.0},
	"bge-m3":            {InputPerMTok: 0.0, OutputPerMTok: 0.0},
}

// Usage is the token-level accounting shape the LME harness threads
// through its provider calls. Mirrors the fields the SDK llm package
// exposes on [llm.CompleteResponse] plus cache buckets for providers
// that break them out. Kept local so score_judge and the cost
// accumulator can share a single shape.
type Usage struct {
	InputTokens  int
	OutputTokens int
	CacheRead    int
	CacheCreate  int
}

// usageFromResponse lifts an [llm.CompleteResponse] into the local
// Usage shape. The SDK provider interface only reports TokensIn /
// TokensOut so cache buckets stay zero; providers that break them out
// can populate the fields at the call site.
func usageFromResponse(resp llm.CompleteResponse) Usage {
	return Usage{InputTokens: resp.TokensIn, OutputTokens: resp.TokensOut}
}

// EstimateUSD returns the approximate USD cost of a single LLM call
// given a model name and token usage. Unknown models price at zero.
func EstimateUSD(model string, u Usage) float64 {
	price, ok := DefaultPricing[model]
	if !ok {
		return 0
	}
	in := float64(u.InputTokens+u.CacheCreate) / 1_000_000 * price.InputPerMTok
	cache := float64(u.CacheRead) / 1_000_000 * price.InputPerMTok * 0.1
	out := float64(u.OutputTokens) / 1_000_000 * price.OutputPerMTok
	return in + cache + out
}

// CostAccumulator threads token-cost totals through concurrent work
// without re-entrant locking. Amounts are stored as microcents
// (USD * 1e8) so int arithmetic never rounds a sub-cent add to zero.
type CostAccumulator struct {
	ingestMicro int64
	agentMicro  int64
	judgeMicro  int64
	mu          sync.Mutex
}

func (c *CostAccumulator) AddIngest(usd float64) {
	c.mu.Lock()
	c.ingestMicro += usdToMicro(usd)
	c.mu.Unlock()
}

func (c *CostAccumulator) AddAgent(usd float64) {
	c.mu.Lock()
	c.agentMicro += usdToMicro(usd)
	c.mu.Unlock()
}

func (c *CostAccumulator) AddJudge(usd float64) {
	c.mu.Lock()
	c.judgeMicro += usdToMicro(usd)
	c.mu.Unlock()
}

// Snapshot returns a CostAccounting whose TotalUSD is exactly the sum of
// the three buckets, free of float rounding drift.
func (c *CostAccumulator) Snapshot() CostAccounting {
	c.mu.Lock()
	ingest, agent, judge := c.ingestMicro, c.agentMicro, c.judgeMicro
	c.mu.Unlock()

	return CostAccounting{
		IngestUSD: microToUSD(ingest),
		AgentUSD:  microToUSD(agent),
		JudgeUSD:  microToUSD(judge),
		TotalUSD:  microToUSD(ingest + agent + judge),
	}
}

// microScale keeps eight decimal places of precision so sub-cent adds
// do not round to zero.
const microScale = 1e8

func usdToMicro(usd float64) int64 {
	return int64(usd * microScale)
}

func microToUSD(m int64) float64 {
	return float64(m) / microScale
}
