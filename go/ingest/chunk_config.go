// SPDX-License-Identifier: Apache-2.0
package ingest

import "fmt"

// Strategy selects how the chunker identifies split points within a
// document body. The initial implementation supports recursive and
// markdown; remaining strategies are reserved for future work.
type Strategy string

const (
	// StrategyRecursive splits at separators in priority order, recursing
	// into sub-chunks until each fits within the token budget.
	StrategyRecursive Strategy = "recursive"
	// StrategyMarkdown splits at heading boundaries, falling back to
	// recursive splitting within sections.
	StrategyMarkdown Strategy = "markdown"
	// StrategyCode splits at function/class boundaries (language-aware).
	StrategyCode Strategy = "code"
	// StrategyTable splits at row boundaries preserving header context.
	StrategyTable Strategy = "table"
	// StrategyConversation splits at speaker-turn boundaries.
	StrategyConversation Strategy = "conversation"
)

// validStrategies enumerates all accepted Strategy values.
var validStrategies = map[Strategy]struct{}{
	StrategyRecursive:    {},
	StrategyMarkdown:     {},
	StrategyCode:         {},
	StrategyTable:        {},
	StrategyConversation: {},
}

// DefaultMaxTokens is the spec-mandated ceiling per chunk.
const DefaultMaxTokens = 512

// DefaultOverlapTokens is the spec-mandated overlap between adjacent
// chunks.
const DefaultOverlapTokens = 64

// DefaultMinTokens is the spec-mandated floor below which a chunk is
// merged into its neighbour.
const DefaultMinTokens = 30

// DefaultSeparators is the spec-mandated separator hierarchy for the
// recursive strategy. Tried in order until one produces chunks within
// the token budget.
var DefaultSeparators = []string{"\n\n", "\n", ". ", " ", ""}

// ChunkConfig controls the segmentation strategy for ingested
// documents. Zero values are NOT valid — use [DefaultChunkConfig] to
// obtain the spec defaults.
type ChunkConfig struct {
	// MaxTokens is the target ceiling per chunk.
	MaxTokens int
	// OverlapTokens is the number of trailing tokens from the previous
	// chunk prepended to the next.
	OverlapTokens int
	// MinTokens is the floor below which a chunk is merged into its
	// neighbour.
	MinTokens int
	// Strategy selects how split points are identified.
	Strategy Strategy
	// Separators is the ordered list of split delimiters for the
	// recursive strategy. Ignored for other strategies.
	Separators []string
}

// DefaultChunkConfig returns the spec-mandated defaults defined in
// spec/INGESTION.md.
func DefaultChunkConfig() ChunkConfig {
	seps := make([]string, len(DefaultSeparators))
	copy(seps, DefaultSeparators)
	return ChunkConfig{
		MaxTokens:     DefaultMaxTokens,
		OverlapTokens: DefaultOverlapTokens,
		MinTokens:     DefaultMinTokens,
		Strategy:      StrategyRecursive,
		Separators:    seps,
	}
}

// ValidateChunkConfig checks that cfg satisfies the constraints defined
// in spec/INGESTION.md. Returns a descriptive error on the first
// violated rule.
func ValidateChunkConfig(cfg ChunkConfig) error {
	if cfg.MaxTokens <= 0 {
		return fmt.Errorf("ingest: maxTokens must be > 0, got %d", cfg.MaxTokens)
	}
	if cfg.OverlapTokens < 0 {
		return fmt.Errorf("ingest: overlapTokens must be >= 0, got %d", cfg.OverlapTokens)
	}
	if cfg.OverlapTokens >= cfg.MaxTokens {
		return fmt.Errorf("ingest: overlapTokens (%d) must be < maxTokens (%d)", cfg.OverlapTokens, cfg.MaxTokens)
	}
	if cfg.MinTokens < 0 {
		return fmt.Errorf("ingest: minTokens must be >= 0, got %d", cfg.MinTokens)
	}
	if cfg.MinTokens >= cfg.MaxTokens {
		return fmt.Errorf("ingest: minTokens (%d) must be < maxTokens (%d)", cfg.MinTokens, cfg.MaxTokens)
	}
	if _, ok := validStrategies[cfg.Strategy]; !ok {
		return fmt.Errorf("ingest: unknown strategy %q", cfg.Strategy)
	}
	if cfg.Strategy == StrategyRecursive && len(cfg.Separators) == 0 {
		return fmt.Errorf("ingest: separators must be non-empty when strategy is %q", StrategyRecursive)
	}
	return nil
}

// EstimateTokens returns a coarse token count using the spec formula:
// ceil(len(text) / 4). This matches the TypeScript implementation
// exactly for any given byte sequence.
func EstimateTokens(text string) int {
	return (len(text) + 3) / 4
}
