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

// ChunkStrategy names a built-in chunking approach that a caller can
// request. The registry uses this to override the default content-type
// routing when a specific strategy is desired.
type ChunkStrategy string

const (
	// ChunkStrategyAuto lets the registry pick the chunker from the content type.
	ChunkStrategyAuto ChunkStrategy = ""
	// ChunkStrategyRecursive forces the recursive separator-hierarchy chunker.
	ChunkStrategyRecursive ChunkStrategy = "recursive"
	// ChunkStrategyMarkdown forces the heading-aware markdown chunker.
	ChunkStrategyMarkdown ChunkStrategy = "markdown"
	// ChunkStrategyCode forces the code-aware chunker.
	ChunkStrategyCode ChunkStrategy = "code"
	// ChunkStrategyTabular forces the tabular (CSV/TSV) chunker.
	ChunkStrategyTabular ChunkStrategy = "tabular"
	// ChunkStrategyPageLevel forces the page-level (form-feed) chunker.
	ChunkStrategyPageLevel ChunkStrategy = "page_level"
)

// ChunkConfigOption applies an optional setting to a ChunkConfig.
type ChunkConfigOption func(*ChunkConfig)

// WithStrategy returns an option that sets the chunking strategy on the
// ChunkConfig's Strategy field using the mapping from ChunkStrategy.
func WithStrategy(s ChunkStrategy) ChunkConfigOption {
	return func(c *ChunkConfig) {
		c.Strategy = Strategy(s)
	}
}

// WithSeparators returns an option that overrides the default separator
// hierarchy used by the recursive chunker.
func WithSeparators(seps []string) ChunkConfigOption {
	return func(c *ChunkConfig) { c.Separators = seps }
}

// NewChunkConfig validates and returns a ChunkConfig for the chunker
// registry. Returns an error when maxTokens < minTokens, overlapTokens
// >= maxTokens, or any value is negative. Applies defaults for
// zero/negative values. Optional ChunkConfigOption values configure
// strategy and separators.
func NewChunkConfig(maxTokens, overlapTokens, minTokens int, opts ...ChunkConfigOption) (ChunkConfig, error) {
	if maxTokens <= 0 {
		maxTokens = DefaultMaxTokens
	}
	if overlapTokens < 0 {
		overlapTokens = DefaultOverlapTokens
	}
	if minTokens < 0 {
		minTokens = DefaultMinTokens
	}
	seps := make([]string, len(DefaultSeparators))
	copy(seps, DefaultSeparators)
	cfg := ChunkConfig{
		MaxTokens:     maxTokens,
		OverlapTokens: overlapTokens,
		MinTokens:     minTokens,
		Strategy:      StrategyRecursive,
		Separators:    seps,
	}
	for _, opt := range opts {
		opt(&cfg)
	}
	if err := ValidateChunkConfig(cfg); err != nil {
		return ChunkConfig{}, err
	}
	return cfg, nil
}
