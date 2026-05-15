// SPDX-License-Identifier: Apache-2.0
package ingest

import (
	"testing"
)

func TestDefaultChunkConfig(t *testing.T) {
	cfg := DefaultChunkConfig()
	if cfg.MaxTokens != 512 {
		t.Fatalf("MaxTokens = %d, want 512", cfg.MaxTokens)
	}
	if cfg.OverlapTokens != 64 {
		t.Fatalf("OverlapTokens = %d, want 64", cfg.OverlapTokens)
	}
	if cfg.MinTokens != 30 {
		t.Fatalf("MinTokens = %d, want 30", cfg.MinTokens)
	}
	if cfg.Strategy != StrategyRecursive {
		t.Fatalf("Strategy = %q, want %q", cfg.Strategy, StrategyRecursive)
	}
	if len(cfg.Separators) != 5 {
		t.Fatalf("Separators length = %d, want 5", len(cfg.Separators))
	}
	expectedSeps := []string{"\n\n", "\n", ". ", " ", ""}
	for i, sep := range cfg.Separators {
		if sep != expectedSeps[i] {
			t.Fatalf("Separators[%d] = %q, want %q", i, sep, expectedSeps[i])
		}
	}
}

func TestDefaultChunkConfig_returnsIndependentSlice(t *testing.T) {
	a := DefaultChunkConfig()
	b := DefaultChunkConfig()
	a.Separators[0] = "MUTATED"
	if b.Separators[0] == "MUTATED" {
		t.Fatal("DefaultChunkConfig returns shared separator slice")
	}
}

func TestValidateChunkConfig(t *testing.T) {
	cases := []struct {
		name    string
		cfg     ChunkConfig
		wantErr bool
	}{
		{
			name:    "spec defaults valid",
			cfg:     DefaultChunkConfig(),
			wantErr: false,
		},
		{
			name: "small chunks no overlap valid",
			cfg: ChunkConfig{
				MaxTokens:     64,
				OverlapTokens: 0,
				MinTokens:     10,
				Strategy:      StrategyRecursive,
				Separators:    []string{"\n\n", "\n", ". ", " ", ""},
			},
			wantErr: false,
		},
		{
			name: "markdown strategy valid",
			cfg: ChunkConfig{
				MaxTokens:     1024,
				OverlapTokens: 128,
				MinTokens:     50,
				Strategy:      StrategyMarkdown,
				Separators:    []string{"\n\n", "\n", ". ", " ", ""},
			},
			wantErr: false,
		},
		{
			name: "markdown strategy empty separators valid",
			cfg: ChunkConfig{
				MaxTokens:     512,
				OverlapTokens: 64,
				MinTokens:     30,
				Strategy:      StrategyMarkdown,
				Separators:    []string{},
			},
			wantErr: false,
		},
		{
			name: "zero maxTokens rejected",
			cfg: ChunkConfig{
				MaxTokens:     0,
				OverlapTokens: 64,
				MinTokens:     30,
				Strategy:      StrategyRecursive,
				Separators:    []string{"\n\n"},
			},
			wantErr: true,
		},
		{
			name: "negative maxTokens rejected",
			cfg: ChunkConfig{
				MaxTokens:     -1,
				OverlapTokens: 64,
				MinTokens:     30,
				Strategy:      StrategyRecursive,
				Separators:    []string{"\n\n"},
			},
			wantErr: true,
		},
		{
			name: "overlap equals max rejected",
			cfg: ChunkConfig{
				MaxTokens:     100,
				OverlapTokens: 100,
				MinTokens:     30,
				Strategy:      StrategyRecursive,
				Separators:    []string{"\n\n"},
			},
			wantErr: true,
		},
		{
			name: "overlap exceeds max rejected",
			cfg: ChunkConfig{
				MaxTokens:     100,
				OverlapTokens: 200,
				MinTokens:     30,
				Strategy:      StrategyRecursive,
				Separators:    []string{"\n\n"},
			},
			wantErr: true,
		},
		{
			name: "negative overlap rejected",
			cfg: ChunkConfig{
				MaxTokens:     512,
				OverlapTokens: -1,
				MinTokens:     30,
				Strategy:      StrategyRecursive,
				Separators:    []string{"\n\n"},
			},
			wantErr: true,
		},
		{
			name: "minTokens equals max rejected",
			cfg: ChunkConfig{
				MaxTokens:     512,
				OverlapTokens: 64,
				MinTokens:     512,
				Strategy:      StrategyRecursive,
				Separators:    []string{"\n\n"},
			},
			wantErr: true,
		},
		{
			name: "negative minTokens rejected",
			cfg: ChunkConfig{
				MaxTokens:     512,
				OverlapTokens: 64,
				MinTokens:     -5,
				Strategy:      StrategyRecursive,
				Separators:    []string{"\n\n"},
			},
			wantErr: true,
		},
		{
			name: "unknown strategy rejected",
			cfg: ChunkConfig{
				MaxTokens:     512,
				OverlapTokens: 64,
				MinTokens:     30,
				Strategy:      "unknown",
				Separators:    []string{"\n\n"},
			},
			wantErr: true,
		},
		{
			name: "empty separators with recursive rejected",
			cfg: ChunkConfig{
				MaxTokens:     512,
				OverlapTokens: 64,
				MinTokens:     30,
				Strategy:      StrategyRecursive,
				Separators:    []string{},
			},
			wantErr: true,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := ValidateChunkConfig(tc.cfg)
			if (err != nil) != tc.wantErr {
				t.Fatalf("ValidateChunkConfig() err = %v, wantErr = %v", err, tc.wantErr)
			}
		})
	}
}

func TestEstimateTokens(t *testing.T) {
	cases := []struct {
		name     string
		input    string
		expected int
	}{
		{"empty string", "", 0},
		{"single char", "a", 1},
		{"two chars", "ab", 1},
		{"three chars", "abc", 1},
		{"four chars", "abcd", 1},
		{"five chars", "abcde", 2},
		{"hello world", "hello world", 3},
		{"sentence", "The quick brown fox jumps over the lazy dog.", 11},
		{"spaced letters", "a b c d e f g h", 4},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := EstimateTokens(tc.input)
			if got != tc.expected {
				t.Fatalf("EstimateTokens(%q) = %d, want %d", tc.input, got, tc.expected)
			}
		})
	}
}

func TestEstimateTokens_monotonic(t *testing.T) {
	prev := EstimateTokens("")
	for i := 1; i <= 100; i++ {
		text := make([]byte, i)
		for j := range text {
			text[j] = 'x'
		}
		cur := EstimateTokens(string(text))
		if cur < prev {
			t.Fatalf("non-monotonic at len %d: %d < %d", i, cur, prev)
		}
		prev = cur
	}
}
