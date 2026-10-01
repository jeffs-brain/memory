// SPDX-License-Identifier: Apache-2.0

package search

import "testing"

// TestSanitiseQuery mirrors the TypeScript SDK's end-to-end
// sanitise pipeline: raw string in, FTS5 MATCH expression out. The
// cases below pin down the backward-compatible jeff behaviours that
// the retrieval layer relies on.
func TestSanitiseQuery(t *testing.T) {
	tests := []struct {
		input    string
		expected string
	}{
		{"hello world", "hello OR world"},
		{"", ""},
		{"   ", ""},
		{`"mill brook"`, `"mill brook"`},
		{"zenco*", "zenco*"},
		{"lleverage AND zenco", "lleverage AND zenco"},
		{"lleverage OR zenco", "lleverage OR zenco"},
		{"lleverage NOT zenco", "lleverage AND NOT zenco"},
		{"what about zenco", "zenco"},
		{"the and or", "the and or"},
		{"(test)", "test"},
		{"relationship between lleverage and zenco", "lleverage OR zenco"},
	}

	for _, tt := range tests {
		got := sanitiseQuery(tt.input)
		if got != tt.expected {
			t.Errorf("sanitiseQuery(%q) = %q, want %q", tt.input, got, tt.expected)
		}
	}
}

// TestParseQuery covers the tokenisation rules: phrases, prefixes,
// explicit boolean operators, and stop word stripping. Unlike the
// golden fixture this is the jeff-flavoured quickcheck that also
// exercises the ported FTS5 compile glue.
func TestParseQuery(t *testing.T) {
	tests := []struct {
		name  string
		input string
		want  []Token
	}{
		{
			name:  "empty",
			input: "",
			want:  nil,
		},
		{
			name:  "single term",
			input: "zenco",
			want:  []Token{{Kind: TokTerm, Text: "zenco"}},
		},
		{
			name:  "stop words stripped",
			input: "what about zenco",
			want:  []Token{{Kind: TokTerm, Text: "zenco"}},
		},
		{
			name:  "all stop words",
			input: "the and or",
			want:  nil,
		},
		{
			name:  "phrase preserved",
			input: `"mill brook"`,
			want:  []Token{{Kind: TokPhrase, Text: "mill brook"}},
		},
		{
			name:  "phrase plus term",
			input: `"mill brook" zenco`,
			want: []Token{
				{Kind: TokPhrase, Text: "mill brook"},
				{Kind: TokTerm, Text: "zenco"},
			},
		},
		{
			name:  "prefix",
			input: "zenco*",
			want:  []Token{{Kind: TokPrefix, Text: "zenco"}},
		},
		{
			name:  "explicit AND",
			input: "lleverage AND zenco",
			want: []Token{
				{Kind: TokTerm, Text: "lleverage"},
				{Kind: TokTerm, Text: "zenco", Operator: "AND"},
			},
		},
		{
			name:  "explicit OR",
			input: "lleverage OR zenco",
			want: []Token{
				{Kind: TokTerm, Text: "lleverage"},
				{Kind: TokTerm, Text: "zenco", Operator: "OR"},
			},
		},
		{
			name:  "explicit NOT",
			input: "lleverage NOT zenco",
			want: []Token{
				{Kind: TokTerm, Text: "lleverage"},
				{Kind: TokTerm, Text: "zenco", Operator: "NOT"},
			},
		},
		{
			name:  "parentheses stripped",
			input: "(test)",
			want:  []Token{{Kind: TokTerm, Text: "test"}},
		},
		{
			name:  "natural language question",
			input: "relationship between lleverage and zenco",
			want: []Token{
				{Kind: TokTerm, Text: "lleverage"},
				{Kind: TokTerm, Text: "zenco"},
			},
		},
		{
			name:  "short term dropped",
			input: "go patterns",
			want:  []Token{{Kind: TokTerm, Text: "patterns"}},
		},
		{
			name:  "dutch stop words stripped",
			input: "wat is de zenco situatie",
			want: []Token{
				{Kind: TokTerm, Text: "zenco"},
				{Kind: TokTerm, Text: "situatie"},
			},
		},
		{
			name:  "lowercase and is a stop word",
			input: "lleverage and zenco",
			want: []Token{
				{Kind: TokTerm, Text: "lleverage"},
				{Kind: TokTerm, Text: "zenco"},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := ParseQuery(tt.input)
			if len(got) != len(tt.want) {
				t.Fatalf("ParseQuery(%q) length = %d, want %d (got %+v)", tt.input, len(got), len(tt.want), got)
			}
			for i := range got {
				if got[i] != tt.want[i] {
					t.Errorf("ParseQuery(%q)[%d] = %+v, want %+v", tt.input, i, got[i], tt.want[i])
				}
			}
		})
	}
}

// TestBuildFTS5Expr covers every operator combination the parser
// can emit. Each case maps a hand-built token slice to the
// expected FTS5 MATCH expression.
func TestBuildFTS5Expr(t *testing.T) {
	tests := []struct {
		name   string
		tokens []Token
		want   string
	}{
		{
			name:   "empty",
			tokens: nil,
			want:   "",
		},
		{
			name:   "single term",
			tokens: []Token{{Kind: TokTerm, Text: "zenco"}},
			want:   "zenco",
		},
		{
			name: "two terms default OR",
			tokens: []Token{
				{Kind: TokTerm, Text: "lleverage"},
				{Kind: TokTerm, Text: "zenco"},
			},
			want: "lleverage OR zenco",
		},
		{
			name: "explicit AND",
			tokens: []Token{
				{Kind: TokTerm, Text: "lleverage"},
				{Kind: TokTerm, Text: "zenco", Operator: "AND"},
			},
			want: "lleverage AND zenco",
		},
		{
			name: "explicit OR",
			tokens: []Token{
				{Kind: TokTerm, Text: "lleverage"},
				{Kind: TokTerm, Text: "zenco", Operator: "OR"},
			},
			want: "lleverage OR zenco",
		},
		{
			name: "NOT rewritten to AND NOT",
			tokens: []Token{
				{Kind: TokTerm, Text: "lleverage"},
				{Kind: TokTerm, Text: "zenco", Operator: "NOT"},
			},
			want: "lleverage AND NOT zenco",
		},
		{
			name: "phrase token",
			tokens: []Token{
				{Kind: TokPhrase, Text: "mill brook"},
			},
			want: `"mill brook"`,
		},
		{
			name: "phrase plus term",
			tokens: []Token{
				{Kind: TokPhrase, Text: "mill brook"},
				{Kind: TokTerm, Text: "zenco"},
			},
			want: `"mill brook" OR zenco`,
		},
		{
			name: "prefix token",
			tokens: []Token{
				{Kind: TokPrefix, Text: "zenco"},
			},
			want: "zenco*",
		},
		{
			name: "prefix plus term",
			tokens: []Token{
				{Kind: TokPrefix, Text: "zenco"},
				{Kind: TokTerm, Text: "power"},
			},
			want: "zenco* OR power",
		},
		{
			name: "three terms default OR",
			tokens: []Token{
				{Kind: TokTerm, Text: "alpha"},
				{Kind: TokTerm, Text: "beta"},
				{Kind: TokTerm, Text: "gamma"},
			},
			want: "alpha OR beta OR gamma",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := BuildFTS5Expr(tt.tokens)
			if got != tt.want {
				t.Errorf("BuildFTS5Expr(%+v) = %q, want %q", tt.tokens, got, tt.want)
			}
		})
	}
}
