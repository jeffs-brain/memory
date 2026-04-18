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
		{`"oude reimer"`, `"oude reimer"`},
		{"bosch*", "bosch*"},
		{"lleverage AND bosch", "lleverage AND bosch"},
		{"lleverage OR bosch", "lleverage OR bosch"},
		{"lleverage NOT bosch", "lleverage AND NOT bosch"},
		{"what about bosch", "bosch"},
		{"the and or", "the and or"},
		{"(test)", "test"},
		{"relationship between lleverage and bosch", "lleverage OR bosch"},
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
			input: "bosch",
			want:  []Token{{Kind: TokTerm, Text: "bosch"}},
		},
		{
			name:  "stop words stripped",
			input: "what about bosch",
			want:  []Token{{Kind: TokTerm, Text: "bosch"}},
		},
		{
			name:  "all stop words",
			input: "the and or",
			want:  nil,
		},
		{
			name:  "phrase preserved",
			input: `"oude reimer"`,
			want:  []Token{{Kind: TokPhrase, Text: "oude reimer"}},
		},
		{
			name:  "phrase plus term",
			input: `"oude reimer" bosch`,
			want: []Token{
				{Kind: TokPhrase, Text: "oude reimer"},
				{Kind: TokTerm, Text: "bosch"},
			},
		},
		{
			name:  "prefix",
			input: "bosch*",
			want:  []Token{{Kind: TokPrefix, Text: "bosch"}},
		},
		{
			name:  "explicit AND",
			input: "lleverage AND bosch",
			want: []Token{
				{Kind: TokTerm, Text: "lleverage"},
				{Kind: TokTerm, Text: "bosch", Operator: "AND"},
			},
		},
		{
			name:  "explicit OR",
			input: "lleverage OR bosch",
			want: []Token{
				{Kind: TokTerm, Text: "lleverage"},
				{Kind: TokTerm, Text: "bosch", Operator: "OR"},
			},
		},
		{
			name:  "explicit NOT",
			input: "lleverage NOT bosch",
			want: []Token{
				{Kind: TokTerm, Text: "lleverage"},
				{Kind: TokTerm, Text: "bosch", Operator: "NOT"},
			},
		},
		{
			name:  "parentheses stripped",
			input: "(test)",
			want:  []Token{{Kind: TokTerm, Text: "test"}},
		},
		{
			name:  "natural language question",
			input: "relationship between lleverage and bosch",
			want: []Token{
				{Kind: TokTerm, Text: "lleverage"},
				{Kind: TokTerm, Text: "bosch"},
			},
		},
		{
			name:  "short term dropped",
			input: "go patterns",
			want:  []Token{{Kind: TokTerm, Text: "patterns"}},
		},
		{
			name:  "dutch stop words stripped",
			input: "wat is de bosch situatie",
			want: []Token{
				{Kind: TokTerm, Text: "bosch"},
				{Kind: TokTerm, Text: "situatie"},
			},
		},
		{
			name:  "lowercase and is a stop word",
			input: "lleverage and bosch",
			want: []Token{
				{Kind: TokTerm, Text: "lleverage"},
				{Kind: TokTerm, Text: "bosch"},
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
			tokens: []Token{{Kind: TokTerm, Text: "bosch"}},
			want:   "bosch",
		},
		{
			name: "two terms default OR",
			tokens: []Token{
				{Kind: TokTerm, Text: "lleverage"},
				{Kind: TokTerm, Text: "bosch"},
			},
			want: "lleverage OR bosch",
		},
		{
			name: "explicit AND",
			tokens: []Token{
				{Kind: TokTerm, Text: "lleverage"},
				{Kind: TokTerm, Text: "bosch", Operator: "AND"},
			},
			want: "lleverage AND bosch",
		},
		{
			name: "explicit OR",
			tokens: []Token{
				{Kind: TokTerm, Text: "lleverage"},
				{Kind: TokTerm, Text: "bosch", Operator: "OR"},
			},
			want: "lleverage OR bosch",
		},
		{
			name: "NOT rewritten to AND NOT",
			tokens: []Token{
				{Kind: TokTerm, Text: "lleverage"},
				{Kind: TokTerm, Text: "bosch", Operator: "NOT"},
			},
			want: "lleverage AND NOT bosch",
		},
		{
			name: "phrase token",
			tokens: []Token{
				{Kind: TokPhrase, Text: "oude reimer"},
			},
			want: `"oude reimer"`,
		},
		{
			name: "phrase plus term",
			tokens: []Token{
				{Kind: TokPhrase, Text: "oude reimer"},
				{Kind: TokTerm, Text: "bosch"},
			},
			want: `"oude reimer" OR bosch`,
		},
		{
			name: "prefix token",
			tokens: []Token{
				{Kind: TokPrefix, Text: "bosch"},
			},
			want: "bosch*",
		},
		{
			name: "prefix plus term",
			tokens: []Token{
				{Kind: TokPrefix, Text: "bosch"},
				{Kind: TokTerm, Text: "power"},
			},
			want: "bosch* OR power",
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
