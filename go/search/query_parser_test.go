// SPDX-License-Identifier: Apache-2.0

package search

import (
	"sort"
	"strings"
	"testing"
)

// TestParseQuery_WithAlias installs a package-level alias map and
// asserts the parser expands a matching bare term into the configured
// alternatives while leaving non-matching terms untouched. The alias
// map is cleared on teardown so the global state does not leak into
// other tests in the package.
func TestParseQuery_WithAlias(t *testing.T) {
	prev := getAliasMap()
	t.Cleanup(func() { SetAliasMap(prev) })

	m := NewAliasMap()
	m.entries["a-ware"] = []string{"royal-aware", "royal-a-ware", "a-ware"}
	m.entries["bosch"] = []string{"bosch", "robert-bosch"}
	SetAliasMap(m)

	tokens := ParseQuery("a-ware production")
	if len(tokens) == 0 {
		t.Fatal("ParseQuery returned no tokens")
	}

	// Collect the surface form of every emitted token. Hyphenated
	// alternatives become phrase tokens so the FTS5 tokenizer
	// (porter+unicode61) still matches the stored documents.
	var got []string
	for _, tok := range tokens {
		switch tok.Kind {
		case TokTerm:
			got = append(got, "term:"+tok.Text)
		case TokPhrase:
			got = append(got, "phrase:"+tok.Text)
		case TokPrefix:
			got = append(got, "prefix:"+tok.Text)
		}
	}
	sort.Strings(got)

	want := []string{
		"phrase:a ware",
		"phrase:royal a ware",
		"phrase:royal aware",
		"term:production",
	}
	sort.Strings(want)

	if len(got) != len(want) {
		t.Fatalf("ParseQuery alias expansion got %d tokens (%v), want %d (%v)", len(got), got, len(want), want)
	}
	for i := range got {
		if got[i] != want[i] {
			t.Errorf("ParseQuery alias expansion token %d = %q, want %q", i, got[i], want[i])
		}
	}

	expr := BuildFTS5Expr(tokens)
	for _, needle := range []string{`"royal aware"`, `"royal a ware"`, `"a ware"`, "production"} {
		if !strings.Contains(expr, needle) {
			t.Errorf("BuildFTS5Expr(%v) = %q, missing %q", tokens, expr, needle)
		}
	}
}

// TestParseQuery_AliasMiss confirms that a token with no alias entry
// is emitted verbatim even when an alias map is installed.
func TestParseQuery_AliasMiss(t *testing.T) {
	prev := getAliasMap()
	t.Cleanup(func() { SetAliasMap(prev) })

	m := NewAliasMap()
	m.entries["a-ware"] = []string{"royal-aware", "a-ware"}
	SetAliasMap(m)

	tokens := ParseQuery("lleverage")
	if len(tokens) != 1 {
		t.Fatalf("ParseQuery(lleverage) = %d tokens, want 1 (%v)", len(tokens), tokens)
	}
	if tokens[0].Text != "lleverage" {
		t.Errorf("ParseQuery(lleverage)[0].Text = %q, want lleverage", tokens[0].Text)
	}
}

// TestParseQuery_AliasCaseInsensitive asserts the parser lowercases
// the token before consulting the alias map, so mixed-case user input
// still triggers the same expansion as the canonical lowercase form.
func TestParseQuery_AliasCaseInsensitive(t *testing.T) {
	prev := getAliasMap()
	t.Cleanup(func() { SetAliasMap(prev) })

	m := NewAliasMap()
	m.entries["bosch"] = []string{"bosch", "robert-bosch"}
	SetAliasMap(m)

	tokens := ParseQuery("BOSCH")
	if len(tokens) != 2 {
		t.Fatalf("ParseQuery(BOSCH) = %d tokens, want 2 (%v)", len(tokens), tokens)
	}
}
