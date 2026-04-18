// SPDX-License-Identifier: Apache-2.0

package search

import (
	"strings"
	"testing"
)

// TestSanitiseQuery_StripsTrailingPunctuation guards against the
// class of bug where a natural-language question ending in
// punctuation (?, !, .) reaches FTS5 as a malformed MATCH expression
// and silently returns zero hits. sanitiseQuery must scrub trailing
// punctuation so the emitted expression parses.
func TestSanitiseQuery_StripsTrailingPunctuation(t *testing.T) {
	cases := []struct {
		name  string
		query string
	}{
		{"trailing ?", "What degree did I graduate with?"},
		{"trailing !", "Amazing meal!"},
		{"trailing .", "I went home."},
		{"trailing ;", "first;"},
		{"mid-question !", "wait! what did I say"},
		{"multi-clause with ,", "I went there, and then I left"},
		{"trailing !?", "really!?"},
		{"dollar sign", "How much is $200 worth"},
		{"hashtag", "What's on #twitter"},
		{"percent", "Got 50% off"},
		{"at sign", "email is jeff@home.lab"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := sanitiseQuery(tc.query)
			for _, c := range []string{"?", "!", ",", ";"} {
				if strings.Contains(got, c) {
					t.Errorf("sanitiseQuery(%q) leaked punctuation %q in output %q", tc.query, c, got)
				}
			}
			if got == "" {
				t.Errorf("sanitiseQuery(%q) must not blank out, got %q", tc.query, got)
			}
		})
	}
}

// TestFallbackSanitise_StripsTrailingPunctuation verifies the
// stop-word fallback path also strips noisy punctuation.
func TestFallbackSanitise_StripsTrailingPunctuation(t *testing.T) {
	got := fallbackSanitise("? ! , ;")
	// All punctuation stripped; fields are empty; empty string
	// returned.
	if strings.ContainsAny(got, "?!,;") {
		t.Errorf("fallbackSanitise leaked punctuation, got %q", got)
	}
}

// TestSanitiseQuery_NaturalLanguageQuestionEndsInFTS5ParseableExpression
// exercises the real-world question shapes the LME harness emits.
// Every question must produce a MATCH expression that parses as
// FTS5 — no bare token may carry trailing `?`/`!`/`,`/`;`.
func TestSanitiseQuery_NaturalLanguageQuestionEndsInFTS5ParseableExpression(t *testing.T) {
	questions := []string{
		"How much did I earn at the Downtown Farmers Market on my most recent visit?",
		"What is the total amount I spent on gifts for my coworker and brother?",
		"I was thinking back to our previous conversation about the Radiation Amplified zombie, and I was wondering if you remembered what we finally decided to name it?",
	}
	for _, q := range questions {
		got := sanitiseQuery(q)
		if got == "" {
			t.Errorf("sanitised blanked out on %q", q)
			continue
		}
		for _, tok := range strings.Fields(got) {
			if strings.ContainsAny(tok, "?!,;") {
				t.Errorf("bare token %q in sanitised output %q carries banned punctuation (from input %q)", tok, got, q)
			}
		}
	}
}
