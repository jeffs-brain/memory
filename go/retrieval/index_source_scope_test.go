// SPDX-License-Identifier: Apache-2.0

package retrieval

import "testing"

func TestExactSearchScopeRawLME(t *testing.T) {
	t.Parallel()

	got, ok := exactSearchScope("raw_lme")
	if !ok {
		t.Fatal("exactSearchScope(raw_lme) = not ok, want ok")
	}
	if got != "raw_lme" {
		t.Fatalf("exactSearchScope(raw_lme) = %q, want raw_lme", got)
	}
}

func TestScopeMatchesFilterRawMatchesRawLME(t *testing.T) {
	t.Parallel()

	if !scopeMatchesFilter("raw_lme", "raw") {
		t.Fatal("scopeMatchesFilter(raw_lme, raw) = false, want true")
	}
	if !scopeMatchesFilter("raw_lme", "raw_lme") {
		t.Fatal("scopeMatchesFilter(raw_lme, raw_lme) = false, want true")
	}
}
