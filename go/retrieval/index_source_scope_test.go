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

func TestExactSearchScopeConversations(t *testing.T) {
	t.Parallel()

	for _, in := range []string{"conversations", "conversation"} {
		got, ok := exactSearchScope(in)
		if !ok {
			t.Fatalf("exactSearchScope(%q) = not ok, want ok", in)
		}
		if got != "conversations" {
			t.Fatalf("exactSearchScope(%q) = %q, want conversations", in, got)
		}
	}
}

func TestScopeMatchesFilterConversations(t *testing.T) {
	t.Parallel()

	if !scopeMatchesFilter("conversations", "conversations") {
		t.Fatal("scopeMatchesFilter(conversations, conversations) = false, want true")
	}
	if !scopeMatchesFilter("conversations", "conversation") {
		t.Fatal("scopeMatchesFilter(conversations, conversation) = false, want true")
	}
	if scopeMatchesFilter("wiki", "conversations") {
		t.Fatal("scopeMatchesFilter(wiki, conversations) = true, want false")
	}
}
