// SPDX-License-Identifier: Apache-2.0

package retrieval

import (
	"math"
	"testing"
)

func TestDetectRetrievalIntent_Preference(t *testing.T) {
	t.Parallel()
	cases := []struct {
		query string
		want  bool
	}{
		{"recommend a coffee shop", true},
		{"Can you suggest a book?", true},
		{"what should i read next", true},
		{"which should I buy, the blue or the red?", true},
		{"tip for next release", true},
		{"any ideas for dinner tonight", true},
		{"advice on cabling", true},
		{"how many invoices did we send", false},
		{"hola amigo", false},
	}
	for _, tc := range cases {
		got := detectRetrievalIntent(tc.query).preferenceQuery
		if got != tc.want {
			t.Errorf("query %q: preferenceQuery=%v, want %v", tc.query, got, tc.want)
		}
	}
}

func TestDetectRetrievalIntent_ConcreteFact(t *testing.T) {
	t.Parallel()
	cases := []struct {
		query string
		want  bool
	}{
		{"how many invoices did we process", true},
		{"count the line items", true},
		{"in total how much was spent", true},
		{"list the clients", true},
		{"what are all the projects", true},
		{"did I pick up milk", true},
		{"have I finished the report", true},
		{"was I booked for dinner", true},
		{"were I the one who ordered", true},
		{"did i travelled to bosch yesterday", true},
		{"How long is my daily commute to work?", true},
		{"What specific languages did you recommend for learning back-end programming?", true},
		{"recommend a flat white", false},
		{"non english text abc xyz", false},
	}
	for _, tc := range cases {
		got := detectRetrievalIntent(tc.query).concreteFactQuery
		if got != tc.want {
			t.Errorf("query %q: concreteFactQuery=%v, want %v", tc.query, got, tc.want)
		}
	}
}

func TestDetectRetrievalIntent_BothCompose(t *testing.T) {
	t.Parallel()
	intent := detectRetrievalIntent("recommend how many to buy")
	if !intent.preferenceQuery {
		t.Fatalf("preference missing")
	}
	if !intent.concreteFactQuery {
		t.Fatalf("concrete-fact missing")
	}
	if intent.label() != "preference+concrete-fact" {
		t.Fatalf("label %q", intent.label())
	}
}

func TestPreferenceMultiplier_GlobalUserPreferenceWins(t *testing.T) {
	t.Parallel()
	r := RetrievedChunk{
		Path:  "memory/global/user-preference-coffee.md",
		Title: "Coffee",
		Text:  "I prefer oat milk",
	}
	text := retrievalResultText(r)
	got := preferenceIntentMultiplier(r, text)
	if got != 2.35 {
		t.Fatalf("user-preference multiplier %v, want 2.35", got)
	}
}

func TestPreferenceMultiplier_GlobalPreferenceNote(t *testing.T) {
	t.Parallel()
	r := RetrievedChunk{
		Path:  "memory/global/notes.md",
		Title: "Notes",
		Text:  "I really love flat whites",
	}
	text := retrievalResultText(r)
	got := preferenceIntentMultiplier(r, text)
	if got != 2.1 {
		t.Fatalf("global preference multiplier %v, want 2.1", got)
	}
}

func TestPreferenceMultiplier_GenericNonGlobal(t *testing.T) {
	t.Parallel()
	r := RetrievedChunk{
		Path: "wiki/guides/tips.md",
		Text: "here are some tips for improving throughput",
	}
	text := retrievalResultText(r)
	got := preferenceIntentMultiplier(r, text)
	if got != 0.82 {
		t.Fatalf("generic non-global multiplier %v, want 0.82", got)
	}
}

func TestPreferenceMultiplier_Rollup(t *testing.T) {
	t.Parallel()
	r := RetrievedChunk{
		Path: "memory/global/roll-up.md",
		Text: "overview summary totalling everything",
	}
	text := retrievalResultText(r)
	got := preferenceIntentMultiplier(r, text)
	if got != 0.9 {
		t.Fatalf("rollup multiplier %v, want 0.9", got)
	}
}

func TestConcreteFactMultiplier_UserFactPath(t *testing.T) {
	t.Parallel()
	r := RetrievedChunk{
		Path: "memory/global/user-fact-birthday.md",
		Text: "no date tag",
	}
	text := retrievalResultText(r)
	got := concreteFactIntentMultiplier("What specific fact is this?", r, text)
	if got != 2.2 {
		t.Fatalf("user-fact multiplier %v, want 2.2", got)
	}
}

func TestConcreteFactMultiplier_QuestionLikeUserFactPenalty(t *testing.T) {
	t.Parallel()
	r := RetrievedChunk{
		Path: "memory/global/user-fact-commute-question.md",
		Text: "What are some tips for staying awake during morning commutes?",
	}
	text := retrievalResultText(r)
	got := concreteFactIntentMultiplier("How long is my daily commute to work?", r, text)
	if math.Abs(got-0.99) > 1e-9 {
		t.Fatalf("question-like user-fact multiplier %v, want 0.99", got)
	}
}

func TestConcreteFactMultiplier_RollupPenalty(t *testing.T) {
	t.Parallel()
	r := RetrievedChunk{
		Path: "wiki/recap.md",
		Text: "summary recap overview totalling monthly figures",
	}
	text := retrievalResultText(r)
	got := concreteFactIntentMultiplier("What is the total amount I spent?", r, text)
	// Neither user-fact path nor atomic event with no date tag
	// guarded by !isRollUp -> concrete-fact false; isRollUp true.
	if got != 0.45 {
		t.Fatalf("rollup multiplier %v, want 0.45", got)
	}
}

func TestConcreteFactMultiplier_ComposedWithGenericPenalty(t *testing.T) {
	t.Parallel()
	r := RetrievedChunk{
		Path: "wiki/tips/misc.md",
		Text: "general guide and tips",
	}
	text := retrievalResultText(r)
	got := concreteFactIntentMultiplier("What is the total amount I spent?", r, text)
	if got != 0.75 {
		t.Fatalf("generic non-global multiplier %v, want 0.75", got)
	}
}

func TestConcreteFactMultiplier_BoostsExplicitDateForActionDateQuery(t *testing.T) {
	t.Parallel()
	r := RetrievedChunk{
		Path: "memory/global/user-fact-acl-date.md",
		Text: "I'm reviewing for ACL, and their submission date was February 1st.",
	}
	text := retrievalResultText(r)
	got := concreteFactIntentMultiplier("When did I submit my research paper on sentiment analysis?", r, text)
	if math.Abs(got-3.19) > 1e-9 {
		t.Fatalf("explicit-date multiplier %v, want 3.19", got)
	}
}

func TestConcreteFactMultiplier_PenalisesMeasurementlessDurationNotes(t *testing.T) {
	t.Parallel()
	r := RetrievedChunk{
		Path: "memory/global/user-commute-summary.md",
		Text: "The user has been driving their car to work every day since mid-January.",
	}
	text := retrievalResultText(r)
	got := concreteFactIntentMultiplier("How long is my daily commute to work?", r, text)
	if math.Abs(got-0.72) > 1e-9 {
		t.Fatalf("measurementless duration multiplier %v, want 0.72", got)
	}
}

func TestReweight_TieBreakByOriginalRank(t *testing.T) {
	t.Parallel()
	results := []RetrievedChunk{
		{Path: "a.md", Score: 1.0, Title: "A"},
		{Path: "b.md", Score: 1.0, Title: "B"},
	}
	// Preference intent that applies no explicit multiplier (no
	// matching text/path) leaves scores intact; the sort must then
	// preserve input order.
	out := reweightSharedMemoryRanking("recommend a thing", results)
	if out[0].Path != "a.md" || out[1].Path != "b.md" {
		t.Fatalf("tie-break broken: %+v", out)
	}
}

func TestReweight_NoIntent_ReturnsInputs(t *testing.T) {
	t.Parallel()
	in := []RetrievedChunk{{Path: "a.md", Score: 0.1}}
	out := reweightSharedMemoryRanking("regular fact lookup", in)
	if len(out) != 1 || out[0].Score != 0.1 {
		t.Fatalf("reweight perturbed no-intent input: %+v", out)
	}
}
