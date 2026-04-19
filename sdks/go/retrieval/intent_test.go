// SPDX-License-Identifier: Apache-2.0

package retrieval

import (
	"math"
	"sort"
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
		{"How often do I see my therapist, Dr. Smith?", true},
		{"What time do I wake up on Saturday mornings?", true},
		{"Where do I initially keep my old sneakers?", true},
		{"What speed is my new internet plan?", true},
		{"What percentage of the countryside property's price is the cost of the renovations I plan to do on my current house?", true},
		{"What was the page count of the two novels I finished in January and March?", true},
		{"Which mode of transport did I take most recently, bus or train?", true},
		{"When did I submit my research paper on sentiment analysis?", true},
		{"What specific languages did you recommend for learning back-end programming?", true},
		{"Can you remind me of the specific back-end programming languages you recommended I learn?", true},
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
	got := concreteFactIntentMultiplier("How many morning commutes did I log?", r, text)
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
		Path: "memory/global/user-commute-note.md",
		Text: "The user has been driving their car to work every day since mid-January.",
	}
	text := retrievalResultText(r)
	got := concreteFactIntentMultiplier("How long is my daily commute to work?", r, text)
	if math.Abs(got-0.72) > 1e-9 {
		t.Fatalf("measurementless duration multiplier %v, want 0.72", got)
	}
}

func TestFocusAlignedConcreteFactMultiplier_BoostsExactProbeMatches(t *testing.T) {
	t.Parallel()
	got := focusAlignedConcreteFactMultiplier(
		"I wanted to follow up on our previous conversation about front-end and back-end development. Can you remind me of the specific back-end programming languages you recommended I learn?",
		"Tips: learn a back-end programming language such as Ruby, Python, or PHP.",
	)
	if got != 1.6 {
		t.Fatalf("focus multiplier %v, want 1.6", got)
	}
}

func TestFocusAlignedConcreteFactMultiplier_BoostsStrongPhraseOverlap(t *testing.T) {
	t.Parallel()
	got := focusAlignedConcreteFactMultiplier(
		"How long is my daily commute to work?",
		"My daily commute takes 45 minutes each way.",
	)
	if got != 1.6 {
		t.Fatalf("focus multiplier %v, want 1.6", got)
	}
}

func TestReweight_ExactRecallPrefersFocusedBackEndLanguageNote(t *testing.T) {
	t.Parallel()
	out := reweightSharedMemoryRanking(
		"I wanted to follow up on our previous conversation about front-end and back-end development. Can you remind me of the specific back-end programming languages you recommended I learn?",
		[]RetrievedChunk{
			{
				Path:  "memory/project/eval-lme/back-end-learning-resources.md",
				Score: 1.0,
				Text:  "Recommended back-end resources include NodeSchool, Udacity, Coursera, Flask, Django, Spring, Hibernate, SQL.",
			},
			{
				Path:  "memory/project/eval-lme/study-tips-for-becoming-full-stack.md",
				Score: 0.8,
				Text:  "Learn a back-end programming language, such as Ruby, Python, or PHP.",
			},
		},
	)
	if out[0].Path != "memory/project/eval-lme/study-tips-for-becoming-full-stack.md" {
		t.Fatalf("top path = %q, want focused language note", out[0].Path)
	}
}

func TestReweight_ExactDurationPrefersPhraseAlignedCommuteFact(t *testing.T) {
	t.Parallel()
	out := reweightSharedMemoryRanking(
		"How long is my daily commute to work?",
		[]RetrievedChunk{
			{
				Path:  "memory/global/user_commute_duration.md",
				Score: 1.0,
				Text:  "Typically has a 30-minute train commute; some days the commute is shorter.",
			},
			{
				Path:  "memory/global/user-fact-2023-05-22-listening-audiobooks-during-daily-commute.md",
				Score: 0.4,
				Text:  "I've been listening to audiobooks during my daily commute to work, which takes 45 minutes each way.",
			},
		},
	)
	if out[0].Path != "memory/global/user-fact-2023-05-22-listening-audiobooks-during-daily-commute.md" {
		t.Fatalf("top path = %q, want direct commute fact", out[0].Path)
	}
}

func TestReweight_FirstPersonDurationPrefersRoutineUserFactOverProjectTips(t *testing.T) {
	t.Parallel()
	out := reweightSharedMemoryRanking(
		"How long is my daily morning commute to work?",
		[]RetrievedChunk{
			{
				Path:  "memory/project/eval-lme/morning-commute-tips.md",
				Score: 1.0,
				Text:  "Tips for staying awake during a 30-minute morning commute.",
			},
			{
				Path:  "memory/global/user-morning-commute-duration.md",
				Score: 0.92,
				Text:  "User is often on a train for a 30-minute morning commute. Some days the commute is shorter, around 15-20 minutes.",
			},
			{
				Path:  "memory/global/user-commute-time.md",
				Score: 0.75,
				Text:  "I listen to audiobooks during my daily commute, which takes 45 minutes each way.",
			},
		},
	)
	if out[0].Path != "memory/global/user-commute-time.md" {
		t.Fatalf("top path = %q, want routine commute fact", out[0].Path)
	}
}

func TestReweight_FirstPersonPropertyPrefersDirectGlobalFactOverProjectGuide(t *testing.T) {
	t.Parallel()
	out := reweightSharedMemoryRanking(
		"What speed is my new internet plan?",
		[]RetrievedChunk{
			{
				Path:  "memory/project/eval-lme/internet-plan-guide.md",
				Score: 1.0,
				Text:  "Guide to choosing a fast home internet plan, with tips about 500 Mbps and 1 Gbps options.",
			},
			{
				Path:  "memory/global/user-fact-internet-plan.md",
				Score: 0.55,
				Text:  "I upgraded my internet plan to 500 Mbps.",
			},
		},
	)
	if out[0].Path != "memory/global/user-fact-internet-plan.md" {
		t.Fatalf("top path = %q, want direct global fact", out[0].Path)
	}
}

func TestReweight_PenalisesSupersededConcreteFactNotes(t *testing.T) {
	t.Parallel()
	out := reweightSharedMemoryRanking(
		"What speed is my new internet plan?",
		[]RetrievedChunk{
			{
				Path:     "memory/global/user-fact-old-internet-plan.md",
				Score:    1.0,
				Text:     "I upgraded my internet plan to 300 Mbps.",
				Metadata: map[string]any{"superseded_by": "user-fact-new-internet-plan.md"},
			},
			{
				Path:  "memory/global/user-fact-new-internet-plan.md",
				Score: 0.72,
				Text:  "I upgraded my internet plan to 500 Mbps.",
			},
		},
	)
	if out[0].Path != "memory/global/user-fact-new-internet-plan.md" {
		t.Fatalf("top path = %q, want current fact", out[0].Path)
	}
}

func TestReweight_CompositeTotalsDiversifyAcrossFocuses(t *testing.T) {
	t.Parallel()
	out := reweightSharedMemoryRanking(
		"What is the total amount I spent on the designer handbag and high-end skincare products?",
		[]RetrievedChunk{
			{
				Path:  "memory/global/coach-handbag-800.md",
				Score: 1.0,
				Text:  "User recently treated themself to a Coach handbag which cost $800 and they are really loving the quality.",
			},
			{
				Path:  "memory/global/user-fact-2023-05-28-recently-invested-some-high-end-products.md",
				Score: 0.78,
				Text:  "I've recently invested $500 in some high-end products during the Nordstrom anniversary sale.",
			},
			{
				Path:  "memory/global/user_ebay_handbag_deal.md",
				Score: 0.63,
				Text:  "The user bought a designer handbag on eBay that originally retailed for $1,500 for $200.",
			},
			{
				Path:  "memory/global/user_high-end-moisturizer.md",
				Score: 0.5,
				Text:  "The user recently splurged on a $150 moisturizer and is asking for affordable alternatives to high-end skincare products.",
			},
		},
	)
	if len(out) < 2 {
		t.Fatalf("unexpected output length %d", len(out))
	}
	topTwo := []string{out[0].Path, out[1].Path}
	sort.Strings(topTwo)
	want := []string{
		"memory/global/coach-handbag-800.md",
		"memory/global/user-fact-2023-05-28-recently-invested-some-high-end-products.md",
	}
	sort.Strings(want)
	if topTwo[0] != want[0] || topTwo[1] != want[1] {
		t.Fatalf("top two = %v, want %v", topTwo, want)
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
