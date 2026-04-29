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
	got := preferenceIntentMultiplier("recommend coffee", r, text)
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
	got := preferenceIntentMultiplier("recommend coffee", r, text)
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
	got := preferenceIntentMultiplier("recommend throughput", r, text)
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
	got := preferenceIntentMultiplier("recommend summary", r, text)
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
	got := concreteFactIntentMultiplier("How many morning commutes did I log?", r, text)
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
	got := concreteFactIntentMultiplier("How many morning commutes did I log?", r, text)
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

func TestConcreteFactMultiplier_BoostsWordNumberDurations(t *testing.T) {
	t.Parallel()
	r := RetrievedChunk{
		Path: "memory/global/user-japan-trip.md",
		Text: "The user spent two weeks travelling solo around Japan.",
	}
	text := retrievalResultText(r)
	got := concreteFactIntentMultiplier("How long was I in Japan for?", r, text)
	if got <= 1.0 {
		t.Fatalf("word-number duration multiplier %v, want boost", got)
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

func TestFocusAlignedConcreteFactMultiplier_BoostsCoverageFacetOverlap(t *testing.T) {
	t.Parallel()
	got := focusAlignedConcreteFactMultiplier(
		"How many attendee questions came from the Atlas Webinar and Beacon Podcast episode?",
		"The Beacon Podcast episode received 17 attendee questions.",
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

func TestReweight_FirstPersonCountCanPromotePersonalProjectFact(t *testing.T) {
	t.Parallel()
	out := reweightSharedMemoryRanking(
		"How many bikes do I currently own?",
		[]RetrievedChunk{
			{
				Path:  "memory/global/user-current-keyboard.md",
				Score: 1.0,
				Text:  "The user is currently using a Logitech keyboard.",
			},
			{
				Path:  "memory/project/eval-lme/user-road-trip-bike-plan.md",
				Score: 0.62,
				Text:  "[Date: 2023-05-25 Thursday May 2023] The user will bring four bikes: a road bike, mountain bike, commuter bike, and a new hybrid bike.",
			},
		},
	)
	if out[0].Path != "memory/project/eval-lme/user-road-trip-bike-plan.md" {
		t.Fatalf("top path = %q, want personal project fact", out[0].Path)
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

func TestReweight_MeasurementTotalPrefersMeasuredUserFactsOverGenericGuides(t *testing.T) {
	t.Parallel()
	out := reweightSharedMemoryRanking(
		"How many hours have I spent playing games in total?",
		[]RetrievedChunk{
			{
				Path:  "memory/project/eval-lme/game-backlog-planning.md",
				Score: 1.0,
				Text:  "General tips for tracking a game backlog and keeping notes about completed titles.",
			},
			{
				Path:  "memory/global/user-fact-2023-04-08-playing-the-last-of-us-part-ii.md",
				Score: 0.5,
				Text:  "I finished playing The Last of Us Part II and it took 30 hours.",
			},
		},
	)
	if out[0].Path != "memory/global/user-fact-2023-04-08-playing-the-last-of-us-part-ii.md" {
		t.Fatalf("top path = %q, want measured user fact", out[0].Path)
	}
}

func TestReweight_MoneyQuestionPrefersCurrencyEvidenceOverGenericGuides(t *testing.T) {
	t.Parallel()
	out := reweightSharedMemoryRanking(
		"Which grocery store did I spend the most money at in the past month?",
		[]RetrievedChunk{
			{
				Path:  "memory/project/eval-lme/grocery-budgeting-guide.md",
				Score: 1.0,
				Text:  "General tips for tracking grocery spending, meal planning, and reducing overspend.",
			},
			{
				Path:  "memory/project/eval-lme/grocery-expense-reference.md",
				Score: 0.58,
				Text:  "Costco: $120 on 2023-02-15. Publix: $85 on 2023-02-18. Walmart: $72 on 2023-02-25.",
			},
		},
	)
	if out[0].Path != "memory/project/eval-lme/grocery-expense-reference.md" {
		t.Fatalf("top path = %q, want explicit currency evidence", out[0].Path)
	}
}

func TestReweight_AggregateMoneyQuestionsPromoteDistinctEvidence(t *testing.T) {
	t.Parallel()
	out := reweightSharedMemoryRanking(
		"Which store did I spend the most money at this month?",
		[]RetrievedChunk{
			{
				Path:  "memory/project/budgeting-guide.md",
				Score: 1.0,
				Text:  "Guide to reducing shopping costs and keeping a store-by-store budget.",
			},
			{
				Path:     "memory/global/user-store-alpha.md",
				Score:    0.58,
				Text:     "The user spent $45 at Alpha Market on groceries.",
				Metadata: map[string]any{"session_id": "alpha", "source_role": "user"},
			},
			{
				Path:     "memory/global/user-store-beta.md",
				Score:    0.54,
				Text:     "The user spent $120 at Beta Foods on pantry staples.",
				Metadata: map[string]any{"session_id": "beta", "source_role": "user"},
			},
			{
				Path:     "memory/global/user-store-gamma.md",
				Score:    0.52,
				Text:     "The user spent $80 at Gamma Grocery.",
				Metadata: map[string]any{"session_id": "gamma", "source_role": "user"},
			},
		},
	)
	if len(out) < 3 {
		t.Fatalf("unexpected output length %d", len(out))
	}
	topThree := map[string]bool{out[0].Path: true, out[1].Path: true, out[2].Path: true}
	for _, want := range []string{
		"memory/global/user-store-alpha.md",
		"memory/global/user-store-beta.md",
		"memory/global/user-store-gamma.md",
	} {
		if !topThree[want] {
			t.Fatalf("top three paths = %v, want transactional evidence %q", topThree, want)
		}
	}
}

func TestReweight_FirstPersonFactPenalisesAssistantOnlyEvidence(t *testing.T) {
	t.Parallel()
	out := reweightSharedMemoryRanking(
		"How many bikes do I own?",
		[]RetrievedChunk{
			{
				Path:     "memory/project/bike-buying-guide.md",
				Score:    1.0,
				Text:     "The assistant suggested comparing three road bikes before buying.",
				Metadata: map[string]any{"source_role": "assistant"},
			},
			{
				Path:     "memory/global/user-bike-count.md",
				Score:    0.62,
				Text:     "The user currently owns two bikes.",
				Metadata: map[string]any{"source_role": "user"},
			},
		},
	)
	if out[0].Path != "memory/global/user-bike-count.md" {
		t.Fatalf("top path = %q, want user-sourced fact", out[0].Path)
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

func TestReweight_CompositeCoverageFacetsPromoteDistinctEvidence(t *testing.T) {
	t.Parallel()
	out := reweightSharedMemoryRanking(
		"What is the total number of attendee questions from the Atlas Webinar and Beacon Podcast episode?",
		[]RetrievedChunk{
			{
				Path:  "memory/global/user-fact-atlas-webinar.md",
				Score: 1.0,
				Text:  "The Atlas Webinar received 11 attendee questions.",
			},
			{
				Path:  "memory/global/user-fact-event-questions.md",
				Score: 0.95,
				Text:  "The event planning guide recommends tracking attendee questions after each launch.",
			},
			{
				Path:  "memory/global/user-fact-beacon-podcast.md",
				Score: 0.4,
				Text:  "The Beacon Podcast episode received 17 attendee questions.",
			},
		},
	)
	if len(out) < 2 {
		t.Fatalf("unexpected output length %d", len(out))
	}
	topTwo := map[string]bool{out[0].Path: true, out[1].Path: true}
	for _, want := range []string{
		"memory/global/user-fact-atlas-webinar.md",
		"memory/global/user-fact-beacon-podcast.md",
	} {
		if !topTwo[want] {
			t.Fatalf("top two paths = %v, want distinct facet fact %q", topTwo, want)
		}
	}
}

func TestReweight_TypeCountQueryDoesNotPromoteLowSignalTypePhrase(t *testing.T) {
	t.Parallel()
	out := reweightSharedMemoryRanking(
		"How many different types of citrus fruits have I used in my cocktail recipes?",
		[]RetrievedChunk{
			{
				Path:  "memory/global/user-fact-wood-types.md",
				Score: 1.0,
				Text:  "The user experimented with different types of wood chips for smoking vegetables.",
			},
			{
				Path:  "memory/project/cocktail-lime.md",
				Score: 0.5,
				Text:  "Cocktail brief: a refreshing drink made with rum, lime juice, and lemon garnish.",
			},
			{
				Path:  "memory/global/user-preference-orange-cocktail.md",
				Score: 0.4,
				Text:  "The user liked the Orange You Glad It's a Whiskey Sour cocktail recipe.",
			},
		},
	)
	if len(out) < 2 {
		t.Fatalf("unexpected output length %d", len(out))
	}
	if out[0].Path == "memory/global/user-fact-wood-types.md" {
		t.Fatalf("top path = %q, want relevant type-context evidence promoted", out[0].Path)
	}
}

func TestReweight_DateDifferencePromotesBothEventFacts(t *testing.T) {
	t.Parallel()
	out := reweightSharedMemoryRanking(
		"How many days ago did I launch my website when I signed a contract with my first client?",
		[]RetrievedChunk{
			{
				Path:  "memory/global/user_first_client_contract.md",
				Score: 1.0,
				Text:  "User signed a contract with their first client on March 1, 2023.",
			},
			{
				Path:  "memory/global/user_online_necklace.md",
				Score: 0.9,
				Text:  "The user bought a necklace from an online jewellery retailer's website.",
			},
			{
				Path:  "memory/global/user_post_launch_priorities.md",
				Score: 0.1,
				Text:  "User launched a website and created a business plan.",
			},
		},
	)
	if len(out) < 2 {
		t.Fatalf("unexpected output length %d", len(out))
	}
	topTwo := map[string]bool{out[0].Path: true, out[1].Path: true}
	for _, want := range []string{
		"memory/global/user_first_client_contract.md",
		"memory/global/user_post_launch_priorities.md",
	} {
		if !topTwo[want] {
			t.Fatalf("top two paths = %v, want date event fact %q", topTwo, want)
		}
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
