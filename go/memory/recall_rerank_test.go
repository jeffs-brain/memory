// SPDX-License-Identifier: Apache-2.0

package memory

import (
	"testing"
	"time"

	"github.com/jeffs-brain/memory/go/brain"
)

func TestRerankRecallHitsForDiversity_EmptyInput(t *testing.T) {
	result := rerankRecallHitsForDiversity(nil, "query", 5, recallQuerySignals{})
	if len(result) != 0 {
		t.Errorf("expected empty result for nil input, got %d", len(result))
	}
}

func TestRerankRecallHitsForDiversity_SingleResult(t *testing.T) {
	memories := []SurfacedMemory{
		makeSurfacedMemory("topic-a", "Topic A", "Description of A", "2025-01-01T10:00:00Z"),
	}
	result := rerankRecallHitsForDiversity(memories, "topic", 5, recallQuerySignals{})
	if len(result) != 1 {
		t.Fatalf("expected 1 result, got %d", len(result))
	}
	if result[0].Topic.Name != "Topic A" {
		t.Errorf("expected Topic A, got %q", result[0].Topic.Name)
	}
}

func TestRerankRecallHitsForDiversity_DiverseTopics(t *testing.T) {
	memories := []SurfacedMemory{
		makeSurfacedMemory("cooking", "Cooking pasta", "Italian recipe carbonara", "2025-03-01T10:00:00Z"),
		makeSurfacedMemory("cooking2", "Cooking risotto", "Italian recipe risotto mushroom", "2025-03-02T10:00:00Z"),
		makeSurfacedMemory("cooking3", "Cooking pizza", "Italian recipe pizza margherita", "2025-03-03T10:00:00Z"),
		makeSurfacedMemory("travel", "Travel Japan", "Tokyo trip planning itinerary", "2025-02-15T10:00:00Z"),
		makeSurfacedMemory("finance", "Budget tracking", "Monthly expenses spreadsheet", "2025-01-20T10:00:00Z"),
	}

	signals := recallQuerySignals{aggregate: true, temporal: true}
	result := rerankRecallHitsForDiversity(memories, "all my activities", 3, signals)

	if len(result) != 3 {
		t.Fatalf("expected 3 results, got %d", len(result))
	}

	// With diversity reranking and aggregate signals, the algorithm
	// should penalise selecting all three cooking memories together.
	// At least one non-cooking memory should appear in the top 3.
	topics := make(map[string]bool)
	for _, m := range result {
		topics[m.Topic.Name] = true
	}
	hasDiversity := topics["Travel Japan"] || topics["Budget tracking"]
	if !hasDiversity {
		t.Errorf("expected diverse results (non-cooking topic in top 3), got: %v",
			namesOf(result))
	}
}

func TestRerankRecallHitsForDiversity_AllSimilarStillReturnsK(t *testing.T) {
	memories := []SurfacedMemory{
		makeSurfacedMemory("go-test-1", "Go testing patterns", "Unit test table driven approach", "2025-04-01T10:00:00Z"),
		makeSurfacedMemory("go-test-2", "Go testing patterns", "Unit test table driven approach", "2025-04-02T10:00:00Z"),
		makeSurfacedMemory("go-test-3", "Go testing patterns", "Unit test table driven approach", "2025-04-03T10:00:00Z"),
	}

	result := rerankRecallHitsForDiversity(memories, "go testing", 3, recallQuerySignals{})
	if len(result) != 3 {
		t.Fatalf("expected 3 results even with identical content, got %d", len(result))
	}
}

func TestRerankRecallHitsForDiversity_TwoIdenticalDedup(t *testing.T) {
	// Two identical memories should both be returned (reranking does
	// not deduplicate by path, it penalises similarity).
	memories := []SurfacedMemory{
		makeSurfacedMemory("same-topic", "Exactly same content", "Same description same words", "2025-05-01T10:00:00Z"),
		makeSurfacedMemory("same-topic-copy", "Exactly same content", "Same description same words", "2025-05-01T10:00:00Z"),
	}

	result := rerankRecallHitsForDiversity(memories, "same content", 2, recallQuerySignals{})
	if len(result) != 2 {
		t.Fatalf("expected 2 results, got %d", len(result))
	}
}

func TestRerankRecallHitsForDiversity_DateBucketBalance(t *testing.T) {
	// Results spanning multiple date buckets should be balanced when
	// aggregate/temporal signals are active.
	memories := []SurfacedMemory{
		makeSurfacedMemory("jan-meeting", "January planning meeting", "Q1 goals reviewed budget allocations", "2025-01-15T10:00:00Z"),
		makeSurfacedMemory("jan-review", "January code review", "Q1 PR review statistics pull request", "2025-01-20T10:00:00Z"),
		makeSurfacedMemory("feb-meeting", "February planning meeting", "Q1 mid-quarter progress report", "2025-02-15T10:00:00Z"),
		makeSurfacedMemory("mar-meeting", "March retrospective meeting", "Q1 retrospective sprint review", "2025-03-15T10:00:00Z"),
		makeSurfacedMemory("jan-standup", "January standup", "Q1 daily standup notes sync", "2025-01-18T10:00:00Z"),
	}

	signals := recallQuerySignals{timeline: true, temporal: true, aggregate: true}
	result := rerankRecallHitsForDiversity(memories, "all meetings timeline", 3, signals)

	if len(result) != 3 {
		t.Fatalf("expected 3 results, got %d", len(result))
	}

	// With date bucket diversity bonus, we should see results from
	// different months rather than all January items.
	buckets := make(map[string]bool)
	for _, m := range result {
		ts, _ := parseTopicTime(m.Topic.Modified)
		buckets[ts.UTC().Format("2006-01")] = true
	}
	if len(buckets) < 2 {
		t.Errorf("expected results from at least 2 different months, got %d: %v",
			len(buckets), namesOf(result))
	}
}

func TestRerankRecallHitsForDiversity_RecentQuerySortsNewestFirst(t *testing.T) {
	memories := []SurfacedMemory{
		makeSurfacedMemory("old", "Old memory", "Description old", "2024-01-01T10:00:00Z"),
		makeSurfacedMemory("mid", "Middle memory", "Description mid", "2024-06-15T10:00:00Z"),
		makeSurfacedMemory("new", "New memory", "Description new", "2025-03-01T10:00:00Z"),
	}

	signals := recallQuerySignals{recent: true, temporal: true}
	result := rerankRecallHitsForDiversity(memories, "most recent updates", 3, signals)

	if len(result) != 3 {
		t.Fatalf("expected 3 results, got %d", len(result))
	}

	// Final ordering should be newest-first for recent queries.
	ts0, _ := parseTopicTime(result[0].Topic.Modified)
	ts1, _ := parseTopicTime(result[1].Topic.Modified)
	ts2, _ := parseTopicTime(result[2].Topic.Modified)
	if !ts0.After(ts1) || !ts1.After(ts2) {
		t.Errorf("expected newest-first order, got: %s, %s, %s",
			result[0].Topic.Modified, result[1].Topic.Modified, result[2].Topic.Modified)
	}
}

func TestRerankRecallHitsForDiversity_TimelineQuerySortsChronologically(t *testing.T) {
	memories := []SurfacedMemory{
		makeSurfacedMemory("new", "March event", "Event in March", "2025-03-01T10:00:00Z"),
		makeSurfacedMemory("old", "January event", "Event in January", "2025-01-01T10:00:00Z"),
		makeSurfacedMemory("mid", "February event", "Event in February", "2025-02-01T10:00:00Z"),
	}

	signals := recallQuerySignals{timeline: true, temporal: true}
	result := rerankRecallHitsForDiversity(memories, "timeline of events last year", 3, signals)

	if len(result) != 3 {
		t.Fatalf("expected 3 results, got %d", len(result))
	}

	// Timeline query should produce chronological (oldest-first) order.
	ts0, _ := parseTopicTime(result[0].Topic.Modified)
	ts1, _ := parseTopicTime(result[1].Topic.Modified)
	ts2, _ := parseTopicTime(result[2].Topic.Modified)
	if !ts0.Before(ts1) || !ts1.Before(ts2) {
		t.Errorf("expected chronological order, got: %s, %s, %s",
			result[0].Topic.Modified, result[1].Topic.Modified, result[2].Topic.Modified)
	}
}

func TestRerankRecallHitsForDiversity_KLargerThanInput(t *testing.T) {
	memories := []SurfacedMemory{
		makeSurfacedMemory("a", "Topic A", "Description A", "2025-01-01T10:00:00Z"),
		makeSurfacedMemory("b", "Topic B", "Description B", "2025-02-01T10:00:00Z"),
	}

	result := rerankRecallHitsForDiversity(memories, "query", 10, recallQuerySignals{})
	if len(result) != 2 {
		t.Fatalf("expected 2 results (all available), got %d", len(result))
	}
}

// ---- classifyQuery tests ----

func TestClassifyQuery_Timeline(t *testing.T) {
	tests := []struct {
		name  string
		query string
		want  recallQuerySignals
	}{
		{
			name:  "yesterday is temporal",
			query: "what happened yesterday",
			want:  recallQuerySignals{timeline: true, temporal: true, aggregate: true, concrete: true},
		},
		{
			name:  "recent is recent",
			query: "most recent updates",
			want:  recallQuerySignals{recent: true, temporal: true, concrete: true},
		},
		{
			name:  "plain query without signal words",
			query: "golang concurrency",
			want:  recallQuerySignals{},
		},
		{
			name:  "aggregate query",
			query: "all my expenses total",
			want:  recallQuerySignals{aggregate: true, concrete: true},
		},
		{
			name:  "patterns triggers aggregate",
			query: "golang concurrency patterns",
			want:  recallQuerySignals{aggregate: true, concrete: false},
		},
		{
			name:  "empty query",
			query: "",
			want:  recallQuerySignals{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := classifyQuery(tt.query)
			if got != tt.want {
				t.Errorf("classifyQuery(%q) = %+v, want %+v", tt.query, got, tt.want)
			}
		})
	}
}

// ---- Jaccard similarity tests ----

func TestJaccardSimilarity(t *testing.T) {
	tests := []struct {
		name string
		left []string
		right []string
		want  float64
	}{
		{"empty left", nil, []string{"a", "b"}, 0},
		{"empty right", []string{"a", "b"}, nil, 0},
		{"both empty", nil, nil, 1.0},
		{"identical", []string{"a", "b", "c"}, []string{"a", "b", "c"}, 1.0},
		{"disjoint", []string{"a", "b"}, []string{"c", "d"}, 0},
		{"partial overlap", []string{"a", "b", "c"}, []string{"b", "c", "d"}, 0.5},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := jaccardSimilarity(tt.left, tt.right)
			if !floatClose(got, tt.want, 0.001) {
				t.Errorf("jaccardSimilarity(%v, %v) = %f, want %f",
					tt.left, tt.right, got, tt.want)
			}
		})
	}
}

// ---- Tokenise tests ----

func TestTokenise(t *testing.T) {
	tests := []struct {
		name  string
		text  string
		limit int
		want  []string
	}{
		{"empty", "", 32, nil},
		{"stop words removed", "the and is a", 32, nil},
		{"short tokens removed", "a b c", 32, nil},
		{
			"stemming ies",
			"batteries activities",
			32,
			[]string{"battery", "activity"},
		},
		{
			"stemming es only when long enough",
			"processes boxes",
			32,
			[]string{"process", "boxe"},
		},
		{
			"stemming s requires length over 4",
			"items tests",
			32,
			[]string{"item", "test"},
		},
		{
			"deduplication",
			"go golang go testing testing",
			32,
			[]string{"go", "golang", "testing"},
		},
		{
			"limit respected",
			"alpha beta gamma delta epsilon",
			2,
			[]string{"alpha", "beta"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := tokenise(tt.text, tt.limit)
			if !stringSliceEqual(got, tt.want) {
				t.Errorf("tokenise(%q, %d) = %v, want %v",
					tt.text, tt.limit, got, tt.want)
			}
		})
	}
}

// ---- normaliseRecency tests ----

func TestNormaliseRecency(t *testing.T) {
	oldest := time.Date(2025, 1, 1, 0, 0, 0, 0, time.UTC)
	newest := time.Date(2025, 12, 31, 0, 0, 0, 0, time.UTC)
	mid := time.Date(2025, 7, 1, 0, 0, 0, 0, time.UTC)

	tests := []struct {
		name string
		ts   time.Time
		want float64
	}{
		{"oldest is 0", oldest, 0},
		{"newest is 1", newest, 1},
		{"middle is ~0.5", mid, 0.5},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := normaliseRecency(tt.ts, oldest, newest)
			if !floatClose(got, tt.want, 0.01) {
				t.Errorf("normaliseRecency(%v) = %f, want ~%f", tt.ts, got, tt.want)
			}
		})
	}
}

func TestNormaliseRecency_EqualTimestamps(t *testing.T) {
	ts := time.Date(2025, 6, 1, 0, 0, 0, 0, time.UTC)
	got := normaliseRecency(ts, ts, ts)
	if got != 1 {
		t.Errorf("expected 1 for equal timestamps, got %f", got)
	}
}

// ---- scoreCandidate tests ----

func TestScoreCandidate_FirstItemReturnsBaseScore(t *testing.T) {
	candidate := rankedRecallHit{
		baseScore:       5.0,
		signatureTokens: []string{"go", "test"},
		topicTokens:     []string{"go"},
	}
	score := scoreCandidate(candidate, nil, recallQuerySignals{})
	if score != 5.0 {
		t.Errorf("expected base score 5.0, got %f", score)
	}
}

func TestScoreCandidate_PenalisesHighSimilarity(t *testing.T) {
	candidate := rankedRecallHit{
		baseScore:       5.0,
		signatureTokens: []string{"go", "test", "unit"},
		topicTokens:     []string{"go", "test"},
	}
	chosen := []rankedRecallHit{
		{
			signatureTokens: []string{"go", "test", "unit"},
			topicTokens:     []string{"go", "test"},
		},
	}
	score := scoreCandidate(candidate, chosen, recallQuerySignals{})
	// Default penalty is 0.35 * 1.0 (perfect Jaccard) = 0.35
	expected := 5.0 - 0.35
	if !floatClose(score, expected, 0.001) {
		t.Errorf("expected ~%f, got %f", expected, score)
	}
}

func TestScoreCandidate_AggregatePenaltyIsHigher(t *testing.T) {
	candidate := rankedRecallHit{
		baseScore:       5.0,
		signatureTokens: []string{"go", "test", "unit"},
		topicTokens:     []string{"go", "test"},
	}
	chosen := []rankedRecallHit{
		{
			signatureTokens: []string{"go", "test", "unit"},
			topicTokens:     []string{"go", "test"},
		},
	}
	signals := recallQuerySignals{aggregate: true}
	score := scoreCandidate(candidate, chosen, signals)
	// Aggregate: sig penalty = 1.1 * 1.0, topic penalty = 0.45 * 1.0
	// Total penalty = 1.1 + 0.45 = 1.55
	expected := 5.0 - 1.1 - 0.45
	if !floatClose(score, expected, 0.001) {
		t.Errorf("expected ~%f, got %f", expected, score)
	}
}

func TestScoreCandidate_DateBucketDiversityBonus(t *testing.T) {
	candidate := rankedRecallHit{
		baseScore:       5.0,
		signatureTokens: []string{"meeting", "review"},
		topicTokens:     []string{"meeting"},
		dateBucket:      "2025-02-15",
	}
	chosen := []rankedRecallHit{
		{
			signatureTokens: []string{"different", "topic"},
			topicTokens:     []string{"different"},
			dateBucket:      "2025-01-15",
		},
	}
	signals := recallQuerySignals{aggregate: true, temporal: true}
	score := scoreCandidate(candidate, chosen, signals)
	// Should include +0.35 date bucket diversity bonus
	// Sig similarity is 0 (disjoint), topic similarity is 0
	// So score = 5.0 + 0.35 (date bucket) + 0.25 (topic < 0.2)
	expected := 5.0 + 0.35 + 0.25
	if !floatClose(score, expected, 0.001) {
		t.Errorf("expected ~%f, got %f", expected, score)
	}
}

func TestScoreCandidate_NoBucketBonusWhenSameDate(t *testing.T) {
	candidate := rankedRecallHit{
		baseScore:       5.0,
		signatureTokens: []string{"meeting", "review"},
		topicTokens:     []string{"meeting"},
		dateBucket:      "2025-01-15",
	}
	chosen := []rankedRecallHit{
		{
			signatureTokens: []string{"different", "topic"},
			topicTokens:     []string{"different"},
			dateBucket:      "2025-01-15",
		},
	}
	signals := recallQuerySignals{aggregate: true, temporal: true}
	score := scoreCandidate(candidate, chosen, signals)
	// Same date bucket: no +0.35 bonus, but still gets +0.25 topic novelty
	expected := 5.0 + 0.25
	if !floatClose(score, expected, 0.001) {
		t.Errorf("expected ~%f, got %f", expected, score)
	}
}

// ---- isTimeSensitiveQuery tests ----

func TestIsTimeSensitiveQuery(t *testing.T) {
	tests := []struct {
		query string
		want  bool
	}{
		{"what happened yesterday", true},
		{"2 weeks ago", true},
		{"last month", true},
		{"first time we met", true},
		{"golang patterns", false},
		{"", false},
		{"   ", false},
		{"January events", true},
		{"monday standup", true},
		{"timeline of project", true},
	}

	for _, tt := range tests {
		t.Run(tt.query, func(t *testing.T) {
			got := isTimeSensitiveQuery(tt.query)
			if got != tt.want {
				t.Errorf("isTimeSensitiveQuery(%q) = %v, want %v",
					tt.query, got, tt.want)
			}
		})
	}
}

// ---- Integration: verify same input produces same ranking ----

func TestRerankDeterministic(t *testing.T) {
	memories := []SurfacedMemory{
		makeSurfacedMemory("auth", "Auth patterns", "OAuth2 PKCE implementation guide", "2025-01-10T10:00:00Z"),
		makeSurfacedMemory("api", "API design", "REST API versioning strategies", "2025-02-10T10:00:00Z"),
		makeSurfacedMemory("db", "Database patterns", "PostgreSQL query optimisation indexes", "2025-03-10T10:00:00Z"),
		makeSurfacedMemory("testing", "Testing patterns", "Integration test fixtures mocking", "2025-04-10T10:00:00Z"),
	}

	signals := recallQuerySignals{aggregate: true, temporal: true}
	first := rerankRecallHitsForDiversity(memories, "all patterns", 3, signals)
	second := rerankRecallHitsForDiversity(memories, "all patterns", 3, signals)

	if len(first) != len(second) {
		t.Fatalf("non-deterministic: first=%d, second=%d", len(first), len(second))
	}
	for i := range first {
		if first[i].Topic.Name != second[i].Topic.Name {
			t.Errorf("non-deterministic at position %d: %q vs %q",
				i, first[i].Topic.Name, second[i].Topic.Name)
		}
	}
}

// ---- Helpers ----

func makeSurfacedMemory(slug, name, description, modified string) SurfacedMemory {
	return SurfacedMemory{
		Path:    brain.Path("memory/project/test/" + slug + ".md"),
		Content: description,
		Topic: TopicFile{
			Name:        name,
			Description: description,
			Path:        brain.Path("memory/project/test/" + slug + ".md"),
			Modified:    modified,
			Scope:       "project",
		},
	}
}

func namesOf(memories []SurfacedMemory) []string {
	names := make([]string, len(memories))
	for i, m := range memories {
		names[i] = m.Topic.Name
	}
	return names
}

func floatClose(a, b, epsilon float64) bool {
	diff := a - b
	if diff < 0 {
		diff = -diff
	}
	return diff < epsilon
}

func stringSliceEqual(a, b []string) bool {
	if len(a) == 0 && len(b) == 0 {
		return true
	}
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
