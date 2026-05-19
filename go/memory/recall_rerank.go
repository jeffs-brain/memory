// SPDX-License-Identifier: Apache-2.0

package memory

import (
	"math"
	"regexp"
	"strings"
	"time"
)

// Query signal classification patterns, ported from the TS recall
// pipeline with identical regex semantics.
var (
	temporalSortPatterns = []*regexp.Regexp{
		regexp.MustCompile(`(?i)\b(?:today|yesterday|tomorrow|tonight)\b`),
		regexp.MustCompile(`(?i)\blast\s+(?:week|month|year|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b`),
		regexp.MustCompile(`(?i)\b\d+\s+(?:day|days|week|weeks|month|months|year|years)\s+ago\b`),
		regexp.MustCompile(`(?i)\b(?:oldest|earlier|before|after|between|since|compared|difference|timeline|history|trend)\b`),
		regexp.MustCompile(`(?i)\bfirst\b`),
		regexp.MustCompile(`\b(?:19|20)\d{2}\b`),
		regexp.MustCompile(`(?i)\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\b`),
		regexp.MustCompile(`(?i)\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b`),
	}

	recentQueryPatterns = []*regexp.Regexp{
		regexp.MustCompile(`(?i)\b(?:latest|most recent|newest|updated|recent|recently|current|currently)\b`),
		regexp.MustCompile(`(?i)\blast\s+time\b`),
	}

	aggregateQueryPatterns = []*regexp.Regexp{
		regexp.MustCompile(`(?i)\b(?:all|across|between|compare|comparison|different|history|timeline|pattern|patterns|period|periods|times|episodes|instances|list|summary|summarise|recap|types|kinds|how much|total|spent|expense|expenses|cost|costs|breaks|appointments|meetings|workshops)\b`),
	}

	concreteQueryPatterns = []*regexp.Regexp{
		regexp.MustCompile(`(?i)^(?:when|where|who|which)\b`),
		regexp.MustCompile(`(?i)\b(?:how much|how many|spent|cost|costs|paid|before|after|between|compare|compared|difference|differences|happened|meeting|workshop|appointment|doctor|bill|expense|expenses|break)\b`),
	}

	genericAdvicePatterns = []*regexp.Regexp{
		regexp.MustCompile(`(?i)\b(?:tip|tips|advice|guidance|guideline|guidelines|best practice|best practices|principle|principles|always|never|generally|usually|remember to|consider|try to)\b`),
	}

	concreteContentPatterns = []*regexp.Regexp{
		regexp.MustCompile(`(?i)(?:` + "£" + `|\$|` + "€" + `)\s?\d+`),
		regexp.MustCompile(`(?i)\b\d+(?:\.\d+)?\s?(?:hour|hours|day|days|week|weeks|month|months|year|years|km|mi|mile|miles|min|mins|minute|minutes)\b`),
		regexp.MustCompile(`(?i)\b(?:19|20)\d{2}\b`),
		regexp.MustCompile(`(?i)\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\b`),
		regexp.MustCompile(`(?i)\b(?:meeting|met|workshop|call|doctor|appointment|spent|paid|bought|travelled|visited|break|holiday|trip)\b`),
	}

	rerankTokenPattern = regexp.MustCompile(`(?i)[a-z0-9]+`)
)

// rerankStopWords extends the package-level stopWords set with
// additional entries from the TS pipeline for recall tokenisation.
var rerankStopWords = map[string]bool{
	"a": true, "an": true, "and": true, "are": true, "as": true,
	"at": true, "be": true, "been": true, "but": true, "by": true,
	"did": true, "do": true, "does": true, "for": true, "from": true,
	"had": true, "has": true, "have": true, "how": true, "i": true,
	"if": true, "in": true, "into": true, "is": true, "it": true,
	"its": true, "me": true, "my": true, "of": true, "on": true,
	"or": true, "our": true, "that": true, "the": true, "their": true,
	"them": true, "then": true, "there": true, "these": true, "they": true,
	"this": true, "those": true, "to": true, "was": true, "we": true,
	"were": true, "what": true, "when": true, "where": true, "which": true,
	"who": true, "with": true, "you": true, "your": true,
	"not": true,
}

// recallQuerySignals captures the classification of a recall query,
// mirroring the TS RecallQuerySignals type.
type recallQuerySignals struct {
	timeline  bool
	recent    bool
	temporal  bool
	aggregate bool
	concrete  bool
}

// rankedRecallHit holds analysis metadata for a single recall
// candidate during diversity-aware reranking.
type rankedRecallHit struct {
	memory          SurfacedMemory
	baseScore       float64
	originalScore   float64
	timestamp       time.Time
	hasTimestamp    bool
	recencyScore    float64
	dateBucket      string
	signatureTokens []string
	topicTokens     []string
}

// rerankRecallHitsForDiversity applies MMR-style greedy selection with
// Jaccard similarity penalties and date-bucket diversity. It mirrors
// the TS rerankRecallHits function exactly, including all constants
// and thresholds.
func rerankRecallHitsForDiversity(
	memories []SurfacedMemory,
	query string,
	k int,
	signals recallQuerySignals,
) []SurfacedMemory {
	if len(memories) <= 1 {
		return orderMemoriesForQuery(memories, signals)
	}

	queryTokens := tokenise(query, 32)
	ranked := analyseAllHits(memories, queryTokens, signals)

	selected := make([]SurfacedMemory, 0, k)
	chosen := make([]rankedRecallHit, 0, k)
	remaining := make([]rankedRecallHit, len(ranked))
	copy(remaining, ranked)

	for len(selected) < k && len(remaining) > 0 {
		nextIdx := selectBestCandidate(remaining, chosen, signals)
		next := remaining[nextIdx]
		remaining = append(remaining[:nextIdx], remaining[nextIdx+1:]...)
		selected = append(selected, next.memory)
		chosen = append(chosen, next)
	}

	return orderMemoriesForQuery(selected, signals)
}

// selectBestCandidate finds the index of the candidate with the
// highest diversity-penalised score.
func selectBestCandidate(
	remaining []rankedRecallHit,
	chosen []rankedRecallHit,
	signals recallQuerySignals,
) int {
	bestIdx := 0
	bestScore := math.Inf(-1)
	for i := range remaining {
		score := scoreCandidate(remaining[i], chosen, signals)
		if score > bestScore {
			bestIdx = i
			bestScore = score
		}
	}
	return bestIdx
}

// scoreCandidate computes the diversity-penalised selection score for
// a candidate relative to the already-chosen set. Mirrors the TS
// scoreRecallCandidate function with identical constants.
func scoreCandidate(
	candidate rankedRecallHit,
	chosen []rankedRecallHit,
	signals recallQuerySignals,
) float64 {
	if len(chosen) == 0 {
		return candidate.baseScore
	}

	score := candidate.baseScore

	maxSigSim := maxJaccardSimilarity(
		candidate.signatureTokens, chosen,
		func(r rankedRecallHit) []string { return r.signatureTokens },
	)
	penalty := signaturePenaltyWeight(signals)
	score -= maxSigSim * penalty

	if signals.aggregate {
		maxTopicSim := maxJaccardSimilarity(
			candidate.topicTokens, chosen,
			func(r rankedRecallHit) []string { return r.topicTokens },
		)
		score -= maxTopicSim * 0.45
		if maxTopicSim < 0.2 {
			score += 0.25
		}
	}

	if (signals.aggregate || signals.temporal) && candidate.dateBucket != "" {
		if !anyMatchingDateBucket(chosen, candidate.dateBucket) {
			score += 0.35
		}
	}

	if signals.recent {
		score += candidate.recencyScore * 0.4
	}

	return score
}

// signaturePenaltyWeight returns the Jaccard similarity penalty
// multiplier based on query signals.
func signaturePenaltyWeight(signals recallQuerySignals) float64 {
	if signals.aggregate {
		return 1.1
	}
	if signals.temporal {
		return 0.75
	}
	return 0.35
}

// anyMatchingDateBucket checks whether any chosen hit shares the same
// date bucket as the candidate.
func anyMatchingDateBucket(chosen []rankedRecallHit, bucket string) bool {
	for i := range chosen {
		if chosen[i].dateBucket == bucket {
			return true
		}
	}
	return false
}

// maxJaccardSimilarity computes the maximum Jaccard similarity between
// a candidate's tokens and the corresponding tokens from all chosen
// hits.
func maxJaccardSimilarity(
	candidateTokens []string,
	chosen []rankedRecallHit,
	getTokens func(rankedRecallHit) []string,
) float64 {
	maxSim := 0.0
	for i := range chosen {
		sim := jaccardSimilarity(candidateTokens, getTokens(chosen[i]))
		if sim > maxSim {
			maxSim = sim
		}
	}
	return maxSim
}

// analyseAllHits computes ranking metadata for every candidate.
func analyseAllHits(
	memories []SurfacedMemory,
	queryTokens []string,
	signals recallQuerySignals,
) []rankedRecallHit {
	maxScore := 0.0
	for _, m := range memories {
		if m.Topic.Modified != "" {
			// We use a score of 1.0 as default since Go recall has
			// no numeric score field; all memories start equal.
		}
	}
	// In the Go pipeline, SurfacedMemory has no numeric score field.
	// We assign a default score of 1.0 to all memories and let the
	// reranking bonuses differentiate them.
	maxScore = 1.0
	scoreScale := clampFloat(maxFloat(maxScore, 1), 1, 4)

	var oldest, newest time.Time
	var haveTimestamps bool
	for _, m := range memories {
		ts, ok := parseTopicTime(m.Topic.Modified)
		if !ok {
			continue
		}
		if !haveTimestamps {
			oldest, newest = ts, ts
			haveTimestamps = true
			continue
		}
		if ts.Before(oldest) {
			oldest = ts
		}
		if ts.After(newest) {
			newest = ts
		}
	}

	ranked := make([]rankedRecallHit, 0, len(memories))
	for _, m := range memories {
		ranked = append(ranked, analyseHit(
			m, queryTokens, signals, scoreScale,
			maxScore, oldest, newest, haveTimestamps,
		))
	}
	return ranked
}

// analyseHit computes the base adjusted score and token signatures for
// a single candidate. Mirrors TS analyseRecallHit.
func analyseHit(
	m SurfacedMemory,
	queryTokens []string,
	signals recallQuerySignals,
	scoreScale float64,
	maxScore float64,
	oldest, newest time.Time,
	haveTimestamps bool,
) rankedRecallHit {
	searchText := buildNoteSearchText(m)
	topicText := buildTopicText(m)
	sigTokens := tokenise(searchText, 48)
	topTokens := tokenise(topicText, 24)

	queryMatches := countOverlap(queryTokens, sigTokens)
	titleMatches := countOverlap(queryTokens, topTokens)
	queryCoverage := safeDivide(float64(queryMatches), float64(len(queryTokens)))
	titleCoverage := safeDivide(float64(titleMatches), float64(len(queryTokens)))

	concreteScore := computeConcreteScore(m)
	genericPenalty := computeGenericPenalty(m)

	ts, hasTS := parseTopicTime(m.Topic.Modified)

	var recencyScore float64
	if signals.recent && hasTS && haveTimestamps {
		recencyScore = normaliseRecency(ts, oldest, newest)
	}

	// Default score is 1.0 for Go pipeline (no search score).
	score := 1.0
	bonus := queryCoverage*1.4 + titleCoverage*0.8 +
		minFloat(float64(queryMatches), 3)*0.15

	if signals.temporal && hasTS {
		bonus += 0.8
	}
	if signals.recent {
		bonus += recencyScore * 0.8
	}
	if signals.concrete {
		bonus += concreteScore*0.55 - genericPenalty*0.75
	} else {
		bonus -= genericPenalty * 0.1
	}

	baseScore := score + normaliseScore(score, maxScore)*0.35 +
		bonus*scoreScale

	var bucket string
	if hasTS {
		bucket = ts.UTC().Format("2006-01-02")
	}

	return rankedRecallHit{
		memory:          m,
		baseScore:       baseScore,
		originalScore:   score,
		timestamp:       ts,
		hasTimestamp:    hasTS,
		recencyScore:    recencyScore,
		dateBucket:      bucket,
		signatureTokens: sigTokens,
		topicTokens:     topTokens,
	}
}

// classifyQuery classifies a recall query into signal flags. Mirrors
// the TS classifyRecallQuery function.
func classifyQuery(query string) recallQuerySignals {
	trimmed := strings.TrimSpace(query)
	timeline := isTimeSensitiveQuery(trimmed)
	recent := trimmed != "" && matchesAny(trimmed, recentQueryPatterns)
	temporal := timeline || recent
	aggregate := trimmed != "" &&
		(timeline || matchesAny(trimmed, aggregateQueryPatterns))
	concrete := trimmed != "" &&
		(temporal || matchesAny(trimmed, concreteQueryPatterns))
	return recallQuerySignals{
		timeline:  timeline,
		recent:    recent,
		temporal:  temporal,
		aggregate: aggregate,
		concrete:  concrete,
	}
}

// isTimeSensitiveQuery returns true when the query contains temporal
// patterns that suggest chronological ordering is important.
func isTimeSensitiveQuery(query string) bool {
	trimmed := strings.TrimSpace(query)
	if trimmed == "" {
		return false
	}
	return matchesAny(trimmed, temporalSortPatterns)
}

// orderMemoriesForQuery applies the final presentation order based on
// query signals.
func orderMemoriesForQuery(
	memories []SurfacedMemory,
	signals recallQuerySignals,
) []SurfacedMemory {
	if signals.recent {
		return sortByRecency(memories)
	}
	if signals.timeline {
		return SortMemoriesChronologically(memories)
	}
	out := make([]SurfacedMemory, len(memories))
	copy(out, memories)
	return out
}

// sortByRecency returns memories sorted newest-first by their
// modified timestamp. Undated memories are appended at the end in
// their original order.
func sortByRecency(memories []SurfacedMemory) []SurfacedMemory {
	type indexed struct {
		mem   SurfacedMemory
		idx   int
		ts    time.Time
		hasTS bool
	}

	items := make([]indexed, len(memories))
	for i, m := range memories {
		ts, ok := parseTopicTime(m.Topic.Modified)
		items[i] = indexed{mem: m, idx: i, ts: ts, hasTS: ok}
	}

	// Stable sort: dated items newest-first, undated preserve order.
	stableSort(items, func(a, b indexed) bool {
		if a.hasTS != b.hasTS {
			return a.hasTS
		}
		if a.hasTS && b.hasTS {
			if !a.ts.Equal(b.ts) {
				return a.ts.After(b.ts)
			}
		}
		return a.idx < b.idx
	})

	out := make([]SurfacedMemory, len(items))
	for i, item := range items {
		out[i] = item.mem
	}
	return out
}

// ---- Token / text utilities ----

// tokenise extracts unique, stemmed, non-stop-word tokens from text.
// Mirrors the TS tokenise function.
func tokenise(text string, limit int) []string {
	matches := rerankTokenPattern.FindAllString(strings.ToLower(text), -1)
	var out []string
	seen := make(map[string]struct{})
	for _, match := range matches {
		token := stemToken(match)
		if len(token) < 2 {
			continue
		}
		if rerankStopWords[token] {
			continue
		}
		if _, dup := seen[token]; dup {
			continue
		}
		seen[token] = struct{}{}
		out = append(out, token)
		if len(out) >= limit {
			break
		}
	}
	return out
}

// stemToken and countOverlap are defined in text_util.go.

// buildNoteSearchText concatenates all searchable fields of a memory.
func buildNoteSearchText(m SurfacedMemory) string {
	parts := []string{m.Topic.Name, m.Topic.Description}
	parts = append(parts, m.Topic.Tags...)
	parts = append(parts, m.Content)
	return strings.Join(parts, "\n")
}

// buildTopicText concatenates the topic-level fields only.
func buildTopicText(m SurfacedMemory) string {
	parts := []string{m.Topic.Name, m.Topic.Description}
	parts = append(parts, m.Topic.Tags...)
	return strings.Join(parts, "\n")
}

// computeConcreteScore assigns a concreteness bonus to a memory based
// on timestamps, session IDs, and concrete content patterns. Mirrors
// the TS concreteNoteScore.
func computeConcreteScore(m SurfacedMemory) float64 {
	text := buildNoteSearchText(m)
	score := 0.0
	if _, ok := parseTopicTime(m.Topic.Modified); ok {
		score += 1.0
	}
	// TS checks for sessionId; in Go we do not have a direct equivalent
	// on SurfacedMemory, so we skip that 0.35 bonus.
	for _, pattern := range concreteContentPatterns {
		if pattern.MatchString(text) {
			score += 0.35
		}
	}
	return score
}

// computeGenericPenalty penalises memories that contain generic advice
// patterns. Mirrors the TS genericAdvicePenalty.
func computeGenericPenalty(m SurfacedMemory) float64 {
	text := buildNoteSearchText(m)
	matchesGeneric := false
	for _, pattern := range genericAdvicePatterns {
		if pattern.MatchString(text) {
			matchesGeneric = true
			break
		}
	}
	if !matchesGeneric {
		return 0
	}
	if _, ok := parseTopicTime(m.Topic.Modified); ok {
		return 0.25
	}
	return 1
}

// normaliseScore normalises a score relative to the maximum.
func normaliseScore(score, maxScore float64) float64 {
	if maxScore <= 0 {
		return 0
	}
	return score / maxScore
}

// normaliseRecency normalises a timestamp to [0, 1] relative to the
// oldest and newest timestamps in the result set.
func normaliseRecency(ts, oldest, newest time.Time) float64 {
	if oldest.Equal(newest) {
		return 1
	}
	totalRange := newest.Sub(oldest).Seconds()
	if totalRange <= 0 {
		return 1
	}
	return ts.Sub(oldest).Seconds() / totalRange
}

// matchesAny returns true if text matches any of the given patterns.
func matchesAny(text string, patterns []*regexp.Regexp) bool {
	for _, p := range patterns {
		if p.MatchString(text) {
			return true
		}
	}
	return false
}

// ---- Numeric helpers ----

func clampFloat(v, lo, hi float64) float64 {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

func maxFloat(a, b float64) float64 {
	if a > b {
		return a
	}
	return b
}

func minFloat(a, b float64) float64 {
	if a < b {
		return a
	}
	return b
}

func safeDivide(num, denom float64) float64 {
	if denom == 0 {
		return 0
	}
	return num / denom
}

// stableSort sorts a slice in-place preserving original order for
// equal elements. Uses insertion sort for small slices which is
// sufficient for recall result sets (typically < 30 items).
func stableSort[T any](s []T, less func(a, b T) bool) {
	for i := 1; i < len(s); i++ {
		for j := i; j > 0 && less(s[j], s[j-1]); j-- {
			s[j], s[j-1] = s[j-1], s[j]
		}
	}
}
