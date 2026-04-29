// SPDX-License-Identifier: Apache-2.0

package retrieval

import (
	"regexp"
	"sort"
	"strings"
)

// The intent regexes below must be kept bit-for-bit identical to the
// TypeScript reference in retrieval/hybrid.ts. Any drift breaks
// cross-SDK parity and will be flagged by the conformance suite.
//
// The `(?i)` flag is applied inline so the patterns remain portable.
// The spec forbids additional flags (no `u`, `g`, `s`); Go's regexp
// engine honours `\b` against ASCII word boundaries which matches the
// TS surface behaviour for every curated pattern.

var (
	preferenceQueryRe             = regexp.MustCompile(`(?i)\b(?:recommend|suggest|recommendation|suggestion|tips?|advice|ideas?|what should i|which should i)\b`)
	enumerationOrTotalQueryRe     = regexp.MustCompile(`(?i)\b(?:how many|count|total|in total|sum|add up|list|what are all)\b`)
	propertyLookupQueryRe         = regexp.MustCompile(`(?i)\b(?:how long is my|how often do i|what time do i|what time is my|where do i|where did i|where have i|where am i|where is my|what speed is my|how fast is my|what percentage(?: of)?|(?:what was the )?page count|what specific|which specific|what exact|which exact|which mode of transport did i|what mode of transport did i|which transport did i|what transport did i)\b`)
	specificRecommendationQueryRe = regexp.MustCompile(`(?i)\b(?:specific|exact)\b`)
	firstPersonFactLookupRe       = regexp.MustCompile(`(?i)\b(?:did i|have i|was i|were i)\b`)
	firstPersonConcreteQueryRe    = regexp.MustCompile(`(?i)\b(?:my|me|i)\b`)
	factLookupVerbRe              = regexp.MustCompile(`(?i)\b(?:pick(?:ed)? up|bought|ordered|spent|earned|sold|drove|travelled|traveled|watched|visited|completed|finished|submitted|booked|take|took|keep|kept|see|saw)\b`)
	moneyEventQueryRe             = regexp.MustCompile(`(?i)\b(?:spent|spend|cost|costed|paid|pay)\b`)
	preferenceNoteRe              = regexp.MustCompile(`(?i)\b(?:prefer(?:s|red)?|like(?:s|d)?|love(?:s|d)?|want(?:s|ed)?|need(?:s|ed)?|avoid(?:s|ed)?|dislike(?:s|d)?|hate(?:s|d)?|enjoy(?:s|ed)?|interested in|looking for)\b`)
	genericNoteRe                 = regexp.MustCompile(`(?i)\b(?:tips?|advice|suggest(?:ion|ed)?s?|recommend(?:ation|ed)?s?|ideas?|options?|guide|tracking|tracker|checklist)\b`)
	rollupNoteRe                  = regexp.MustCompile(`(?i)\b(?:roll-?up|summary|recap|overview|aggregate|combined|overall|in total|totalled?|totalling)\b`)
	atomicEventNoteRe             = regexp.MustCompile(`(?i)\b(?:i|we)\s+(?:picked up|bought|ordered|spent|earned|sold|drove|travelled|traveled|went|watched|visited|completed|finished|started|booked|got|took|submitted)\b`)
	dateTagRe                     = regexp.MustCompile(`(?i)\[(?:date|observed on):`)
	questionLikeNoteRe            = regexp.MustCompile(`(?i)(?:^|\n)(?:what\s+(?:are|is|should|could)|which\s+(?:should|would)|how\s+(?:can|should|could|long)|can\s+you|could\s+you|should\s+i|would\s+you|when\s+did|where\s+(?:can|should)|why\s+(?:is|does|did))\b`)
	durationQueryRe               = regexp.MustCompile(`(?i)\bhow long\b`)
	bodyAbsoluteDateRe            = regexp.MustCompile(`(?i)\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}(?:st|nd|rd|th)?\b`)
	measurementValueRe            = regexp.MustCompile(`(?i)\b(?:(?:\d+(?:\.\d+)?|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)(?:\s*-\s*(?:\d+(?:\.\d+)?|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve))?)(?:\s+|-)(?:minutes?|hours?|days?|weeks?|months?|years?)\b`)
	measurementUnitQueryRe        = regexp.MustCompile(`(?i)\b(?:minutes?|hours?|days?|weeks?|months?|years?)\b`)
	currencyValueRe               = regexp.MustCompile(`(?i)(?:[$€£]\s?\d|\b\d+(?:\.\d{1,2})?\s?(?:dollars?|usd|eur|gbp|pounds?|euros?)\b)`)
	numericValueRe                = regexp.MustCompile(`(?i)\b(?:\d+(?:\.\d+)?|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\b`)
	valueComparisonQueryRe        = regexp.MustCompile(`(?i)\b(?:most|least|highest|lowest|largest|smallest|biggest|which|where)\b`)
	routineScopeQueryRe           = regexp.MustCompile(`(?i)\b(?:daily|every|weekday|each way)\b`)
	routineScopeNoteRe            = regexp.MustCompile(`(?i)\b(?:daily commute|every day|every weekday|weekday|weekdays|each way)\b`)
	segmentQualifierNoteRe        = regexp.MustCompile(`(?i)\b(?:morning commute|often|some days?|sometimes|around)\b`)
	supersededFrontmatterRe       = regexp.MustCompile(`(?im)(?:^\s*superseded_by:\s*\S+|\bsuperseded by\b|\breplaced by\b|\bstatus\s*:\s*superseded\b|\bno longer current\b)`)
)

// retrievalIntent captures the outcome of the regex-driven intent
// detection step. One per query; reused across every candidate.
type retrievalIntent struct {
	preferenceQuery   bool
	concreteFactQuery bool
}

// label returns a compact trace string describing the detected
// intent. Empty when the query is English but matched nothing.
func (r retrievalIntent) label() string {
	parts := make([]string, 0, 2)
	if r.preferenceQuery {
		parts = append(parts, "preference")
	}
	if r.concreteFactQuery {
		parts = append(parts, "concrete-fact")
	}
	return strings.Join(parts, "+")
}

// detectRetrievalIntent applies the spec's patterns to the lowercased
// query. Non-English queries bypass every regex and therefore always
// report both intents as false, which is the intended v1.0
// behaviour.
func detectRetrievalIntent(query string) retrievalIntent {
	normalised := strings.ToLower(query)
	return retrievalIntent{
		preferenceQuery: preferenceQueryRe.MatchString(normalised),
		concreteFactQuery: enumerationOrTotalQueryRe.MatchString(normalised) ||
			propertyLookupQueryRe.MatchString(normalised) ||
			len(deriveActionDateProbes(query)) > 0 ||
			hasSpecificRecallCue(normalised) ||
			(firstPersonConcreteQueryRe.MatchString(normalised) && moneyEventQueryRe.MatchString(normalised)) ||
			(firstPersonFactLookupRe.MatchString(normalised) && factLookupVerbRe.MatchString(normalised)),
	}
}

func hasSpecificRecallCue(normalisedQuery string) bool {
	if normalisedQuery == "" {
		return false
	}
	if !strings.Contains(normalisedQuery, "remind me") && !strings.Contains(normalisedQuery, "remember") {
		return false
	}
	return strings.Contains(normalisedQuery, "the specific") || strings.Contains(normalisedQuery, "the exact")
}

// reweightSharedMemoryRanking applies multiplicative score adjustments
// to fused results when the query pattern matches. The tie-breaker
// on the final sort reuses the original fused rank so ports stay
// deterministic.
func reweightSharedMemoryRanking(query string, results []RetrievedChunk) []RetrievedChunk {
	if len(results) == 0 {
		return results
	}
	intent := detectRetrievalIntent(query)
	if !intent.preferenceQuery && !intent.concreteFactQuery {
		return results
	}

	type indexed struct {
		chunk RetrievedChunk
		index int
	}
	copied := make([]indexed, len(results))
	for i := range results {
		r := results[i]
		r.Score = r.Score * retrievalIntentMultiplier(intent, query, r)
		copied[i] = indexed{chunk: r, index: i}
	}

	sort.SliceStable(copied, func(i, j int) bool {
		if copied[i].chunk.Score != copied[j].chunk.Score {
			return copied[i].chunk.Score > copied[j].chunk.Score
		}
		return copied[i].index < copied[j].index
	})

	out := make([]RetrievedChunk, len(copied))
	for i, c := range copied {
		out[i] = c.chunk
	}
	out = diversifyCompositeConcreteRanking(query, out)
	out = diversifyAggregateEvidenceRanking(query, out)
	out = diversifyPreferencePersonalContextRanking(query, out)
	return diversifyDateDifferenceRanking(query, out)
}

func retrievalIntentMultiplier(intent retrievalIntent, query string, r RetrievedChunk) float64 {
	multiplier := 1.0
	text := retrievalResultText(r)
	if intent.preferenceQuery {
		multiplier *= preferenceIntentMultiplier(query, r, text)
	}
	if intent.concreteFactQuery {
		multiplier *= concreteFactIntentMultiplier(query, r, text)
		multiplier *= focusAlignedConcreteFactMultiplier(query, text)
		multiplier *= staleSupersededConcreteFactMultiplier(r, text)
		multiplier *= firstPersonConcreteFactMultiplier(query, r, text)
	}
	return multiplier
}

func preferenceIntentMultiplier(query string, r RetrievedChunk, text string) float64 {
	path := strings.ToLower(r.Path)
	sourceRole := chunkSourceRole(r)
	isGlobalPreferenceNote := strings.Contains(path, "memory/global/") &&
		(strings.Contains(path, "user-preference-") || preferenceNoteRe.MatchString(text))
	if isGlobalPreferenceNote {
		if strings.Contains(path, "user-preference-") {
			return 2.35
		}
		return 2.1
	}
	if sourceRole == "assistant" && assistantOutputQuestion(strings.ToLower(query)) {
		return 1.35
	}
	if !strings.Contains(path, "memory/global/") && genericNoteRe.MatchString(text) {
		return 0.82
	}
	if rollupNoteRe.MatchString(text) {
		return 0.9
	}
	return 1.0
}

func concreteFactIntentMultiplier(query string, r RetrievedChunk, text string) float64 {
	path := strings.ToLower(r.Path)
	isRollUp := rollupNoteRe.MatchString(text)
	isQuestionLikeNote := questionLikeNoteRe.MatchString(text) && genericNoteRe.MatchString(text)
	isConcreteFact := isConcreteFactLike(path, text)

	multiplier := 1.0
	if isConcreteFact {
		multiplier *= 2.2
	}
	if len(deriveActionDateProbes(query)) > 0 {
		if bodyAbsoluteDateRe.MatchString(text) {
			multiplier *= 1.45
		} else {
			multiplier *= 0.78
		}
	}
	if durationQueryRe.MatchString(query) {
		if measurementValueRe.MatchString(text) {
			multiplier *= 1.35
		} else {
			multiplier *= 0.72
		}
	}
	if measurementTotalQuery(query) {
		if measurementValueRe.MatchString(text) {
			multiplier *= 1.55
		} else if genericNoteRe.MatchString(text) || questionLikeNoteRe.MatchString(text) {
			multiplier *= 0.68
		}
	}
	if moneyEventQueryRe.MatchString(query) {
		if currencyValueRe.MatchString(text) {
			multiplier *= 1.45
		} else if genericNoteRe.MatchString(text) || questionLikeNoteRe.MatchString(text) {
			multiplier *= 0.62
		}
	}
	if isQuestionLikeNote {
		multiplier *= 0.45
	}
	if isRollUp {
		multiplier *= 0.45
	}
	if !isConcreteFact && !strings.Contains(path, "memory/global/") && genericNoteRe.MatchString(text) {
		multiplier *= 0.75
	}
	return multiplier
}

func measurementTotalQuery(query string) bool {
	return enumerationOrTotalQueryRe.MatchString(query) && measurementUnitQueryRe.MatchString(query)
}

func focusAlignedConcreteFactMultiplier(query, text string) float64 {
	phrases := buildPriorityBM25Queries(query)
	if len(phrases) == 0 {
		phrases = filteredPhraseProbes(query)
	}
	if len(phrases) == 0 {
		return 1.0
	}

	loweredText := normaliseFocusAlignmentText(text)
	best := 0.0
	for _, phrase := range phrases {
		if score := focusAlignmentScore(loweredText, phrase); score > best {
			best = score
		}
	}
	if isTypeContextCountQuery(query) && best < 0.25 {
		return 0.18
	}

	switch {
	case best >= 0.99:
		return 1.6
	case best >= 0.66:
		return 1.25
	default:
		return 1.0
	}
}

func firstPersonConcreteFactMultiplier(query string, r RetrievedChunk, text string) float64 {
	normalisedQuery := strings.ToLower(strings.TrimSpace(query))
	if normalisedQuery == "" {
		return 1.0
	}
	if !firstPersonConcreteQueryRe.MatchString(normalisedQuery) && !firstPersonFactLookupRe.MatchString(normalisedQuery) {
		return 1.0
	}

	path := strings.ToLower(r.Path)
	sourceRole := chunkSourceRole(r)
	isGlobal := strings.Contains(path, "memory/global/")
	isDirectFact := isConcreteFactLike(path, text)
	isPersonalMemory := strings.Contains(path, "user") ||
		strings.Contains(text, "the user ") ||
		strings.Contains(text, "user ")
	multiplier := 1.0
	switch {
	case isGlobal && isDirectFact:
		multiplier *= 1.35
	case isDirectFact && isPersonalMemory:
		multiplier *= 1.25
	case isGlobal:
		multiplier *= 1.22
	case isDirectFact:
		multiplier *= 0.88
	default:
		multiplier *= 0.58
	}
	if !isGlobal && genericNoteRe.MatchString(text) {
		multiplier *= 0.82
	}
	switch sourceRole {
	case "user":
		multiplier *= 1.18
	case "assistant":
		if !assistantOutputQuestion(normalisedQuery) {
			multiplier *= 0.72
		}
	}
	if durationQueryRe.MatchString(normalisedQuery) && routineScopeQueryRe.MatchString(normalisedQuery) {
		if routineScopeNoteRe.MatchString(text) {
			multiplier *= 1.25
		}
		if segmentQualifierNoteRe.MatchString(text) && !routineScopeNoteRe.MatchString(text) {
			multiplier *= 0.15
		}
	}
	return multiplier
}

func assistantOutputQuestion(normalisedQuery string) bool {
	if normalisedQuery == "" {
		return false
	}
	return strings.Contains(normalisedQuery, "recommended") ||
		strings.Contains(normalisedQuery, "suggested") ||
		strings.Contains(normalisedQuery, "you recommended") ||
		strings.Contains(normalisedQuery, "you suggested") ||
		strings.Contains(normalisedQuery, "previous chat") ||
		strings.Contains(normalisedQuery, "previous conversation") ||
		strings.Contains(normalisedQuery, "remind me")
}

func staleSupersededConcreteFactMultiplier(r RetrievedChunk, text string) float64 {
	if metadataHasNonEmptyString(r.Metadata, "superseded_by", "supersededBy") || supersededFrontmatterRe.MatchString(text) {
		return 0.18
	}
	return 1.0
}

func metadataHasNonEmptyString(metadata map[string]any, keys ...string) bool {
	return metadataStringValue(metadata, keys...) != ""
}

func metadataStringValue(metadata map[string]any, keys ...string) string {
	if metadata == nil {
		return ""
	}
	for _, key := range keys {
		value, ok := metadata[key]
		if !ok {
			continue
		}
		text, ok := value.(string)
		if ok && strings.TrimSpace(text) != "" {
			return strings.TrimSpace(text)
		}
	}
	return ""
}

func chunkSourceRole(r RetrievedChunk) string {
	role := metadataStringValue(r.Metadata, "source_role", "sourceRole")
	switch strings.ToLower(strings.TrimSpace(role)) {
	case "user", "assistant", "mixed":
		return strings.ToLower(strings.TrimSpace(role))
	default:
		return ""
	}
}

func isConcreteFactLike(path, text string) bool {
	if strings.Contains(path, "user-fact-") || strings.Contains(path, "milestone-") {
		return true
	}
	if rollupNoteRe.MatchString(text) {
		return false
	}
	return dateTagRe.MatchString(text) || atomicEventNoteRe.MatchString(text)
}

type compositeFocusMatch struct {
	index int
	score float64
}

func diversifyCompositeConcreteRanking(query string, results []RetrievedChunk) []RetrievedChunk {
	focuses := filteredPhraseProbes(query)
	if len(results) < 3 || len(focuses) < 2 || !isCompositeConcreteQuery(query) {
		return results
	}

	primary := make([]RetrievedChunk, 0, len(focuses))
	secondary := make([]RetrievedChunk, 0, len(results))
	nearMisses := make([]RetrievedChunk, 0, len(results))
	duplicates := make([]RetrievedChunk, 0, len(results))
	covered := make(map[int]bool, len(focuses))
	for _, result := range results {
		match := bestCompositeFocusMatch(retrievalResultText(result), focuses)
		if match.index >= 0 && match.score >= 0.5 {
			if !covered[match.index] {
				primary = append(primary, result)
				covered[match.index] = true
				continue
			}
			duplicates = append(duplicates, result)
			continue
		}
		if match.index >= 0 && match.score >= 0.25 {
			nearMisses = append(nearMisses, result)
			continue
		}
		secondary = append(secondary, result)
	}
	out := append(primary, secondary...)
	out = append(out, nearMisses...)
	out = append(out, duplicates...)
	return out
}

func diversifyAggregateEvidenceRanking(query string, results []RetrievedChunk) []RetrievedChunk {
	if len(results) < 2 || !isAggregateEvidenceQuery(query) || isTypeContextCountQuery(query) {
		return results
	}
	focuses := aggregateFocusProbes(query)
	type candidate struct {
		index    int
		chunk    RetrievedChunk
		session  string
		focus    int
		strength float64
	}
	candidates := make([]candidate, 0, len(results))
	for i, result := range results {
		text := retrievalResultText(result)
		focus, focusScore := bestAggregateFocus(text, focuses)
		strength := aggregateEvidenceStrength(query, result, text, focusScore)
		if strength < 1.25 {
			continue
		}
		candidates = append(candidates, candidate{
			index:    i,
			chunk:    result,
			session:  chunkSessionID(result),
			focus:    focus,
			strength: strength,
		})
	}
	if len(candidates) == 0 {
		return results
	}
	sort.SliceStable(candidates, func(i, j int) bool {
		if candidates[i].strength != candidates[j].strength {
			return candidates[i].strength > candidates[j].strength
		}
		return candidates[i].index < candidates[j].index
	})

	const maxAggregateCoveragePromotions = 8
	selected := make(map[int]bool, len(candidates))
	coveredSessions := make(map[string]bool, len(candidates))
	coveredFocuses := make(map[int]bool, len(focuses))
	primary := make([]RetrievedChunk, 0, maxAggregateCoveragePromotions)
	for _, c := range candidates {
		if len(primary) == maxAggregateCoveragePromotions {
			break
		}
		sessionCovered := c.session != "" && coveredSessions[c.session]
		focusCovered := c.focus >= 0 && coveredFocuses[c.focus]
		if (sessionCovered || focusCovered) && c.strength < 3.4 {
			continue
		}
		selected[c.index] = true
		if c.session != "" {
			coveredSessions[c.session] = true
		}
		if c.focus >= 0 {
			coveredFocuses[c.focus] = true
		}
		primary = append(primary, c.chunk)
	}
	if len(primary) == 0 || (len(primary) == 1 && candidates[0].strength < 2.0) {
		return results
	}
	out := make([]RetrievedChunk, 0, len(results))
	out = append(out, primary...)
	for i, result := range results {
		if selected[i] {
			continue
		}
		out = append(out, result)
	}
	return out
}

func isAggregateEvidenceQuery(query string) bool {
	lowered := strings.ToLower(strings.TrimSpace(query))
	if lowered == "" {
		return false
	}
	return enumerationOrTotalQueryRe.MatchString(lowered) ||
		(moneyEventQueryRe.MatchString(lowered) && valueComparisonQueryRe.MatchString(lowered)) ||
		(measurementUnitQueryRe.MatchString(lowered) && valueComparisonQueryRe.MatchString(lowered))
}

func aggregateFocusProbes(query string) []string {
	out := make([]string, 0, maxCoverageFacetQueries+maxDerivedSubQueries)
	out = append(out, deriveCoverageFacetQueries(query)...)
	out = append(out, filteredPhraseProbes(query)...)
	out = append(out, deriveMoneyFocusProbes(query)...)
	return dedupeTrimmedStrings(out)
}

func bestAggregateFocus(text string, focuses []string) (int, float64) {
	if len(focuses) == 0 {
		return -1, 0
	}
	match := bestCompositeFocusMatch(text, focuses)
	return match.index, match.score
}

func aggregateEvidenceStrength(query string, r RetrievedChunk, text string, focusScore float64) float64 {
	loweredQuery := strings.ToLower(strings.TrimSpace(query))
	path := strings.ToLower(r.Path)
	hasFocuses := len(aggregateFocusProbes(query)) > 0
	strength := 0.0
	switch {
	case focusScore >= 0.99:
		strength += 2.0
	case focusScore >= 0.66:
		strength += 1.35
	case focusScore >= 0.5:
		strength += 0.8
	case focusScore >= 0.25:
		strength += 0.35
	case !hasFocuses:
		strength += 0.6
	}
	if moneyEventQueryRe.MatchString(loweredQuery) && currencyValueRe.MatchString(text) {
		strength += 2.4
	} else if measurementTotalQuery(loweredQuery) && measurementValueRe.MatchString(text) {
		strength += 2.0
	} else if numericValueRe.MatchString(text) {
		strength += 1.0
	}
	if isConcreteFactLike(path, text) {
		strength += 1.2
	}
	if strings.Contains(path, "memory/global/") {
		strength += 0.65
	}
	switch chunkSourceRole(r) {
	case "user":
		strength += 0.8
	case "assistant":
		if !assistantOutputQuestion(loweredQuery) {
			strength -= 0.6
		}
	}
	if genericNoteRe.MatchString(text) {
		strength -= 0.45
	}
	if questionLikeNoteRe.MatchString(text) {
		strength -= 0.75
	}
	if rollupNoteRe.MatchString(text) {
		strength -= 0.35
	}
	if hasFocuses && focusScore < 0.25 && !currencyValueRe.MatchString(text) && !measurementValueRe.MatchString(text) {
		strength -= 1.5
	}
	return strength
}

func diversifyDateDifferenceRanking(query string, results []RetrievedChunk) []RetrievedChunk {
	if len(results) < 3 {
		return results
	}
	match := dateArithmeticWhenRe.FindStringSubmatch(query)
	if len(match) < 3 {
		return results
	}
	events := [][]string{
		questionTokens(match[1]),
		questionTokens(match[2]),
	}
	return promoteMatchingCoverage(results, len(events), func(result RetrievedChunk) int {
		text := retrievalResultText(result)
		for i, tokens := range events {
			if matchesEventEvidence(text, tokens) {
				return i
			}
		}
		return -1
	})
}

func promoteMatchingCoverage(results []RetrievedChunk, slots int, matchIndex func(RetrievedChunk) int) []RetrievedChunk {
	if slots <= 0 {
		return results
	}
	primary := make([]RetrievedChunk, 0, slots)
	secondary := make([]RetrievedChunk, 0, len(results))
	covered := map[int]bool{}
	for _, result := range results {
		idx := matchIndex(result)
		if idx >= 0 && !covered[idx] {
			covered[idx] = true
			primary = append(primary, result)
			continue
		}
		secondary = append(secondary, result)
	}
	if len(primary) == 0 {
		return results
	}
	out := append(primary, secondary...)
	return out
}

func matchesEventEvidence(text string, tokens []string) bool {
	if len(tokens) == 0 {
		return false
	}
	for i, token := range tokens {
		if token == "" {
			continue
		}
		if strings.Contains(text, token) {
			continue
		}
		if i == 0 {
			past := inflectProbeVerbPast(token)
			if past != "" && strings.Contains(text, past) {
				continue
			}
		}
		return false
	}
	return true
}

func diversifyPreferencePersonalContextRanking(query string, results []RetrievedChunk) []RetrievedChunk {
	if len(results) < 3 || !preferenceQueryRe.MatchString(query) {
		return results
	}
	focuses := buildPriorityBM25Queries(query)
	if len(focuses) == 0 {
		focuses = filteredPhraseProbes(query)
	}
	if len(focuses) == 0 {
		return results
	}

	primary := make([]RetrievedChunk, 0, len(focuses))
	secondary := make([]RetrievedChunk, 0, len(results))
	covered := make(map[int]bool, len(focuses))
	for _, result := range results {
		text := retrievalResultText(result)
		match := bestCompositeFocusMatch(text, focuses)
		if isDirectPersonalContextEvidence(result, text) && match.index >= 0 && match.score >= 0.5 && !covered[match.index] {
			covered[match.index] = true
			primary = append(primary, result)
			continue
		}
		secondary = append(secondary, result)
	}
	if len(primary) == 0 {
		return results
	}
	out := append(primary, secondary...)
	return out
}

func isDirectPersonalContextEvidence(r RetrievedChunk, text string) bool {
	path := strings.ToLower(r.Path)
	if strings.Contains(path, "user-preference-") {
		return false
	}
	if strings.Contains(path, "memory/project/") && genericNoteRe.MatchString(text) {
		return false
	}
	return strings.Contains(path, "memory/global/") &&
		(strings.Contains(path, "user-fact-") || strings.Contains(path, "user_") || strings.Contains(path, "user-") ||
			strings.Contains(text, "the user ") || strings.Contains(text, "user ") || strings.Contains(text, "i "))
}

func isCompositeConcreteQuery(query string) bool {
	lowered := strings.ToLower(strings.TrimSpace(query))
	hasCompositeSeparator := strings.Contains(lowered, " and ") || strings.Contains(lowered, " plus ") || strings.Contains(lowered, " or ")
	if !hasCompositeSeparator {
		return false
	}
	return enumerationOrTotalQueryRe.MatchString(lowered) ||
		propertyLookupQueryRe.MatchString(lowered) ||
		firstPersonFactLookupRe.MatchString(lowered)
}

func bestCompositeFocusMatch(text string, focuses []string) compositeFocusMatch {
	best := compositeFocusMatch{index: -1, score: 0}
	loweredText := normaliseFocusAlignmentText(text)
	for i, focus := range focuses {
		score := focusAlignmentScore(loweredText, focus)
		if score > best.score {
			best = compositeFocusMatch{index: i, score: score}
		}
	}
	return best
}

func focusAlignmentScore(loweredText, phrase string) float64 {
	loweredPhrase := normaliseFocusAlignmentText(phrase)
	if loweredPhrase == "" {
		return 0
	}
	if strings.Contains(loweredText, loweredPhrase) {
		return 1.0
	}

	textTokens := tokenSetWithSingulars(strings.Fields(loweredText))
	phraseTokens := singularFocusTokens(strings.Fields(loweredPhrase))
	if len(phraseTokens) == 0 {
		return 0
	}

	matched := 0
	for _, token := range phraseTokens {
		if textTokens[token] {
			matched++
		}
	}
	return float64(matched) / float64(len(phraseTokens))
}

func tokenSetWithSingulars(tokens []string) map[string]bool {
	out := make(map[string]bool, len(tokens)*2)
	for _, token := range tokens {
		if token == "" {
			continue
		}
		out[token] = true
		if singular := singularProbeEntityToken(token); singular != "" {
			out[singular] = true
		}
	}
	return out
}

func singularFocusTokens(tokens []string) []string {
	out := make([]string, 0, len(tokens))
	for _, token := range tokens {
		if token == "" {
			continue
		}
		singular := singularProbeEntityToken(token)
		if singular == "" {
			singular = token
		}
		out = append(out, singular)
	}
	return out
}

func normaliseFocusAlignmentText(raw string) string {
	if raw == "" {
		return ""
	}
	replacer := strings.NewReplacer(
		"-", " ",
		"/", " ",
		"(", " ",
		")", " ",
		"[", " ",
		"]", " ",
		",", " ",
		".", " ",
		":", " ",
		";", " ",
		"?", " ",
		"!", " ",
	)
	return strings.Join(strings.Fields(strings.ToLower(replacer.Replace(raw))), " ")
}

func tokenSet(tokens []string) map[string]bool {
	out := make(map[string]bool, len(tokens))
	for _, token := range tokens {
		if token == "" {
			continue
		}
		out[token] = true
	}
	return out
}

func retrievalResultText(r RetrievedChunk) string {
	var b strings.Builder
	b.WriteString(r.Path)
	b.WriteByte('\n')
	b.WriteString(r.Title)
	b.WriteByte('\n')
	b.WriteString(r.Summary)
	b.WriteByte('\n')
	b.WriteString(r.Text)
	return strings.ToLower(b.String())
}
