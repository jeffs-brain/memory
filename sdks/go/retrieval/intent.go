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
	measurementValueRe            = regexp.MustCompile(`(?i)\b\d+(?:\.\d+)?(?:\s*-\s*\d+(?:\.\d+)?)?(?:\s+|-)(?:minutes?|hours?|days?|weeks?|months?|years?)\b`)
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
	return diversifyCompositeConcreteRanking(query, out)
}

func retrievalIntentMultiplier(intent retrievalIntent, query string, r RetrievedChunk) float64 {
	multiplier := 1.0
	text := retrievalResultText(r)
	if intent.preferenceQuery {
		multiplier *= preferenceIntentMultiplier(r, text)
	}
	if intent.concreteFactQuery {
		multiplier *= concreteFactIntentMultiplier(query, r, text)
		multiplier *= focusAlignedConcreteFactMultiplier(query, text)
		multiplier *= staleSupersededConcreteFactMultiplier(r, text)
		multiplier *= firstPersonConcreteFactMultiplier(query, r, text)
	}
	return multiplier
}

func preferenceIntentMultiplier(r RetrievedChunk, text string) float64 {
	path := strings.ToLower(r.Path)
	isGlobalPreferenceNote := strings.Contains(path, "memory/global/") &&
		(strings.Contains(path, "user-preference-") || preferenceNoteRe.MatchString(text))
	if isGlobalPreferenceNote {
		if strings.Contains(path, "user-preference-") {
			return 2.35
		}
		return 2.1
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

func focusAlignedConcreteFactMultiplier(query, text string) float64 {
	phrases := derivePrioritySubQueries(query)
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
	isGlobal := strings.Contains(path, "memory/global/")
	isDirectFact := isConcreteFactLike(path, text)
	multiplier := 1.0
	switch {
	case isGlobal && isDirectFact:
		multiplier *= 1.35
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

func staleSupersededConcreteFactMultiplier(r RetrievedChunk, text string) float64 {
	if metadataHasNonEmptyString(r.Metadata, "superseded_by", "supersededBy") || supersededFrontmatterRe.MatchString(text) {
		return 0.18
	}
	return 1.0
}

func metadataHasNonEmptyString(metadata map[string]any, keys ...string) bool {
	if metadata == nil {
		return false
	}
	for _, key := range keys {
		value, ok := metadata[key]
		if !ok {
			continue
		}
		text, ok := value.(string)
		if ok && strings.TrimSpace(text) != "" {
			return true
		}
	}
	return false
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

	textTokens := tokenSet(strings.Fields(loweredText))
	phraseTokens := strings.Fields(loweredPhrase)
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
