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
	preferenceQueryRe           = regexp.MustCompile(`(?i)\b(?:recommend|suggest|recommendation|suggestion|tips?|advice|ideas?|what should i|which should i)\b`)
	enumerationOrTotalQueryRe   = regexp.MustCompile(`(?i)\b(?:how many|count|total|in total|sum|add up|list|what are all)\b`)
	firstPersonFactLookupRe     = regexp.MustCompile(`(?i)\b(?:did i|have i|was i|were i)\b`)
	factLookupVerbRe            = regexp.MustCompile(`(?i)\b(?:pick(?:ed)? up|bought|ordered|spent|earned|sold|drove|travelled|traveled|watched|visited|completed|finished|submitted|booked)\b`)
	preferenceNoteRe            = regexp.MustCompile(`(?i)\b(?:prefer(?:s|red)?|like(?:s|d)?|love(?:s|d)?|want(?:s|ed)?|need(?:s|ed)?|avoid(?:s|ed)?|dislike(?:s|d)?|hate(?:s|d)?|enjoy(?:s|ed)?|interested in|looking for)\b`)
	genericNoteRe               = regexp.MustCompile(`(?i)\b(?:tips?|advice|suggest(?:ion|ed)?s?|recommend(?:ation|ed)?s?|ideas?|options?|guide|tracking|tracker|checklist)\b`)
	rollupNoteRe                = regexp.MustCompile(`(?i)\b(?:roll-?up|summary|recap|overview|aggregate|combined|overall|in total|totalled?|totalling)\b`)
	atomicEventNoteRe           = regexp.MustCompile(`(?i)\b(?:i|we)\s+(?:picked up|bought|ordered|spent|earned|sold|drove|travelled|traveled|went|watched|visited|completed|finished|started|booked|got|took|submitted)\b`)
	dateTagRe                   = regexp.MustCompile(`(?i)\[(?:date|observed on):`)
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
			(firstPersonFactLookupRe.MatchString(normalised) && factLookupVerbRe.MatchString(normalised)),
	}
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
		r.Score = r.Score * retrievalIntentMultiplier(intent, r)
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
	return out
}

func retrievalIntentMultiplier(intent retrievalIntent, r RetrievedChunk) float64 {
	multiplier := 1.0
	text := retrievalResultText(r)
	if intent.preferenceQuery {
		multiplier *= preferenceIntentMultiplier(r, text)
	}
	if intent.concreteFactQuery {
		multiplier *= concreteFactIntentMultiplier(r, text)
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

func concreteFactIntentMultiplier(r RetrievedChunk, text string) float64 {
	path := strings.ToLower(r.Path)
	isRollUp := rollupNoteRe.MatchString(text)
	isConcreteFact := strings.Contains(path, "user-fact-") ||
		strings.Contains(path, "milestone-") ||
		(!isRollUp && (dateTagRe.MatchString(text) || atomicEventNoteRe.MatchString(text)))

	multiplier := 1.0
	if isConcreteFact {
		multiplier *= 2.2
	}
	if isRollUp {
		multiplier *= 0.45
	}
	if !isConcreteFact && !strings.Contains(path, "memory/global/") && genericNoteRe.MatchString(text) {
		multiplier *= 0.75
	}
	return multiplier
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
