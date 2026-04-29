// SPDX-License-Identifier: Apache-2.0

package retrieval

import (
	"regexp"
	"sort"
	"strings"
	"time"
)

type statePromotionTrace struct {
	Intent     bool
	Promotions int
}

type statePromotionCandidate struct {
	chunk RetrievedChunk
	index int
	score int
}

var (
	currentStateQueryRe = regexp.MustCompile(`(?i)\b(?:current(?:ly)?|now|still|these days|at the moment|where do i (?:go|take|attend)|where am i (?:going|taking|attending)|what .* am i (?:learning|playing|using)|which .* am i (?:using|taking|attending)|how many .* do i own|what .* do i own|which .* do i own|do i own)\b`)
	stateEvidenceRe     = regexp.MustCompile(`(?i)\b(?:current(?:ly)?|now|still|own(?:s)?|have|has|use|attend|go to|take|learn(?:ing)?|play(?:ing)?|prefer|favo(?:u)?rite|settled on)\b`)
	staleStateRe        = regexp.MustCompile(`(?i)\b(?:used to|previously|formerly|old|no longer|sold|cancelled|canceled|stopped|replaced|switched away|planning|considering|interested in|might|maybe)\b`)
	stateWordRe         = regexp.MustCompile(`[a-z0-9][a-z0-9'-]*`)
)

var stateRankStopWords = map[string]bool{
	"about": true, "and": true, "are": true, "current": true, "currently": true,
	"does": true, "have": true, "how": true, "many": true, "now": true,
	"own": true, "owns": true, "still": true, "that": true, "the": true,
	"these": true, "what": true, "where": true, "which": true, "with": true,
}

func promoteCurrentStateEvidence(query, questionDate string, chunks []RetrievedChunk) ([]RetrievedChunk, statePromotionTrace) {
	if len(chunks) <= 1 || !currentStateQueryRe.MatchString(query) {
		return chunks, statePromotionTrace{}
	}
	queryTokens := stateTokenSet(query)
	if len(queryTokens) == 0 {
		return chunks, statePromotionTrace{Intent: true}
	}
	anchor, hasAnchor := parseStateAnchor(questionDate)

	candidates := make([]statePromotionCandidate, 0, len(chunks))
	for i, chunk := range chunks {
		score := stateEvidenceScore(chunk, queryTokens, anchor, hasAnchor)
		candidates = append(candidates, statePromotionCandidate{chunk: chunk, index: i, score: score})
	}

	sort.SliceStable(candidates, func(i, j int) bool {
		if candidates[i].score != candidates[j].score {
			return candidates[i].score > candidates[j].score
		}
		return candidates[i].index < candidates[j].index
	})

	out := make([]RetrievedChunk, len(candidates))
	promotions := 0
	for i, candidate := range candidates {
		out[i] = candidate.chunk
		if candidate.index != i && candidate.score > 0 {
			promotions++
		}
	}
	return out, statePromotionTrace{Intent: true, Promotions: promotions}
}

func stateEvidenceScore(chunk RetrievedChunk, queryTokens map[string]struct{}, anchor time.Time, hasAnchor bool) int {
	text := retrievalResultText(chunk)
	lower := strings.ToLower(text)
	stateKey := metadataStringValue(chunk.Metadata, "state_key", "stateKey")
	stateKind := metadataStringValue(chunk.Metadata, "state_kind", "stateKind")
	stateSubject := metadataStringValue(chunk.Metadata, "state_subject", "stateSubject")
	claimStatus := metadataStringValue(chunk.Metadata, "claim_status", "claimStatus")

	score := 0
	if stateKey != "" {
		score += 8
	}
	if stateKind != "" {
		score += 3
	}
	if stateSubject != "" {
		score += sharedStateTokenCount(queryTokens, stateTokenSet(stateSubject)) * 3
	}
	if stateEvidenceRe.MatchString(lower) {
		score += 2
	}
	score += sharedStateTokenCount(queryTokens, stateTokenSet(lower))

	if strings.EqualFold(claimStatus, "superseded") {
		score -= 8
	}
	if staleStateRe.MatchString(lower) {
		score -= 5
	}
	if hasAnchor && !stateValidityContains(chunk, anchor) {
		return 0
	}
	if score < 0 {
		return 0
	}
	return score
}

func stateValidityContains(chunk RetrievedChunk, anchor time.Time) bool {
	fromRaw := metadataStringValue(chunk.Metadata, "valid_from", "validFrom")
	toRaw := metadataStringValue(chunk.Metadata, "valid_to", "validTo")
	from, hasFrom := parseStateAnchor(fromRaw)
	to, hasTo := parseStateAnchor(toRaw)
	if hasFrom && anchor.Before(from) {
		return false
	}
	if hasTo && anchor.After(to) {
		return false
	}
	return true
}

func parseStateAnchor(raw string) (time.Time, bool) {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return time.Time{}, false
	}
	for _, layout := range []string{
		time.RFC3339,
		"2006-01-02",
		"2006/01/02",
		"2006/01/02 (Mon) 15:04",
		"2006/01/02 15:04",
		"2006-01-02 15:04",
	} {
		if parsed, err := time.Parse(layout, raw); err == nil {
			return parsed, true
		}
	}
	if len(raw) >= 10 {
		return parseStateAnchor(raw[:10])
	}
	return time.Time{}, false
}

func stateTokenSet(text string) map[string]struct{} {
	out := make(map[string]struct{})
	for _, token := range stateWordRe.FindAllString(strings.ToLower(text), -1) {
		token = strings.Trim(token, "'-")
		if len(token) < 3 || stateRankStopWords[token] {
			continue
		}
		out[token] = struct{}{}
	}
	return out
}

func sharedStateTokenCount(a, b map[string]struct{}) int {
	if len(a) == 0 || len(b) == 0 {
		return 0
	}
	count := 0
	for token := range a {
		if _, ok := b[token]; ok {
			count++
		}
	}
	return count
}
