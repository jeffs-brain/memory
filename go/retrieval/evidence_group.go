// SPDX-License-Identifier: Apache-2.0

package retrieval

import (
	"fmt"
	"math"
	"regexp"
	"sort"
	"strconv"
	"strings"
)

type evidenceKind string

const (
	evidenceKindAtomic  evidenceKind = "atomic"
	evidenceKindRollup  evidenceKind = "rollup"
	evidenceKindRecap   evidenceKind = "recap"
	evidenceKindPlan    evidenceKind = "plan"
	evidenceKindUnknown evidenceKind = "unknown"
)

type aggregateEvidenceTrace struct {
	Groups     int
	Suppressed int
}

type aggregateEvidenceInfo struct {
	chunk       RetrievedChunk
	kind        evidenceKind
	group       string
	metric      string
	action      string
	amounts     []evidenceAmount
	focusTokens map[string]struct{}
}

type evidenceAmount struct {
	metric   string
	unit     string
	currency string
	value    int64
}

var (
	aggregateEvidenceQueryRe = regexp.MustCompile(`(?i)\b(?:total|sum|combined|altogether|overall|in\s+total|how\s+many|how\s+much|amount|count|spent|cost|raised|earned|donated)\b`)
	aggregatePlanQueryRe     = regexp.MustCompile(`(?i)\b(?:plan|planned|budget|budgeted|expect|expected|intend|intended|estimate|estimated|target|goal)\b`)
	evidenceMoneyRe          = regexp.MustCompile(`([$£€])\s?(\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)`)
	evidenceQuantityRe       = regexp.MustCompile(`(?i)\b(\d{1,6}(?:\.\d+)?)\s+(hours?|hrs?|days?|weeks?|months?|items?|books?|issues?|sessions?|classes?|lessons?|rides?|trips?|events?|songs?|recipes?|gifts?|donations?)\b`)
	evidenceWordRe           = regexp.MustCompile(`[a-z0-9][a-z0-9'-]*`)
)

var evidenceActionPatterns = []struct {
	action string
	re     *regexp.Regexp
}{
	{"donation", regexp.MustCompile(`(?i)\b(?:donat(?:e|ed|ion|ions)|fundrais(?:e|ed|ing)|raised|pledged)\b`)},
	{"purchase", regexp.MustCompile(`(?i)\b(?:spent|cost|paid|bought|purchased|ordered)\b`)},
	{"gift", regexp.MustCompile(`(?i)\b(?:gift|gifts|gave|sent|present|presents)\b`)},
	{"sale", regexp.MustCompile(`(?i)\b(?:sold|sale|resold)\b`)},
	{"earning", regexp.MustCompile(`(?i)\b(?:earned|income|revenue|made)\b`)},
	{"completion", regexp.MustCompile(`(?i)\b(?:completed|finished|read|watched|attended|visited|rode|tried)\b`)},
}

var evidenceStopWords = map[string]bool{
	"a": true, "about": true, "all": true, "also": true, "am": true, "an": true,
	"and": true, "are": true, "as": true, "at": true, "be": true, "been": true,
	"by": true, "count": true, "current": true, "currently": true, "date": true,
	"did": true, "do": true, "does": true, "for": true, "from": true, "had": true,
	"has": true, "have": true, "how": true, "i": true, "in": true, "is": true,
	"it": true, "me": true, "my": true, "of": true, "on": true, "overall": true,
	"recap": true, "same": true, "session": true, "so": true, "summary": true,
	"that": true, "the": true, "this": true, "to": true, "total": true, "was": true,
	"were": true, "what": true, "with": true,
	"spent": true, "spend": true, "paid": true, "pay": true, "bought": true,
	"buy": true, "purchased": true, "purchase": true, "ordered": true,
	"order": true, "donated": true, "donate": true, "raised": true,
	"raise": true, "earned": true, "earn": true, "sold": true, "sale": true,
	"completed": true, "finished": true, "attended": true, "visited": true,
}

func groupAggregateEvidence(query string, chunks []RetrievedChunk) ([]RetrievedChunk, aggregateEvidenceTrace) {
	if len(chunks) <= 1 || !shouldGroupAggregateEvidence(query) {
		return chunks, aggregateEvidenceTrace{}
	}

	infos := make([]aggregateEvidenceInfo, 0, len(chunks))
	for _, chunk := range chunks {
		info := classifyAggregateEvidence(chunk)
		info.chunk = annotateAggregateEvidence(chunk, info)
		infos = append(infos, info)
	}

	atomic := make([]aggregateEvidenceInfo, 0, len(infos))
	for _, info := range infos {
		if info.kind == evidenceKindAtomic {
			atomic = append(atomic, info)
		}
	}

	trace := aggregateEvidenceTrace{Groups: countEvidenceGroups(infos)}
	if len(atomic) == 0 {
		out := make([]RetrievedChunk, 0, len(infos))
		for _, info := range infos {
			out = append(out, info.chunk)
		}
		return out, trace
	}

	wantsPlans := aggregatePlanQueryRe.MatchString(query)
	out := make([]RetrievedChunk, 0, len(infos))
	for _, info := range infos {
		if shouldSuppressAggregateEvidence(info, atomic, wantsPlans) {
			trace.Suppressed++
			continue
		}
		out = append(out, info.chunk)
	}
	return out, trace
}

func shouldGroupAggregateEvidence(query string) bool {
	return aggregateEvidenceQueryRe.MatchString(query)
}

func classifyAggregateEvidence(chunk RetrievedChunk) aggregateEvidenceInfo {
	text := retrievalResultText(chunk)
	lower := strings.ToLower(text)
	amounts := extractEvidenceAmounts(lower)
	action := classifyEvidenceAction(lower)
	kind := evidenceKindUnknown

	switch {
	case isPlanEvidence(lower):
		kind = evidenceKindPlan
	case isRollupEvidence(lower):
		kind = evidenceKindRollup
	case isRecapEvidence(lower):
		kind = evidenceKindRecap
	case len(amounts) > 0 && action != "":
		kind = evidenceKindAtomic
	case len(amounts) > 0 && hasAtomicPastSignal(lower):
		kind = evidenceKindAtomic
	}

	metric := ""
	if len(amounts) > 0 {
		metric = amounts[0].metric
	}
	group := buildEvidenceGroup(chunk, kind, metric, action, amounts, lower)
	return aggregateEvidenceInfo{
		kind:        kind,
		group:       group,
		metric:      metric,
		action:      action,
		amounts:     amounts,
		focusTokens: evidenceFocusTokens(lower),
	}
}

func annotateAggregateEvidence(chunk RetrievedChunk, info aggregateEvidenceInfo) RetrievedChunk {
	if info.kind == evidenceKindUnknown {
		return chunk
	}
	meta := cloneChunkMetadata(chunk.Metadata)
	meta["evidence_kind"] = string(info.kind)
	if info.group != "" {
		meta["evidence_group"] = info.group
	}
	chunk.Metadata = meta
	return chunk
}

func isRollupEvidence(text string) bool {
	return strings.Contains(text, "total") ||
		strings.Contains(text, "overall") ||
		strings.Contains(text, "so far") ||
		strings.Contains(text, "combined") ||
		strings.Contains(text, "altogether") ||
		strings.Contains(text, "in all")
}

func isRecapEvidence(text string) bool {
	return strings.Contains(text, "recap") ||
		strings.Contains(text, "summary") ||
		strings.Contains(text, "tracker") ||
		strings.Contains(text, "ledger") ||
		strings.Contains(text, "bookkeeping")
}

func isPlanEvidence(text string) bool {
	if regexp.MustCompile(`(?i)\b(?:spent|paid|bought|purchased|donated|raised|earned|sold|completed|finished)\b`).MatchString(text) {
		return false
	}
	return regexp.MustCompile(`(?i)\b(?:plan(?:s|ned|ning)?|budget(?:ed)?|intend(?:ed|ing)?|expect(?:ed|ing)?|aim(?:ed|ing)?|target|goal|would|could|might|may)\b`).MatchString(text)
}

func hasAtomicPastSignal(text string) bool {
	return regexp.MustCompile(`(?i)\b(?:spent|paid|bought|purchased|donated|raised|earned|sold|completed|finished|gave|sent|received|attended|visited)\b`).MatchString(text)
}

func classifyEvidenceAction(text string) string {
	for _, pattern := range evidenceActionPatterns {
		if pattern.re.MatchString(text) {
			return pattern.action
		}
	}
	return ""
}

func extractEvidenceAmounts(text string) []evidenceAmount {
	var out []evidenceAmount
	for _, match := range evidenceMoneyRe.FindAllStringSubmatch(text, -1) {
		if len(match) < 3 {
			continue
		}
		value, ok := parseDecimalAmountCents(match[2])
		if !ok {
			continue
		}
		out = append(out, evidenceAmount{
			metric:   "money",
			currency: currencyCode(match[1]),
			value:    value,
		})
	}
	for _, match := range evidenceQuantityRe.FindAllStringSubmatch(text, -1) {
		if len(match) < 3 {
			continue
		}
		value, ok := parseDecimalQuantity(match[1])
		if !ok {
			continue
		}
		out = append(out, evidenceAmount{
			metric: "quantity",
			unit:   normaliseEvidenceUnit(match[2]),
			value:  value,
		})
	}
	return out
}

func parseDecimalAmountCents(raw string) (int64, bool) {
	value, ok := parseDecimalQuantity(raw)
	if !ok {
		return 0, false
	}
	return value, true
}

func parseDecimalQuantity(raw string) (int64, bool) {
	cleaned := strings.ReplaceAll(strings.TrimSpace(raw), ",", "")
	if cleaned == "" {
		return 0, false
	}
	parsed, err := strconv.ParseFloat(cleaned, 64)
	if err != nil || math.IsNaN(parsed) || math.IsInf(parsed, 0) {
		return 0, false
	}
	return int64(math.Round(parsed * 100)), true
}

func currencyCode(symbol string) string {
	switch symbol {
	case "$":
		return "usd"
	case "£":
		return "gbp"
	case "€":
		return "eur"
	default:
		return "money"
	}
}

func normaliseEvidenceUnit(unit string) string {
	unit = strings.ToLower(strings.TrimSpace(unit))
	switch unit {
	case "hr", "hrs", "hour", "hours":
		return "hours"
	case "day", "days":
		return "days"
	case "week", "weeks":
		return "weeks"
	case "month", "months":
		return "months"
	case "item", "items":
		return "items"
	case "gift", "gifts":
		return "gifts"
	case "donation", "donations":
		return "donations"
	default:
		return strings.TrimSuffix(unit, "s")
	}
}

func buildEvidenceGroup(chunk RetrievedChunk, kind evidenceKind, metric, action string, amounts []evidenceAmount, text string) string {
	parts := []string{string(kind)}
	if metric != "" {
		parts = append(parts, metric)
	}
	if action != "" {
		parts = append(parts, action)
	}
	if len(amounts) > 0 {
		parts = append(parts, amounts[0].groupSegment())
	}
	if date := aggregateEvidenceDate(chunk); date != "" {
		parts = append(parts, date)
	}
	tokens := sortedEvidenceTokens(evidenceFocusTokens(text), 4)
	if len(tokens) > 0 {
		parts = append(parts, strings.Join(tokens, "-"))
	}
	return strings.Join(parts, "|")
}

func (a evidenceAmount) groupSegment() string {
	switch a.metric {
	case "money":
		return fmt.Sprintf("%s-%d", a.currency, a.value)
	case "quantity":
		return fmt.Sprintf("%s-%d", a.unit, a.value)
	default:
		return fmt.Sprintf("%s-%d", a.metric, a.value)
	}
}

func aggregateEvidenceDate(chunk RetrievedChunk) string {
	for _, key := range []string{"event_date", "eventDate", "observed_on", "observedOn", "session_date", "sessionDate", "modified"} {
		if value := metadataStringValue(chunk.Metadata, key); value != "" {
			if len(value) >= 10 {
				return value[:10]
			}
			return value
		}
	}
	return ""
}

func shouldSuppressAggregateEvidence(info aggregateEvidenceInfo, atomic []aggregateEvidenceInfo, wantsPlans bool) bool {
	if len(info.amounts) != 1 {
		return false
	}
	switch info.kind {
	case evidenceKindRollup, evidenceKindRecap:
		return overlapsAtomicEvidence(info, atomic)
	case evidenceKindPlan:
		return !wantsPlans && overlapsAtomicEvidence(info, atomic)
	default:
		return false
	}
}

func overlapsAtomicEvidence(info aggregateEvidenceInfo, atomic []aggregateEvidenceInfo) bool {
	if len(info.amounts) == 0 {
		return false
	}
	for _, amount := range info.amounts {
		if equalsAnyAtomicAmount(amount, atomic, info) {
			return true
		}
		if equalsAtomicAmountSum(amount, atomic, info) {
			return true
		}
	}
	return false
}

func equalsAnyAtomicAmount(amount evidenceAmount, atomic []aggregateEvidenceInfo, info aggregateEvidenceInfo) bool {
	for _, candidate := range atomic {
		if !compatibleEvidence(info, candidate) {
			continue
		}
		if sharedEvidenceTokens(info.focusTokens, candidate.focusTokens) < 2 {
			continue
		}
		for _, atomicAmount := range candidate.amounts {
			if sameEvidenceAmount(amount, atomicAmount) {
				return true
			}
		}
	}
	return false
}

func equalsAtomicAmountSum(amount evidenceAmount, atomic []aggregateEvidenceInfo, info aggregateEvidenceInfo) bool {
	var total int64
	var count int
	for _, candidate := range atomic {
		if !compatibleEvidence(info, candidate) {
			continue
		}
		for _, atomicAmount := range candidate.amounts {
			if amount.metric == atomicAmount.metric && amount.currency == atomicAmount.currency && amount.unit == atomicAmount.unit {
				total += atomicAmount.value
				count++
			}
		}
	}
	return count > 1 && total == amount.value
}

func compatibleEvidence(a, b aggregateEvidenceInfo) bool {
	if a.metric != "" && b.metric != "" && a.metric != b.metric {
		return false
	}
	shared := sharedEvidenceTokens(a.focusTokens, b.focusTokens)
	if a.action != "" && b.action != "" && a.action == b.action {
		return shared >= 1
	}
	return shared >= 2
}

func sameEvidenceAmount(a, b evidenceAmount) bool {
	return a.metric == b.metric && a.currency == b.currency && a.unit == b.unit && a.value == b.value
}

func evidenceFocusTokens(text string) map[string]struct{} {
	out := make(map[string]struct{})
	for _, raw := range evidenceWordRe.FindAllString(strings.ToLower(text), -1) {
		token := strings.Trim(raw, "'-")
		if len(token) < 3 || evidenceStopWords[token] {
			continue
		}
		out[token] = struct{}{}
	}
	return out
}

func sharedEvidenceTokens(a, b map[string]struct{}) int {
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

func sortedEvidenceTokens(tokens map[string]struct{}, limit int) []string {
	out := make([]string, 0, len(tokens))
	for token := range tokens {
		out = append(out, token)
	}
	sort.Strings(out)
	if limit > 0 && len(out) > limit {
		out = out[:limit]
	}
	return out
}

func countEvidenceGroups(infos []aggregateEvidenceInfo) int {
	seen := make(map[string]bool, len(infos))
	for _, info := range infos {
		if info.kind == evidenceKindUnknown || info.group == "" {
			continue
		}
		seen[info.group] = true
	}
	return len(seen)
}
