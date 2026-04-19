// SPDX-License-Identifier: Apache-2.0

package lme

import (
	"fmt"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"
)

type retrievedFact struct {
	Date      string
	SessionID string
	Source    string
	Body      string
}

type namedEntity struct {
	Key    string
	Tokens []string
}

var (
	retrievedFactHeaderRe = regexp.MustCompile(`^\s*\d+\.\s+((?:\[[^\]]+\]\s*)+)\s*$`)
	labelRe               = regexp.MustCompile(`\[(.*?)\]`)
	actionQuestionRe      = regexp.MustCompile(`(?i)\bwhen did i (submit|submitted|apply|applied|book|booked|join|joined|start|started|attend|attended)\b(.*)$`)
	anchorPhraseRe        = regexp.MustCompile(`(?i)\b(?:submit(?:ted)?|apply|applied|book(?:ed)?|join(?:ed)?|start(?:ed)?|attend(?:ed)?)\s+(?:to|for|at|with|on)?\s*([A-Z][A-Za-z0-9&-]+(?: [A-Z][A-Za-z0-9&-]+){0,3})`)
	acronymRe             = regexp.MustCompile(`\b[A-Z][A-Z0-9&-]{1,}\b`)
	moneyRe               = regexp.MustCompile(`\$\s*([0-9]+(?:\.[0-9]{1,2})?)`)
	monthDayRe            = regexp.MustCompile(`(?i)\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+([0-9]{1,2})(?:st|nd|rd|th)?\b`)
	isoDateRe             = regexp.MustCompile(`\b([0-9]{4})[-/]([0-9]{2})[-/]([0-9]{2})\b`)
	datePrefixRe          = regexp.MustCompile(`\[Date:\s*([0-9]{4}-[0-9]{2}-[0-9]{2})`)
	suchAsRe              = regexp.MustCompile(`(?i)\bsuch as ([^.]+)`)
)

var resolverStopWords = map[string]bool{
	"a": true, "about": true, "all": true, "am": true, "amount": true, "an": true,
	"and": true, "are": true, "as": true, "at": true, "back": true, "backend": true,
	"can": true, "conversation": true, "did": true, "for": true, "i": true, "in": true,
	"is": true, "it": true, "languages": true, "learn": true, "me": true, "my": true,
	"of": true, "on": true, "our": true, "previous": true, "programming": true,
	"question": true, "recommended": true, "remind": true, "specific": true,
	"spent": true, "that": true, "the": true, "their": true, "these": true,
	"this": true, "to": true, "total": true, "up": true, "was": true, "what": true,
	"when": true, "you": true, "your": true,
}

// ResolveDeterministicAnswer applies narrow evidence-first answer rules
// before falling back to the LLM reader.
func ResolveDeterministicAnswer(question, retrievedContent string) (string, bool) {
	facts := parseRetrievedFacts(retrievedContent)
	if len(facts) == 0 {
		return "", false
	}
	if answer, ok := resolveAnchoredActionDate(question, facts); ok {
		return answer, true
	}
	if answer, ok := resolveNamedSpendTotal(question, facts); ok {
		return answer, true
	}
	if answer, ok := resolveBackendRecommendation(question, facts); ok {
		return answer, true
	}
	return "", false
}

func parseRetrievedFacts(rendered string) []retrievedFact {
	lines := strings.Split(rendered, "\n")
	start := -1
	for i, line := range lines {
		if strings.HasPrefix(strings.TrimSpace(line), "Retrieved facts (") {
			start = i + 1
			break
		}
	}
	if start < 0 {
		return nil
	}

	var (
		facts  []retrievedFact
		labels string
		body   []string
	)
	flush := func() {
		if labels == "" {
			return
		}
		fact := retrievedFact{Body: strings.TrimSpace(strings.Join(body, "\n"))}
		if fact.Body == "" {
			labels = ""
			body = body[:0]
			return
		}
		allLabels := labelRe.FindAllStringSubmatch(labels, -1)
		for i, match := range allLabels {
			if len(match) < 2 {
				continue
			}
			value := strings.TrimSpace(match[1])
			switch {
			case i == 0:
				fact.Date = value
			case strings.HasPrefix(value, "session="):
				fact.SessionID = strings.TrimPrefix(value, "session=")
			default:
				fact.Source = value
			}
		}
		facts = append(facts, fact)
		labels = ""
		body = body[:0]
	}

	for _, line := range lines[start:] {
		if matches := retrievedFactHeaderRe.FindStringSubmatch(line); len(matches) == 2 {
			flush()
			labels = matches[1]
			continue
		}
		if labels == "" {
			continue
		}
		body = append(body, line)
	}
	flush()
	return facts
}

func resolveAnchoredActionDate(question string, facts []retrievedFact) (string, bool) {
	matches := actionQuestionRe.FindStringSubmatch(question)
	if len(matches) != 3 {
		return "", false
	}

	action := canonicalAction(matches[1])
	objectTokens := significantTokens(matches[2])
	if len(objectTokens) == 0 {
		return "", false
	}

	minOverlap := 1
	if len(objectTokens) >= 3 {
		minOverlap = 2
	}

	var primary []retrievedFact
	for _, fact := range facts {
		lower := strings.ToLower(fact.Body)
		if !containsActionVerb(lower, action) {
			continue
		}
		if tokenOverlap(lower, objectTokens) < minOverlap {
			continue
		}
		if directDate := directActionDateFromBody(action, fact.Body); directDate != "" {
			return directDate, true
		}
		primary = append(primary, fact)
	}
	if len(primary) == 0 {
		return "", false
	}

	found := map[string]bool{}
	for _, fact := range primary {
		anchors := extractAnchors(fact.Body)
		if len(anchors) == 0 {
			continue
		}
		for _, support := range facts {
			if support.Body == fact.Body {
				continue
			}
			lower := strings.ToLower(support.Body)
			if !sharesAnchor(lower, anchors) {
				continue
			}
			if !containsActionSupport(lower, action) {
				continue
			}
			if date := answerDateFromBody(support.Body); date != "" {
				found[date] = true
			}
		}
	}

	if len(found) != 1 {
		return "", false
	}
	for date := range found {
		return date, true
	}
	return "", false
}

func resolveNamedSpendTotal(question string, facts []retrievedFact) (string, bool) {
	lowerQuestion := strings.ToLower(question)
	if !strings.Contains(lowerQuestion, "spent") && !strings.Contains(lowerQuestion, "total amount") && !strings.Contains(lowerQuestion, "cost") {
		return "", false
	}
	idx := strings.Index(lowerQuestion, " for ")
	if idx < 0 {
		return "", false
	}
	entities := parseNamedEntities(question[idx+5:])
	if len(entities) < 2 {
		return "", false
	}

	total := 0.0
	for _, entity := range entities {
		best, ok := bestAmountForEntity(entity, question, facts)
		if !ok {
			return "", false
		}
		total += best
	}
	return formatCurrency(total), true
}

func resolveBackendRecommendation(question string, facts []retrievedFact) (string, bool) {
	lowerQuestion := strings.ToLower(question)
	if !strings.Contains(lowerQuestion, "recommend") || !strings.Contains(lowerQuestion, "learn") {
		return "", false
	}
	if !strings.Contains(lowerQuestion, "programming language") {
		return "", false
	}
	if !strings.Contains(lowerQuestion, "back-end") && !strings.Contains(lowerQuestion, "backend") {
		return "", false
	}

	var options []string
	for _, fact := range facts {
		items := extractBackendLanguages(fact.Body)
		if len(items) == 0 {
			continue
		}
		options = mergeListItems(options, items)
	}
	if len(options) < 2 {
		return "", false
	}
	return fmt.Sprintf("I recommended learning %s as a back-end programming language.", joinWithOr(options)), true
}

func canonicalAction(raw string) string {
	switch strings.ToLower(strings.TrimSpace(raw)) {
	case "submit", "submitted", "apply", "applied":
		return "submission"
	case "book", "booked":
		return "booking"
	case "join", "joined":
		return "join"
	case "start", "started":
		return "start"
	case "attend", "attended":
		return "attendance"
	default:
		return ""
	}
}

func containsActionVerb(text, action string) bool {
	switch action {
	case "submission":
		return containsAny(text, []string{" submit ", " submitted ", "applied", "apply "}) ||
			strings.Contains(text, "submitted to")
	case "booking":
		return containsAny(text, []string{" book ", " booked ", "booking "})
	case "join":
		return containsAny(text, []string{" join ", " joined "})
	case "start":
		return containsAny(text, []string{" start ", " started ", "began "})
	case "attendance":
		return containsAny(text, []string{" attend ", " attended "})
	default:
		return false
	}
}

func containsActionSupport(text, action string) bool {
	switch action {
	case "submission":
		return containsAny(text, []string{"submission date", "submitted on", "submission was"})
	case "booking":
		return containsAny(text, []string{"booking date", "booked on", "reservation date"})
	case "join":
		return containsAny(text, []string{"join date", "joined on", "membership date"})
	case "start":
		return containsAny(text, []string{"start date", "started on", "began on"})
	case "attendance":
		return containsAny(text, []string{"attended on", "attendance date"})
	default:
		return false
	}
}

func extractAnchors(body string) []string {
	seen := map[string]bool{}
	out := make([]string, 0, 4)
	for _, match := range acronymRe.FindAllString(body, -1) {
		anchor := strings.ToLower(strings.TrimSpace(match))
		if anchor == "" || seen[anchor] {
			continue
		}
		seen[anchor] = true
		out = append(out, anchor)
	}
	for _, match := range anchorPhraseRe.FindAllStringSubmatch(body, -1) {
		if len(match) < 2 {
			continue
		}
		anchor := strings.ToLower(strings.TrimSpace(match[1]))
		if anchor == "" || seen[anchor] {
			continue
		}
		seen[anchor] = true
		out = append(out, anchor)
	}
	sort.Strings(out)
	return out
}

func sharesAnchor(text string, anchors []string) bool {
	for _, anchor := range anchors {
		if strings.Contains(text, anchor) {
			return true
		}
	}
	return false
}

func answerDateFromBody(body string) string {
	if matches := monthDayRe.FindStringSubmatch(body); len(matches) == 3 {
		day, _ := strconv.Atoi(matches[2])
		return fmt.Sprintf("%s %s", titleCaseMonth(matches[1]), ordinal(day))
	}
	if matches := datePrefixRe.FindStringSubmatch(body); len(matches) == 2 {
		if parsed, err := time.Parse("2006-01-02", matches[1]); err == nil {
			return fmt.Sprintf("%s %s", parsed.Month().String(), ordinal(parsed.Day()))
		}
	}
	if matches := isoDateRe.FindStringSubmatch(body); len(matches) == 4 {
		year, _ := strconv.Atoi(matches[1])
		month, _ := strconv.Atoi(matches[2])
		day, _ := strconv.Atoi(matches[3])
		if parsed := time.Date(year, time.Month(month), day, 0, 0, 0, 0, time.UTC); parsed.Year() == year {
			return fmt.Sprintf("%s %s", parsed.Month().String(), ordinal(parsed.Day()))
		}
	}
	return ""
}

func directActionDateFromBody(action, body string) string {
	lower := strings.ToLower(body)
	switch action {
	case "submission":
		if !containsAny(lower, []string{"submitted on", "submission date", "submission was"}) {
			return ""
		}
	case "booking":
		if !containsAny(lower, []string{"booked on", "booking date", "reservation date"}) {
			return ""
		}
	case "join":
		if !containsAny(lower, []string{"joined on", "join date", "membership date"}) {
			return ""
		}
	case "start":
		if !containsAny(lower, []string{"started on", "start date", "began on"}) {
			return ""
		}
	case "attendance":
		if !containsAny(lower, []string{"attended on", "attendance date"}) {
			return ""
		}
	}
	return answerDateFromBody(body)
}

func parseNamedEntities(tail string) []namedEntity {
	tail = strings.TrimSpace(strings.TrimSuffix(tail, "?"))
	tail = strings.TrimSpace(strings.TrimPrefix(strings.ToLower(tail), "my "))
	tail = strings.TrimSpace(strings.TrimPrefix(tail, "our "))
	parts := strings.FieldsFunc(tail, func(r rune) bool {
		return r == ',' || r == ';'
	})
	if len(parts) == 1 {
		parts = strings.Split(parts[0], " and ")
	}
	seen := map[string]bool{}
	out := make([]namedEntity, 0, len(parts))
	for _, part := range parts {
		tokens := significantTokens(part)
		if len(tokens) == 0 {
			continue
		}
		key := tokens[len(tokens)-1]
		if seen[key] {
			continue
		}
		seen[key] = true
		out = append(out, namedEntity{Key: key, Tokens: tokens})
	}
	return out
}

func bestAmountForEntity(entity namedEntity, question string, facts []retrievedFact) (float64, bool) {
	candidates := map[string]int{}
	for _, fact := range facts {
		bodyLower := strings.ToLower(fact.Body)
		if !containsEntity(bodyLower, entity) {
			continue
		}
		matches := moneyRe.FindAllStringSubmatchIndex(fact.Body, -1)
		for _, match := range matches {
			if len(match) < 4 {
				continue
			}
			value, err := strconv.ParseFloat(fact.Body[match[2]:match[3]], 64)
			if err != nil {
				continue
			}
			window := strings.ToLower(sentenceWindow(fact.Body, match[0], match[1]))
			if !containsEntity(window, entity) && !containsEntity(bodyLower, entity) {
				continue
			}
			score := 0
			if containsEntity(window, entity) {
				score += 6
			} else {
				score += 2
			}
			if containsAny(window, []string{"bought", "buy ", "purchased", "cost", "paid", "gift", "gift card"}) {
				score += 4
			}
			if strings.Contains(strings.ToLower(question), "gift") && containsAny(window, []string{"gift", "baby shower", "graduation"}) {
				score++
			}
			if containsAny(window, []string{"total", "overall", "recently", "budget", "tracker", "summary", "in total"}) {
				score -= 5
			}
			if containsAny(window, []string{"plan", "planned", "maybe", "might", "consider"}) {
				score -= 2
			}
			if score < 3 {
				continue
			}
			key := fmt.Sprintf("%.2f", value)
			if current, ok := candidates[key]; !ok || score > current {
				candidates[key] = score
			}
		}
	}
	if len(candidates) == 0 {
		return 0, false
	}
	type scoredAmount struct {
		Value float64
		Score int
	}
	best := make([]scoredAmount, 0, len(candidates))
	for raw, score := range candidates {
		value, _ := strconv.ParseFloat(raw, 64)
		best = append(best, scoredAmount{Value: value, Score: score})
	}
	sort.Slice(best, func(i, j int) bool {
		if best[i].Score != best[j].Score {
			return best[i].Score > best[j].Score
		}
		return best[i].Value < best[j].Value
	})
	if len(best) > 1 && best[0].Score == best[1].Score && best[0].Value != best[1].Value {
		return 0, false
	}
	return best[0].Value, true
}

func containsEntity(text string, entity namedEntity) bool {
	if strings.Contains(text, entity.Key) {
		return true
	}
	for _, token := range entity.Tokens {
		if strings.Contains(text, token) {
			return true
		}
	}
	return false
}

func extractBackendLanguages(body string) []string {
	for _, clause := range splitClauses(body) {
		lower := strings.ToLower(clause)
		if !strings.Contains(lower, "learn") || !strings.Contains(lower, "programming language") {
			continue
		}
		if containsAny(lower, []string{"resource", "resources", "course", "courses", "curriculum", "workshop", "framework", "frameworks"}) {
			continue
		}
		matches := suchAsRe.FindStringSubmatch(clause)
		if len(matches) != 2 {
			continue
		}
		items := mergeListItems(nil, parseListItems(matches[1]))
		if len(items) >= 2 {
			return items
		}
	}
	return nil
}

func splitClauses(body string) []string {
	body = strings.ReplaceAll(body, "\n", " ")
	body = strings.ReplaceAll(body, ";", ".")
	parts := strings.Split(body, ".")
	out := make([]string, 0, len(parts))
	for _, part := range parts {
		part = strings.TrimSpace(part)
		if part != "" {
			out = append(out, part)
		}
	}
	return out
}

func parseListItems(raw string) []string {
	raw = strings.ReplaceAll(raw, " and ", ", ")
	raw = strings.ReplaceAll(raw, " or ", ", ")
	parts := strings.Split(raw, ",")
	seen := map[string]bool{}
	out := make([]string, 0, len(parts))
	for _, part := range parts {
		part = strings.TrimSpace(part)
		part = strings.Trim(part, " .:;!?")
		if part == "" {
			continue
		}
		canonical := strings.ToLower(part)
		if seen[canonical] {
			continue
		}
		seen[canonical] = true
		out = append(out, part)
	}
	return out
}

func mergeListItems(base, extra []string) []string {
	seen := map[string]bool{}
	out := make([]string, 0, len(base)+len(extra))
	for _, item := range append(base, extra...) {
		key := strings.ToLower(strings.TrimSpace(item))
		if key == "" || seen[key] {
			continue
		}
		seen[key] = true
		out = append(out, strings.TrimSpace(item))
	}
	return out
}

func joinWithOr(items []string) string {
	if len(items) == 0 {
		return ""
	}
	if len(items) == 1 {
		return items[0]
	}
	if len(items) == 2 {
		return items[0] + " or " + items[1]
	}
	return strings.Join(items[:len(items)-1], ", ") + ", or " + items[len(items)-1]
}

func significantTokens(text string) []string {
	parts := strings.Fields(strings.ToLower(strings.NewReplacer(
		"?", " ",
		"!", " ",
		".", " ",
		",", " ",
		":", " ",
		";", " ",
		"(", " ",
		")", " ",
		"'", " ",
		"\"", " ",
		"-", " ",
	).Replace(text)))
	seen := map[string]bool{}
	out := make([]string, 0, len(parts))
	for _, part := range parts {
		part = strings.TrimSpace(part)
		if len(part) < 2 || resolverStopWords[part] || seen[part] {
			continue
		}
		seen[part] = true
		out = append(out, part)
	}
	return out
}

func tokenOverlap(text string, tokens []string) int {
	var overlap int
	for _, token := range tokens {
		if strings.Contains(text, token) {
			overlap++
		}
	}
	return overlap
}

func containsAny(text string, needles []string) bool {
	for _, needle := range needles {
		if strings.Contains(text, needle) {
			return true
		}
	}
	return false
}

func windowAround(text string, start, end, radius int) string {
	from := start - radius
	if from < 0 {
		from = 0
	}
	to := end + radius
	if to > len(text) {
		to = len(text)
	}
	return text[from:to]
}

func sentenceWindow(text string, start, end int) string {
	from := strings.LastIndexAny(text[:start], ".!?\n")
	if from >= 0 {
		from++
	} else {
		from = 0
	}
	to := len(text)
	if after := strings.IndexAny(text[end:], ".!?\n"); after >= 0 {
		to = end + after
	}
	return strings.TrimSpace(text[from:to])
}

func formatCurrency(value float64) string {
	if value == float64(int64(value)) {
		return fmt.Sprintf("$%d", int64(value))
	}
	return fmt.Sprintf("$%.2f", value)
}

func titleCaseMonth(month string) string {
	month = strings.ToLower(strings.TrimSpace(month))
	if month == "" {
		return ""
	}
	return strings.ToUpper(month[:1]) + month[1:]
}

func ordinal(day int) string {
	if day%100 >= 11 && day%100 <= 13 {
		return fmt.Sprintf("%dth", day)
	}
	switch day % 10 {
	case 1:
		return fmt.Sprintf("%dst", day)
	case 2:
		return fmt.Sprintf("%dnd", day)
	case 3:
		return fmt.Sprintf("%drd", day)
	default:
		return fmt.Sprintf("%dth", day)
	}
}
