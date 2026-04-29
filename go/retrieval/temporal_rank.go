// SPDX-License-Identifier: Apache-2.0

package retrieval

import (
	"regexp"
	"sort"
	"strings"
	"time"

	"github.com/jeffs-brain/memory/go/query"
)

var recencyQueryHints = []string{
	"most recent",
	"latest",
	"last time",
	"currently",
	"current",
	"now",
	"newest",
}

var bodyMonthDayRe = regexp.MustCompile(`(?i)\b(?:on\s+)?(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?\s+(\d{1,2})(?:st|nd|rd|th)?\b`)

var earliestQueryHints = []string{
	"earliest",
	"first",
	"initial",
	"original",
	"at first",
}

var candidateDateLayouts = []string{
	time.RFC3339,
	"2006-01-02 15:04:05-07:00",
	"2006-01-02 15:04:05",
	"2006/01/02 (Mon) 15:04",
	"2006/01/02 15:04",
	"2006/01/02",
	"2006-01-02",
}

func reweightTemporalRanking(question, questionDate string, results []RetrievedChunk) []RetrievedChunk {
	if len(results) == 0 {
		return results
	}

	if anchor, ok := parseCandidateTime(questionDate); ok {
		filterFutureEvents := shouldFilterFutureEventFacts(question)
		filtered := make([]RetrievedChunk, 0, len(results))
		for _, chunk := range results {
			if filterFutureEvents {
				eventDate, hasEventDate := extractBodyMonthDayEventDate(chunk.Text, anchor)
				if hasEventDate && candidateDateAfterAnchorDate(eventDate, anchor) {
					continue
				}
			}
			candidate, hasTime := extractCandidateTime(chunk)
			if hasTime && candidateDateAfterAnchorDate(candidate, anchor) {
				continue
			}
			filtered = append(filtered, chunk)
		}
		results = filtered
		if len(results) == 0 {
			return results
		}
	}

	expansion := query.ExpandTemporal(question, questionDate)
	hintTimes := make([]time.Time, 0, len(expansion.DateHints))
	for _, hint := range expansion.DateHints {
		parsed, ok := parseCandidateTime(hint)
		if ok {
			hintTimes = append(hintTimes, parsed)
		}
	}

	lower := strings.ToLower(question)
	if dateArithmeticQueryRe.MatchString(lower) {
		return results
	}
	wantsRecency := containsAnyHint(lower, recencyQueryHints)
	wantsEarliest := !wantsRecency && containsAnyHint(lower, earliestQueryHints)
	if !wantsRecency && !wantsEarliest && len(hintTimes) == 0 {
		return results
	}

	candidateTimes := make([]time.Time, len(results))
	hasTime := make([]bool, len(results))
	var minTime time.Time
	var maxTime time.Time
	for i, chunk := range results {
		parsed, ok := extractCandidateTime(chunk)
		if !ok {
			continue
		}
		candidateTimes[i] = parsed
		hasTime[i] = true
		if minTime.IsZero() || parsed.Before(minTime) {
			minTime = parsed
		}
		if maxTime.IsZero() || parsed.After(maxTime) {
			maxTime = parsed
		}
	}
	if minTime.IsZero() || maxTime.IsZero() {
		return results
	}

	type scoredChunk struct {
		chunk RetrievedChunk
		index int
		score float64
	}
	scored := make([]scoredChunk, 0, len(results))
	for i, chunk := range results {
		multiplier := 1.0
		if hasTime[i] && len(hintTimes) > 0 {
			multiplier *= temporalHintMultiplier(candidateTimes[i], hintTimes)
		}
		if hasTime[i] && maxTime.After(minTime) {
			norm := float64(candidateTimes[i].Unix()-minTime.Unix()) / float64(maxTime.Unix()-minTime.Unix())
			if wantsRecency {
				multiplier *= 1.0 + 0.25*norm
			}
			if wantsEarliest {
				multiplier *= 1.0 + 0.25*(1.0-norm)
			}
		} else if !hasTime[i] && (wantsRecency || wantsEarliest) {
			multiplier *= 0.95
		}
		next := chunk
		next.Score = chunk.Score * multiplier
		scored = append(scored, scoredChunk{
			chunk: next,
			index: i,
			score: next.Score,
		})
	}
	sort.SliceStable(scored, func(i, j int) bool {
		if scored[i].score != scored[j].score {
			return scored[i].score > scored[j].score
		}
		return scored[i].index < scored[j].index
	})

	out := make([]RetrievedChunk, 0, len(scored))
	for _, item := range scored {
		out = append(out, item.chunk)
	}
	return out
}

func shouldFilterFutureEventFacts(question string) bool {
	lower := strings.ToLower(question)
	if strings.Contains(lower, "plan") ||
		strings.Contains(lower, "planning") ||
		strings.Contains(lower, "upcoming") ||
		strings.Contains(lower, "future") ||
		strings.Contains(lower, "will i") ||
		strings.Contains(lower, "going to") {
		return false
	}
	fundraisingQuery := strings.Contains(lower, "charity") ||
		strings.Contains(lower, "fundrais") ||
		strings.Contains(lower, "donation") ||
		strings.Contains(lower, "donations") ||
		strings.Contains(lower, "raised") ||
		strings.Contains(lower, "raise")
	totalQuery := strings.Contains(lower, "how much") ||
		strings.Contains(lower, "total")
	return fundraisingQuery && totalQuery
}

func extractBodyMonthDayEventDate(text string, anchor time.Time) (time.Time, bool) {
	body := stripFrontmatterBody(text)
	for _, match := range bodyMonthDayRe.FindAllStringSubmatch(body, -1) {
		if len(match) < 3 {
			continue
		}
		month, ok := monthNameNumber(match[1])
		if !ok {
			continue
		}
		day, ok := parseSmallDay(match[2])
		if !ok {
			continue
		}
		candidate := time.Date(anchor.Year(), month, day, 0, 0, 0, 0, time.UTC)
		return candidate, true
	}
	return time.Time{}, false
}

func monthNameNumber(raw string) (time.Month, bool) {
	switch strings.ToLower(strings.Trim(strings.TrimSpace(raw), ".")) {
	case "jan", "january":
		return time.January, true
	case "feb", "february":
		return time.February, true
	case "mar", "march":
		return time.March, true
	case "apr", "april":
		return time.April, true
	case "may":
		return time.May, true
	case "jun", "june":
		return time.June, true
	case "jul", "july":
		return time.July, true
	case "aug", "august":
		return time.August, true
	case "sep", "sept", "september":
		return time.September, true
	case "oct", "october":
		return time.October, true
	case "nov", "november":
		return time.November, true
	case "dec", "december":
		return time.December, true
	default:
		return time.January, false
	}
}

func parseSmallDay(raw string) (int, bool) {
	day := 0
	for _, r := range strings.TrimSpace(raw) {
		if r < '0' || r > '9' {
			return 0, false
		}
		day = day*10 + int(r-'0')
	}
	return day, day >= 1 && day <= 31
}

func candidateDateAfterAnchorDate(candidate, anchor time.Time) bool {
	candidateYear, candidateMonth, candidateDay := candidate.Date()
	anchorYear, anchorMonth, anchorDay := anchor.Date()
	candidateDate := time.Date(candidateYear, candidateMonth, candidateDay, 0, 0, 0, 0, time.UTC)
	anchorDate := time.Date(anchorYear, anchorMonth, anchorDay, 0, 0, 0, 0, time.UTC)
	return candidateDate.After(anchorDate)
}

func temporalHintMultiplier(candidate time.Time, hints []time.Time) float64 {
	nearestDays := 1e9
	for _, hint := range hints {
		diff := candidate.Sub(hint)
		if diff < 0 {
			diff = -diff
		}
		days := diff.Hours() / 24.0
		if days < nearestDays {
			nearestDays = days
		}
	}
	switch {
	case nearestDays <= 1:
		return 1.35
	case nearestDays <= 7:
		return 1.20
	case nearestDays <= 30:
		return 1.08
	default:
		return 0.92
	}
}

func containsAnyHint(text string, hints []string) bool {
	for _, hint := range hints {
		if strings.Contains(text, hint) {
			return true
		}
	}
	return false
}

func extractCandidateTime(chunk RetrievedChunk) (time.Time, bool) {
	if parsed, ok := extractMetadataTime(chunk.Metadata); ok {
		return parsed, true
	}
	return extractTimeFromText(chunk.Text)
}

func extractMetadataTime(metadata map[string]any) (time.Time, bool) {
	if metadata == nil {
		return time.Time{}, false
	}
	for _, key := range []string{"eventDate", "event_date", "observedOn", "observed_on", "sessionDate", "session_date", "modified"} {
		value, ok := metadata[key]
		if !ok {
			continue
		}
		text, ok := value.(string)
		if !ok {
			continue
		}
		parsed, ok := parseCandidateTime(text)
		if ok {
			return parsed, true
		}
	}
	return time.Time{}, false
}

func extractTimeFromText(text string) (time.Time, bool) {
	for _, line := range strings.Split(text, "\n") {
		trimmed := strings.TrimSpace(line)
		lower := strings.ToLower(trimmed)
		switch {
		case strings.HasPrefix(lower, "[observed on "):
			value := strings.TrimSuffix(strings.TrimSpace(trimmed[len("[Observed on "):]), "]")
			if parsed, ok := parseCandidateTime(value); ok {
				return parsed, true
			}
		case strings.HasPrefix(lower, "[observed on:"):
			value := strings.TrimSuffix(strings.TrimSpace(trimmed[len("[Observed on:"):]), "]")
			if parsed, ok := parseCandidateTime(value); ok {
				return parsed, true
			}
		case strings.HasPrefix(lower, "[date:"):
			value := strings.TrimSuffix(strings.TrimSpace(trimmed[len("[Date:"):]), "]")
			if parsed, ok := parseCandidateTime(value); ok {
				return parsed, true
			}
		default:
			for _, key := range []string{"event_date:", "session_date:", "observed_on:", "modified:"} {
				if !strings.HasPrefix(lower, key) {
					continue
				}
				value := strings.TrimSpace(trimmed[len(key):])
				if parsed, ok := parseCandidateTime(value); ok {
					return parsed, true
				}
			}
		}
	}
	return time.Time{}, false
}

func parseCandidateTime(value string) (time.Time, bool) {
	trimmed := strings.TrimSpace(value)
	if trimmed == "" {
		return time.Time{}, false
	}
	for _, layout := range candidateDateLayouts {
		if parsed, err := time.Parse(layout, trimmed); err == nil {
			return parsed.UTC(), true
		}
	}
	return time.Time{}, false
}
