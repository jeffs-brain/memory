// SPDX-License-Identifier: Apache-2.0

package query

import (
	"fmt"
	"regexp"
	"strconv"
	"strings"
	"time"
)

// TemporalExpansion holds the result of resolving temporal references.
type TemporalExpansion struct {
	OriginalQuery string
	ExpandedQuery string
	DateHints     []string // resolved date strings to help retrieval
	Resolved      bool
}

// ExpandTemporal resolves relative temporal references in a question using
// the question date as the anchor point. Returns the original query
// enriched with resolved date information.
//
// The questionDate format is the LME form "2023/04/10 (Mon) 23:07" by
// default; [parseQuestionDate] also accepts a handful of sibling formats.
//
// The expander is deliberately English-only and recognises exactly the
// three patterns documented in spec/QUERY-DSL.md under "Temporal
// expansion": relative time phrases ("N days/weeks/months ago"), last
// weekday ("last Monday" etc), and ordering hints ("first", "most recent",
// etc). Additional recognisers are a spec concern; do not add more here
// without updating the spec first.
func ExpandTemporal(question, questionDate string) TemporalExpansion {
	result := TemporalExpansion{
		OriginalQuery: question,
		ExpandedQuery: question,
	}

	anchor, err := parseQuestionDate(questionDate)
	if err != nil {
		return result
	}

	var hints []string

	// Resolve "N weeks/days/months ago" patterns.
	question, resolved := resolveRelativeTime(question, anchor)
	hints = append(hints, resolved...)

	// Resolve "last <weekday>" patterns.
	question, resolved = resolveLastWeekday(question, anchor)
	hints = append(hints, resolved...)

	// Resolve "first" / "most recent" ordering hints.
	question = annotateOrdering(question)

	if len(hints) > 0 {
		result.ExpandedQuery = question
		result.DateHints = hints
		result.Resolved = true
	}

	return result
}

// parseQuestionDate parses the LME date format "2023/04/10 (Mon) 23:07"
// and a handful of common siblings.
func parseQuestionDate(s string) (time.Time, error) {
	s = strings.TrimSpace(s)
	if s == "" {
		return time.Time{}, fmt.Errorf("empty date")
	}

	// Try the full LME format first.
	formats := []string{
		"2006/01/02 (Mon) 15:04",
		"2006/01/02 15:04",
		"2006/01/02",
		"2006-01-02",
	}
	for _, f := range formats {
		if t, err := time.Parse(f, s); err == nil {
			return t, nil
		}
	}
	return time.Time{}, fmt.Errorf("unrecognised date format: %q", s)
}

// relativeTimePattern mirrors the RELATIVE_TIME_RE recogniser from the TS
// reference. Matching is case-insensitive.
var relativeTimePattern = regexp.MustCompile(
	`(?i)(\d+)\s+(day|days|week|weeks|month|months)\s+ago`)

func resolveRelativeTime(question string, anchor time.Time) (string, []string) {
	var hints []string

	expanded := relativeTimePattern.ReplaceAllStringFunc(question, func(match string) string {
		parts := relativeTimePattern.FindStringSubmatch(match)
		if len(parts) < 3 {
			return match
		}
		n, err := strconv.Atoi(parts[1])
		if err != nil {
			return match
		}
		unit := strings.ToLower(parts[2])

		var resolved time.Time
		switch {
		case strings.HasPrefix(unit, "day"):
			resolved = anchor.AddDate(0, 0, -n)
		case strings.HasPrefix(unit, "week"):
			resolved = anchor.AddDate(0, 0, -n*7)
		case strings.HasPrefix(unit, "month"):
			resolved = anchor.AddDate(0, -n, 0)
		default:
			return match
		}

		dateStr := resolved.Format("2006/01/02")
		hints = append(hints, dateStr)
		return fmt.Sprintf("%s (around %s)", match, dateStr)
	})

	return expanded, hints
}

// lastWeekdayPattern mirrors the LAST_WEEKDAY_RE recogniser from the TS
// reference.
var lastWeekdayPattern = regexp.MustCompile(
	`(?i)last\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)`)

var weekdayMap = map[string]time.Weekday{
	"monday": time.Monday, "tuesday": time.Tuesday,
	"wednesday": time.Wednesday, "thursday": time.Thursday,
	"friday": time.Friday, "saturday": time.Saturday,
	"sunday": time.Sunday,
}

func resolveLastWeekday(question string, anchor time.Time) (string, []string) {
	var hints []string

	expanded := lastWeekdayPattern.ReplaceAllStringFunc(question, func(match string) string {
		parts := lastWeekdayPattern.FindStringSubmatch(match)
		if len(parts) < 2 {
			return match
		}
		target, ok := weekdayMap[strings.ToLower(parts[1])]
		if !ok {
			return match
		}

		// Find the most recent occurrence of the target weekday before
		// the anchor. The anchor day itself is never returned, so "last
		// monday" on a Monday resolves to seven days earlier.
		d := anchor
		for i := 1; i <= 7; i++ {
			d = d.AddDate(0, 0, -1)
			if d.Weekday() == target {
				dateStr := d.Format("2006/01/02")
				hints = append(hints, dateStr)
				return fmt.Sprintf("%s (%s)", match, dateStr)
			}
		}
		return match
	})

	return expanded, hints
}

// annotateOrdering adds hints for temporal ordering questions. Mirrors the
// annotateOrdering helper from the TS reference implementation.
func annotateOrdering(question string) string {
	lower := strings.ToLower(question)
	if strings.Contains(lower, "first") || strings.Contains(lower, "earlier") ||
		strings.Contains(lower, "before") {
		return question + " [Note: look for the earliest dated event]"
	}
	if strings.Contains(lower, "most recent") || strings.Contains(lower, "latest") ||
		strings.Contains(lower, "last time") {
		return question + " [Note: look for the most recently dated event]"
	}
	return question
}
