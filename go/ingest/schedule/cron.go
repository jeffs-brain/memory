// SPDX-License-Identifier: Apache-2.0
package schedule

import (
	"fmt"
	"strconv"
	"strings"
	"time"
)

// CronSchedule represents a parsed 5-field cron expression.
type CronSchedule struct {
	Minute     []int // 0-59
	Hour       []int // 0-23
	DayOfMonth []int // 1-31
	Month      []int // 1-12
	DayOfWeek  []int // 0-6 (Sunday=0)
}

// ParseCron parses a standard 5-field cron expression.
// Supports: numbers, ranges (1-5), steps (*/5), lists (1,3,5), and *.
func ParseCron(expression string) (CronSchedule, error) {
	fields := strings.Fields(expression)
	if len(fields) != 5 {
		return CronSchedule{}, fmt.Errorf("cron: expected 5 fields, got %d in %q", len(fields), expression)
	}

	minute, err := parseField(fields[0], 0, 59)
	if err != nil {
		return CronSchedule{}, fmt.Errorf("cron: minute field: %w", err)
	}
	hour, err := parseField(fields[1], 0, 23)
	if err != nil {
		return CronSchedule{}, fmt.Errorf("cron: hour field: %w", err)
	}
	dom, err := parseField(fields[2], 1, 31)
	if err != nil {
		return CronSchedule{}, fmt.Errorf("cron: day-of-month field: %w", err)
	}
	month, err := parseField(fields[3], 1, 12)
	if err != nil {
		return CronSchedule{}, fmt.Errorf("cron: month field: %w", err)
	}
	dow, err := parseField(fields[4], 0, 6)
	if err != nil {
		return CronSchedule{}, fmt.Errorf("cron: day-of-week field: %w", err)
	}

	return CronSchedule{
		Minute:     minute,
		Hour:       hour,
		DayOfMonth: dom,
		Month:      month,
		DayOfWeek:  dow,
	}, nil
}

// IsValid reports whether expression is a valid 5-field cron expression.
func IsValid(expression string) bool {
	_, err := ParseCron(expression)
	return err == nil
}

// NextOccurrence computes the next time the schedule fires after the
// given reference time. It iterates minute-by-minute up to 4 years out
// to find the next match.
//
// DOM+DOW union semantics: per POSIX cron, when both day-of-month and
// day-of-week are non-wildcard (i.e. not full-range), a date matches if
// EITHER condition is true. When one or both are wildcard, standard
// intersection applies.
func NextOccurrence(sched CronSchedule, after time.Time) time.Time {
	// Start one minute after `after`, zeroing seconds.
	t := after.Truncate(time.Minute).Add(time.Minute)

	minuteSet := toSet(sched.Minute)
	hourSet := toSet(sched.Hour)
	domSet := toSet(sched.DayOfMonth)
	monthSet := toSet(sched.Month)
	dowSet := toSet(sched.DayOfWeek)

	domIsWild := len(sched.DayOfMonth) == 31
	dowIsWild := len(sched.DayOfWeek) == 7

	// Search up to 4 years (≈2.1M minutes) to find a match.
	limit := after.Add(4 * 365 * 24 * time.Hour)
	for t.Before(limit) {
		dayMatch := matchDay(domSet, dowSet, domIsWild, dowIsWild, t.Day(), int(t.Weekday()))
		if monthSet[int(t.Month())] &&
			dayMatch &&
			hourSet[t.Hour()] &&
			minuteSet[t.Minute()] {
			return t
		}

		// Skip ahead intelligently.
		if !monthSet[int(t.Month())] {
			// Jump to the first day of the next month.
			t = time.Date(t.Year(), t.Month()+1, 1, 0, 0, 0, 0, t.Location())
			continue
		}
		if !dayMatch {
			// Jump to the next day.
			t = time.Date(t.Year(), t.Month(), t.Day()+1, 0, 0, 0, 0, t.Location())
			continue
		}
		if !hourSet[t.Hour()] {
			// Jump to the next hour.
			t = time.Date(t.Year(), t.Month(), t.Day(), t.Hour()+1, 0, 0, 0, t.Location())
			continue
		}
		t = t.Add(time.Minute)
	}

	// Fallback: should not happen for valid schedules.
	return limit
}

// matchDay implements POSIX cron union semantics for day-of-month and
// day-of-week. When both fields are restricted (non-wildcard), the day
// matches if EITHER the DOM or DOW matches. When one or both are
// wildcard, standard AND logic applies.
func matchDay(domSet, dowSet map[int]bool, domIsWild, dowIsWild bool, day, weekday int) bool {
	if !domIsWild && !dowIsWild {
		return domSet[day] || dowSet[weekday]
	}
	return domSet[day] && dowSet[weekday]
}

// parseField parses a single cron field (e.g., "*/5", "1-3", "1,3,5", "*").
func parseField(field string, min, max int) ([]int, error) {
	var result []int

	for _, part := range strings.Split(field, ",") {
		part = strings.TrimSpace(part)
		stepParts := strings.SplitN(part, "/", 2)
		rangePart := stepParts[0]
		step := 1

		if len(stepParts) == 2 {
			var err error
			step, err = strconv.Atoi(stepParts[1])
			if err != nil || step < 1 {
				return nil, fmt.Errorf("invalid step %q", stepParts[1])
			}
		}

		var rangeStart, rangeEnd int

		if rangePart == "*" {
			rangeStart = min
			rangeEnd = max
		} else if idx := strings.Index(rangePart, "-"); idx >= 0 {
			var err error
			rangeStart, err = strconv.Atoi(rangePart[:idx])
			if err != nil {
				return nil, fmt.Errorf("invalid range start %q", rangePart[:idx])
			}
			rangeEnd, err = strconv.Atoi(rangePart[idx+1:])
			if err != nil {
				return nil, fmt.Errorf("invalid range end %q", rangePart[idx+1:])
			}
		} else {
			val, err := strconv.Atoi(rangePart)
			if err != nil {
				return nil, fmt.Errorf("invalid value %q", rangePart)
			}
			rangeStart = val
			rangeEnd = val
		}

		if rangeStart < min || rangeEnd > max || rangeStart > rangeEnd {
			return nil, fmt.Errorf("value out of range [%d-%d]: %d-%d", min, max, rangeStart, rangeEnd)
		}

		for i := rangeStart; i <= rangeEnd; i += step {
			result = append(result, i)
		}
	}

	if len(result) == 0 {
		return nil, fmt.Errorf("empty field")
	}
	return deduplicate(result), nil
}

// deduplicate removes duplicate values from a sorted-ish slice
// produced by parseField, preserving order.
func deduplicate(vals []int) []int {
	seen := make(map[int]bool, len(vals))
	out := make([]int, 0, len(vals))
	for _, v := range vals {
		if !seen[v] {
			seen[v] = true
			out = append(out, v)
		}
	}
	return out
}

func toSet(vals []int) map[int]bool {
	s := make(map[int]bool, len(vals))
	for _, v := range vals {
		s[v] = true
	}
	return s
}
