// SPDX-License-Identifier: Apache-2.0
package schedule

import (
	"testing"
	"time"
)

func TestParseCronEveryHour(t *testing.T) {
	sched, err := ParseCron("0 * * * *")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(sched.Minute) != 1 || sched.Minute[0] != 0 {
		t.Fatalf("expected minute [0], got %v", sched.Minute)
	}
	if len(sched.Hour) != 24 {
		t.Fatalf("expected 24 hours, got %d", len(sched.Hour))
	}
}

func TestParseCronMondayMorning(t *testing.T) {
	sched, err := ParseCron("30 2 * * 1")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if sched.Minute[0] != 30 {
		t.Fatalf("expected minute 30, got %v", sched.Minute)
	}
	if sched.Hour[0] != 2 {
		t.Fatalf("expected hour 2, got %v", sched.Hour)
	}
	if sched.DayOfWeek[0] != 1 {
		t.Fatalf("expected dow 1, got %v", sched.DayOfWeek)
	}
}

func TestParseCronEvery5Minutes(t *testing.T) {
	sched, err := ParseCron("*/5 * * * *")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	expected := []int{0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55}
	if len(sched.Minute) != len(expected) {
		t.Fatalf("expected %d minutes, got %d: %v", len(expected), len(sched.Minute), sched.Minute)
	}
	for i, v := range expected {
		if sched.Minute[i] != v {
			t.Fatalf("minute %d: expected %d, got %d", i, v, sched.Minute[i])
		}
	}
}

func TestParseCronInvalidExpression(t *testing.T) {
	tests := []string{
		"",
		"* * *",
		"60 * * * *",
		"* 25 * * *",
		"* * 32 * *",
		"* * * 13 *",
		"* * * * 7",
		"abc * * * *",
	}
	for _, expr := range tests {
		_, err := ParseCron(expr)
		if err == nil {
			t.Errorf("expected error for %q", expr)
		}
	}
}

func TestIsValid(t *testing.T) {
	if !IsValid("0 * * * *") {
		t.Fatal("expected valid")
	}
	if IsValid("invalid") {
		t.Fatal("expected invalid")
	}
}

func TestNextOccurrenceEveryHour(t *testing.T) {
	sched, _ := ParseCron("0 * * * *")
	ref := time.Date(2026, 5, 15, 10, 30, 0, 0, time.UTC)
	next := NextOccurrence(sched, ref)

	expected := time.Date(2026, 5, 15, 11, 0, 0, 0, time.UTC)
	if !next.Equal(expected) {
		t.Fatalf("expected %v, got %v", expected, next)
	}
}

func TestNextOccurrenceExactMinute(t *testing.T) {
	sched, _ := ParseCron("0 * * * *")
	// When we're exactly at minute 0, the next occurrence should be the next hour.
	ref := time.Date(2026, 5, 15, 10, 0, 0, 0, time.UTC)
	next := NextOccurrence(sched, ref)

	expected := time.Date(2026, 5, 15, 11, 0, 0, 0, time.UTC)
	if !next.Equal(expected) {
		t.Fatalf("expected %v, got %v", expected, next)
	}
}

func TestNextOccurrenceEvery5Minutes(t *testing.T) {
	sched, _ := ParseCron("*/5 * * * *")
	ref := time.Date(2026, 5, 15, 10, 12, 0, 0, time.UTC)
	next := NextOccurrence(sched, ref)

	expected := time.Date(2026, 5, 15, 10, 15, 0, 0, time.UTC)
	if !next.Equal(expected) {
		t.Fatalf("expected %v, got %v", expected, next)
	}
}

func TestNextOccurrenceSpecificDay(t *testing.T) {
	// Every Monday at 2:30 AM.
	sched, _ := ParseCron("30 2 * * 1")
	// Thursday May 15, 2025.
	ref := time.Date(2025, 5, 15, 10, 0, 0, 0, time.UTC)
	next := NextOccurrence(sched, ref)

	// Next Monday is May 19, 2025.
	expected := time.Date(2025, 5, 19, 2, 30, 0, 0, time.UTC)
	if !next.Equal(expected) {
		t.Fatalf("expected %v, got %v", expected, next)
	}
}

func TestParseCronRangeAndList(t *testing.T) {
	sched, err := ParseCron("0 9-17 * * 1-5")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// Hours 9 through 17.
	if len(sched.Hour) != 9 {
		t.Fatalf("expected 9 hours, got %d: %v", len(sched.Hour), sched.Hour)
	}
	// Monday through Friday.
	if len(sched.DayOfWeek) != 5 {
		t.Fatalf("expected 5 days, got %d: %v", len(sched.DayOfWeek), sched.DayOfWeek)
	}
}
