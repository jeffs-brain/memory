// SPDX-License-Identifier: Apache-2.0

package memory

import (
	"testing"
	"time"
)

func TestDeriveHeuristicConfidence(t *testing.T) {
	// Fixed reference time to ensure deterministic tests.
	now := time.Date(2025, 6, 1, 12, 0, 0, 0, time.UTC)

	tests := []struct {
		name       string
		current    string
		observedAt time.Time
		createdAt  time.Time
		want       string
	}{
		{
			name:       "fresh memory (1 day old) retains full confidence",
			current:    "high",
			observedAt: now.AddDate(0, 0, -1),
			createdAt:  now.AddDate(0, 0, -1),
			want:       "high",
		},
		{
			name:       "30 day old, recently accessed retains full confidence (reinforced via strong span)",
			current:    "medium",
			observedAt: now.AddDate(0, 0, -5),
			createdAt:  now.AddDate(0, 0, -60),
			want:       "high",
		},
		{
			name:       "100 day old, never accessed since creation is reduced (high to medium)",
			current:    "high",
			observedAt: now.AddDate(0, 0, -100),
			createdAt:  now.AddDate(0, 0, -100),
			want:       "medium",
		},
		{
			name:       "100 day old, never accessed since creation is reduced (medium to low)",
			current:    "medium",
			observedAt: now.AddDate(0, 0, -100),
			createdAt:  now.AddDate(0, 0, -100),
			want:       "low",
		},
		{
			name:       "200 day old, never accessed forces minimum confidence",
			current:    "high",
			observedAt: now.AddDate(0, 0, -200),
			createdAt:  now.AddDate(0, 0, -200),
			want:       "low",
		},
		{
			name:       "200 day old, accessed yesterday retains full confidence (reinforced)",
			current:    "medium",
			observedAt: now.AddDate(0, 0, -1),
			createdAt:  now.AddDate(0, 0, -200),
			want:       "high",
		},
		{
			name:       "future observedAt date handled gracefully (negative age = fresh)",
			current:    "high",
			observedAt: now.AddDate(0, 0, 5),
			createdAt:  now.AddDate(0, 0, -10),
			want:       "high",
		},
		{
			name:       "future createdAt date handled gracefully (negative reinforcement clamped to 0)",
			current:    "low",
			observedAt: now.AddDate(0, 0, -1),
			createdAt:  now.AddDate(0, 0, 5),
			want:       "low",
		},
		{
			name:       "exactly 90 days old (at boundary) demotes high to medium",
			current:    "high",
			observedAt: now.AddDate(0, 0, -90),
			createdAt:  now.AddDate(0, 0, -90),
			want:       "medium",
		},
		{
			name:       "exactly 90 days old (at boundary) demotes medium to low",
			current:    "medium",
			observedAt: now.AddDate(0, 0, -90),
			createdAt:  now.AddDate(0, 0, -90),
			want:       "low",
		},
		{
			name:       "exactly 90 days old (at boundary) low stays low",
			current:    "low",
			observedAt: now.AddDate(0, 0, -90),
			createdAt:  now.AddDate(0, 0, -90),
			want:       "low",
		},
		{
			name:       "exactly 180 days old (at boundary) forces low regardless of current",
			current:    "high",
			observedAt: now.AddDate(0, 0, -180),
			createdAt:  now.AddDate(0, 0, -180),
			want:       "low",
		},
		{
			name:       "89 days old is not stale",
			current:    "high",
			observedAt: now.AddDate(0, 0, -89),
			createdAt:  now.AddDate(0, 0, -89),
			want:       "high",
		},
		{
			name:       "179 days old with high current is stale but not deep stale",
			current:    "high",
			observedAt: now.AddDate(0, 0, -179),
			createdAt:  now.AddDate(0, 0, -179),
			want:       "medium",
		},
		{
			name:       "reinforcement of exactly 14 days promotes low to medium",
			current:    "low",
			observedAt: now.AddDate(0, 0, -5),
			createdAt:  now.AddDate(0, 0, -19),
			want:       "medium",
		},
		{
			name:       "reinforcement of exactly 14 days keeps high as high",
			current:    "high",
			observedAt: now.AddDate(0, 0, -5),
			createdAt:  now.AddDate(0, 0, -19),
			want:       "high",
		},
		{
			name:       "reinforcement of exactly 45 days promotes to high unconditionally",
			current:    "low",
			observedAt: now.AddDate(0, 0, -5),
			createdAt:  now.AddDate(0, 0, -50),
			want:       "high",
		},
		{
			name:       "reinforcement of 13 days does not promote",
			current:    "low",
			observedAt: now.AddDate(0, 0, -5),
			createdAt:  now.AddDate(0, 0, -18),
			want:       "low",
		},
		{
			name:       "unrecognised confidence value normalises to low",
			current:    "invalid",
			observedAt: now.AddDate(0, 0, -5),
			createdAt:  now.AddDate(0, 0, -5),
			want:       "low",
		},
		{
			name:       "empty confidence value normalises to low",
			current:    "",
			observedAt: now.AddDate(0, 0, -5),
			createdAt:  now.AddDate(0, 0, -5),
			want:       "low",
		},
		{
			name:       "stale with reinforcement: age takes priority over reinforcement",
			current:    "high",
			observedAt: now.AddDate(0, 0, -95),
			createdAt:  now.AddDate(0, 0, -140),
			want:       "medium",
		},
		{
			name:       "deep stale with reinforcement: deep staleness always wins",
			current:    "high",
			observedAt: now.AddDate(0, 0, -185),
			createdAt:  now.AddDate(0, 0, -230),
			want:       "low",
		},
		{
			name:       "zero time observedAt treats memory as maximally stale",
			current:    "high",
			observedAt: time.Time{},
			createdAt:  time.Time{},
			want:       "low",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := DeriveHeuristicConfidence(tc.current, tc.observedAt, tc.createdAt, now)
			if got != tc.want {
				t.Errorf("DeriveHeuristicConfidence(%q, observedAt=%v, createdAt=%v, now=%v) = %q, want %q",
					tc.current, tc.observedAt, tc.createdAt, now, got, tc.want)
			}
		})
	}
}

func TestDiffDaysTruncated(t *testing.T) {
	tests := []struct {
		name  string
		start time.Time
		end   time.Time
		want  int
	}{
		{
			name:  "same instant",
			start: time.Date(2025, 1, 1, 0, 0, 0, 0, time.UTC),
			end:   time.Date(2025, 1, 1, 0, 0, 0, 0, time.UTC),
			want:  0,
		},
		{
			name:  "exactly one day",
			start: time.Date(2025, 1, 1, 0, 0, 0, 0, time.UTC),
			end:   time.Date(2025, 1, 2, 0, 0, 0, 0, time.UTC),
			want:  1,
		},
		{
			name:  "23 hours is 0 days",
			start: time.Date(2025, 1, 1, 0, 0, 0, 0, time.UTC),
			end:   time.Date(2025, 1, 1, 23, 59, 59, 0, time.UTC),
			want:  0,
		},
		{
			name:  "negative direction",
			start: time.Date(2025, 1, 10, 0, 0, 0, 0, time.UTC),
			end:   time.Date(2025, 1, 1, 0, 0, 0, 0, time.UTC),
			want:  -9,
		},
		{
			name:  "90 days exactly",
			start: time.Date(2025, 1, 1, 0, 0, 0, 0, time.UTC),
			end:   time.Date(2025, 4, 1, 0, 0, 0, 0, time.UTC),
			want:  90,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := diffDaysTruncated(tc.start, tc.end)
			if got != tc.want {
				t.Errorf("diffDaysTruncated(%v, %v) = %d, want %d", tc.start, tc.end, got, tc.want)
			}
		})
	}
}

func TestNormaliseConfidenceLevel(t *testing.T) {
	tests := []struct {
		input string
		want  string
	}{
		{"high", "high"},
		{"medium", "medium"},
		{"low", "low"},
		{"", "low"},
		{"invalid", "low"},
		{"HIGH", "low"},
		{"Medium", "low"},
	}

	for _, tc := range tests {
		t.Run(tc.input, func(t *testing.T) {
			got := normaliseConfidenceLevel(tc.input)
			if got != tc.want {
				t.Errorf("normaliseConfidenceLevel(%q) = %q, want %q", tc.input, got, tc.want)
			}
		})
	}
}
