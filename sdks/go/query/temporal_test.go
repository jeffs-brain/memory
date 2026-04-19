// SPDX-License-Identifier: Apache-2.0

package query

import (
	"strings"
	"testing"
)

func TestExpandTemporal_TwoWeeksAgo(t *testing.T) {
	// 2023/04/10 is a Monday. Two weeks ago = 2023/03/27.
	exp := ExpandTemporal(
		"What did we discuss 2 weeks ago?",
		"2023/04/10 (Mon) 23:07",
	)
	if !exp.Resolved {
		t.Fatal("expected Resolved to be true")
	}
	if !strings.Contains(exp.ExpandedQuery, "2023/03/27") {
		t.Errorf("expected expanded query to contain 2023/03/27, got %q", exp.ExpandedQuery)
	}
	if len(exp.DateHints) != 1 || exp.DateHints[0] != "2023/03/27" {
		t.Errorf("expected DateHints [2023/03/27], got %v", exp.DateHints)
	}
}

func TestExpandTemporal_TwoWeeksAgo_NumberWord(t *testing.T) {
	exp := ExpandTemporal(
		"What did we discuss two weeks ago?",
		"2023/04/10 (Mon) 23:07",
	)
	if !exp.Resolved {
		t.Fatal("expected Resolved to be true")
	}
	if !strings.Contains(exp.ExpandedQuery, "2023/03/27") {
		t.Errorf("expected expanded query to contain 2023/03/27, got %q", exp.ExpandedQuery)
	}
	if len(exp.DateHints) != 1 || exp.DateHints[0] != "2023/03/27" {
		t.Errorf("expected DateHints [2023/03/27], got %v", exp.DateHints)
	}
}

func TestExpandTemporal_ThreeDaysAgo(t *testing.T) {
	// 2023/04/10 minus 3 days = 2023/04/07.
	exp := ExpandTemporal(
		"Who messaged me 3 days ago?",
		"2023/04/10 (Mon) 23:07",
	)
	if !exp.Resolved {
		t.Fatal("expected Resolved to be true")
	}
	if !strings.Contains(exp.ExpandedQuery, "2023/04/07") {
		t.Errorf("expected expanded query to contain 2023/04/07, got %q", exp.ExpandedQuery)
	}
	if len(exp.DateHints) != 1 || exp.DateHints[0] != "2023/04/07" {
		t.Errorf("expected DateHints [2023/04/07], got %v", exp.DateHints)
	}
}

func TestExpandTemporal_OneMonthAgo(t *testing.T) {
	// 2023/04/10 minus 1 month = 2023/03/10.
	exp := ExpandTemporal(
		"What happened 1 month ago?",
		"2023/04/10 (Mon) 23:07",
	)
	if !exp.Resolved {
		t.Fatal("expected Resolved to be true")
	}
	if !strings.Contains(exp.ExpandedQuery, "2023/03/10") {
		t.Errorf("expected expanded query to contain 2023/03/10, got %q", exp.ExpandedQuery)
	}
	if len(exp.DateHints) != 1 || exp.DateHints[0] != "2023/03/10" {
		t.Errorf("expected DateHints [2023/03/10], got %v", exp.DateHints)
	}
}

func TestExpandTemporal_LastSaturday(t *testing.T) {
	// 2023/04/10 is a Monday. Last Saturday = 2023/04/08.
	exp := ExpandTemporal(
		"What was said last Saturday?",
		"2023/04/10 (Mon) 23:07",
	)
	if !exp.Resolved {
		t.Fatal("expected Resolved to be true")
	}
	if !strings.Contains(exp.ExpandedQuery, "2023/04/08") {
		t.Errorf("expected expanded query to contain 2023/04/08, got %q", exp.ExpandedQuery)
	}
	if len(exp.DateHints) != 1 || exp.DateHints[0] != "2023/04/08" {
		t.Errorf("expected DateHints [2023/04/08], got %v", exp.DateHints)
	}
}

func TestExpandTemporal_Today(t *testing.T) {
	exp := ExpandTemporal(
		"What did I do today?",
		"2023/04/10 (Mon) 23:07",
	)
	if !exp.Resolved {
		t.Fatal("expected Resolved to be true")
	}
	if !strings.Contains(exp.ExpandedQuery, "2023/04/10") {
		t.Errorf("expected expanded query to contain 2023/04/10, got %q", exp.ExpandedQuery)
	}
	if len(exp.DateHints) != 1 || exp.DateHints[0] != "2023/04/10" {
		t.Errorf("expected DateHints [2023/04/10], got %v", exp.DateHints)
	}
}

func TestExpandTemporal_Yesterday(t *testing.T) {
	exp := ExpandTemporal(
		"What did I do yesterday?",
		"2023/04/10 (Mon) 23:07",
	)
	if !exp.Resolved {
		t.Fatal("expected Resolved to be true")
	}
	if !strings.Contains(exp.ExpandedQuery, "2023/04/09") {
		t.Errorf("expected expanded query to contain 2023/04/09, got %q", exp.ExpandedQuery)
	}
	if len(exp.DateHints) != 1 || exp.DateHints[0] != "2023/04/09" {
		t.Errorf("expected DateHints [2023/04/09], got %v", exp.DateHints)
	}
}

func TestExpandTemporal_LastWeek(t *testing.T) {
	exp := ExpandTemporal(
		"Where did I volunteer last week?",
		"2023/04/10 (Mon) 23:07",
	)
	if !exp.Resolved {
		t.Fatal("expected Resolved to be true")
	}
	if !strings.Contains(exp.ExpandedQuery, "2023/04/03") || !strings.Contains(exp.ExpandedQuery, "2023/04/09") {
		t.Errorf("expected expanded query to contain last-week bounds, got %q", exp.ExpandedQuery)
	}
	if len(exp.DateHints) != 7 {
		t.Fatalf("expected 7 DateHints, got %v", exp.DateHints)
	}
	if exp.DateHints[0] != "2023/04/03" || exp.DateHints[6] != "2023/04/09" {
		t.Errorf("expected DateHints to span 2023/04/03..2023/04/09, got %v", exp.DateHints)
	}
}

func TestExpandTemporal_NoTemporalReferences(t *testing.T) {
	exp := ExpandTemporal(
		"What is the capital of France?",
		"2023/04/10 (Mon) 23:07",
	)
	if exp.Resolved {
		t.Error("expected Resolved to be false for question without temporal references")
	}
	if exp.ExpandedQuery != exp.OriginalQuery {
		t.Errorf("expected query unchanged, got %q", exp.ExpandedQuery)
	}
	if len(exp.DateHints) != 0 {
		t.Errorf("expected no DateHints, got %v", exp.DateHints)
	}
}

func TestExpandTemporal_EmptyQuestionDate(t *testing.T) {
	exp := ExpandTemporal("What happened 2 weeks ago?", "")
	if exp.Resolved {
		t.Error("expected Resolved to be false for empty question date")
	}
	if exp.ExpandedQuery != exp.OriginalQuery {
		t.Errorf("expected query unchanged, got %q", exp.ExpandedQuery)
	}
}

func TestParseQuestionDate_AllFormats(t *testing.T) {
	tests := []struct {
		input   string
		wantDay int
		wantMon int
		wantYr  int
	}{
		{"2023/04/10 (Mon) 23:07", 10, 4, 2023},
		{"2023/04/10 23:07", 10, 4, 2023},
		{"2023/04/10", 10, 4, 2023},
		{"2023-04-10", 10, 4, 2023},
	}
	for _, tt := range tests {
		parsed, err := parseQuestionDate(tt.input)
		if err != nil {
			t.Errorf("parseQuestionDate(%q): unexpected error: %v", tt.input, err)
			continue
		}
		if parsed.Day() != tt.wantDay || int(parsed.Month()) != tt.wantMon || parsed.Year() != tt.wantYr {
			t.Errorf("parseQuestionDate(%q) = %v, want %d/%d/%d",
				tt.input, parsed, tt.wantYr, tt.wantMon, tt.wantDay)
		}
	}
}

func TestParseQuestionDate_Invalid(t *testing.T) {
	tests := []string{
		"",
		"not a date",
		"10/04/2023",
		"April 10, 2023",
	}
	for _, input := range tests {
		_, err := parseQuestionDate(input)
		if err == nil {
			t.Errorf("parseQuestionDate(%q): expected error, got nil", input)
		}
	}
}

func TestAnnotateOrdering_First(t *testing.T) {
	result := annotateOrdering("When did we first discuss the project?")
	if !strings.Contains(result, "earliest dated event") {
		t.Errorf("expected earliest hint, got %q", result)
	}
}

func TestAnnotateOrdering_MostRecent(t *testing.T) {
	result := annotateOrdering("What was the most recent update?")
	if !strings.Contains(result, "most recently dated event") {
		t.Errorf("expected most recent hint, got %q", result)
	}
}

func TestAnnotateOrdering_NoOrdering(t *testing.T) {
	input := "What colour is the sky?"
	result := annotateOrdering(input)
	if result != input {
		t.Errorf("expected no annotation, got %q", result)
	}
}

func TestExpandTemporal_MultipleReferences(t *testing.T) {
	// 2023/04/10 (Mon). 2 weeks ago = 2023/03/27. 3 days ago = 2023/04/07.
	exp := ExpandTemporal(
		"Compare what happened 2 weeks ago with 3 days ago",
		"2023/04/10 (Mon) 23:07",
	)
	if !exp.Resolved {
		t.Fatal("expected Resolved to be true")
	}
	if !strings.Contains(exp.ExpandedQuery, "2023/03/27") {
		t.Errorf("expected expanded query to contain 2023/03/27, got %q", exp.ExpandedQuery)
	}
	if !strings.Contains(exp.ExpandedQuery, "2023/04/07") {
		t.Errorf("expected expanded query to contain 2023/04/07, got %q", exp.ExpandedQuery)
	}
	if len(exp.DateHints) != 2 {
		t.Errorf("expected 2 DateHints, got %v", exp.DateHints)
	}
}
