// SPDX-License-Identifier: Apache-2.0

package lme

import "testing"

func TestNormalise(t *testing.T) {
	cases := []struct {
		input string
		want  string
	}{
		{"Hello World!", "hello world"},
		{"It's   a  test.", "it s a test"},
		{"  spaces  ", "spaces"},
		{"42", "42"},
		{"café", "café"},
		{"", ""},
		{"UPPER", "upper"},
		{"a--b..c", "a b c"},
	}
	for _, tc := range cases {
		t.Run(tc.input, func(t *testing.T) {
			got := normalise(tc.input)
			if got != tc.want {
				t.Fatalf("normalise(%q) = %q, want %q", tc.input, got, tc.want)
			}
		})
	}
}

func TestExactMatch(t *testing.T) {
	cases := []struct {
		name  string
		agent string
		truth string
		want  bool
	}{
		{"identical", "red", "red", true},
		{"case insensitive", "Red", "red", true},
		{"containment", "The answer is red", "red", true},
		{"whitespace", "the  red   car", "the red car", true},
		{"punctuation strips to words", "It's red!", "it s red", true},
		{"punctuation mismatch contraction", "It's red!", "its red", false},
		{"numbers", "42", "42", true},
		{"mismatch", "blue", "red", false},
		{"empty agent", "", "red", false},
		{"empty truth", "red", "", false},
		{"both empty", "", "", false},
		{"partial word no match", "redirect", "red car", false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := exactMatch(tc.agent, tc.truth)
			if got != tc.want {
				t.Fatalf("exactMatch(%q, %q) = %v, want %v", tc.agent, tc.truth, got, tc.want)
			}
		})
	}
}

func TestScoreExactMatch(t *testing.T) {
	outcomes := []QuestionOutcome{
		{ID: "q1", Category: "single-session", AgentAnswer: "red", GroundTruth: "red"},
		{ID: "q2", Category: "single-session", AgentAnswer: "blue", GroundTruth: "red"},
		{ID: "q3", Category: "temporal", AgentAnswer: "2024", GroundTruth: "2024"},
		{ID: "q4", Category: "temporal", AgentAnswer: "wrong", GroundTruth: "2024"},
		{ID: "q5", Category: "temporal", AgentAnswer: "", GroundTruth: "2024", Error: "timeout"},
	}

	result := ScoreExactMatch(outcomes)

	if result.QuestionsRun != 5 {
		t.Errorf("QuestionsRun = %d, want 5", result.QuestionsRun)
	}

	wantOverall := 0.4
	if result.OverallScore != wantOverall {
		t.Errorf("OverallScore = %v, want %v", result.OverallScore, wantOverall)
	}

	ss := result.ByCategory["single-session"]
	if ss.Correct != 1 || ss.Incorrect != 1 || ss.Run != 2 {
		t.Errorf("single-session = %+v, want Correct=1, Incorrect=1, Run=2", ss)
	}
	if ss.Score != 0.5 {
		t.Errorf("single-session Score = %v, want 0.5", ss.Score)
	}

	tmp := result.ByCategory["temporal"]
	if tmp.Correct != 1 || tmp.Incorrect != 2 || tmp.Run != 3 {
		t.Errorf("temporal = %+v, want Correct=1, Incorrect=2, Run=3", tmp)
	}
}

func TestScoreExactMatch_Empty(t *testing.T) {
	result := ScoreExactMatch(nil)
	if result.QuestionsRun != 0 {
		t.Errorf("QuestionsRun = %d, want 0", result.QuestionsRun)
	}
	if result.OverallScore != 0 {
		t.Errorf("OverallScore = %v, want 0", result.OverallScore)
	}
}
