// SPDX-License-Identifier: Apache-2.0

package lme

import (
	"strings"
	"unicode"
)

// ScoreExactMatch evaluates question outcomes using normalised
// exact-match comparison against ground-truth answers. This is the
// Phase 0 lower-bound scorer; the LLM judge supersedes it for
// partial/paraphrase credit.
func ScoreExactMatch(outcomes []QuestionOutcome) *LMEResult {
	byCategory := make(map[string]*Category)
	correct := 0

	for i, o := range outcomes {
		cat, ok := byCategory[o.Category]
		if !ok {
			cat = &Category{}
			byCategory[o.Category] = cat
		}
		cat.Run++

		if o.Error != "" {
			cat.Incorrect++
			continue
		}

		if exactMatch(o.AgentAnswer, o.GroundTruth) {
			cat.Correct++
			correct++
			outcomes[i].JudgeVerdict = "correct"
		} else {
			cat.Incorrect++
			outcomes[i].JudgeVerdict = "incorrect"
		}
	}

	catMap := make(map[string]Category, len(byCategory))
	for name, cat := range byCategory {
		if cat.Run > 0 {
			cat.Score = float64(cat.Correct) / float64(cat.Run)
		}
		catMap[name] = *cat
	}

	overall := 0.0
	if len(outcomes) > 0 {
		overall = float64(correct) / float64(len(outcomes))
	}

	taskAvg := 0.0
	if len(catMap) > 0 {
		sum := 0.0
		for _, cat := range catMap {
			sum += cat.Score
		}
		taskAvg = sum / float64(len(catMap))
	}

	return &LMEResult{
		QuestionsRun:    len(outcomes),
		OverallScore:    overall,
		TaskAvgScore:    taskAvg,
		ExactMatchScore: overall,
		ByCategory:      catMap,
		Questions:       outcomes,
	}
}

// exactMatch returns true if the normalised agent answer contains the
// normalised ground truth.
func exactMatch(agent, truth string) bool {
	na := normalise(agent)
	nt := normalise(truth)
	if na == "" || nt == "" {
		return false
	}
	return strings.Contains(na, nt)
}

// normalise prepares a string for exact-match comparison: lowercase,
// strip non-alphanumeric characters (except spaces), collapse whitespace
// runs.
func normalise(s string) string {
	s = strings.ToLower(s)
	var b strings.Builder
	b.Grow(len(s))
	prevSpace := false
	for _, r := range s {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			b.WriteRune(r)
			prevSpace = false
		} else if !prevSpace {
			b.WriteByte(' ')
			prevSpace = true
		}
	}
	return strings.TrimSpace(b.String())
}
