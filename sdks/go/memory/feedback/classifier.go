// SPDX-License-Identifier: Apache-2.0

// Package feedback implements regex-based classification of implicit
// user feedback on surfaced memories. It is consumed by the memory
// reinforcement loop; the classifier is intentionally conservative so
// a false positive never demotes a good memory.
package feedback

import (
	"regexp"
	"strings"

	"github.com/jeffs-brain/memory/go/brain"
)

// Reaction classifies implicit user feedback on surfaced memories.
type Reaction string

const (
	ReactionReinforced Reaction = "reinforced"
	ReactionCorrected  Reaction = "corrected"
	ReactionNeutral    Reaction = "neutral"
)

// FeedbackEvent records a detected reaction on a specific memory.
type FeedbackEvent struct {
	MemoryPath brain.Path
	Reaction   Reaction
	Confidence float64
	Pattern    string
	Snippet    string
}

// ClassifyResult holds the classification outcome for one turn.
type ClassifyResult struct {
	Events      []FeedbackEvent
	TurnContent string
}

// Classifier detects implicit user feedback from the next user turn.
type Classifier struct {
	positive []*regexp.Regexp
	negative []*regexp.Regexp
}

// NewClassifier creates a regex-based feedback classifier.
func NewClassifier() *Classifier {
	return &Classifier{
		positive: compilePatterns(positivePatterns),
		negative: compilePatterns(negativePatterns),
	}
}

// positivePatterns detect reinforcement signals.
var positivePatterns = []string{
	`(?i)\b(perfect|exactly|great|thanks|correct|right|yes)\b`,
	`(?i)\bthat('s| is| was) (right|correct|helpful|useful|what i needed)\b`,
	`(?i)\b(good|nice) (memory|recall|find)\b`,
	`(?i)\byou remembered\b`,
	`(?i)\bthat helps\b`,
	`(?i)\bspot on\b`,
}

// negativePatterns detect correction signals.
var negativePatterns = []string{
	`(?i)\b(wrong|incorrect|no|nope|not right)\b`,
	`(?i)\bthat('s| is| was) (wrong|incorrect|outdated|old|stale)\b`,
	`(?i)\b(forget|remove|delete) (that|this|it)\b`,
	`(?i)\bnot what i (meant|asked|wanted)\b`,
	`(?i)\btry again\b`,
	`(?i)\bthat('s| is) (not|no longer) (true|accurate|relevant)\b`,
	`(?i)\bactually[,.]?\s`,
}

func compilePatterns(patterns []string) []*regexp.Regexp {
	out := make([]*regexp.Regexp, 0, len(patterns))
	for _, p := range patterns {
		if r, err := regexp.Compile(p); err == nil {
			out = append(out, r)
		}
	}
	return out
}

// Classify analyses a user turn to detect feedback on memories that
// were surfaced in the previous turn.
func (c *Classifier) Classify(userInput string, surfacedThisTurn []brain.Path) ClassifyResult {
	result := ClassifyResult{
		TurnContent: truncateSnippet(userInput, 500),
	}

	if len(surfacedThisTurn) == 0 || strings.TrimSpace(userInput) == "" {
		return result
	}

	reaction, confidence, pattern := c.detectReaction(userInput)

	for _, path := range surfacedThisTurn {
		result.Events = append(result.Events, FeedbackEvent{
			MemoryPath: path,
			Reaction:   reaction,
			Confidence: confidence,
			Pattern:    pattern,
			Snippet:    truncateSnippet(userInput, 200),
		})
	}

	return result
}

func (c *Classifier) detectReaction(input string) (Reaction, float64, string) {
	posMatches := 0
	posPattern := ""
	for _, r := range c.positive {
		if loc := r.FindString(input); loc != "" {
			posMatches++
			if posPattern == "" {
				posPattern = loc
			}
		}
	}

	negMatches := 0
	negPattern := ""
	for _, r := range c.negative {
		if loc := r.FindString(input); loc != "" {
			negMatches++
			if negPattern == "" {
				negPattern = loc
			}
		}
	}

	if posMatches == 0 && negMatches == 0 {
		return ReactionNeutral, 0.0, ""
	}

	if posMatches > negMatches {
		conf := clamp(float64(posMatches) * 0.3)
		return ReactionReinforced, conf, posPattern
	}
	if negMatches > posMatches {
		conf := clamp(float64(negMatches) * 0.3)
		return ReactionCorrected, conf, negPattern
	}

	return ReactionNeutral, 0.2, ""
}

func clamp(v float64) float64 {
	if v > 1.0 {
		return 1.0
	}
	return v
}

func truncateSnippet(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}
