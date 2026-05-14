// SPDX-License-Identifier: Apache-2.0

package memory

import (
	"fmt"
	"regexp"
	"strings"
)

// Correction is the structured result of DetectCorrection.
type Correction struct {
	Snippet string
	Phrase  string
}

// CorrectionReminderOptions configures BuildCorrectionReminderWithOptions.
type CorrectionReminderOptions struct {
	SearchTool    string
	CreateTool    string
	UpdateTool    string
	RemoveTool    string
	MentionChange bool
}

var correctionPhrases = []string{
	"that's wrong",
	"thats wrong",
	"you got that wrong",
	"you have that wrong",
	"you're wrong",
	"youre wrong",
	"you are wrong",
	"actually it's",
	"actually its",
	"actually it is",
	"it's actually",
	"its actually",
	"it is actually",
	"stop saying",
	"don't say",
	"do not say",
	"not right",
	"wrong, it's",
	"wrong, its",
	"wrong it's",
	"i never said",
	"i didn't say",
	"i did not say",
	"that's not",
	"thats not",
	"that is not",
	"correct that",
	"correction:",
	"please correct",
	"please update",
	"please remove",
	"please forget",
	"forget that",
	"forget what",
	"remember instead",
	"the correct",
}

var soloCorrectionPatterns = []*regexp.Regexp{
	regexp.MustCompile(`(?i)^\s*no[\s,\-]+(it'?s|the|that|it is|its|that is|its actually)\b`),
	regexp.MustCompile(`(?i)^\s*(wrong|incorrect)[\s,!\.]`),
	regexp.MustCompile(`(?i)^\s*actually[,\s\-]`),
}

var falsePositiveSubstrings = []string{
	"no problem",
	"no idea",
	"no worries",
	"no rush",
	"no big deal",
	"wrong end of the stick",
	"wrong number",
	"wrong place",
	"nothing wrong",
	"not wrong",
	"no, before that",
	"no, but",
}

// DetectCorrection returns a Correction when latestUserText looks like a
// correction of a previously stated fact or preference.
func DetectCorrection(latestUserText string) (Correction, bool) {
	if latestUserText == "" {
		return Correction{}, false
	}

	clean := strings.ToLower(latestUserText)
	clean = strings.TrimSpace(clean)
	if clean == "" {
		return Correction{}, false
	}

	for _, fp := range falsePositiveSubstrings {
		if strings.Contains(clean, fp) {
			return Correction{}, false
		}
	}

	for _, phrase := range correctionPhrases {
		if idx := strings.Index(clean, phrase); idx >= 0 {
			return Correction{
				Snippet: extractCorrectionSnippet(latestUserText, idx),
				Phrase:  phrase,
			}, true
		}
	}

	for _, re := range soloCorrectionPatterns {
		if loc := re.FindStringIndex(clean); loc != nil {
			return Correction{
				Snippet: extractCorrectionSnippet(latestUserText, loc[0]),
				Phrase:  re.String(),
			}, true
		}
	}

	return Correction{}, false
}

// BuildCorrectionReminder returns a default memory-tool reminder for a
// correction-looking turn, or the empty string when no correction is detected.
func BuildCorrectionReminder(userInput string) string {
	return BuildCorrectionReminderWithOptions(userInput, CorrectionReminderOptions{
		SearchTool:    "memory_search",
		UpdateTool:    "memory_update",
		RemoveTool:    "memory_remove",
		CreateTool:    "memory_create",
		MentionChange: true,
	})
}

// BuildCorrectionReminderWithOptions returns a per-turn instruction that
// nudges an assistant to repair stale memory before answering.
func BuildCorrectionReminderWithOptions(userInput string, opts CorrectionReminderOptions) string {
	c, ok := DetectCorrection(userInput)
	if !ok {
		return ""
	}
	snippet := c.Snippet
	if snippet == "" {
		snippet = userInput
	}

	searchTool := defaultString(opts.SearchTool, "memory_search")
	updateTool := defaultString(opts.UpdateTool, "memory_update")
	removeTool := defaultString(opts.RemoveTool, "memory_remove")
	createTool := defaultString(opts.CreateTool, "memory_create")

	reminder := fmt.Sprintf(
		"User correction detected (%q). Before answering, call %s for the disputed topic. If a stale entry exists, call %s or %s with a reason. If no entry exists yet but the correction is durable, call %s.",
		snippet,
		searchTool,
		updateTool,
		removeTool,
		createTool,
	)
	if opts.MentionChange {
		reminder += " Mention in your reply what you changed."
	}
	return reminder
}

func extractCorrectionSnippet(original string, matchStart int) string {
	const maxLen = 160
	runes := []rune(original)
	if len(runes) <= maxLen {
		return strings.TrimSpace(string(runes))
	}

	runeStart := 0
	for i := range original {
		if i >= matchStart {
			break
		}
		runeStart++
	}
	start := runeStart - maxLen/3
	if start < 0 {
		start = 0
	}
	end := start + maxLen
	if end > len(runes) {
		end = len(runes)
		start = end - maxLen
		if start < 0 {
			start = 0
		}
	}
	prefix := ""
	if start > 0 {
		prefix = "..."
	}
	suffix := ""
	if end < len(runes) {
		suffix = "..."
	}
	return strings.TrimSpace(prefix + string(runes[start:end]) + suffix)
}

func defaultString(value, fallback string) string {
	value = strings.TrimSpace(value)
	if value == "" {
		return fallback
	}
	return value
}
