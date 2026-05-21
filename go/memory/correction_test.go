// SPDX-License-Identifier: Apache-2.0

package memory

import (
	"strings"
	"testing"
)

func TestDetectCorrectionPositive(t *testing.T) {
	cases := []string{
		"No, it's actually 7am, not 8am.",
		"That's wrong, the meeting is on Tuesday.",
		"You got that wrong, the project is called Roo not Roe.",
		"Actually, it's blue not red.",
		"Stop saying I live in London, I live in Amersfoort.",
		"Wrong, it's Anthropic not OpenAI.",
		"I never said I wanted Slack messages, I wanted Telegram.",
		"That's not what I meant, please update memory.",
		"Please forget that I work at Google.",
		"Correction: my wife's name is Nadia, not Nadya.",
		"You're wrong about my role.",
		"Actually it's a Pi 4B, not a Pi 5.",
		"I didn't say that. I never use docker compose for this.",
		"It is actually 4 children, not 3.",
	}
	for _, c := range cases {
		got, ok := DetectCorrection(c)
		if !ok {
			t.Errorf("expected correction for %q", c)
			continue
		}
		if got.Snippet == "" {
			t.Errorf("expected snippet for %q", c)
		}
	}
}

func TestDetectCorrectionNegative(t *testing.T) {
	cases := []string{
		"No problem, I'll handle it.",
		"No idea what that means.",
		"No worries, take your time.",
		"You got the wrong end of the stick earlier.",
		"There's nothing wrong with the build.",
		"Not wrong per se, just not what I'd pick.",
		"No, before that I asked you to set up Slack.",
		"I think you're right about the route.",
		"Yes, that's right.",
		"Can you fix the bug in app.go?",
		"What is the status of the deploy?",
		"I see no problem with that approach.",
		"No, but seriously can you try again?",
	}
	for _, c := range cases {
		got, ok := DetectCorrection(c)
		if ok {
			t.Errorf("unexpected correction for %q: phrase=%q", c, got.Phrase)
		}
	}
}

func TestDetectCorrectionEmptyInput(t *testing.T) {
	if _, ok := DetectCorrection(""); ok {
		t.Errorf("expected no correction for empty input")
	}
	if _, ok := DetectCorrection("   \n\t  "); ok {
		t.Errorf("expected no correction for whitespace-only input")
	}
}

func TestDetectCorrectionSnippetCarriesOriginalCasing(t *testing.T) {
	input := "Actually it's Anthropic, not OpenAI - get it right next time."
	got, ok := DetectCorrection(input)
	if !ok {
		t.Fatalf("expected correction")
	}
	if !strings.Contains(strings.ToLower(got.Snippet), "anthropic") {
		t.Errorf("snippet %q should contain anthropic", got.Snippet)
	}
	if !strings.Contains(got.Snippet, "Anthropic") {
		t.Errorf("snippet %q should preserve original casing", got.Snippet)
	}
}

func TestBuildCorrectionReminder(t *testing.T) {
	got := BuildCorrectionReminder("Actually it's 7am not 8am, please update memory.")
	for _, want := range []string{"memory_search", "memory_update", "memory_remove", "memory_create", "Mention"} {
		if !strings.Contains(got, want) {
			t.Fatalf("reminder %q missing %q", got, want)
		}
	}
}

func TestBuildCorrectionReminderWithOptions(t *testing.T) {
	got := BuildCorrectionReminderWithOptions("Wrong, it is Tuesday.", CorrectionReminderOptions{
		SearchTool: "lookup",
		UpdateTool: "patch",
		RemoveTool: "retire",
		CreateTool: "create",
	})
	for _, want := range []string{"lookup", "patch", "retire", "create"} {
		if !strings.Contains(got, want) {
			t.Fatalf("reminder %q missing %q", got, want)
		}
	}
	if strings.Contains(got, "Mention") {
		t.Fatalf("unexpected mention instruction: %q", got)
	}
}
