// SPDX-License-Identifier: Apache-2.0

package feedback

import (
	"math"
	"strings"
	"testing"

	"github.com/jeffs-brain/memory/go/brain"
)

func TestClassify_PositiveSignal(t *testing.T) {
	c := NewClassifier()
	paths := []brain.Path{"memory/global/test.md"}
	result := c.Classify("perfect, thanks for that", paths)

	if len(result.Events) != 1 {
		t.Fatalf("expected 1 event, got %d", len(result.Events))
	}
	ev := result.Events[0]
	if ev.Reaction != ReactionReinforced {
		t.Errorf("reaction = %q, want %q", ev.Reaction, ReactionReinforced)
	}
	if ev.Confidence <= 0 {
		t.Error("expected positive confidence")
	}
}

func TestClassify_NegativeSignal(t *testing.T) {
	c := NewClassifier()
	paths := []brain.Path{"memory/global/test.md"}
	result := c.Classify("that's wrong, it was updated last week", paths)

	if len(result.Events) != 1 {
		t.Fatalf("expected 1 event, got %d", len(result.Events))
	}
	ev := result.Events[0]
	if ev.Reaction != ReactionCorrected {
		t.Errorf("reaction = %q, want %q", ev.Reaction, ReactionCorrected)
	}
}

func TestClassify_NeutralWhenNoPatterns(t *testing.T) {
	c := NewClassifier()
	paths := []brain.Path{"memory/global/test.md"}
	result := c.Classify("can you show me the deployment logs", paths)

	if len(result.Events) != 1 {
		t.Fatalf("expected 1 event, got %d", len(result.Events))
	}
	ev := result.Events[0]
	if ev.Reaction != ReactionNeutral {
		t.Errorf("reaction = %q, want %q", ev.Reaction, ReactionNeutral)
	}
	if ev.Confidence != 0.0 {
		t.Errorf("confidence = %f, want 0.0", ev.Confidence)
	}
}

func TestClassify_PositiveWinsWhenMoreMatches(t *testing.T) {
	c := NewClassifier()
	paths := []brain.Path{"memory/global/test.md"}
	result := c.Classify("no but perfect, that's helpful, spot on", paths)

	if len(result.Events) != 1 {
		t.Fatalf("expected 1 event, got %d", len(result.Events))
	}
	ev := result.Events[0]
	if ev.Reaction != ReactionReinforced {
		t.Errorf("reaction = %q, want %q", ev.Reaction, ReactionReinforced)
	}
}

func TestClassify_NegativeWinsWhenMoreMatches(t *testing.T) {
	c := NewClassifier()
	paths := []brain.Path{"memory/global/test.md"}
	result := c.Classify("yes but that's wrong, incorrect, try again", paths)

	if len(result.Events) != 1 {
		t.Fatalf("expected 1 event, got %d", len(result.Events))
	}
	ev := result.Events[0]
	if ev.Reaction != ReactionCorrected {
		t.Errorf("reaction = %q, want %q", ev.Reaction, ReactionCorrected)
	}
}

func TestClassify_TieGoesToNeutral(t *testing.T) {
	c := NewClassifier()
	paths := []brain.Path{"memory/global/test.md"}
	result := c.Classify("yes and no", paths)

	if len(result.Events) != 1 {
		t.Fatalf("expected 1 event, got %d", len(result.Events))
	}
	ev := result.Events[0]
	if ev.Reaction != ReactionNeutral {
		t.Errorf("reaction = %q, want %q", ev.Reaction, ReactionNeutral)
	}
	if ev.Confidence != 0.2 {
		t.Errorf("confidence = %f, want 0.2", ev.Confidence)
	}
}

func TestClassify_EmptyInput(t *testing.T) {
	c := NewClassifier()
	paths := []brain.Path{"memory/global/test.md"}
	result := c.Classify("", paths)

	if len(result.Events) != 0 {
		t.Errorf("expected 0 events for empty input, got %d", len(result.Events))
	}
}

func TestClassify_EmptySurfacedPaths(t *testing.T) {
	c := NewClassifier()
	result := c.Classify("perfect, thanks", nil)

	if len(result.Events) != 0 {
		t.Errorf("expected 0 events for nil surfaced paths, got %d", len(result.Events))
	}
}

func TestClassify_MultipleSurfacedPaths(t *testing.T) {
	c := NewClassifier()
	paths := []brain.Path{
		"memory/global/topic-a.md",
		"memory/global/topic-b.md",
		"memory/project/foo/bar.md",
	}
	result := c.Classify("great, thanks", paths)

	if len(result.Events) != 3 {
		t.Fatalf("expected 3 events, got %d", len(result.Events))
	}
	for i, ev := range result.Events {
		if ev.MemoryPath != paths[i] {
			t.Errorf("event[%d].MemoryPath = %q, want %q", i, ev.MemoryPath, paths[i])
		}
		if ev.Reaction != ReactionReinforced {
			t.Errorf("event[%d].Reaction = %q, want %q", i, ev.Reaction, ReactionReinforced)
		}
	}
}

func TestClassify_ConfidenceScalesWithMatchCount(t *testing.T) {
	c := NewClassifier()
	paths := []brain.Path{"memory/global/test.md"}

	r1 := c.Classify("thanks", paths)
	r2 := c.Classify("thanks, that's helpful, you remembered, that helps, spot on", paths)

	if len(r1.Events) != 1 || len(r2.Events) != 1 {
		t.Fatal("expected 1 event each")
	}

	c1 := r1.Events[0].Confidence
	c2 := r2.Events[0].Confidence

	if c1 >= c2 {
		t.Errorf("single-match confidence (%f) should be less than multi-match (%f)", c1, c2)
	}
	if math.Abs(c1-0.3) > 0.001 {
		t.Errorf("single-match confidence = %f, want ~0.3", c1)
	}
	if c2 > 1.0 {
		t.Errorf("confidence should be clamped to 1.0, got %f", c2)
	}
}

func TestClassify_PatternCaptured(t *testing.T) {
	c := NewClassifier()
	paths := []brain.Path{"memory/global/test.md"}
	result := c.Classify("you remembered that well", paths)

	if len(result.Events) != 1 {
		t.Fatalf("expected 1 event, got %d", len(result.Events))
	}
	ev := result.Events[0]
	if ev.Pattern == "" {
		t.Error("expected non-empty pattern")
	}
	if ev.Reaction != ReactionReinforced {
		t.Errorf("reaction = %q, want %q", ev.Reaction, ReactionReinforced)
	}
}

func TestClassify_SnippetTruncation(t *testing.T) {
	c := NewClassifier()
	paths := []brain.Path{"memory/global/test.md"}

	longInput := "perfect " + strings.Repeat("x", 300)
	result := c.Classify(longInput, paths)

	if len(result.Events) != 1 {
		t.Fatalf("expected 1 event, got %d", len(result.Events))
	}

	ev := result.Events[0]
	if len(ev.Snippet) > 203 {
		t.Errorf("snippet length = %d, want <= 203 (200 + '...')", len(ev.Snippet))
	}
	if !strings.HasSuffix(ev.Snippet, "...") {
		t.Errorf("truncated snippet should end with '...', got suffix %q", ev.Snippet[len(ev.Snippet)-10:])
	}

	if len(result.TurnContent) > 503 {
		t.Errorf("TurnContent length = %d, want <= 503", len(result.TurnContent))
	}
}
