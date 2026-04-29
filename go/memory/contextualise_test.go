// SPDX-License-Identifier: Apache-2.0

package memory

import (
	"context"
	"fmt"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/jeffs-brain/memory/go/llm"
)

// stubProvider is a deterministic test [llm.Provider] that records the
// request payload so tests can assert on the prompt and count calls.
type stubProvider struct {
	reply string
	err   error
	calls int32

	lastSystem string
	lastUser   string
}

func (s *stubProvider) Complete(_ context.Context, req llm.CompleteRequest) (llm.CompleteResponse, error) {
	atomic.AddInt32(&s.calls, 1)
	for _, m := range req.Messages {
		if m.Role == llm.RoleSystem {
			s.lastSystem = m.Content
		}
		if m.Role == llm.RoleUser {
			s.lastUser = m.Content
		}
	}
	if s.err != nil {
		return llm.CompleteResponse{}, s.err
	}
	return llm.CompleteResponse{Text: s.reply}, nil
}

func (s *stubProvider) CompleteStream(_ context.Context, _ llm.CompleteRequest) (<-chan llm.StreamChunk, error) {
	return nil, fmt.Errorf("not implemented")
}

func (s *stubProvider) Close() error { return nil }

func TestContextualiser_BuildPrefixReturnsTrimmedSinglePara(t *testing.T) {
	p := &stubProvider{reply: "  Context: On Monday the user discussed\n  the blue car they bought.  "}
	c := NewContextualiser(ContextualiserConfig{Provider: p, CacheDir: t.TempDir()})
	if c == nil {
		t.Fatal("NewContextualiser returned nil")
	}
	got := c.BuildPrefix(context.Background(), "sess-1", "session_date=2024-03-25", "User bought a blue car.")
	if got == "" {
		t.Fatal("expected non-empty prefix")
	}
	if strings.Contains(got, "\n") {
		t.Errorf("prefix should be single line, got %q", got)
	}
	if strings.HasPrefix(strings.ToLower(got), "context:") {
		t.Errorf("prefix should not carry the marker label, got %q", got)
	}
	if !strings.Contains(got, "blue car") {
		t.Errorf("prefix should retain key content, got %q", got)
	}
}

func TestContextualiser_CachesByFingerprint(t *testing.T) {
	p := &stubProvider{reply: "On Monday the user discussed cars."}
	c := NewContextualiser(ContextualiserConfig{Provider: p, CacheDir: t.TempDir()})
	ctx := context.Background()

	first := c.BuildPrefix(ctx, "sess-1", "summary", "body text")
	second := c.BuildPrefix(ctx, "sess-1", "summary", "body text")

	if first == "" || first != second {
		t.Fatalf("cached call mismatch: first=%q second=%q", first, second)
	}
	if got := atomic.LoadInt32(&p.calls); got != 1 {
		t.Errorf("provider called %d times, want 1 (second call should hit cache)", got)
	}
}

func TestContextualiser_CacheBustsOnModelChange(t *testing.T) {
	dir := t.TempDir()
	p := &stubProvider{reply: "first-model prefix."}
	c1 := NewContextualiser(ContextualiserConfig{Provider: p, Model: "model-a", CacheDir: dir})
	_ = c1.BuildPrefix(context.Background(), "sess-1", "", "body text")

	p2 := &stubProvider{reply: "second-model prefix."}
	c2 := NewContextualiser(ContextualiserConfig{Provider: p2, Model: "model-b", CacheDir: dir})
	got := c2.BuildPrefix(context.Background(), "sess-1", "", "body text")

	if got != "second-model prefix." {
		t.Errorf("model-b should hit provider fresh, got %q", got)
	}
	if got := atomic.LoadInt32(&p2.calls); got != 1 {
		t.Errorf("model-b provider calls = %d, want 1", got)
	}
}

func TestContextualiser_FailsOpenOnProviderError(t *testing.T) {
	p := &stubProvider{err: fmt.Errorf("boom")}
	c := NewContextualiser(ContextualiserConfig{Provider: p, CacheDir: t.TempDir()})
	got := c.BuildPrefix(context.Background(), "sess-1", "summary", "body text")
	if got != "" {
		t.Errorf("expected empty prefix on provider error, got %q", got)
	}
}

func TestContextualiser_NilReceiverIsDisabled(t *testing.T) {
	var c *Contextualiser
	if c.Enabled() {
		t.Error("nil receiver should not be Enabled")
	}
	if got := c.BuildPrefix(context.Background(), "sess", "", "body"); got != "" {
		t.Errorf("nil receiver BuildPrefix = %q, want empty", got)
	}
}

func TestApplyContextualPrefix_Shape(t *testing.T) {
	cases := []struct {
		name   string
		prefix string
		body   string
		want   string
	}{
		{"empty prefix passes body through", "", "fact body", "fact body"},
		{"prefix prepended with marker", "session context", "fact body", "Context: session context\n\nfact body"},
		{"whitespace-only prefix treated as empty", "   ", "fact body", "fact body"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := ApplyContextualPrefix(tc.prefix, tc.body)
			if got != tc.want {
				t.Errorf("ApplyContextualPrefix(%q, %q) = %q, want %q", tc.prefix, tc.body, got, tc.want)
			}
		})
	}
}

func TestBuildTopicFileContent_WritesSessionDate(t *testing.T) {
	em := ExtractedMemory{
		Action:      "create",
		Filename:    "fact.md",
		Name:        "Test fact",
		Description: "desc",
		Type:        "project",
		Content:     "body",
		Scope:       "project",
		SessionID:   "sess-42",
		SessionDate: "2024-03-25",
		ObservedOn:  "2024-03-25T09:00:00Z",
	}
	got := string(buildTopicFileContent(em))
	if !strings.Contains(got, "session_date: 2024-03-25\n") {
		t.Errorf("missing session_date frontmatter in:\n%s", got)
	}
	if !strings.Contains(got, "session_id: sess-42\n") {
		t.Errorf("missing session_id frontmatter in:\n%s", got)
	}
}

func TestBuildTopicFileContent_AppliesContextPrefix(t *testing.T) {
	em := ExtractedMemory{
		Action:        "create",
		Filename:      "fact.md",
		Name:          "Test fact",
		Type:          "project",
		Content:       "User bought a blue car.",
		Scope:         "project",
		ContextPrefix: "On Monday the user discussed cars.",
	}
	got := string(buildTopicFileContent(em))
	if !strings.Contains(got, "Context: On Monday the user discussed cars.\n\nUser bought a blue car.") {
		t.Errorf("context prefix not prepended correctly in:\n%s", got)
	}
}

func TestExtractSessionSummary_PrefersSystemOverUser(t *testing.T) {
	msgs := []Message{
		{Role: RoleSystem, Content: "This conversation took place on 2024-03-25."},
		{Role: RoleUser, Content: "What time is it?"},
	}
	got := extractSessionSummary(msgs)
	if !strings.Contains(got, "2024-03-25") {
		t.Errorf("expected system date in summary, got %q", got)
	}
}
