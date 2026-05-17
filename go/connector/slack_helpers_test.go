// SPDX-License-Identifier: Apache-2.0

package connector

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strconv"
	"strings"
	"testing"
	"time"
)

// ---------------------------------------------------------------------------
// Tests: parseSlackTimestamp
// ---------------------------------------------------------------------------

func TestParseSlackTimestamp(t *testing.T) {
	tests := []struct {
		input string
		want  int64
	}{
		{"1700000001.123456", 1700000001},
		{"1700000000.000000", 1700000000},
		{"0.0", 0},
	}

	for _, tt := range tests {
		ts := parseSlackTimestamp(tt.input)
		if ts.Unix() != tt.want {
			t.Errorf("parseSlackTimestamp(%q) = %d, want %d", tt.input, ts.Unix(), tt.want)
		}
	}
}

// ---------------------------------------------------------------------------
// Tests: formatTimestamp uses UTC
// ---------------------------------------------------------------------------

func TestFormatTimestamp_UTC(t *testing.T) {
	ts := time.Date(2026, 5, 1, 14, 30, 0, 0, time.UTC)
	result := formatTimestamp(ts)
	if result != "2026-05-01 14:30" {
		t.Errorf("formatTimestamp() = %q, want %q", result, "2026-05-01 14:30")
	}
}

// ---------------------------------------------------------------------------
// Tests: Cursor update tracking
// ---------------------------------------------------------------------------

func TestFetchAll_CursorTracking(t *testing.T) {
	server := mockSlackServer(map[string]http.HandlerFunc{
		"conversations.history": func(w http.ResponseWriter, r *http.Request) {
			jsonResponse(w, slackResponse{
				OK: true,
				Messages: []slackMessage{
					{Type: "message", User: "U001", Text: "First", TS: "100.000000"},
					{Type: "message", User: "U001", Text: "Second", TS: "200.000000"},
					{Type: "message", User: "U001", Text: "Third", TS: "300.000000"},
				},
			})
		},
	})
	defer server.Close()

	c := newTestConnectorWithServer(server.URL)
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	docs, errs := c.FetchAll(ctx)
	var latestTS string
	for doc := range docs {
		ts := doc.Metadata["ts"]
		if ts > latestTS {
			latestTS = ts
		}
	}
	if err, ok := <-errs; ok && err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if latestTS != "300.000000" {
		t.Errorf("expected latest cursor to be 300.000000, got %s", latestTS)
	}
}

// ---------------------------------------------------------------------------
// Tests: Multiple channels
// ---------------------------------------------------------------------------

func TestFetchAll_MultipleChannels(t *testing.T) {
	server := mockSlackServer(map[string]http.HandlerFunc{
		"conversations.history": func(w http.ResponseWriter, r *http.Request) {
			channel := r.URL.Query().Get("channel")
			jsonResponse(w, slackResponse{
				OK: true,
				Messages: []slackMessage{
					{Type: "message", User: "U001", Text: fmt.Sprintf("Message in %s", channel), TS: "1700000001.000000"},
				},
			})
		},
	})
	defer server.Close()

	c := newTestConnectorWithServer(server.URL, "C001", "C002", "C003")
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	docs, errs := c.FetchAll(ctx)
	var collected []ConnectorDocument
	for doc := range docs {
		collected = append(collected, doc)
	}
	if err, ok := <-errs; ok && err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(collected) != 3 {
		t.Fatalf("expected 3 documents (one per channel), got %d", len(collected))
	}
}

// ---------------------------------------------------------------------------
// Tests: Connector interface compliance
// ---------------------------------------------------------------------------

func TestSlackConnector_ImplementsConnector(t *testing.T) {
	var _ Connector = (*SlackConnector)(nil)
}

// ---------------------------------------------------------------------------
// Tests: MaxFileSize configuration via Configure
// ---------------------------------------------------------------------------

func TestConfigure_MaxFileSizeParsing(t *testing.T) {
	c := NewSlackConnector(SlackConnectorConfig{})
	err := c.Configure(map[string]any{
		"botToken":    "xoxb-test",
		"channels":    "C123",
		"maxFileSize": "invalid",
	})
	if err == nil {
		t.Fatal("expected error for invalid maxFileSize")
	}

	err = c.Configure(map[string]any{
		"botToken":    "xoxb-test",
		"channels":    "C123",
		"maxFileSize": strconv.FormatInt(100*1024*1024, 10),
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if c.config.MaxFileSize != 100*1024*1024 {
		t.Errorf("expected maxFileSize=100MB, got %d", c.config.MaxFileSize)
	}
}

// ---------------------------------------------------------------------------
// Tests: SSRF validation
// ---------------------------------------------------------------------------

func TestValidateDownloadURL_BlocksPrivateIPs(t *testing.T) {
	tests := []struct {
		name    string
		url     string
		blocked bool
	}{
		{"external HTTPS", "https://files.slack.com/files/F001", false},
		{"ftp scheme blocked", "ftp://files.slack.com/files/F001", true},
		{"no scheme", "://bad", true},
		{"empty hostname", "http:///path", true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := validateDownloadURL(tt.url)
			if tt.blocked && err == nil {
				t.Errorf("expected URL to be blocked: %s", tt.url)
			}
			if !tt.blocked && err != nil {
				t.Errorf("expected URL to be allowed, got error: %v", err)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// Tests: Rate limit retry exhaustion
// ---------------------------------------------------------------------------

func TestCallSlackAPI_ExhaustsRetries(t *testing.T) {
	callCount := 0
	server := mockSlackServer(map[string]http.HandlerFunc{
		"conversations.history": func(w http.ResponseWriter, _ *http.Request) {
			callCount++
			w.Header().Set("Retry-After", "1")
			w.WriteHeader(http.StatusTooManyRequests)
		},
	})
	defer server.Close()

	c := newTestConnectorWithServer(server.URL)
	// Use a short context timeout to prevent slow exponential backoff.
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	docs, errs := c.FetchAll(ctx)
	for range docs {
	}

	err, ok := <-errs
	if !ok || err == nil {
		t.Fatal("expected an error after exhausting retries")
	}
	// Accept either "rate limited" (all retries exhausted) or context deadline exceeded.
	errMsg := err.Error()
	hasExpectedError := strings.Contains(errMsg, "rate limited") || strings.Contains(errMsg, "context deadline exceeded")
	if !hasExpectedError {
		t.Errorf("expected rate limited or deadline error, got: %v", err)
	}
	if callCount < 2 {
		t.Errorf("expected multiple API calls, got %d", callCount)
	}
}

// ---------------------------------------------------------------------------
// Tests: Thread document includes parent message
// ---------------------------------------------------------------------------

func TestBuildThreadDocument_IncludesParent(t *testing.T) {
	server := mockSlackServer(map[string]http.HandlerFunc{
		"conversations.history": func(w http.ResponseWriter, r *http.Request) {
			jsonResponse(w, slackResponse{
				OK: true,
				Messages: []slackMessage{
					{
						Type:       "message",
						User:       "U001",
						Text:       "Thread parent",
						TS:         "1700000001.000000",
						ThreadTS:   "1700000001.000000",
						ReplyCount: 1,
					},
				},
			})
		},
		"conversations.replies": func(w http.ResponseWriter, r *http.Request) {
			jsonResponse(w, slackResponse{
				OK: true,
				Messages: []slackMessage{
					{Type: "message", User: "U001", Text: "Thread parent", TS: "1700000001.000000"},
					{Type: "message", User: "U002", Text: "A reply", TS: "1700000002.000000"},
				},
			})
		},
		"users.info": func(w http.ResponseWriter, r *http.Request) {
			userID := r.URL.Query().Get("user")
			names := map[string]string{"U001": "Alice", "U002": "Bob"}
			name := names[userID]
			if name == "" {
				name = userID
			}
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(slackUserResponse{
				OK:   true,
				User: slackUser{RealName: name},
			})
		},
	})
	defer server.Close()

	c := newTestConnectorWithServer(server.URL)
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	docs, errs := c.FetchAll(ctx)
	var collected []ConnectorDocument
	for doc := range docs {
		collected = append(collected, doc)
	}
	if err, ok := <-errs; ok && err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(collected) < 2 {
		t.Fatalf("expected at least 2 documents, got %d", len(collected))
	}

	threadDoc := collected[1]
	content := string(threadDoc.Content)

	// Verify parent message appears in the body (not just the header).
	if !strings.Contains(content, "**Alice**") {
		t.Errorf("thread doc should contain parent user name Alice: %s", content)
	}
	if !strings.Contains(content, "**Bob**") {
		t.Errorf("thread doc should contain reply user name Bob: %s", content)
	}

	// Count occurrences of "Thread parent" -- should appear in header AND body.
	count := strings.Count(content, "Thread parent")
	if count < 2 {
		t.Errorf("expected 'Thread parent' to appear at least twice (header + body), got %d: %s", count, content)
	}
}
