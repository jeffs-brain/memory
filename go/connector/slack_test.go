// SPDX-License-Identifier: Apache-2.0

package connector

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"testing"
	"time"
)

// ---------------------------------------------------------------------------
// Mock HTTP server helpers
// ---------------------------------------------------------------------------

// mockSlackServer creates a test server that responds to Slack API
// endpoints with pre-configured responses. Handlers are keyed by
// method path (e.g. "conversations.history").
func mockSlackServer(handlers map[string]http.HandlerFunc) *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		for path, handler := range handlers {
			if strings.HasSuffix(r.URL.Path, path) {
				handler(w, r)
				return
			}
		}
		http.NotFound(w, r)
	}))
}

func jsonResponse(w http.ResponseWriter, data slackResponse) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(data)
}

func newTestConnector(serverURL string) *SlackConnector {
	return NewSlackConnector(SlackConnectorConfig{
		BotToken:       "xoxb-test-token",
		Channels:       []string{"C123ABC"},
		IncludeThreads: true,
		IncludeFiles:   true,
		MaxFileSize:    50 * 1024 * 1024,
		HTTPClient:     &http.Client{Timeout: 5 * time.Second},
	})
}

// replaceSlackBaseURL patches API calls to use the test server.
// Because the connector hard-codes slack.com URLs, we use a custom
// HTTPDoer that rewrites the host.
type rewritingClient struct {
	baseURL string
	client  *http.Client
}

func (rc *rewritingClient) Do(req *http.Request) (*http.Response, error) {
	req.URL.Scheme = "http"
	req.URL.Host = strings.TrimPrefix(rc.baseURL, "http://")
	return rc.client.Do(req)
}

func newTestConnectorWithServer(serverURL string, channels ...string) *SlackConnector {
	if len(channels) == 0 {
		channels = []string{"C123ABC"}
	}
	return NewSlackConnector(SlackConnectorConfig{
		BotToken:       "xoxb-test-token",
		Channels:       channels,
		IncludeThreads: true,
		IncludeFiles:   true,
		MaxFileSize:    50 * 1024 * 1024,
		HTTPClient:     &rewritingClient{baseURL: serverURL, client: &http.Client{Timeout: 5 * time.Second}},
	})
}

// ---------------------------------------------------------------------------
// Tests: Name and Configure
// ---------------------------------------------------------------------------

func TestSlackConnector_Name(t *testing.T) {
	c := NewSlackConnector(SlackConnectorConfig{})
	if c.Name() != "slack" {
		t.Fatalf("expected name 'slack', got %q", c.Name())
	}
}

func TestConfigure_ValidatesRequiredFields(t *testing.T) {
	c := NewSlackConnector(SlackConnectorConfig{})

	// Missing botToken.
	err := c.Configure(map[string]string{"channels": "C123"})
	if err == nil || !strings.Contains(err.Error(), "botToken is required") {
		t.Fatalf("expected botToken validation error, got %v", err)
	}

	// Missing channels.
	err = c.Configure(map[string]string{"botToken": "xoxb-123"})
	if err == nil || !strings.Contains(err.Error(), "at least one channel") {
		t.Fatalf("expected channels validation error, got %v", err)
	}

	// Valid config.
	err = c.Configure(map[string]string{
		"botToken": "xoxb-123",
		"channels": "C123,C456",
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

// ---------------------------------------------------------------------------
// Tests: FetchAll
// ---------------------------------------------------------------------------

func TestFetchAll_SingleChannelMessages(t *testing.T) {
	server := mockSlackServer(map[string]http.HandlerFunc{
		"conversations.history": func(w http.ResponseWriter, r *http.Request) {
			jsonResponse(w, slackResponse{
				OK: true,
				Messages: []slackMessage{
					{Type: "message", User: "U001", Text: "Hello world", TS: "1700000001.000000"},
					{Type: "message", User: "U002", Text: "Good morning", TS: "1700000002.000000"},
					{Type: "message", User: "U003", Text: "How are you?", TS: "1700000003.000000"},
				},
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

	if len(collected) != 3 {
		t.Fatalf("expected 3 documents, got %d", len(collected))
	}

	// Verify first message.
	first := collected[0]
	if first.ExternalID != "C123ABC:1700000001.000000" {
		t.Errorf("unexpected externalID: %s", first.ExternalID)
	}
	if string(first.Content) != "Hello world" {
		t.Errorf("unexpected content: %s", string(first.Content))
	}
	if first.Metadata["source"] != "slack" {
		t.Errorf("expected source=slack, got %s", first.Metadata["source"])
	}
}

func TestFetchAll_PaginatedMessages(t *testing.T) {
	callCount := 0
	server := mockSlackServer(map[string]http.HandlerFunc{
		"conversations.history": func(w http.ResponseWriter, r *http.Request) {
			callCount++
			cursor := r.URL.Query().Get("cursor")

			switch {
			case cursor == "" && callCount == 1:
				jsonResponse(w, slackResponse{
					OK: true,
					Messages: []slackMessage{
						{Type: "message", User: "U001", Text: "Page 1 msg 1", TS: "1700000001.000000"},
						{Type: "message", User: "U001", Text: "Page 1 msg 2", TS: "1700000002.000000"},
					},
					ResponseMetadata: slackRespMeta{NextCursor: "cursor_page2"},
				})
			default:
				jsonResponse(w, slackResponse{
					OK: true,
					Messages: []slackMessage{
						{Type: "message", User: "U002", Text: "Page 2 msg 1", TS: "1700000003.000000"},
						{Type: "message", User: "U002", Text: "Page 2 msg 2", TS: "1700000004.000000"},
					},
				})
			}
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

	if len(collected) != 4 {
		t.Fatalf("expected 4 documents from 2 pages, got %d", len(collected))
	}
}

// ---------------------------------------------------------------------------
// Tests: Thread reconstruction
// ---------------------------------------------------------------------------

func TestFetchAll_ThreadReconstruction(t *testing.T) {
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
						ReplyCount: 2,
					},
				},
			})
		},
		"conversations.replies": func(w http.ResponseWriter, r *http.Request) {
			jsonResponse(w, slackResponse{
				OK: true,
				Messages: []slackMessage{
					{Type: "message", User: "U001", Text: "Thread parent", TS: "1700000001.000000"},
					{Type: "message", User: "U002", Text: "Reply one", TS: "1700000002.000000"},
					{Type: "message", User: "U003", Text: "Reply two", TS: "1700000003.000000"},
				},
			})
		},
		"users.info": func(w http.ResponseWriter, r *http.Request) {
			userID := r.URL.Query().Get("user")
			names := map[string]string{"U001": "Alice", "U002": "Bob", "U003": "Charlie"}
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

	// Should have: 1 message doc + 1 thread doc = 2.
	if len(collected) != 2 {
		t.Fatalf("expected 2 documents (message + thread), got %d", len(collected))
	}

	threadDoc := collected[1]
	if threadDoc.Metadata["type"] != "thread" {
		t.Errorf("expected type=thread, got %s", threadDoc.Metadata["type"])
	}
	content := string(threadDoc.Content)
	if !strings.Contains(content, "## Thread: Thread parent") {
		t.Errorf("thread doc missing parent text: %s", content)
	}
	if !strings.Contains(content, "Reply one") {
		t.Errorf("thread doc missing reply one: %s", content)
	}
	if !strings.Contains(content, "Reply two") {
		t.Errorf("thread doc missing reply two: %s", content)
	}
}

// ---------------------------------------------------------------------------
// Tests: File attachment download
// ---------------------------------------------------------------------------

func TestFetchAll_FileAttachmentDownload(t *testing.T) {
	fileContent := []byte("file content bytes")

	server := mockSlackServer(map[string]http.HandlerFunc{
		"conversations.history": func(w http.ResponseWriter, r *http.Request) {
			jsonResponse(w, slackResponse{
				OK: true,
				Messages: []slackMessage{
					{
						Type: "message",
						User: "U001",
						Text: "Check this file",
						TS:   "1700000001.000000",
						Files: []slackFile{
							{
								ID:                 "F001",
								Name:               "report.pdf",
								MIMEType:           "application/pdf",
								Size:               int64(len(fileContent)),
								URLPrivateDownload: "http://localhost/files/F001",
							},
						},
					},
				},
			})
		},
		"/files/F001": func(w http.ResponseWriter, r *http.Request) {
			// Verify auth header.
			auth := r.Header.Get("Authorization")
			if auth != "Bearer xoxb-test-token" {
				http.Error(w, "unauthorized", http.StatusUnauthorized)
				return
			}
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write(fileContent)
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

	// Should have: 1 message + 1 file = 2.
	if len(collected) != 2 {
		t.Fatalf("expected 2 documents (message + file), got %d", len(collected))
	}

	fileDoc := collected[1]
	if fileDoc.ExternalID != "C123ABC:file:F001" {
		t.Errorf("unexpected file externalID: %s", fileDoc.ExternalID)
	}
	if string(fileDoc.Content) != string(fileContent) {
		t.Errorf("unexpected file content: %s", string(fileDoc.Content))
	}
	if fileDoc.MIME != "application/pdf" {
		t.Errorf("unexpected MIME: %s", fileDoc.MIME)
	}
}

func TestFetchAll_FileTooLargeSkipped(t *testing.T) {
	server := mockSlackServer(map[string]http.HandlerFunc{
		"conversations.history": func(w http.ResponseWriter, r *http.Request) {
			jsonResponse(w, slackResponse{
				OK: true,
				Messages: []slackMessage{
					{
						Type: "message",
						User: "U001",
						Text: "Big file",
						TS:   "1700000001.000000",
						Files: []slackFile{
							{
								ID:                 "F002",
								Name:               "huge.zip",
								MIMEType:           "application/zip",
								Size:               100 * 1024 * 1024, // 100 MB > 50 MB limit
								URLPrivateDownload: "http://localhost/files/F002",
							},
						},
					},
				},
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

	// Only the message, no file (skipped due to size).
	if len(collected) != 1 {
		t.Fatalf("expected 1 document (message only, file skipped), got %d", len(collected))
	}
}

// ---------------------------------------------------------------------------
// Tests: Incremental sync with cursor
// ---------------------------------------------------------------------------

func TestFetchSince_IncrementalSync(t *testing.T) {
	server := mockSlackServer(map[string]http.HandlerFunc{
		"conversations.history": func(w http.ResponseWriter, r *http.Request) {
			oldest := r.URL.Query().Get("oldest")
			if oldest != "1700000002.000000" {
				t.Errorf("expected oldest=1700000002.000000, got %s", oldest)
			}
			jsonResponse(w, slackResponse{
				OK: true,
				Messages: []slackMessage{
					{Type: "message", User: "U001", Text: "New message", TS: "1700000003.000000"},
				},
			})
		},
	})
	defer server.Close()

	c := newTestConnectorWithServer(server.URL)
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	cursor := SyncCursor{Value: "1700000002.000000", UpdatedAt: time.Now()}
	docs, errs := c.FetchSince(ctx, cursor)
	var collected []ConnectorDocument
	for doc := range docs {
		collected = append(collected, doc)
	}
	if err, ok := <-errs; ok && err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(collected) != 1 {
		t.Fatalf("expected 1 document, got %d", len(collected))
	}
}

// ---------------------------------------------------------------------------
// Tests: Rate limit 429 handling
// ---------------------------------------------------------------------------

func TestFetchAll_RateLimit429Handling(t *testing.T) {
	callCount := 0
	server := mockSlackServer(map[string]http.HandlerFunc{
		"conversations.history": func(w http.ResponseWriter, r *http.Request) {
			callCount++
			if callCount == 1 {
				w.Header().Set("Retry-After", "1")
				w.WriteHeader(http.StatusTooManyRequests)
				return
			}
			jsonResponse(w, slackResponse{
				OK: true,
				Messages: []slackMessage{
					{Type: "message", User: "U001", Text: "After retry", TS: "1700000001.000000"},
				},
			})
		},
	})
	defer server.Close()

	c := newTestConnectorWithServer(server.URL)
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()

	docs, errs := c.FetchAll(ctx)
	var collected []ConnectorDocument
	for doc := range docs {
		collected = append(collected, doc)
	}
	if err, ok := <-errs; ok && err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(collected) != 1 {
		t.Fatalf("expected 1 document after retry, got %d", len(collected))
	}
	if callCount < 2 {
		t.Errorf("expected at least 2 API calls (initial + retry), got %d", callCount)
	}
}

// ---------------------------------------------------------------------------
// Tests: Empty channel
// ---------------------------------------------------------------------------

func TestFetchAll_EmptyChannel(t *testing.T) {
	server := mockSlackServer(map[string]http.HandlerFunc{
		"conversations.history": func(w http.ResponseWriter, r *http.Request) {
			jsonResponse(w, slackResponse{
				OK:       true,
				Messages: []slackMessage{},
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

	if len(collected) != 0 {
		t.Fatalf("expected 0 documents for empty channel, got %d", len(collected))
	}
}

// ---------------------------------------------------------------------------
// Tests: Slack API error
// ---------------------------------------------------------------------------

func TestFetchAll_SlackAPIError(t *testing.T) {
	server := mockSlackServer(map[string]http.HandlerFunc{
		"conversations.history": func(w http.ResponseWriter, r *http.Request) {
			jsonResponse(w, slackResponse{
				OK:    false,
				Error: "channel_not_found",
			})
		},
	})
	defer server.Close()

	c := newTestConnectorWithServer(server.URL)
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	docs, errs := c.FetchAll(ctx)
	// Drain docs channel.
	for range docs {
	}

	err, ok := <-errs
	if !ok || err == nil {
		t.Fatal("expected an error for API failure")
	}
	if !strings.Contains(err.Error(), "channel_not_found") {
		t.Errorf("error should mention channel_not_found: %v", err)
	}
}

// ---------------------------------------------------------------------------
// Tests: Stop
// ---------------------------------------------------------------------------

func TestStop_Idempotent(t *testing.T) {
	c := NewSlackConnector(SlackConnectorConfig{BotToken: "xoxb-test"})
	if err := c.Stop(); err != nil {
		t.Fatalf("first stop failed: %v", err)
	}
	if err := c.Stop(); err != nil {
		t.Fatalf("second stop should be idempotent: %v", err)
	}
}

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
	err := c.Configure(map[string]string{
		"botToken":    "xoxb-test",
		"channels":    "C123",
		"maxFileSize": "invalid",
	})
	if err == nil {
		t.Fatal("expected error for invalid maxFileSize")
	}

	err = c.Configure(map[string]string{
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
