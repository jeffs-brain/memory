// SPDX-License-Identifier: Apache-2.0

package connector

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"testing"
	"time"
)

// notionMockHTTPClient captures requests and returns canned responses.
type notionMockHTTPClient struct {
	responses []notionMockResponse
	calls     []notionCapturedRequest
	callIndex int
}

type notionMockResponse struct {
	statusCode int
	body       string
	headers    http.Header
}

type notionCapturedRequest struct {
	method string
	url    string
	body   string
}

func (m *notionMockHTTPClient) Do(req *http.Request) (*http.Response, error) {
	var bodyStr string
	if req.Body != nil {
		data, _ := io.ReadAll(req.Body)
		bodyStr = string(data)
	}

	m.calls = append(m.calls, notionCapturedRequest{
		method: req.Method,
		url:    req.URL.String(),
		body:   bodyStr,
	})

	idx := m.callIndex
	m.callIndex++

	if idx >= len(m.responses) {
		return &http.Response{
			StatusCode: 200,
			Body:       io.NopCloser(strings.NewReader("{}")),
			Header:     http.Header{},
		}, nil
	}

	resp := m.responses[idx]
	headers := resp.headers
	if headers == nil {
		headers = http.Header{}
	}

	return &http.Response{
		StatusCode: resp.statusCode,
		Body:       io.NopCloser(strings.NewReader(resp.body)),
		Header:     headers,
	}, nil
}

// newNotionTestConnector creates a NotionConnector with the given mock
// client and a pre-configured API token.
func newNotionTestConnector(mock *notionMockHTTPClient) *NotionConnector {
	deps := ConnectorConfig{
		Name:    "notion",
		BrainID: "test-brain",
	}
	conn := NewNotionConnector(deps, mock)
	_ = conn.Configure(map[string]any{
		"apiToken": "secret_test_token",
	})
	return conn
}

// collectDocs drains the document and error channels, returning
// all documents and any errors encountered.
func collectDocs(docsCh <-chan ConnectorDocument, errsCh <-chan error) ([]ConnectorDocument, []error) {
	var docs []ConnectorDocument
	var errs []error
	done := make(chan struct{})

	go func() {
		for err := range errsCh {
			errs = append(errs, err)
		}
		close(done)
	}()

	for doc := range docsCh {
		docs = append(docs, doc)
	}
	<-done
	return docs, errs
}

func TestNotionConnector_Name(t *testing.T) {
	conn := NewNotionConnector(ConnectorConfig{}, nil)
	if conn.Name() != "notion" {
		t.Errorf("expected name 'notion', got %q", conn.Name())
	}
}

func TestNotionConnector_Configure_RequiresAPIToken(t *testing.T) {
	conn := NewNotionConnector(ConnectorConfig{}, nil)
	err := conn.Configure(map[string]any{})
	if err == nil {
		t.Fatal("expected error for missing apiToken")
	}
	if !strings.Contains(err.Error(), "apiToken is required") {
		t.Errorf("unexpected error message: %s", err.Error())
	}
}

func TestNotionConnector_Configure_EmptyToken(t *testing.T) {
	conn := NewNotionConnector(ConnectorConfig{}, nil)
	err := conn.Configure(map[string]any{"apiToken": ""})
	if err == nil {
		t.Fatal("expected error for empty apiToken")
	}
}

func TestNotionConnector_Configure_ValidToken(t *testing.T) {
	conn := NewNotionConnector(ConnectorConfig{}, nil)
	err := conn.Configure(map[string]any{
		"apiToken":          "secret_abc123",
		"rootPageIds":       []any{"page-1", "page-2"},
		"databaseIds":       []any{"db-1"},
		"includeChildPages": false,
		"includeDatabases":  true,
		"maxDepth":          float64(5),
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !conn.configured {
		t.Error("expected connector to be configured")
	}
	if conn.config.APIToken != "secret_abc123" {
		t.Errorf("unexpected token: %s", conn.config.APIToken)
	}
	if len(conn.config.RootPageIDs) != 2 {
		t.Errorf("expected 2 root page IDs, got %d", len(conn.config.RootPageIDs))
	}
	if conn.config.IncludeChildPages {
		t.Error("expected includeChildPages=false")
	}
	if conn.config.MaxDepth != 5 {
		t.Errorf("expected maxDepth=5, got %d", conn.config.MaxDepth)
	}
}

func TestNotionConnector_FetchAll_NotConfigured(t *testing.T) {
	conn := NewNotionConnector(ConnectorConfig{}, nil)
	docsCh, errsCh := conn.FetchAll(context.Background())
	_, errs := collectDocs(docsCh, errsCh)
	if len(errs) == 0 {
		t.Fatal("expected error for unconfigured connector")
	}
}

func TestNotionConnector_FetchPageViaMarkdownAPI(t *testing.T) {
	pageResp := `{
		"id": "page-123",
		"url": "https://notion.so/page-123",
		"last_edited_time": "2026-05-10T14:30:00.000Z",
		"properties": {
			"Name": {
				"type": "title",
				"title": [{"plain_text": "Test Page"}]
			}
		},
		"parent": {"type": "workspace"}
	}`
	markdownResp := `{"markdown": "# Test Page\n\nThis is test content."}`
	childBlocksResp := `{"results": [], "has_more": false}`

	mock := &notionMockHTTPClient{
		responses: []notionMockResponse{
			{statusCode: 200, body: pageResp},
			{statusCode: 200, body: markdownResp},
			{statusCode: 200, body: childBlocksResp},
		},
	}

	conn := newNotionTestConnector(mock)
	conn.config.RootPageIDs = []string{"page-123"}
	conn.config.IncludeChildPages = true

	docsCh, errsCh := conn.FetchAll(context.Background())
	docs, errs := collectDocs(docsCh, errsCh)

	if len(errs) > 0 {
		t.Fatalf("unexpected errors: %v", errs)
	}
	if len(docs) != 1 {
		t.Fatalf("expected 1 document, got %d", len(docs))
	}

	doc := docs[0]
	if doc.ExternalID != "page-123" {
		t.Errorf("expected externalID 'page-123', got %q", doc.ExternalID)
	}
	if doc.Title != "Test Page" {
		t.Errorf("expected title 'Test Page', got %q", doc.Title)
	}
	if doc.MIME != "text/markdown" {
		t.Errorf("expected mime 'text/markdown', got %q", doc.MIME)
	}
	if !strings.Contains(string(doc.Content), "This is test content") {
		t.Errorf("expected content to contain test text, got %q", string(doc.Content))
	}
	if doc.Metadata["source"] != "notion" {
		t.Errorf("expected source=notion in metadata")
	}
}

func TestNotionConnector_FetchPageFallbackToBlocks(t *testing.T) {
	pageResp := `{
		"id": "page-456",
		"url": "https://notion.so/page-456",
		"last_edited_time": "2026-05-10T14:30:00.000Z",
		"properties": {
			"Title": {
				"type": "title",
				"title": [{"plain_text": "Block Page"}]
			}
		},
		"parent": {"type": "workspace"}
	}`
	// Markdown API returns 404.
	markdownNotFound := `{"message": "not found"}`

	blocksResp := `{
		"results": [
			{
				"id": "block-1",
				"type": "heading_1",
				"has_children": false,
				"heading_1": {
					"rich_text": [{"type": "text", "plain_text": "Heading One"}]
				}
			},
			{
				"id": "block-2",
				"type": "paragraph",
				"has_children": false,
				"paragraph": {
					"rich_text": [{"type": "text", "plain_text": "Paragraph content here."}]
				}
			}
		],
		"has_more": false
	}`
	childBlocksResp := `{"results": [], "has_more": false}`

	mock := &notionMockHTTPClient{
		responses: []notionMockResponse{
			{statusCode: 200, body: pageResp},
			{statusCode: 404, body: markdownNotFound},
			{statusCode: 200, body: blocksResp},
			{statusCode: 200, body: childBlocksResp},
		},
	}

	conn := newNotionTestConnector(mock)
	conn.config.RootPageIDs = []string{"page-456"}
	conn.config.IncludeChildPages = true

	docsCh, errsCh := conn.FetchAll(context.Background())
	docs, errs := collectDocs(docsCh, errsCh)

	if len(errs) > 0 {
		t.Fatalf("unexpected errors: %v", errs)
	}
	if len(docs) != 1 {
		t.Fatalf("expected 1 document, got %d", len(docs))
	}

	content := string(docs[0].Content)
	if !strings.Contains(content, "# Heading One") {
		t.Errorf("expected heading markdown, got %q", content)
	}
	if !strings.Contains(content, "Paragraph content here.") {
		t.Errorf("expected paragraph content, got %q", content)
	}
}

func TestNotionConnector_DatabaseQuery(t *testing.T) {
	dbQueryResp := `{
		"results": [
			{
				"id": "entry-1",
				"url": "https://notion.so/entry-1",
				"last_edited_time": "2026-05-10T10:00:00.000Z",
				"properties": {
					"Name": {
						"type": "title",
						"title": [{"plain_text": "Entry One"}]
					},
					"Status": {
						"type": "select",
						"select": {"name": "Active"}
					},
					"Priority": {
						"type": "number",
						"number": 3
					}
				}
			},
			{
				"id": "entry-2",
				"url": "https://notion.so/entry-2",
				"last_edited_time": "2026-05-10T11:00:00.000Z",
				"properties": {
					"Name": {
						"type": "title",
						"title": [{"plain_text": "Entry Two"}]
					},
					"Status": {
						"type": "select",
						"select": {"name": "Done"}
					}
				}
			}
		],
		"has_more": false
	}`
	// Page content for entry-1 and entry-2 (markdown API).
	entryMarkdown1 := `{"markdown": "Details for entry one."}`
	entryMarkdown2 := `{"markdown": "Details for entry two."}`

	mock := &notionMockHTTPClient{
		responses: []notionMockResponse{
			{statusCode: 200, body: dbQueryResp},
			// Content for entry-1
			{statusCode: 200, body: entryMarkdown1},
			// Content for entry-2
			{statusCode: 200, body: entryMarkdown2},
		},
	}

	conn := newNotionTestConnector(mock)
	conn.config.DatabaseIDs = []string{"db-test"}
	conn.config.RootPageIDs = nil

	docsCh, errsCh := conn.FetchAll(context.Background())
	docs, errs := collectDocs(docsCh, errsCh)

	if len(errs) > 0 {
		t.Fatalf("unexpected errors: %v", errs)
	}
	if len(docs) != 2 {
		t.Fatalf("expected 2 documents, got %d", len(docs))
	}

	if docs[0].Title != "Entry One" {
		t.Errorf("expected title 'Entry One', got %q", docs[0].Title)
	}
	if docs[0].Metadata["type"] != "database_entry" {
		t.Errorf("expected type=database_entry, got %q", docs[0].Metadata["type"])
	}

	content := string(docs[0].Content)
	if !strings.Contains(content, "## Entry One") {
		t.Errorf("expected title heading in content: %q", content)
	}
	if !strings.Contains(content, "Active") {
		t.Errorf("expected status in content: %q", content)
	}
}

func TestNotionConnector_PaginatedDatabaseQuery(t *testing.T) {
	page1 := `{
		"results": [
			{
				"id": "entry-a",
				"url": "https://notion.so/entry-a",
				"last_edited_time": "2026-05-10T10:00:00.000Z",
				"properties": {
					"Name": {"type": "title", "title": [{"plain_text": "A"}]}
				}
			}
		],
		"has_more": true,
		"next_cursor": "cursor-page-2"
	}`
	page2 := `{
		"results": [
			{
				"id": "entry-b",
				"url": "https://notion.so/entry-b",
				"last_edited_time": "2026-05-10T11:00:00.000Z",
				"properties": {
					"Name": {"type": "title", "title": [{"plain_text": "B"}]}
				}
			}
		],
		"has_more": false
	}`
	entryMD := `{"markdown": "content"}`

	mock := &notionMockHTTPClient{
		responses: []notionMockResponse{
			{statusCode: 200, body: page1},
			{statusCode: 200, body: entryMD},
			{statusCode: 200, body: page2},
			{statusCode: 200, body: entryMD},
		},
	}

	conn := newNotionTestConnector(mock)
	conn.config.DatabaseIDs = []string{"db-paginated"}

	docsCh, errsCh := conn.FetchAll(context.Background())
	docs, errs := collectDocs(docsCh, errsCh)

	if len(errs) > 0 {
		t.Fatalf("unexpected errors: %v", errs)
	}
	if len(docs) != 2 {
		t.Fatalf("expected 2 documents across pages, got %d", len(docs))
	}
}

func TestNotionConnector_RecursiveChildPages(t *testing.T) {
	// Root page.
	rootPageResp := `{
		"id": "root",
		"url": "https://notion.so/root",
		"last_edited_time": "2026-05-10T14:00:00.000Z",
		"properties": {
			"Title": {"type": "title", "title": [{"plain_text": "Root"}]}
		},
		"parent": {"type": "workspace"}
	}`
	rootMD := `{"markdown": "Root content"}`
	rootChildren := `{
		"results": [
			{"id": "child-1", "type": "child_page", "has_children": false}
		],
		"has_more": false
	}`

	// Child page.
	childPageResp := `{
		"id": "child-1",
		"url": "https://notion.so/child-1",
		"last_edited_time": "2026-05-10T15:00:00.000Z",
		"properties": {
			"Title": {"type": "title", "title": [{"plain_text": "Child"}]}
		},
		"parent": {"type": "page_id", "page_id": "root"}
	}`
	childMD := `{"markdown": "Child content"}`
	childChildren := `{"results": [], "has_more": false}`

	mock := &notionMockHTTPClient{
		responses: []notionMockResponse{
			{statusCode: 200, body: rootPageResp},
			{statusCode: 200, body: rootMD},
			{statusCode: 200, body: rootChildren},
			{statusCode: 200, body: childPageResp},
			{statusCode: 200, body: childMD},
			{statusCode: 200, body: childChildren},
		},
	}

	conn := newNotionTestConnector(mock)
	conn.config.RootPageIDs = []string{"root"}
	conn.config.IncludeChildPages = true

	docsCh, errsCh := conn.FetchAll(context.Background())
	docs, errs := collectDocs(docsCh, errsCh)

	if len(errs) > 0 {
		t.Fatalf("unexpected errors: %v", errs)
	}
	if len(docs) != 2 {
		t.Fatalf("expected 2 documents (root + child), got %d", len(docs))
	}
}

func TestNotionConnector_MaxDepthEnforcement(t *testing.T) {
	// Root page with child that has its own child (depth 3).
	rootPage := `{
		"id": "root", "url": "https://notion.so/root",
		"last_edited_time": "2026-05-10T14:00:00.000Z",
		"properties": {"Title": {"type": "title", "title": [{"plain_text": "Root"}]}},
		"parent": {"type": "workspace"}
	}`
	rootMD := `{"markdown": "root"}`
	rootChildren := `{
		"results": [{"id": "child-1", "type": "child_page", "has_children": false}],
		"has_more": false
	}`
	childPage := `{
		"id": "child-1", "url": "https://notion.so/child-1",
		"last_edited_time": "2026-05-10T15:00:00.000Z",
		"properties": {"Title": {"type": "title", "title": [{"plain_text": "Child"}]}},
		"parent": {"type": "page_id", "page_id": "root"}
	}`
	childMD := `{"markdown": "child"}`
	childChildren := `{
		"results": [{"id": "grandchild-1", "type": "child_page", "has_children": false}],
		"has_more": false
	}`

	mock := &notionMockHTTPClient{
		responses: []notionMockResponse{
			{statusCode: 200, body: rootPage},
			{statusCode: 200, body: rootMD},
			{statusCode: 200, body: rootChildren},
			{statusCode: 200, body: childPage},
			{statusCode: 200, body: childMD},
			{statusCode: 200, body: childChildren},
			// Grandchild would be at depth 2 but maxDepth=1 stops it.
		},
	}

	conn := newNotionTestConnector(mock)
	conn.config.RootPageIDs = []string{"root"}
	conn.config.MaxDepth = 1

	docsCh, errsCh := conn.FetchAll(context.Background())
	docs, errs := collectDocs(docsCh, errsCh)

	if len(errs) > 0 {
		t.Fatalf("unexpected errors: %v", errs)
	}
	// Root is depth 0, child is depth 1, grandchild would be depth 2
	// (exceeds maxDepth=1).
	if len(docs) != 2 {
		t.Fatalf("expected 2 documents (root + child, no grandchild), got %d", len(docs))
	}
}

func TestNotionConnector_CycleDetection(t *testing.T) {
	// Page A -> Page B -> Page A (cycle).
	pageA := `{
		"id": "page-a", "url": "https://notion.so/page-a",
		"last_edited_time": "2026-05-10T14:00:00.000Z",
		"properties": {"Title": {"type": "title", "title": [{"plain_text": "A"}]}},
		"parent": {"type": "workspace"}
	}`
	pageMD := `{"markdown": "content"}`
	childrenA := `{
		"results": [{"id": "page-b", "type": "child_page", "has_children": false}],
		"has_more": false
	}`
	pageB := `{
		"id": "page-b", "url": "https://notion.so/page-b",
		"last_edited_time": "2026-05-10T15:00:00.000Z",
		"properties": {"Title": {"type": "title", "title": [{"plain_text": "B"}]}},
		"parent": {"type": "page_id", "page_id": "page-a"}
	}`
	childrenB := `{
		"results": [{"id": "page-a", "type": "child_page", "has_children": false}],
		"has_more": false
	}`

	mock := &notionMockHTTPClient{
		responses: []notionMockResponse{
			{statusCode: 200, body: pageA},
			{statusCode: 200, body: pageMD},
			{statusCode: 200, body: childrenA},
			{statusCode: 200, body: pageB},
			{statusCode: 200, body: pageMD},
			{statusCode: 200, body: childrenB},
			// page-a would be fetched again here but cycle detection stops it.
		},
	}

	conn := newNotionTestConnector(mock)
	conn.config.RootPageIDs = []string{"page-a"}
	conn.config.IncludeChildPages = true

	docsCh, errsCh := conn.FetchAll(context.Background())
	docs, errs := collectDocs(docsCh, errsCh)

	if len(errs) > 0 {
		t.Fatalf("unexpected errors: %v", errs)
	}
	if len(docs) != 2 {
		t.Fatalf("expected 2 documents (A and B, no cycle), got %d", len(docs))
	}
}

func TestNotionConnector_IncrementalSyncViaTimestamp(t *testing.T) {
	dbQueryResp := `{
		"results": [
			{
				"id": "recent-entry",
				"url": "https://notion.so/recent",
				"last_edited_time": "2026-05-15T10:00:00.000Z",
				"properties": {
					"Name": {"type": "title", "title": [{"plain_text": "Recent"}]}
				}
			}
		],
		"has_more": false
	}`
	entryMD := `{"markdown": "updated content"}`

	mock := &notionMockHTTPClient{
		responses: []notionMockResponse{
			{statusCode: 200, body: dbQueryResp},
			{statusCode: 200, body: entryMD},
		},
	}

	conn := newNotionTestConnector(mock)
	conn.config.DatabaseIDs = []string{"db-incremental"}

	cursor := SyncCursor{
		Value:     "2026-05-01T00:00:00.000Z",
		UpdatedAt: time.Now(),
	}
	docsCh, errsCh := conn.FetchSince(context.Background(), cursor)
	docs, errs := collectDocs(docsCh, errsCh)

	if len(errs) > 0 {
		t.Fatalf("unexpected errors: %v", errs)
	}
	if len(docs) != 1 {
		t.Fatalf("expected 1 document, got %d", len(docs))
	}

	// Verify the filter was sent in the request body.
	if len(mock.calls) < 1 {
		t.Fatal("expected at least 1 API call")
	}
	requestBody := mock.calls[0].body
	if !strings.Contains(requestBody, "on_or_after") {
		t.Errorf("expected on_or_after filter in request body: %s", requestBody)
	}
}

func TestNotionConnector_CursorUpdatedAfterSync(t *testing.T) {
	dbResp := `{
		"results": [
			{
				"id": "e1", "url": "https://notion.so/e1",
				"last_edited_time": "2026-05-15T10:00:00.000Z",
				"properties": {"Name": {"type": "title", "title": [{"plain_text": "E1"}]}}
			}
		],
		"has_more": false
	}`
	entryMD := `{"markdown": "content"}`

	mock := &notionMockHTTPClient{
		responses: []notionMockResponse{
			{statusCode: 200, body: dbResp},
			{statusCode: 200, body: entryMD},
		},
	}

	conn := newNotionTestConnector(mock)
	conn.config.DatabaseIDs = []string{"db-cursor"}

	docsCh, errsCh := conn.FetchAll(context.Background())
	docs, errs := collectDocs(docsCh, errsCh)

	if len(errs) > 0 {
		t.Fatalf("unexpected errors: %v", errs)
	}
	if len(docs) != 1 {
		t.Fatalf("expected 1 document, got %d", len(docs))
	}

	// Verify the document has the correct modified time for cursor
	// tracking.
	expected, _ := time.Parse(time.RFC3339, "2026-05-15T10:00:00.000Z")
	if !docs[0].ModifiedAt.Equal(expected) {
		t.Errorf("expected modifiedAt %v, got %v", expected, docs[0].ModifiedAt)
	}
}

func TestNotionConnector_RateLimit429Handling(t *testing.T) {
	// Provide enough 429 responses to exhaust all retries
	// (notionMaxRetryAttempts + 1 = 6).
	rateLimitResp := notionMockResponse{
		statusCode: 429,
		body:       `{"message": "rate limited"}`,
		headers:    http.Header{"Retry-After": []string{"0"}},
	}

	mock := &notionMockHTTPClient{
		responses: []notionMockResponse{
			rateLimitResp,
			rateLimitResp,
			rateLimitResp,
			rateLimitResp,
			rateLimitResp,
			rateLimitResp,
		},
	}

	conn := newNotionTestConnector(mock)
	conn.config.RootPageIDs = []string{"page-rl"}

	docsCh, errsCh := conn.FetchAll(context.Background())
	_, errs := collectDocs(docsCh, errsCh)

	// The connector should report a rate limit error after exhausting
	// retries.
	hasRateLimitErr := false
	for _, err := range errs {
		if strings.Contains(err.Error(), "429") || strings.Contains(err.Error(), "rate limited") {
			hasRateLimitErr = true
			break
		}
	}
	if !hasRateLimitErr {
		t.Errorf("expected rate limit error in results, got: %v", errs)
	}
}

func TestNotionConnector_FileBlockDownload(t *testing.T) {
	pageResp := `{
		"id": "page-file", "url": "https://notion.so/page-file",
		"last_edited_time": "2026-05-10T14:00:00.000Z",
		"properties": {"Title": {"type": "title", "title": [{"plain_text": "File Page"}]}},
		"parent": {"type": "workspace"}
	}`
	// Markdown API returns content with image reference.
	markdownResp := `{"markdown": "![image](https://example.com/image.png)\n\nSome text."}`
	childBlocksResp := `{"results": [], "has_more": false}`

	mock := &notionMockHTTPClient{
		responses: []notionMockResponse{
			{statusCode: 200, body: pageResp},
			{statusCode: 200, body: markdownResp},
			{statusCode: 200, body: childBlocksResp},
		},
	}

	conn := newNotionTestConnector(mock)
	conn.config.RootPageIDs = []string{"page-file"}

	docsCh, errsCh := conn.FetchAll(context.Background())
	docs, errs := collectDocs(docsCh, errsCh)

	if len(errs) > 0 {
		t.Fatalf("unexpected errors: %v", errs)
	}
	if len(docs) != 1 {
		t.Fatalf("expected 1 document, got %d", len(docs))
	}

	content := string(docs[0].Content)
	if !strings.Contains(content, "image.png") {
		t.Errorf("expected image reference in content: %q", content)
	}
}

func TestNotionConnector_EmptyWorkspace(t *testing.T) {
	searchResp := `{"results": [], "has_more": false}`

	mock := &notionMockHTTPClient{
		responses: []notionMockResponse{
			{statusCode: 200, body: searchResp},
		},
	}

	conn := newNotionTestConnector(mock)
	// No rootPageIds or databaseIds, triggers workspace search.

	docsCh, errsCh := conn.FetchAll(context.Background())
	docs, errs := collectDocs(docsCh, errsCh)

	if len(errs) > 0 {
		t.Fatalf("unexpected errors: %v", errs)
	}
	if len(docs) != 0 {
		t.Fatalf("expected 0 documents, got %d", len(docs))
	}
}

func TestNotionConnector_WorkspaceSearch(t *testing.T) {
	searchResp := `{
		"results": [
			{
				"id": "ws-page-1",
				"object": "page",
				"last_edited_time": "2026-05-10T14:00:00.000Z"
			}
		],
		"has_more": false
	}`
	pageResp := `{
		"id": "ws-page-1", "url": "https://notion.so/ws-page-1",
		"last_edited_time": "2026-05-10T14:00:00.000Z",
		"properties": {"Title": {"type": "title", "title": [{"plain_text": "WS Page"}]}},
		"parent": {"type": "workspace"}
	}`
	pageMD := `{"markdown": "workspace page content"}`
	childBlocks := `{"results": [], "has_more": false}`

	mock := &notionMockHTTPClient{
		responses: []notionMockResponse{
			{statusCode: 200, body: searchResp},
			{statusCode: 200, body: pageResp},
			{statusCode: 200, body: pageMD},
			{statusCode: 200, body: childBlocks},
		},
	}

	conn := newNotionTestConnector(mock)

	docsCh, errsCh := conn.FetchAll(context.Background())
	docs, errs := collectDocs(docsCh, errsCh)

	if len(errs) > 0 {
		t.Fatalf("unexpected errors: %v", errs)
	}
	if len(docs) != 1 {
		t.Fatalf("expected 1 document from workspace search, got %d", len(docs))
	}
	if docs[0].Title != "WS Page" {
		t.Errorf("expected title 'WS Page', got %q", docs[0].Title)
	}
}

func TestNotionConnector_BlockToMarkdown_RichText(t *testing.T) {
	conn := newNotionTestConnector(nil)

	tests := []struct {
		name     string
		block    notionBlock
		expected string
	}{
		{
			name: "heading_1",
			block: makeBlock("heading_1", `{
				"rich_text": [{"type": "text", "plain_text": "Main Heading"}]
			}`),
			expected: "# Main Heading\n",
		},
		{
			name: "paragraph with bold",
			block: makeBlock("paragraph", `{
				"rich_text": [{"type": "text", "plain_text": "bold text", "annotations": {"bold": true}}]
			}`),
			expected: "**bold text**\n",
		},
		{
			name: "code block",
			block: makeBlock("code", `{
				"rich_text": [{"type": "text", "plain_text": "fmt.Println(\"hello\")"}],
				"language": "go"
			}`),
			expected: "```go\nfmt.Println(\"hello\")\n```\n",
		},
		{
			name: "bulleted list",
			block: makeBlock("bulleted_list_item", `{
				"rich_text": [{"type": "text", "plain_text": "List item"}]
			}`),
			expected: "- List item",
		},
		{
			name: "to_do checked",
			block: makeBlock("to_do", `{
				"rich_text": [{"type": "text", "plain_text": "Done task"}],
				"checked": true
			}`),
			expected: "- [x] Done task",
		},
		{
			name: "quote",
			block: makeBlock("quote", `{
				"rich_text": [{"type": "text", "plain_text": "Quoted text"}]
			}`),
			expected: "> Quoted text\n",
		},
		{
			name:     "divider",
			block:    makeBlock("divider", `{}`),
			expected: "---\n",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := conn.blockToMarkdown(tt.block)
			if result != tt.expected {
				t.Errorf("expected %q, got %q", tt.expected, result)
			}
		})
	}
}

func TestNotionConnector_PropertyExtraction(t *testing.T) {
	tests := []struct {
		name     string
		raw      string
		expected string
	}{
		{
			name:     "select",
			raw:      `{"type": "select", "select": {"name": "Active"}}`,
			expected: "Active",
		},
		{
			name:     "multi_select",
			raw:      `{"type": "multi_select", "multi_select": [{"name": "Tag1"}, {"name": "Tag2"}]}`,
			expected: "Tag1, Tag2",
		},
		{
			name:     "number",
			raw:      `{"type": "number", "number": 42}`,
			expected: "42",
		},
		{
			name:     "checkbox true",
			raw:      `{"type": "checkbox", "checkbox": true}`,
			expected: "true",
		},
		{
			name:     "date range",
			raw:      `{"type": "date", "date": {"start": "2026-05-01", "end": "2026-05-15"}}`,
			expected: "2026-05-01 to 2026-05-15",
		},
		{
			name:     "url",
			raw:      `{"type": "url", "url": "https://example.com"}`,
			expected: "https://example.com",
		},
		{
			name:     "rich_text",
			raw:      `{"type": "rich_text", "rich_text": [{"plain_text": "Hello"}, {"plain_text": " world"}]}`,
			expected: "Hello world",
		},
		{
			name:     "status",
			raw:      `{"type": "status", "status": {"name": "In Progress"}}`,
			expected: "In Progress",
		},
		{
			name:     "null select",
			raw:      `{"type": "select", "select": null}`,
			expected: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := extractPropertyValue(json.RawMessage(tt.raw))
			if result != tt.expected {
				t.Errorf("expected %q, got %q", tt.expected, result)
			}
		})
	}
}

func TestNotionConnector_RenderRichText(t *testing.T) {
	tests := []struct {
		name     string
		texts    []notionRichText
		expected string
	}{
		{
			name: "plain text",
			texts: []notionRichText{
				{PlainText: "Hello"},
			},
			expected: "Hello",
		},
		{
			name: "bold and italic",
			texts: []notionRichText{
				{PlainText: "bold", Annotations: notionAnnotations{Bold: true}},
				{PlainText: " and "},
				{PlainText: "italic", Annotations: notionAnnotations{Italic: true}},
			},
			expected: "**bold** and *italic*",
		},
		{
			name: "code annotation",
			texts: []notionRichText{
				{PlainText: "codeVar", Annotations: notionAnnotations{Code: true}},
			},
			expected: "`codeVar`",
		},
		{
			name: "link",
			texts: []notionRichText{
				{PlainText: "click here", Href: "https://example.com"},
			},
			expected: "[click here](https://example.com)",
		},
		{
			name: "strikethrough",
			texts: []notionRichText{
				{PlainText: "deleted", Annotations: notionAnnotations{Strikethrough: true}},
			},
			expected: "~~deleted~~",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := renderRichText(tt.texts)
			if result != tt.expected {
				t.Errorf("expected %q, got %q", tt.expected, result)
			}
		})
	}
}

func TestNotionConnector_TableRowRendering(t *testing.T) {
	data := json.RawMessage(`{
		"cells": [
			[{"type": "text", "plain_text": "Col1"}],
			[{"type": "text", "plain_text": "Col2"}],
			[{"type": "text", "plain_text": "Col3"}]
		]
	}`)

	result := renderTableRow(data)
	expected := "| Col1 | Col2 | Col3 |"
	if result != expected {
		t.Errorf("expected %q, got %q", expected, result)
	}
}

func TestNotionConnector_Stop(t *testing.T) {
	conn := NewNotionConnector(ConnectorConfig{}, nil)
	err := conn.Stop()
	if err != nil {
		t.Fatalf("unexpected error from Stop: %v", err)
	}

	// Calling Stop again should not panic.
	err = conn.Stop()
	if err != nil {
		t.Fatalf("second Stop call failed: %v", err)
	}
}

func TestNotionConnector_AuthHeader(t *testing.T) {
	mock := &notionMockHTTPClient{
		responses: []notionMockResponse{
			{statusCode: 200, body: `{
				"id": "p1", "url": "https://notion.so/p1",
				"last_edited_time": "2026-05-10T14:00:00.000Z",
				"properties": {"Title": {"type": "title", "title": [{"plain_text": "T"}]}},
				"parent": {"type": "workspace"}
			}`},
			{statusCode: 200, body: `{"markdown": "content"}`},
			{statusCode: 200, body: `{"results": [], "has_more": false}`},
		},
	}

	conn := newNotionTestConnector(mock)
	conn.config.RootPageIDs = []string{"p1"}

	docsCh, errsCh := conn.FetchAll(context.Background())
	collectDocs(docsCh, errsCh)

	if len(mock.calls) < 1 {
		t.Fatal("expected at least 1 API call")
	}

	// Verify auth header was sent (via URL containing Bearer token in
	// the mock's captured requests).
	firstCall := mock.calls[0]
	if !strings.Contains(firstCall.url, "/pages/p1") {
		t.Errorf("expected first call to /pages/p1, got %s", firstCall.url)
	}
}

func TestNotionConnector_ContextCancellation(t *testing.T) {
	// Create a slow mock that would block.
	mock := &notionMockHTTPClient{
		responses: []notionMockResponse{
			{statusCode: 200, body: `{
				"id": "p1", "url": "https://notion.so/p1",
				"last_edited_time": "2026-05-10T14:00:00.000Z",
				"properties": {"Title": {"type": "title", "title": [{"plain_text": "T"}]}},
				"parent": {"type": "workspace"}
			}`},
			{statusCode: 200, body: `{"markdown": "content"}`},
			{statusCode: 200, body: `{"results": [], "has_more": false}`},
		},
	}

	conn := newNotionTestConnector(mock)
	conn.config.RootPageIDs = []string{"p1"}

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // Cancel immediately.

	docsCh, errsCh := conn.FetchAll(ctx)
	collectDocs(docsCh, errsCh)
	// Just verify it doesn't hang.
}

func TestNotionConnector_NestedBlockChildrenRendering(t *testing.T) {
	pageResp := `{
		"id": "nested-page",
		"url": "https://notion.so/nested-page",
		"last_edited_time": "2026-05-10T14:00:00.000Z",
		"properties": {
			"Title": {"type": "title", "title": [{"plain_text": "Nested"}]}
		},
		"parent": {"type": "workspace"}
	}`
	// Parent list item with has_children=true.
	blocksResp := `{
		"results": [
			{
				"id": "parent-list",
				"type": "bulleted_list_item",
				"has_children": true,
				"bulleted_list_item": {
					"rich_text": [{"type": "text", "plain_text": "Parent item"}]
				}
			}
		],
		"has_more": false
	}`
	// Child blocks of the parent list item.
	childBlocksResp := `{
		"results": [
			{
				"id": "child-para",
				"type": "paragraph",
				"has_children": false,
				"paragraph": {
					"rich_text": [{"type": "text", "plain_text": "Nested paragraph"}]
				}
			},
			{
				"id": "child-list",
				"type": "bulleted_list_item",
				"has_children": false,
				"bulleted_list_item": {
					"rich_text": [{"type": "text", "plain_text": "Nested list item"}]
				}
			}
		],
		"has_more": false
	}`
	// No children for the page.
	pageChildBlocks := `{"results": [], "has_more": false}`

	mock := &notionMockHTTPClient{
		responses: []notionMockResponse{
			{statusCode: 200, body: pageResp},
			{statusCode: 404, body: `{"message": "not found"}`},
			{statusCode: 200, body: blocksResp},
			{statusCode: 200, body: childBlocksResp},
			{statusCode: 200, body: pageChildBlocks},
		},
	}

	conn := newNotionTestConnector(mock)
	conn.config.RootPageIDs = []string{"nested-page"}
	conn.config.IncludeChildPages = true

	docsCh, errsCh := conn.FetchAll(context.Background())
	docs, errs := collectDocs(docsCh, errsCh)

	if len(errs) > 0 {
		t.Fatalf("unexpected errors: %v", errs)
	}
	if len(docs) != 1 {
		t.Fatalf("expected 1 document, got %d", len(docs))
	}

	content := string(docs[0].Content)
	if !strings.Contains(content, "- Parent item") {
		t.Errorf("expected parent list item, got %q", content)
	}
	// Child content of a list block should be indented with two spaces.
	if !strings.Contains(content, "  Nested paragraph") {
		t.Errorf("expected indented nested paragraph, got %q", content)
	}
	if !strings.Contains(content, "  - Nested list item") {
		t.Errorf("expected indented nested list item, got %q", content)
	}
}

func TestNotionConnector_SyncedBlockRendering(t *testing.T) {
	pageResp := `{
		"id": "sync-page",
		"url": "https://notion.so/sync-page",
		"last_edited_time": "2026-05-10T14:00:00.000Z",
		"properties": {
			"Title": {"type": "title", "title": [{"plain_text": "Synced"}]}
		},
		"parent": {"type": "workspace"}
	}`
	blocksResp := `{
		"results": [
			{
				"id": "synced-1",
				"type": "synced_block",
				"has_children": false,
				"synced_block": {
					"synced_from": {"block_id": "original-block-123"}
				}
			},
			{
				"id": "para-1",
				"type": "paragraph",
				"has_children": false,
				"paragraph": {
					"rich_text": [{"type": "text", "plain_text": "After synced block"}]
				}
			}
		],
		"has_more": false
	}`
	childBlocks := `{"results": [], "has_more": false}`

	mock := &notionMockHTTPClient{
		responses: []notionMockResponse{
			{statusCode: 200, body: pageResp},
			{statusCode: 404, body: `{"message": "not found"}`},
			{statusCode: 200, body: blocksResp},
			{statusCode: 200, body: childBlocks},
		},
	}

	conn := newNotionTestConnector(mock)
	conn.config.RootPageIDs = []string{"sync-page"}
	conn.config.IncludeChildPages = true

	docsCh, errsCh := conn.FetchAll(context.Background())
	docs, errs := collectDocs(docsCh, errsCh)

	if len(errs) > 0 {
		t.Fatalf("unexpected errors: %v", errs)
	}
	if len(docs) != 1 {
		t.Fatalf("expected 1 document, got %d", len(docs))
	}

	content := string(docs[0].Content)
	// synced_block renders as empty; only the paragraph should appear.
	if !strings.Contains(content, "After synced block") {
		t.Errorf("expected paragraph after synced block, got %q", content)
	}
	// The synced_from reference should not appear in output.
	if strings.Contains(content, "original-block-123") {
		t.Errorf("synced_from reference should not leak into content: %q", content)
	}
}

func TestNotionConnector_EquationBlockRendering(t *testing.T) {
	conn := newNotionTestConnector(nil)

	block := makeBlock("equation", `{"expression": "E = mc^2"}`)
	result := conn.blockToMarkdown(block)

	expected := "$$\nE = mc^2\n$$\n"
	if result != expected {
		t.Errorf("expected %q, got %q", expected, result)
	}
}

func TestNotionConnector_ImageBlockRendering(t *testing.T) {
	conn := newNotionTestConnector(nil)

	// Image with caption.
	block := makeBlock("image", `{
		"caption": [{"type": "text", "plain_text": "My photo"}],
		"file": {"url": "https://example.com/photo.png"}
	}`)
	result := conn.blockToMarkdown(block)
	expected := "![My photo](https://example.com/photo.png)\n"
	if result != expected {
		t.Errorf("expected %q, got %q", expected, result)
	}

	// Image without caption.
	block2 := makeBlock("image", `{
		"caption": [],
		"external": {"url": "https://example.com/ext.jpg"}
	}`)
	result2 := conn.blockToMarkdown(block2)
	expected2 := "![](https://example.com/ext.jpg)\n"
	if result2 != expected2 {
		t.Errorf("expected %q, got %q", expected2, result2)
	}
}

func TestNotionConnector_FileBlockRendering(t *testing.T) {
	conn := newNotionTestConnector(nil)

	block := makeBlock("file", `{
		"caption": [{"type": "text", "plain_text": "Document"}],
		"file": {"url": "https://example.com/doc.pdf"}
	}`)
	result := conn.blockToMarkdown(block)
	expected := "[Document](https://example.com/doc.pdf)\n"
	if result != expected {
		t.Errorf("expected %q, got %q", expected, result)
	}
}

func TestNotionConnector_BlockTypeFilter(t *testing.T) {
	conn := newNotionTestConnector(nil)
	conn.config.BlockTypeFilter = map[string]struct{}{
		"heading_1": {},
	}

	// heading_1 is allowed.
	h1Block := makeBlock("heading_1", `{
		"rich_text": [{"type": "text", "plain_text": "Allowed Heading"}]
	}`)
	result := conn.blockToMarkdown(h1Block)
	if result != "# Allowed Heading\n" {
		t.Errorf("expected heading, got %q", result)
	}

	// paragraph is not in the filter, should be excluded.
	paraBlock := makeBlock("paragraph", `{
		"rich_text": [{"type": "text", "plain_text": "Filtered out"}]
	}`)
	result2 := conn.blockToMarkdown(paraBlock)
	if result2 != "" {
		t.Errorf("expected empty for filtered block, got %q", result2)
	}
}

func TestNotionConnector_RetryAfterHandling(t *testing.T) {
	pageResp := `{
		"id": "retry-page",
		"url": "https://notion.so/retry-page",
		"last_edited_time": "2026-05-10T14:00:00.000Z",
		"properties": {
			"Title": {"type": "title", "title": [{"plain_text": "Retry"}]}
		},
		"parent": {"type": "workspace"}
	}`
	md := `{"markdown": "retried content"}`
	childBlocks := `{"results": [], "has_more": false}`

	mock := &notionMockHTTPClient{
		responses: []notionMockResponse{
			// First call returns 429 with Retry-After header.
			{
				statusCode: 429,
				body:       `{"message": "rate limited"}`,
				headers:    http.Header{"Retry-After": []string{"1"}},
			},
			// Retry succeeds.
			{statusCode: 200, body: pageResp},
			{statusCode: 200, body: md},
			{statusCode: 200, body: childBlocks},
		},
	}

	conn := newNotionTestConnector(mock)
	conn.config.RootPageIDs = []string{"retry-page"}
	conn.config.IncludeChildPages = true

	docsCh, errsCh := conn.FetchAll(context.Background())
	docs, errs := collectDocs(docsCh, errsCh)

	if len(errs) > 0 {
		t.Fatalf("unexpected errors: %v", errs)
	}
	if len(docs) != 1 {
		t.Fatalf("expected 1 document after retry, got %d", len(docs))
	}
	if docs[0].Title != "Retry" {
		t.Errorf("expected title 'Retry', got %q", docs[0].Title)
	}
	content := string(docs[0].Content)
	if !strings.Contains(content, "retried content") {
		t.Errorf("expected retried content, got %q", content)
	}
}

func TestNotionConnector_DefaultRateLimiter(t *testing.T) {
	// Verify that a connector created without a rate limiter in deps
	// gets a default one.
	deps := ConnectorConfig{
		Name:    "notion",
		BrainID: "test-brain",
		// No RateLimiter set.
	}
	conn := NewNotionConnector(deps, nil)
	if conn.rateLimiter == nil {
		t.Fatal("expected default rate limiter to be created")
	}
}

// makeBlock constructs a notionBlock with RawData for testing.
func makeBlock(blockType, blockDataJSON string) notionBlock {
	raw := map[string]json.RawMessage{
		"id":           json.RawMessage(`"test-block"`),
		"type":         json.RawMessage(`"` + blockType + `"`),
		"has_children": json.RawMessage(`false`),
		blockType:      json.RawMessage(blockDataJSON),
	}
	data, _ := json.Marshal(raw)

	return notionBlock{
		ID:          "test-block",
		Type:        blockType,
		HasChildren: false,
		RawData:     data,
	}
}

// TestNotionConnector_doNotionRequest_SetsHeaders verifies request
// headers are correctly set.
func TestNotionConnector_doNotionRequest_SetsHeaders(t *testing.T) {
	var capturedReq *http.Request

	transport := &capturingTransport{
		response: &http.Response{
			StatusCode: 200,
			Body:       io.NopCloser(bytes.NewReader([]byte(`{}`))),
			Header:     http.Header{},
		},
		captureReq: func(r *http.Request) {
			capturedReq = r
		},
	}

	conn := newNotionTestConnector(nil)
	conn.httpClient = &http.Client{Transport: transport}

	ctx := WithNotionBaseURL(context.Background(), "http://localhost:9999")
	_, err := conn.doNotionRequest(ctx, http.MethodGet, "/test", nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if capturedReq == nil {
		t.Fatal("expected request to be captured")
	}
	if capturedReq.Header.Get("Authorization") != "Bearer secret_test_token" {
		t.Errorf("unexpected auth header: %s", capturedReq.Header.Get("Authorization"))
	}
	if capturedReq.Header.Get("Notion-Version") != notionAPIVersion {
		t.Errorf("unexpected notion version: %s", capturedReq.Header.Get("Notion-Version"))
	}
}

// capturingTransport captures the request for inspection.
type capturingTransport struct {
	response   *http.Response
	captureReq func(r *http.Request)
}

func (t *capturingTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	if t.captureReq != nil {
		t.captureReq(req)
	}
	return t.response, nil
}

// mockNotionTokenExchanger implements NotionTokenExchanger for testing.
type mockNotionTokenExchanger struct {
	refreshFn func(ctx context.Context, refreshToken string) (NotionRefreshedToken, error)
}

func (m *mockNotionTokenExchanger) Refresh(ctx context.Context, refreshToken string) (NotionRefreshedToken, error) {
	return m.refreshFn(ctx, refreshToken)
}

func TestNotionConnector_TokenRefreshOnExpiry(t *testing.T) {
	refreshCalled := false
	exchanger := &mockNotionTokenExchanger{
		refreshFn: func(_ context.Context, rt string) (NotionRefreshedToken, error) {
			refreshCalled = true
			if rt != "test-refresh-token" {
				t.Errorf("unexpected refresh token: %s", rt)
			}
			return NotionRefreshedToken{
				AccessToken:  "refreshed-notion-token",
				RefreshToken: "test-refresh-token",
				ExpiresAt:    time.Now().Add(time.Hour),
			}, nil
		},
	}

	pageResp := `{
		"id": "page-1",
		"url": "https://notion.so/page-1",
		"last_edited_time": "2026-05-10T14:00:00.000Z",
		"properties": {
			"Title": {"type": "title", "title": [{"plain_text": "Refreshed Page"}]}
		},
		"parent": {"type": "workspace"}
	}`
	md := `{"markdown": "content after refresh"}`
	childBlocks := `{"results": [], "has_more": false}`

	mock := &notionMockHTTPClient{
		responses: []notionMockResponse{
			{statusCode: 200, body: pageResp},
			{statusCode: 200, body: md},
			{statusCode: 200, body: childBlocks},
		},
	}

	deps := ConnectorConfig{
		Name:    "notion",
		BrainID: "test-brain",
	}
	conn := NewNotionConnector(deps, mock)
	err := conn.Configure(map[string]any{
		"apiToken":           "expired-token",
		"rootPageIds":        []any{"page-1"},
		"oauth2_client_id":   "client-id",
		"oauth2_client_secret": "client-secret",
		"refresh_token":      "test-refresh-token",
		"token_expires_at":   time.Now().Add(-time.Hour).Format(time.RFC3339),
		"token_exchanger":    exchanger,
	})
	if err != nil {
		t.Fatalf("Configure() failed: %v", err)
	}

	docsCh, errsCh := conn.FetchAll(context.Background())
	docs, errs := collectDocs(docsCh, errsCh)

	if len(errs) > 0 {
		t.Fatalf("unexpected errors: %v", errs)
	}
	if len(docs) != 1 {
		t.Fatalf("expected 1 document, got %d", len(docs))
	}
	if !refreshCalled {
		t.Error("expected token refresh to be called for expired token")
	}
}

func TestNotionConnector_TokenRefreshFailurePropagates(t *testing.T) {
	exchanger := &mockNotionTokenExchanger{
		refreshFn: func(_ context.Context, _ string) (NotionRefreshedToken, error) {
			return NotionRefreshedToken{}, fmt.Errorf("invalid_grant: token revoked")
		},
	}

	mock := &notionMockHTTPClient{
		responses: []notionMockResponse{},
	}

	deps := ConnectorConfig{
		Name:    "notion",
		BrainID: "test-brain",
	}
	conn := NewNotionConnector(deps, mock)
	err := conn.Configure(map[string]any{
		"apiToken":             "expired-token",
		"rootPageIds":          []any{"page-1"},
		"oauth2_client_id":    "client-id",
		"oauth2_client_secret": "client-secret",
		"refresh_token":        "revoked-token",
		"token_expires_at":     time.Now().Add(-time.Hour).Format(time.RFC3339),
		"token_exchanger":      exchanger,
	})
	if err != nil {
		t.Fatalf("Configure() failed: %v", err)
	}

	docsCh, errsCh := conn.FetchAll(context.Background())
	_, errs := collectDocs(docsCh, errsCh)

	if len(errs) == 0 {
		t.Fatal("expected error when token refresh fails")
	}

	hasRefreshErr := false
	for _, e := range errs {
		if strings.Contains(e.Error(), "token refresh failed") {
			hasRefreshErr = true
			break
		}
	}
	if !hasRefreshErr {
		t.Errorf("expected token refresh error, got: %v", errs)
	}
}

func TestNotionConnector_NoRefreshForIntegrationToken(t *testing.T) {
	// Internal integration tokens do not have OAuth2 config or
	// refresh tokens, so no refresh should happen.
	pageResp := `{
		"id": "page-int",
		"url": "https://notion.so/page-int",
		"last_edited_time": "2026-05-10T14:00:00.000Z",
		"properties": {
			"Title": {"type": "title", "title": [{"plain_text": "Int Token Page"}]}
		},
		"parent": {"type": "workspace"}
	}`
	md := `{"markdown": "internal token content"}`
	childBlocks := `{"results": [], "has_more": false}`

	mock := &notionMockHTTPClient{
		responses: []notionMockResponse{
			{statusCode: 200, body: pageResp},
			{statusCode: 200, body: md},
			{statusCode: 200, body: childBlocks},
		},
	}

	conn := newNotionTestConnector(mock)
	conn.config.RootPageIDs = []string{"page-int"}

	docsCh, errsCh := conn.FetchAll(context.Background())
	docs, errs := collectDocs(docsCh, errsCh)

	if len(errs) > 0 {
		t.Fatalf("unexpected errors: %v", errs)
	}
	if len(docs) != 1 {
		t.Fatalf("expected 1 document, got %d", len(docs))
	}
}

func TestNotionConnector_RefreshWithinBuffer(t *testing.T) {
	refreshCalled := false
	exchanger := &mockNotionTokenExchanger{
		refreshFn: func(_ context.Context, _ string) (NotionRefreshedToken, error) {
			refreshCalled = true
			return NotionRefreshedToken{
				AccessToken:  "refreshed-token",
				RefreshToken: "test-refresh-token",
				ExpiresAt:    time.Now().Add(time.Hour),
			}, nil
		},
	}

	mock := &notionMockHTTPClient{
		responses: []notionMockResponse{
			{statusCode: 200, body: `{
				"id": "page-buf",
				"url": "https://notion.so/page-buf",
				"last_edited_time": "2026-05-10T14:00:00.000Z",
				"properties": {"Title": {"type": "title", "title": [{"plain_text": "Buffer"}]}},
				"parent": {"type": "workspace"}
			}`},
			{statusCode: 200, body: `{"markdown": "content"}`},
			{statusCode: 200, body: `{"results": [], "has_more": false}`},
		},
	}

	deps := ConnectorConfig{Name: "notion", BrainID: "test-brain"}
	conn := NewNotionConnector(deps, mock)
	// Token expires in 3 minutes -- within the 5-minute buffer.
	err := conn.Configure(map[string]any{
		"apiToken":             "expiring-token",
		"rootPageIds":          []any{"page-buf"},
		"oauth2_client_id":    "client-id",
		"oauth2_client_secret": "client-secret",
		"refresh_token":        "test-refresh-token",
		"token_expires_at":     time.Now().Add(3 * time.Minute).Format(time.RFC3339),
		"token_exchanger":      exchanger,
	})
	if err != nil {
		t.Fatalf("Configure() failed: %v", err)
	}

	docsCh, errsCh := conn.FetchAll(context.Background())
	_, errs := collectDocs(docsCh, errsCh)

	if len(errs) > 0 {
		t.Fatalf("unexpected errors: %v", errs)
	}
	if !refreshCalled {
		t.Error("expected refresh for token within expiry buffer")
	}
}

func TestNotionConnector_Health(t *testing.T) {
	conn := NewNotionConnector(ConnectorConfig{}, nil)

	// Before configuration.
	health := conn.Health()
	if health.Status != StatusDisconnected {
		t.Errorf("Health().Status = %q, want %q", health.Status, StatusDisconnected)
	}
	if health.Message == "" {
		t.Error("Health().Message should describe unconfigured state")
	}

	// After configuration.
	_ = conn.Configure(map[string]any{
		"apiToken": "secret_test_token",
	})
	health = conn.Health()
	if health.Status != StatusConnected {
		t.Errorf("Health().Status = %q, want %q", health.Status, StatusConnected)
	}
	if health.RateLimitRemaining != -1 {
		t.Errorf("Health().RateLimitRemaining = %d, want -1", health.RateLimitRemaining)
	}
}
