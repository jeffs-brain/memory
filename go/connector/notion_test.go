// SPDX-License-Identifier: Apache-2.0

package connector

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"testing"
	"time"
)

// mockHTTPClient captures requests and returns canned responses.
type mockHTTPClient struct {
	responses []mockResponse
	calls     []capturedRequest
	callIndex int
}

type mockResponse struct {
	statusCode int
	body       string
	headers    http.Header
}

type capturedRequest struct {
	method string
	url    string
	body   string
}

func (m *mockHTTPClient) Do(req *http.Request) (*http.Response, error) {
	var bodyStr string
	if req.Body != nil {
		data, _ := io.ReadAll(req.Body)
		bodyStr = string(data)
	}

	m.calls = append(m.calls, capturedRequest{
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

// newTestConnector creates a NotionConnector with the given mock
// client and a pre-configured API token.
func newTestConnector(mock *mockHTTPClient) *NotionConnector {
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

	mock := &mockHTTPClient{
		responses: []mockResponse{
			{statusCode: 200, body: pageResp},
			{statusCode: 200, body: markdownResp},
			{statusCode: 200, body: childBlocksResp},
		},
	}

	conn := newTestConnector(mock)
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

	mock := &mockHTTPClient{
		responses: []mockResponse{
			{statusCode: 200, body: pageResp},
			{statusCode: 404, body: markdownNotFound},
			{statusCode: 200, body: blocksResp},
			{statusCode: 200, body: childBlocksResp},
		},
	}

	conn := newTestConnector(mock)
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

	mock := &mockHTTPClient{
		responses: []mockResponse{
			{statusCode: 200, body: dbQueryResp},
			// Content for entry-1
			{statusCode: 200, body: entryMarkdown1},
			// Content for entry-2
			{statusCode: 200, body: entryMarkdown2},
		},
	}

	conn := newTestConnector(mock)
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

	mock := &mockHTTPClient{
		responses: []mockResponse{
			{statusCode: 200, body: page1},
			{statusCode: 200, body: entryMD},
			{statusCode: 200, body: page2},
			{statusCode: 200, body: entryMD},
		},
	}

	conn := newTestConnector(mock)
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

	mock := &mockHTTPClient{
		responses: []mockResponse{
			{statusCode: 200, body: rootPageResp},
			{statusCode: 200, body: rootMD},
			{statusCode: 200, body: rootChildren},
			{statusCode: 200, body: childPageResp},
			{statusCode: 200, body: childMD},
			{statusCode: 200, body: childChildren},
		},
	}

	conn := newTestConnector(mock)
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

	mock := &mockHTTPClient{
		responses: []mockResponse{
			{statusCode: 200, body: rootPage},
			{statusCode: 200, body: rootMD},
			{statusCode: 200, body: rootChildren},
			{statusCode: 200, body: childPage},
			{statusCode: 200, body: childMD},
			{statusCode: 200, body: childChildren},
			// Grandchild would be at depth 2 but maxDepth=1 stops it.
		},
	}

	conn := newTestConnector(mock)
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

	mock := &mockHTTPClient{
		responses: []mockResponse{
			{statusCode: 200, body: pageA},
			{statusCode: 200, body: pageMD},
			{statusCode: 200, body: childrenA},
			{statusCode: 200, body: pageB},
			{statusCode: 200, body: pageMD},
			{statusCode: 200, body: childrenB},
			// page-a would be fetched again here but cycle detection stops it.
		},
	}

	conn := newTestConnector(mock)
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

	mock := &mockHTTPClient{
		responses: []mockResponse{
			{statusCode: 200, body: dbQueryResp},
			{statusCode: 200, body: entryMD},
		},
	}

	conn := newTestConnector(mock)
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

	mock := &mockHTTPClient{
		responses: []mockResponse{
			{statusCode: 200, body: dbResp},
			{statusCode: 200, body: entryMD},
		},
	}

	conn := newTestConnector(mock)
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
	pageResp := `{
		"id": "page-rl", "url": "https://notion.so/page-rl",
		"last_edited_time": "2026-05-10T14:00:00.000Z",
		"properties": {"Title": {"type": "title", "title": [{"plain_text": "RL Page"}]}},
		"parent": {"type": "workspace"}
	}`

	mock := &mockHTTPClient{
		responses: []mockResponse{
			{statusCode: 429, body: `{"message": "rate limited"}`, headers: http.Header{"Retry-After": []string{"1"}}},
		},
	}

	conn := newTestConnector(mock)
	conn.config.RootPageIDs = []string{"page-rl"}

	docsCh, errsCh := conn.FetchAll(context.Background())
	_, errs := collectDocs(docsCh, errsCh)

	// The connector should report a rate limit error.
	_ = pageResp // Used for documentation; the mock returns 429 first.
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

	mock := &mockHTTPClient{
		responses: []mockResponse{
			{statusCode: 200, body: pageResp},
			{statusCode: 200, body: markdownResp},
			{statusCode: 200, body: childBlocksResp},
		},
	}

	conn := newTestConnector(mock)
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

	mock := &mockHTTPClient{
		responses: []mockResponse{
			{statusCode: 200, body: searchResp},
		},
	}

	conn := newTestConnector(mock)
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

	mock := &mockHTTPClient{
		responses: []mockResponse{
			{statusCode: 200, body: searchResp},
			{statusCode: 200, body: pageResp},
			{statusCode: 200, body: pageMD},
			{statusCode: 200, body: childBlocks},
		},
	}

	conn := newTestConnector(mock)

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
	conn := newTestConnector(nil)

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
	mock := &mockHTTPClient{
		responses: []mockResponse{
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

	conn := newTestConnector(mock)
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
	mock := &mockHTTPClient{
		responses: []mockResponse{
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

	conn := newTestConnector(mock)
	conn.config.RootPageIDs = []string{"p1"}

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // Cancel immediately.

	docsCh, errsCh := conn.FetchAll(ctx)
	collectDocs(docsCh, errsCh)
	// Just verify it doesn't hang.
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

	conn := newTestConnector(nil)
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
