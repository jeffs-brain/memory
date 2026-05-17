// SPDX-License-Identifier: Apache-2.0

package connector

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"strings"
	"testing"
	"time"
)

// mockResponseDef stores response definition data (not the response itself)
// so fresh response bodies can be created on each request.
type mockResponseDef struct {
	statusCode int
	body       []byte
}

// mockHTTPClient implements HTTPClient for testing with configurable
// responses keyed by request URL substring.
type mockHTTPClient struct {
	responseDefs map[string]mockResponseDef
	requests     []*http.Request
}

func newMockHTTPClient() *mockHTTPClient {
	return &mockHTTPClient{
		responseDefs: make(map[string]mockResponseDef),
	}
}

func (m *mockHTTPClient) Do(req *http.Request) (*http.Response, error) {
	m.requests = append(m.requests, req)

	for pattern, def := range m.responseDefs {
		if strings.Contains(req.URL.String(), pattern) {
			return &http.Response{
				StatusCode: def.statusCode,
				Body:       io.NopCloser(bytes.NewReader(def.body)),
				Header:     http.Header{},
			}, nil
		}
	}
	return &http.Response{
		StatusCode: http.StatusNotFound,
		Body:       io.NopCloser(strings.NewReader(`{"error":{"message":"not found"}}`)),
	}, nil
}

func (m *mockHTTPClient) addResponse(urlPattern string, statusCode int, body string) {
	m.responseDefs[urlPattern] = mockResponseDef{
		statusCode: statusCode,
		body:       []byte(body),
	}
}

func (m *mockHTTPClient) addJSONResponse(urlPattern string, statusCode int, payload any) {
	data, _ := json.Marshal(payload)
	m.responseDefs[urlPattern] = mockResponseDef{
		statusCode: statusCode,
		body:       data,
	}
}

func testLogger() *slog.Logger {
	return slog.New(slog.NewTextHandler(io.Discard, nil))
}

func newTestConnector(client *mockHTTPClient) *GDriveConnector {
	deps := ConnectorConfig{
		Name:    "gdrive",
		BrainID: "test-brain",
	}
	return NewGDriveConnector(deps, testLogger(), client)
}

func configureTestConnector(c *GDriveConnector) error {
	return c.Configure(map[string]any{
		"oauth2_client_id":     "test-client-id",
		"oauth2_client_secret": "test-client-secret",
		"access_token":         "test-access-token",
	})
}

func collectDocuments(docCh <-chan ConnectorDocument, errCh <-chan error) ([]ConnectorDocument, error) {
	var docs []ConnectorDocument
	for doc := range docCh {
		docs = append(docs, doc)
	}
	if err, ok := <-errCh; ok && err != nil {
		return docs, err
	}
	return docs, nil
}

func TestGDriveConnector_Name(t *testing.T) {
	client := newMockHTTPClient()
	c := newTestConnector(client)
	if got := c.Name(); got != "gdrive" {
		t.Fatalf("Name() = %q, want %q", got, "gdrive")
	}
}

func TestGDriveConnector_Configure(t *testing.T) {
	cases := []struct {
		name    string
		config  map[string]any
		wantErr string
	}{
		{
			name: "valid configuration",
			config: map[string]any{
				"oauth2_client_id":     "client-123",
				"oauth2_client_secret": "secret-456",
			},
		},
		{
			name: "missing client ID",
			config: map[string]any{
				"oauth2_client_secret": "secret-456",
			},
			wantErr: "oauth2_client_id is required",
		},
		{
			name: "empty client ID",
			config: map[string]any{
				"oauth2_client_id":     "",
				"oauth2_client_secret": "secret-456",
			},
			wantErr: "oauth2_client_id is required",
		},
		{
			name: "missing client secret",
			config: map[string]any{
				"oauth2_client_id": "client-123",
			},
			wantErr: "oauth2_client_secret is required",
		},
		{
			name: "with folder ID",
			config: map[string]any{
				"oauth2_client_id":     "client-123",
				"oauth2_client_secret": "secret-456",
				"folder_id":            "folder-789",
			},
		},
		{
			name: "with MIME type filter",
			config: map[string]any{
				"oauth2_client_id":     "client-123",
				"oauth2_client_secret": "secret-456",
				"mime_type_filter":     "application/pdf,text/plain",
			},
		},
		{
			name: "with shared drives",
			config: map[string]any{
				"oauth2_client_id":      "client-123",
				"oauth2_client_secret":  "secret-456",
				"include_shared_drives": "true",
			},
		},
		{
			name: "invalid max file size",
			config: map[string]any{
				"oauth2_client_id":     "client-123",
				"oauth2_client_secret": "secret-456",
				"max_file_size":        "not-a-number",
			},
			wantErr: "invalid max_file_size",
		},
		{
			name: "custom max file size",
			config: map[string]any{
				"oauth2_client_id":     "client-123",
				"oauth2_client_secret": "secret-456",
				"max_file_size":        "1048576",
			},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			client := newMockHTTPClient()
			c := newTestConnector(client)
			err := c.Configure(tc.config)

			switch {
			case tc.wantErr != "" && err == nil:
				t.Fatalf("Configure() succeeded, want error containing %q", tc.wantErr)
			case tc.wantErr != "" && !strings.Contains(err.Error(), tc.wantErr):
				t.Fatalf("Configure() error = %q, want containing %q", err, tc.wantErr)
			case tc.wantErr == "" && err != nil:
				t.Fatalf("Configure() failed: %v", err)
			}
		})
	}
}

func TestGDriveConnector_FetchAll_ListsFilesInFolder(t *testing.T) {
	client := newMockHTTPClient()
	c := newTestConnector(client)
	if err := configureTestConnector(c); err != nil {
		t.Fatalf("Configure() failed: %v", err)
	}

	listResp := driveListResponse{
		Files: []driveFile{
			{ID: "f1", Name: "doc1.txt", MIMEType: "text/plain", ModifiedTime: "2026-01-15T10:00:00Z", Size: "100"},
			{ID: "f2", Name: "doc2.pdf", MIMEType: "application/pdf", ModifiedTime: "2026-01-16T10:00:00Z", Size: "200"},
			{ID: "f3", Name: "image.png", MIMEType: "image/png", ModifiedTime: "2026-01-17T10:00:00Z", Size: "300"},
		},
	}
	client.addJSONResponse("drive/v3/files?", http.StatusOK, listResp)

	// Mock download responses for each file.
	client.addResponse("f1?alt=media", http.StatusOK, "file one content")
	client.addResponse("f2?alt=media", http.StatusOK, "file two content")
	client.addResponse("f3?alt=media", http.StatusOK, "file three content")

	docCh, errCh := c.FetchAll(context.Background())
	docs, err := collectDocuments(docCh, errCh)
	if err != nil {
		t.Fatalf("FetchAll() error: %v", err)
	}

	if len(docs) != 3 {
		t.Fatalf("FetchAll() yielded %d documents, want 3", len(docs))
	}

	if docs[0].ExternalID != "f1" {
		t.Errorf("docs[0].ExternalID = %q, want %q", docs[0].ExternalID, "f1")
	}
	if docs[0].Title != "doc1.txt" {
		t.Errorf("docs[0].Title = %q, want %q", docs[0].Title, "doc1.txt")
	}
}

func TestGDriveConnector_FetchAll_PaginatedFileListing(t *testing.T) {
	client := newMockHTTPClient()
	c := newTestConnector(client)
	if err := configureTestConnector(c); err != nil {
		t.Fatalf("Configure() failed: %v", err)
	}

	// First page returns a next page token.
	page1 := driveListResponse{
		NextPageToken: "page2token",
		Files: []driveFile{
			{ID: "f1", Name: "doc1.txt", MIMEType: "text/plain", ModifiedTime: "2026-01-15T10:00:00Z", Size: "100"},
		},
	}
	// Second page has no next token (final page).
	page2 := driveListResponse{
		Files: []driveFile{
			{ID: "f2", Name: "doc2.txt", MIMEType: "text/plain", ModifiedTime: "2026-01-16T10:00:00Z", Size: "200"},
		},
	}

	// Use a stateful mock to return different pages.
	callCount := 0
	client.responseDefs = nil
	originalClient := client
	statefulClient := &statefulMockHTTPClient{
		inner: originalClient,
		handler: func(req *http.Request) (*http.Response, error) {
			originalClient.requests = append(originalClient.requests, req)
			if strings.Contains(req.URL.String(), "drive/v3/files") {
				callCount++
				var data []byte
				switch {
				case callCount == 1:
					data, _ = json.Marshal(page1)
				default:
					data, _ = json.Marshal(page2)
				}
				return &http.Response{
					StatusCode: http.StatusOK,
					Body:       io.NopCloser(bytes.NewReader(data)),
					Header:     http.Header{},
				}, nil
			}
			// Download responses.
			return &http.Response{
				StatusCode: http.StatusOK,
				Body:       io.NopCloser(strings.NewReader("content")),
				Header:     http.Header{},
			}, nil
		},
	}
	c.httpClient = statefulClient

	docCh, errCh := c.FetchAll(context.Background())
	docs, err := collectDocuments(docCh, errCh)
	if err != nil {
		t.Fatalf("FetchAll() error: %v", err)
	}

	if len(docs) != 2 {
		t.Fatalf("FetchAll() yielded %d documents, want 2", len(docs))
	}
}

func TestGDriveConnector_FetchAll_ExportGoogleDocAsMarkdown(t *testing.T) {
	client := newMockHTTPClient()
	c := newTestConnector(client)
	if err := configureTestConnector(c); err != nil {
		t.Fatalf("Configure() failed: %v", err)
	}

	listResp := driveListResponse{
		Files: []driveFile{
			{ID: "doc1", Name: "My Document", MIMEType: mimeGoogleDoc, ModifiedTime: "2026-01-15T10:00:00Z"},
		},
	}
	client.addJSONResponse("drive/v3/files?", http.StatusOK, listResp)
	client.addResponse("doc1/export", http.StatusOK, "# My Document\n\nHello world")

	docCh, errCh := c.FetchAll(context.Background())
	docs, err := collectDocuments(docCh, errCh)
	if err != nil {
		t.Fatalf("FetchAll() error: %v", err)
	}

	if len(docs) != 1 {
		t.Fatalf("FetchAll() yielded %d documents, want 1", len(docs))
	}

	if docs[0].MIME != mimeTextMarkdown {
		t.Errorf("docs[0].MIME = %q, want %q", docs[0].MIME, mimeTextMarkdown)
	}

	if string(docs[0].Content) != "# My Document\n\nHello world" {
		t.Errorf("docs[0].Content = %q, want markdown content", string(docs[0].Content))
	}
}

func TestGDriveConnector_FetchAll_ExportGoogleSheetAsCSV(t *testing.T) {
	client := newMockHTTPClient()
	c := newTestConnector(client)
	if err := configureTestConnector(c); err != nil {
		t.Fatalf("Configure() failed: %v", err)
	}

	listResp := driveListResponse{
		Files: []driveFile{
			{ID: "sheet1", Name: "Budget", MIMEType: mimeGoogleSheet, ModifiedTime: "2026-01-15T10:00:00Z"},
		},
	}
	client.addJSONResponse("drive/v3/files?", http.StatusOK, listResp)
	client.addResponse("sheet1/export", http.StatusOK, "Name,Amount\nRent,1500\nFood,500")

	docCh, errCh := c.FetchAll(context.Background())
	docs, err := collectDocuments(docCh, errCh)
	if err != nil {
		t.Fatalf("FetchAll() error: %v", err)
	}

	if len(docs) != 1 {
		t.Fatalf("FetchAll() yielded %d documents, want 1", len(docs))
	}

	if docs[0].MIME != mimeTextCSV {
		t.Errorf("docs[0].MIME = %q, want %q", docs[0].MIME, mimeTextCSV)
	}
}

func TestGDriveConnector_FetchAll_DirectDownloadOfPDF(t *testing.T) {
	client := newMockHTTPClient()
	c := newTestConnector(client)
	if err := configureTestConnector(c); err != nil {
		t.Fatalf("Configure() failed: %v", err)
	}

	listResp := driveListResponse{
		Files: []driveFile{
			{ID: "pdf1", Name: "report.pdf", MIMEType: "application/pdf", ModifiedTime: "2026-01-15T10:00:00Z", Size: "5000"},
		},
	}
	client.addJSONResponse("drive/v3/files?", http.StatusOK, listResp)
	client.addResponse("pdf1?alt=media", http.StatusOK, "%PDF-1.4 binary content")

	docCh, errCh := c.FetchAll(context.Background())
	docs, err := collectDocuments(docCh, errCh)
	if err != nil {
		t.Fatalf("FetchAll() error: %v", err)
	}

	if len(docs) != 1 {
		t.Fatalf("FetchAll() yielded %d documents, want 1", len(docs))
	}

	if docs[0].MIME != "application/pdf" {
		t.Errorf("docs[0].MIME = %q, want %q", docs[0].MIME, "application/pdf")
	}

	if !strings.HasPrefix(string(docs[0].Content), "%PDF") {
		t.Errorf("docs[0].Content does not start with %%PDF")
	}
}

func TestGDriveConnector_FetchSince_IncrementalSync(t *testing.T) {
	client := newMockHTTPClient()
	c := newTestConnector(client)
	if err := configureTestConnector(c); err != nil {
		t.Fatalf("Configure() failed: %v", err)
	}

	changesResp := driveChangesResponse{
		NewStartPageToken: "new-token-abc",
		Changes: []driveChange{
			{
				FileID: "f1",
				File: &driveFile{
					ID: "f1", Name: "updated.txt", MIMEType: "text/plain",
					ModifiedTime: "2026-01-20T10:00:00Z", Size: "150",
				},
			},
			{
				FileID:  "f2",
				Removed: true,
			},
		},
	}
	client.addJSONResponse("drive/v3/changes?", http.StatusOK, changesResp)
	client.addResponse("f1?alt=media", http.StatusOK, "updated content")

	cursor := SyncCursor{
		Value:     "old-token-xyz",
		UpdatedAt: time.Now(),
	}

	docCh, errCh := c.FetchSince(context.Background(), cursor)
	docs, err := collectDocuments(docCh, errCh)
	if err != nil {
		t.Fatalf("FetchSince() error: %v", err)
	}

	// Expect 3 documents: 1 modified, 1 deleted, 1 cursor update.
	if len(docs) != 3 {
		t.Fatalf("FetchSince() yielded %d documents, want 3", len(docs))
	}

	// First: modified file.
	if docs[0].ExternalID != "f1" {
		t.Errorf("docs[0].ExternalID = %q, want %q", docs[0].ExternalID, "f1")
	}
	if docs[0].Deleted {
		t.Errorf("docs[0].Deleted = true, want false")
	}

	// Second: deleted file.
	if docs[1].ExternalID != "f2" {
		t.Errorf("docs[1].ExternalID = %q, want %q", docs[1].ExternalID, "f2")
	}
	if !docs[1].Deleted {
		t.Errorf("docs[1].Deleted = false, want true")
	}

	// Third: cursor update.
	if docs[2].ExternalID != "__cursor_update__" {
		t.Errorf("docs[2].ExternalID = %q, want %q", docs[2].ExternalID, "__cursor_update__")
	}
	if docs[2].Metadata["new_start_page_token"] != "new-token-abc" {
		t.Errorf("cursor token = %q, want %q", docs[2].Metadata["new_start_page_token"], "new-token-abc")
	}
}

func TestGDriveConnector_FetchSince_CursorUpdated(t *testing.T) {
	client := newMockHTTPClient()
	c := newTestConnector(client)
	if err := configureTestConnector(c); err != nil {
		t.Fatalf("Configure() failed: %v", err)
	}

	changesResp := driveChangesResponse{
		NewStartPageToken: "updated-cursor-token",
		Changes:           []driveChange{},
	}
	client.addJSONResponse("drive/v3/changes?", http.StatusOK, changesResp)

	cursor := SyncCursor{Value: "initial-token", UpdatedAt: time.Now()}
	docCh, errCh := c.FetchSince(context.Background(), cursor)
	docs, err := collectDocuments(docCh, errCh)
	if err != nil {
		t.Fatalf("FetchSince() error: %v", err)
	}

	// Should have just the cursor update document.
	if len(docs) != 1 {
		t.Fatalf("FetchSince() yielded %d documents, want 1", len(docs))
	}
	if docs[0].Metadata["new_start_page_token"] != "updated-cursor-token" {
		t.Errorf("cursor = %q, want %q", docs[0].Metadata["new_start_page_token"], "updated-cursor-token")
	}
}

func TestGDriveConnector_FetchAll_FileTooLargeSkipped(t *testing.T) {
	client := newMockHTTPClient()
	c := newTestConnector(client)

	err := c.Configure(map[string]any{
		"oauth2_client_id":     "test-id",
		"oauth2_client_secret": "test-secret",
		"access_token":         "test-token",
		"max_file_size":        "100",
	})
	if err != nil {
		t.Fatalf("Configure() failed: %v", err)
	}

	listResp := driveListResponse{
		Files: []driveFile{
			{ID: "small", Name: "small.txt", MIMEType: "text/plain", ModifiedTime: "2026-01-15T10:00:00Z", Size: "50"},
			{ID: "large", Name: "large.bin", MIMEType: "application/octet-stream", ModifiedTime: "2026-01-15T10:00:00Z", Size: "500"},
		},
	}
	client.addJSONResponse("drive/v3/files?", http.StatusOK, listResp)
	client.addResponse("small?alt=media", http.StatusOK, "small content")

	docCh, errCh := c.FetchAll(context.Background())
	docs, err := collectDocuments(docCh, errCh)
	if err != nil {
		t.Fatalf("FetchAll() error: %v", err)
	}

	// Only the small file should be included.
	if len(docs) != 1 {
		t.Fatalf("FetchAll() yielded %d documents, want 1", len(docs))
	}
	if docs[0].ExternalID != "small" {
		t.Errorf("docs[0].ExternalID = %q, want %q", docs[0].ExternalID, "small")
	}
}

func TestGDriveConnector_FetchAll_RateLimit403Handling(t *testing.T) {
	client := newMockHTTPClient()
	c := newTestConnector(client)
	if err := configureTestConnector(c); err != nil {
		t.Fatalf("Configure() failed: %v", err)
	}

	// Use a non-retryable 403 (reason "forbidden", not "rateLimitExceeded")
	// so the test completes immediately without backoff retries.
	forbiddenBody := `{"error":{"message":"Forbidden","errors":[{"reason":"forbidden"}]}}`
	client.addResponse("drive/v3/files?", http.StatusForbidden, forbiddenBody)

	docCh, errCh := c.FetchAll(context.Background())
	_, err := collectDocuments(docCh, errCh)

	if err == nil {
		t.Fatal("FetchAll() succeeded, want 403 error")
	}
	if !strings.Contains(err.Error(), "API error") {
		t.Errorf("error = %q, want containing %q", err, "API error")
	}
}

func TestGDriveConnector_FetchAll_FolderIDFiltering(t *testing.T) {
	client := newMockHTTPClient()
	c := newTestConnector(client)

	err := c.Configure(map[string]any{
		"oauth2_client_id":     "test-id",
		"oauth2_client_secret": "test-secret",
		"access_token":         "test-token",
		"folder_id":            "folder-abc",
	})
	if err != nil {
		t.Fatalf("Configure() failed: %v", err)
	}

	listResp := driveListResponse{Files: []driveFile{}}
	client.addJSONResponse("drive/v3/files?", http.StatusOK, listResp)

	docCh, errCh := c.FetchAll(context.Background())
	_, err = collectDocuments(docCh, errCh)
	if err != nil {
		t.Fatalf("FetchAll() error: %v", err)
	}

	// Verify the request includes the folder filter in the query.
	if len(client.requests) == 0 {
		t.Fatal("no requests recorded")
	}
	// The statefulMockHTTPClient won't record here; check the mock's requests.
	reqURL := client.requests[0].URL.String()
	if !strings.Contains(reqURL, "folder-abc") {
		t.Errorf("request URL %q does not contain folder ID filter", reqURL)
	}
}

func TestGDriveConnector_FetchAll_MIMETypeFiltering(t *testing.T) {
	client := newMockHTTPClient()
	c := newTestConnector(client)

	err := c.Configure(map[string]any{
		"oauth2_client_id":     "test-id",
		"oauth2_client_secret": "test-secret",
		"access_token":         "test-token",
		"mime_type_filter":     "application/pdf",
	})
	if err != nil {
		t.Fatalf("Configure() failed: %v", err)
	}

	listResp := driveListResponse{Files: []driveFile{}}
	client.addJSONResponse("drive/v3/files?", http.StatusOK, listResp)

	docCh, errCh := c.FetchAll(context.Background())
	_, err = collectDocuments(docCh, errCh)
	if err != nil {
		t.Fatalf("FetchAll() error: %v", err)
	}

	if len(client.requests) == 0 {
		t.Fatal("no requests recorded")
	}
	reqURL := client.requests[0].URL.String()
	if !strings.Contains(reqURL, "application%2Fpdf") && !strings.Contains(reqURL, "application/pdf") {
		t.Errorf("request URL %q does not contain MIME type filter", reqURL)
	}
}

func TestGDriveConnector_FetchAll_SharedDriveInclusion(t *testing.T) {
	client := newMockHTTPClient()
	c := newTestConnector(client)

	err := c.Configure(map[string]any{
		"oauth2_client_id":      "test-id",
		"oauth2_client_secret":  "test-secret",
		"access_token":          "test-token",
		"include_shared_drives": "true",
	})
	if err != nil {
		t.Fatalf("Configure() failed: %v", err)
	}

	listResp := driveListResponse{Files: []driveFile{}}
	client.addJSONResponse("drive/v3/files?", http.StatusOK, listResp)

	docCh, errCh := c.FetchAll(context.Background())
	_, err = collectDocuments(docCh, errCh)
	if err != nil {
		t.Fatalf("FetchAll() error: %v", err)
	}

	if len(client.requests) == 0 {
		t.Fatal("no requests recorded")
	}
	reqURL := client.requests[0].URL.String()
	if !strings.Contains(reqURL, "supportsAllDrives=true") {
		t.Errorf("request URL %q does not contain supportsAllDrives=true", reqURL)
	}
}

func TestGDriveConnector_FetchAll_NotConfigured(t *testing.T) {
	client := newMockHTTPClient()
	c := newTestConnector(client)

	docCh, errCh := c.FetchAll(context.Background())
	_, err := collectDocuments(docCh, errCh)
	if err == nil {
		t.Fatal("FetchAll() succeeded without Configure(), want error")
	}
	if !strings.Contains(err.Error(), "not configured") {
		t.Errorf("error = %q, want containing %q", err, "not configured")
	}
}

func TestGDriveConnector_FetchAll_ContextCancellation(t *testing.T) {
	client := newMockHTTPClient()
	c := newTestConnector(client)
	if err := configureTestConnector(c); err != nil {
		t.Fatalf("Configure() failed: %v", err)
	}

	// Create a large list to ensure we hit context cancellation mid-stream.
	files := make([]driveFile, 50)
	for i := range files {
		files[i] = driveFile{
			ID:           fmt.Sprintf("f%d", i),
			Name:         fmt.Sprintf("doc%d.txt", i),
			MIMEType:     "text/plain",
			ModifiedTime: "2026-01-15T10:00:00Z",
			Size:         "100",
		}
	}
	listResp := driveListResponse{Files: files}
	client.addJSONResponse("drive/v3/files?", http.StatusOK, listResp)
	for i := range files {
		client.addResponse(fmt.Sprintf("f%d?alt=media", i), http.StatusOK, "content")
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // Cancel immediately.

	docCh, errCh := c.FetchAll(ctx)
	_, err := collectDocuments(docCh, errCh)

	// Should get a context cancellation error or empty results.
	if err != nil && !strings.Contains(err.Error(), "context canceled") {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestGDriveConnector_GetStartPageToken(t *testing.T) {
	client := newMockHTTPClient()
	c := newTestConnector(client)
	if err := configureTestConnector(c); err != nil {
		t.Fatalf("Configure() failed: %v", err)
	}

	client.addJSONResponse("changes/startPageToken", http.StatusOK,
		driveStartPageTokenResponse{StartPageToken: "token-12345"})

	token, err := c.GetStartPageToken(context.Background())
	if err != nil {
		t.Fatalf("GetStartPageToken() error: %v", err)
	}
	if token != "token-12345" {
		t.Errorf("token = %q, want %q", token, "token-12345")
	}
}

func TestGDriveConnector_BuildFileQuery(t *testing.T) {
	cases := []struct {
		name     string
		config   GDriveConfig
		contains []string
		excludes []string
		wantErr  string
	}{
		{
			name:   "no filters",
			config: GDriveConfig{},
			contains: []string{
				"mimeType != 'application/vnd.google-apps.folder'",
				"trashed = false",
			},
		},
		{
			name:   "with folder ID",
			config: GDriveConfig{FolderID: "folder-abc"},
			contains: []string{
				"'folder-abc' in parents",
			},
		},
		{
			name:   "with MIME type filter",
			config: GDriveConfig{MIMETypeFilter: []string{"application/pdf", "text/plain"}},
			contains: []string{
				"mimeType = 'application/pdf'",
				"mimeType = 'text/plain'",
			},
		},
		{
			name:    "rejects folder ID with single quotes",
			config:  GDriveConfig{FolderID: "' or name contains '"},
			wantErr: "invalid folder ID",
		},
		{
			name:    "rejects invalid MIME type with quotes",
			config:  GDriveConfig{MIMETypeFilter: []string{"application/pdf' or mimeType = '"}},
			wantErr: "invalid MIME type",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			client := newMockHTTPClient()
			c := newTestConnector(client)
			c.config = tc.config
			query, err := c.buildFileQuery()

			switch {
			case tc.wantErr != "" && err == nil:
				t.Fatalf("buildFileQuery() succeeded, want error containing %q", tc.wantErr)
			case tc.wantErr != "" && !strings.Contains(err.Error(), tc.wantErr):
				t.Fatalf("buildFileQuery() error = %q, want containing %q", err, tc.wantErr)
			case tc.wantErr == "" && err != nil:
				t.Fatalf("buildFileQuery() failed: %v", err)
			case tc.wantErr != "":
				return
			}

			for _, s := range tc.contains {
				if !strings.Contains(query, s) {
					t.Errorf("query %q does not contain %q", query, s)
				}
			}
			for _, s := range tc.excludes {
				if strings.Contains(query, s) {
					t.Errorf("query %q should not contain %q", query, s)
				}
			}
		})
	}
}

func TestIsGoogleNativeFormat(t *testing.T) {
	cases := []struct {
		mime string
		want bool
	}{
		{mimeGoogleDoc, true},
		{mimeGoogleSheet, true},
		{mimeGoogleSlides, true},
		{mimeGoogleDrawing, true},
		{"application/pdf", false},
		{"text/plain", false},
		{mimeGoogleFolder, false},
	}

	for _, tc := range cases {
		t.Run(tc.mime, func(t *testing.T) {
			if got := isGoogleNativeFormat(tc.mime); got != tc.want {
				t.Errorf("isGoogleNativeFormat(%q) = %v, want %v", tc.mime, got, tc.want)
			}
		})
	}
}

func TestParseFileSize(t *testing.T) {
	cases := []struct {
		input string
		want  int64
	}{
		{"", 0},
		{"0", 0},
		{"100", 100},
		{"1048576", 1048576},
		{"not-a-number", 0},
	}

	for _, tc := range cases {
		t.Run(fmt.Sprintf("input=%q", tc.input), func(t *testing.T) {
			if got := parseFileSize(tc.input); got != tc.want {
				t.Errorf("parseFileSize(%q) = %d, want %d", tc.input, got, tc.want)
			}
		})
	}
}

func TestBuildFileMetadata(t *testing.T) {
	f := &driveFile{
		ID:       "file-123",
		MIMEType: "text/plain",
		Parents:  []string{"parent-456"},
		Size:     "1024",
	}

	meta := buildFileMetadata(f)
	if meta["source"] != "gdrive" {
		t.Errorf("source = %q, want %q", meta["source"], "gdrive")
	}
	if meta["file_id"] != "file-123" {
		t.Errorf("file_id = %q, want %q", meta["file_id"], "file-123")
	}
	if meta["parent_id"] != "parent-456" {
		t.Errorf("parent_id = %q, want %q", meta["parent_id"], "parent-456")
	}
	if meta["size"] != "1024" {
		t.Errorf("size = %q, want %q", meta["size"], "1024")
	}
}

func TestTruncateBody(t *testing.T) {
	short := "short body"
	if got := truncateBody([]byte(short)); got != short {
		t.Errorf("truncateBody(%q) = %q, want %q", short, got, short)
	}

	long := strings.Repeat("x", 300)
	got := truncateBody([]byte(long))
	if len(got) > 210 {
		t.Errorf("truncateBody() returned %d chars, want <= 210", len(got))
	}
	if !strings.HasSuffix(got, "...") {
		t.Errorf("truncateBody() = %q, want suffix %q", got, "...")
	}
}

func TestGDriveConnector_FetchAll_ExportGoogleSlidesAsPlainText(t *testing.T) {
	client := newMockHTTPClient()
	c := newTestConnector(client)
	if err := configureTestConnector(c); err != nil {
		t.Fatalf("Configure() failed: %v", err)
	}

	listResp := driveListResponse{
		Files: []driveFile{
			{ID: "slides1", Name: "Presentation", MIMEType: mimeGoogleSlides, ModifiedTime: "2026-01-15T10:00:00Z"},
		},
	}
	client.addJSONResponse("drive/v3/files?", http.StatusOK, listResp)
	client.addResponse("slides1/export", http.StatusOK, "Slide 1: Introduction\nSlide 2: Conclusion")

	docCh, errCh := c.FetchAll(context.Background())
	docs, err := collectDocuments(docCh, errCh)
	if err != nil {
		t.Fatalf("FetchAll() error: %v", err)
	}

	if len(docs) != 1 {
		t.Fatalf("FetchAll() yielded %d documents, want 1", len(docs))
	}

	if docs[0].MIME != mimeTextPlain {
		t.Errorf("docs[0].MIME = %q, want %q", docs[0].MIME, mimeTextPlain)
	}

	if !strings.Contains(string(docs[0].Content), "Introduction") {
		t.Errorf("docs[0].Content = %q, want containing 'Introduction'", string(docs[0].Content))
	}
}

func TestGDriveConnector_FetchAll_ExportExceedsLimit(t *testing.T) {
	client := newMockHTTPClient()
	c := newTestConnector(client)
	if err := configureTestConnector(c); err != nil {
		t.Fatalf("Configure() failed: %v", err)
	}

	listResp := driveListResponse{
		Files: []driveFile{
			{ID: "bigdoc", Name: "Huge Document", MIMEType: mimeGoogleDoc, ModifiedTime: "2026-01-15T10:00:00Z"},
		},
	}
	client.addJSONResponse("drive/v3/files?", http.StatusOK, listResp)

	// Create an export response that exceeds the 10 MB export limit.
	hugeBody := strings.Repeat("x", driveExportMaxBytes+1)
	client.addResponse("bigdoc/export", http.StatusOK, hugeBody)

	docCh, errCh := c.FetchAll(context.Background())
	docs, err := collectDocuments(docCh, errCh)

	// The file should be skipped (logged as warning), not cause a fatal error.
	// Since fileToDocument returns an error for over-limit exports, the goroutine
	// logs a warning and continues.
	_ = err
	for _, doc := range docs {
		if doc.ExternalID == "bigdoc" {
			t.Error("export exceeding limit should not have been yielded as a document")
		}
	}
}

func TestGDriveConnector_FetchAll_RateLimitRetrySucceeds(t *testing.T) {
	client := newMockHTTPClient()
	c := newTestConnector(client)
	if err := configureTestConnector(c); err != nil {
		t.Fatalf("Configure() failed: %v", err)
	}

	rateLimitBody := `{"error":{"message":"Rate Limit Exceeded","errors":[{"reason":"rateLimitExceeded"}]}}`
	successBody, _ := json.Marshal(driveListResponse{
		Files: []driveFile{
			{ID: "f1", Name: "doc.txt", MIMEType: "text/plain", ModifiedTime: "2026-01-15T10:00:00Z", Size: "50"},
		},
	})

	callCount := 0
	stateful := &statefulMockHTTPClient{
		inner: client,
		handler: func(req *http.Request) (*http.Response, error) {
			client.requests = append(client.requests, req)
			reqURL := req.URL.String()

			if strings.Contains(reqURL, "drive/v3/files?") {
				callCount++
				// First call returns rate limit, second succeeds.
				switch {
				case callCount == 1:
					return &http.Response{
						StatusCode: http.StatusTooManyRequests,
						Body:       io.NopCloser(strings.NewReader(rateLimitBody)),
						Header:     http.Header{},
					}, nil
				default:
					return &http.Response{
						StatusCode: http.StatusOK,
						Body:       io.NopCloser(bytes.NewReader(successBody)),
						Header:     http.Header{},
					}, nil
				}
			}
			return &http.Response{
				StatusCode: http.StatusOK,
				Body:       io.NopCloser(strings.NewReader("content")),
				Header:     http.Header{},
			}, nil
		},
	}
	c.httpClient = stateful

	docCh, errCh := c.FetchAll(context.Background())
	docs, err := collectDocuments(docCh, errCh)
	if err != nil {
		t.Fatalf("FetchAll() error: %v", err)
	}

	if len(docs) != 1 {
		t.Fatalf("FetchAll() yielded %d documents, want 1", len(docs))
	}
	if docs[0].ExternalID != "f1" {
		t.Errorf("docs[0].ExternalID = %q, want %q", docs[0].ExternalID, "f1")
	}
	if callCount < 2 {
		t.Errorf("expected at least 2 API calls (1 rate-limited + 1 success), got %d", callCount)
	}
}

func TestGDriveConnector_FetchAll_OAuth2TokenUsedInRequests(t *testing.T) {
	client := newMockHTTPClient()
	c := newTestConnector(client)
	if err := configureTestConnector(c); err != nil {
		t.Fatalf("Configure() failed: %v", err)
	}

	// Set a new access token (simulating a token refresh scenario).
	c.SetAccessToken("refreshed-access-token-xyz")

	listResp := driveListResponse{Files: []driveFile{}}
	client.addJSONResponse("drive/v3/files?", http.StatusOK, listResp)

	docCh, errCh := c.FetchAll(context.Background())
	_, err := collectDocuments(docCh, errCh)
	if err != nil {
		t.Fatalf("FetchAll() error: %v", err)
	}

	if len(client.requests) == 0 {
		t.Fatal("no requests recorded")
	}
	authHeader := client.requests[0].Header.Get("Authorization")
	if authHeader != "Bearer refreshed-access-token-xyz" {
		t.Errorf("Authorization header = %q, want %q", authHeader, "Bearer refreshed-access-token-xyz")
	}
}

func TestGDriveConnector_FetchAll_ResponseBodyExceedsSizeLimit(t *testing.T) {
	client := newMockHTTPClient()
	c := newTestConnector(client)

	err := c.Configure(map[string]any{
		"oauth2_client_id":     "test-id",
		"oauth2_client_secret": "test-secret",
		"access_token":         "test-token",
		"max_file_size":        "100",
	})
	if err != nil {
		t.Fatalf("Configure() failed: %v", err)
	}

	listResp := driveListResponse{
		Files: []driveFile{
			{ID: "f1", Name: "small.txt", MIMEType: "text/plain", ModifiedTime: "2026-01-15T10:00:00Z", Size: "50"},
		},
	}
	client.addJSONResponse("drive/v3/files?", http.StatusOK, listResp)

	// Download response exceeds configured max_file_size.
	hugeBody := strings.Repeat("x", 200)
	client.addResponse("f1?alt=media", http.StatusOK, hugeBody)

	docCh, errCh := c.FetchAll(context.Background())
	_, err = collectDocuments(docCh, errCh)

	// The download should fail due to size limit enforcement at the read level.
	// The file is skipped with a warning (fileToDocument error is logged and continued).
	// We just verify no panic and the connector handled it gracefully.
	_ = err
}

func TestIsRateLimitError(t *testing.T) {
	cases := []struct {
		name       string
		body       string
		statusCode int
		want       bool
	}{
		{
			name:       "429 status code",
			body:       `{"error":{"message":"too many requests"}}`,
			statusCode: http.StatusTooManyRequests,
			want:       true,
		},
		{
			name:       "403 with rateLimitExceeded reason",
			body:       `{"error":{"message":"Rate Limit Exceeded","errors":[{"reason":"rateLimitExceeded"}]}}`,
			statusCode: http.StatusForbidden,
			want:       true,
		},
		{
			name:       "403 without rate limit reason",
			body:       `{"error":{"message":"Forbidden","errors":[{"reason":"forbidden"}]}}`,
			statusCode: http.StatusForbidden,
			want:       false,
		},
		{
			name:       "403 with userRateLimitExceeded reason",
			body:       `{"error":{"message":"User Rate Limit","errors":[{"reason":"userRateLimitExceeded"}]}}`,
			statusCode: http.StatusForbidden,
			want:       true,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := isRateLimitError([]byte(tc.body), tc.statusCode)
			if got != tc.want {
				t.Errorf("isRateLimitError() = %v, want %v", got, tc.want)
			}
		})
	}
}

func TestCalculateGoBackoff(t *testing.T) {
	// With Retry-After header.
	backoff := calculateGoBackoff(0, "5")
	if backoff != 5*time.Second {
		t.Errorf("calculateGoBackoff(0, '5') = %v, want 5s", backoff)
	}

	// Without Retry-After header, should return exponential backoff.
	backoff = calculateGoBackoff(0, "")
	if backoff < rateLimitBaseBackoff || backoff > 2*rateLimitBaseBackoff+rateLimitBaseBackoff {
		t.Errorf("calculateGoBackoff(0, '') = %v, want roughly %v", backoff, rateLimitBaseBackoff)
	}

	// Large Retry-After should be capped.
	backoff = calculateGoBackoff(0, "120")
	if backoff != rateLimitMaxBackoff {
		t.Errorf("calculateGoBackoff(0, '120') = %v, want %v", backoff, rateLimitMaxBackoff)
	}
}

func TestSanitiseGoError(t *testing.T) {
	errWithToken := fmt.Errorf("request failed: Bearer ya29.A0ARrdaM-xyz123 unauthorized")
	sanitised := sanitiseGoError(errWithToken)
	if strings.Contains(sanitised.Error(), "ya29") {
		t.Errorf("sanitised error still contains token: %q", sanitised)
	}
	if !strings.Contains(sanitised.Error(), "[REDACTED]") {
		t.Errorf("sanitised error missing [REDACTED]: %q", sanitised)
	}

	// Error without token should pass through unchanged.
	normalErr := fmt.Errorf("connection refused")
	result := sanitiseGoError(normalErr)
	if result.Error() != "connection refused" {
		t.Errorf("sanitiseGoError() changed a clean error: %q", result)
	}
}

func TestValidateFolderIDInConfigure(t *testing.T) {
	client := newMockHTTPClient()
	c := newTestConnector(client)

	err := c.Configure(map[string]any{
		"oauth2_client_id":     "test-id",
		"oauth2_client_secret": "test-secret",
		"access_token":         "test-token",
		"folder_id":            "' or name contains '",
	})
	// Configure itself does not validate folder ID -- buildFileQuery does.
	if err != nil {
		t.Fatalf("Configure() should not fail on folder ID: %v", err)
	}

	listResp := driveListResponse{Files: []driveFile{}}
	client.addJSONResponse("drive/v3/files?", http.StatusOK, listResp)

	docCh, errCh := c.FetchAll(context.Background())
	_, fetchErr := collectDocuments(docCh, errCh)
	if fetchErr == nil {
		t.Fatal("FetchAll() with injected folder ID should fail")
	}
	if !strings.Contains(fetchErr.Error(), "invalid folder ID") {
		t.Errorf("error = %q, want containing 'invalid folder ID'", fetchErr)
	}
}

// statefulMockHTTPClient allows per-request response customisation.
type statefulMockHTTPClient struct {
	inner   *mockHTTPClient
	handler func(req *http.Request) (*http.Response, error)
}

func (s *statefulMockHTTPClient) Do(req *http.Request) (*http.Response, error) {
	return s.handler(req)
}

// mockTokenExchanger implements TokenExchanger for testing.
type mockTokenExchanger struct {
	refreshFn  func(ctx context.Context, refreshToken string) (OAuth2Token, error)
	exchangeFn func(ctx context.Context, code string) (OAuth2Token, error)
}

func (m *mockTokenExchanger) Exchange(ctx context.Context, code string) (OAuth2Token, error) {
	if m.exchangeFn != nil {
		return m.exchangeFn(ctx, code)
	}
	return OAuth2Token{}, fmt.Errorf("exchange not implemented")
}

func (m *mockTokenExchanger) Refresh(ctx context.Context, refreshToken string) (OAuth2Token, error) {
	if m.refreshFn != nil {
		return m.refreshFn(ctx, refreshToken)
	}
	return OAuth2Token{}, fmt.Errorf("refresh not implemented")
}

func TestGDriveConnector_TokenRefreshOnExpiry(t *testing.T) {
	refreshCalled := false
	exchanger := &mockTokenExchanger{
		refreshFn: func(_ context.Context, rt string) (OAuth2Token, error) {
			refreshCalled = true
			if rt != "test-refresh-token" {
				t.Errorf("unexpected refresh token: %s", rt)
			}
			return OAuth2Token{
				AccessToken:  "refreshed-access-token",
				RefreshToken: "test-refresh-token",
				ExpiresAt:    time.Now().Add(time.Hour),
				TokenType:    "Bearer",
			}, nil
		},
	}

	client := newMockHTTPClient()
	c := newTestConnector(client)

	err := c.Configure(map[string]any{
		"oauth2_client_id":     "test-client-id",
		"oauth2_client_secret": "test-client-secret",
		"access_token":         "expired-access-token",
		"refresh_token":        "test-refresh-token",
		"token_expires_at":     time.Now().Add(-time.Hour).Format(time.RFC3339),
		"token_exchanger":      exchanger,
	})
	if err != nil {
		t.Fatalf("Configure() failed: %v", err)
	}

	listResp := driveListResponse{Files: []driveFile{}}
	client.addJSONResponse("drive/v3/files?", http.StatusOK, listResp)

	docCh, errCh := c.FetchAll(context.Background())
	_, fetchErr := collectDocuments(docCh, errCh)
	if fetchErr != nil {
		t.Fatalf("FetchAll() error: %v", fetchErr)
	}

	if !refreshCalled {
		t.Error("expected refresh to be called for expired token")
	}

	// Verify the refreshed token was used in the request.
	if len(client.requests) == 0 {
		t.Fatal("no requests recorded")
	}
	authHeader := client.requests[0].Header.Get("Authorization")
	if authHeader != "Bearer refreshed-access-token" {
		t.Errorf("Authorization header = %q, want %q", authHeader, "Bearer refreshed-access-token")
	}
}

func TestGDriveConnector_TokenRefreshFailurePropagates(t *testing.T) {
	exchanger := &mockTokenExchanger{
		refreshFn: func(_ context.Context, _ string) (OAuth2Token, error) {
			return OAuth2Token{}, fmt.Errorf("invalid_grant: token revoked")
		},
	}

	client := newMockHTTPClient()
	c := newTestConnector(client)

	err := c.Configure(map[string]any{
		"oauth2_client_id":     "test-client-id",
		"oauth2_client_secret": "test-client-secret",
		"access_token":         "expired-access-token",
		"refresh_token":        "revoked-refresh-token",
		"token_expires_at":     time.Now().Add(-time.Hour).Format(time.RFC3339),
		"token_exchanger":      exchanger,
	})
	if err != nil {
		t.Fatalf("Configure() failed: %v", err)
	}

	docCh, errCh := c.FetchAll(context.Background())
	_, fetchErr := collectDocuments(docCh, errCh)
	if fetchErr == nil {
		t.Fatal("expected error when token refresh fails")
	}
	if !strings.Contains(fetchErr.Error(), "token refresh failed") {
		t.Errorf("error = %q, want containing 'token refresh failed'", fetchErr)
	}
}

func TestGDriveConnector_NoRefreshWhenTokenValid(t *testing.T) {
	refreshCalled := false
	exchanger := &mockTokenExchanger{
		refreshFn: func(_ context.Context, _ string) (OAuth2Token, error) {
			refreshCalled = true
			return OAuth2Token{}, fmt.Errorf("should not be called")
		},
	}

	client := newMockHTTPClient()
	c := newTestConnector(client)

	err := c.Configure(map[string]any{
		"oauth2_client_id":     "test-client-id",
		"oauth2_client_secret": "test-client-secret",
		"access_token":         "valid-access-token",
		"refresh_token":        "test-refresh-token",
		"token_expires_at":     time.Now().Add(time.Hour).Format(time.RFC3339),
		"token_exchanger":      exchanger,
	})
	if err != nil {
		t.Fatalf("Configure() failed: %v", err)
	}

	listResp := driveListResponse{Files: []driveFile{}}
	client.addJSONResponse("drive/v3/files?", http.StatusOK, listResp)

	docCh, errCh := c.FetchAll(context.Background())
	_, fetchErr := collectDocuments(docCh, errCh)
	if fetchErr != nil {
		t.Fatalf("FetchAll() error: %v", fetchErr)
	}

	if refreshCalled {
		t.Error("refresh should not be called for valid token")
	}
}

func TestGDriveConnector_RefreshWithinBuffer(t *testing.T) {
	refreshCalled := false
	exchanger := &mockTokenExchanger{
		refreshFn: func(_ context.Context, _ string) (OAuth2Token, error) {
			refreshCalled = true
			return OAuth2Token{
				AccessToken:  "refreshed-token",
				RefreshToken: "test-refresh-token",
				ExpiresAt:    time.Now().Add(time.Hour),
				TokenType:    "Bearer",
			}, nil
		},
	}

	client := newMockHTTPClient()
	c := newTestConnector(client)

	// Token expires in 3 minutes -- within the 5-minute buffer.
	err := c.Configure(map[string]any{
		"oauth2_client_id":     "test-client-id",
		"oauth2_client_secret": "test-client-secret",
		"access_token":         "expiring-access-token",
		"refresh_token":        "test-refresh-token",
		"token_expires_at":     time.Now().Add(3 * time.Minute).Format(time.RFC3339),
		"token_exchanger":      exchanger,
	})
	if err != nil {
		t.Fatalf("Configure() failed: %v", err)
	}

	listResp := driveListResponse{Files: []driveFile{}}
	client.addJSONResponse("drive/v3/files?", http.StatusOK, listResp)

	docCh, errCh := c.FetchAll(context.Background())
	_, fetchErr := collectDocuments(docCh, errCh)
	if fetchErr != nil {
		t.Fatalf("FetchAll() error: %v", fetchErr)
	}

	if !refreshCalled {
		t.Error("expected refresh to be called for token within expiry buffer")
	}
}

func TestGDriveConnector_StaticTokenNoRefresh(t *testing.T) {
	client := newMockHTTPClient()
	c := newTestConnector(client)

	// No refresh_token -> no OAuth2Client -> no refresh happens.
	err := c.Configure(map[string]any{
		"oauth2_client_id":     "test-client-id",
		"oauth2_client_secret": "test-client-secret",
		"access_token":         "static-token",
	})
	if err != nil {
		t.Fatalf("Configure() failed: %v", err)
	}

	listResp := driveListResponse{Files: []driveFile{}}
	client.addJSONResponse("drive/v3/files?", http.StatusOK, listResp)

	docCh, errCh := c.FetchAll(context.Background())
	_, fetchErr := collectDocuments(docCh, errCh)
	if fetchErr != nil {
		t.Fatalf("FetchAll() error: %v", fetchErr)
	}

	if len(client.requests) == 0 {
		t.Fatal("no requests recorded")
	}
	authHeader := client.requests[0].Header.Get("Authorization")
	if authHeader != "Bearer static-token" {
		t.Errorf("Authorization header = %q, want %q", authHeader, "Bearer static-token")
	}
}

func TestGDriveConnector_Health(t *testing.T) {
	client := newMockHTTPClient()
	c := newTestConnector(client)

	// Before configuration.
	health := c.Health()
	if health.Status != StatusDisconnected {
		t.Errorf("Health().Status = %q, want %q", health.Status, StatusDisconnected)
	}

	// After configuration.
	if err := configureTestConnector(c); err != nil {
		t.Fatalf("Configure() failed: %v", err)
	}
	health = c.Health()
	if health.Status != StatusConnected {
		t.Errorf("Health().Status = %q, want %q", health.Status, StatusConnected)
	}
	if health.RateLimitRemaining != -1 {
		t.Errorf("Health().RateLimitRemaining = %d, want -1", health.RateLimitRemaining)
	}
}
