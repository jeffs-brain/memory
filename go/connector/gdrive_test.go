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

// mockHTTPClient implements HTTPClient for testing with configurable
// responses keyed by request URL substring.
type mockHTTPClient struct {
	responses map[string]*http.Response
	requests  []*http.Request
}

func newMockHTTPClient() *mockHTTPClient {
	return &mockHTTPClient{
		responses: make(map[string]*http.Response),
	}
}

func (m *mockHTTPClient) Do(req *http.Request) (*http.Response, error) {
	m.requests = append(m.requests, req)

	for pattern, resp := range m.responses {
		if strings.Contains(req.URL.String(), pattern) {
			return resp, nil
		}
	}
	return &http.Response{
		StatusCode: http.StatusNotFound,
		Body:       io.NopCloser(strings.NewReader(`{"error":{"message":"not found"}}`)),
	}, nil
}

func (m *mockHTTPClient) addResponse(urlPattern string, statusCode int, body string) {
	m.responses[urlPattern] = &http.Response{
		StatusCode: statusCode,
		Body:       io.NopCloser(strings.NewReader(body)),
		Header:     http.Header{},
	}
}

func (m *mockHTTPClient) addJSONResponse(urlPattern string, statusCode int, payload any) {
	data, _ := json.Marshal(payload)
	m.responses[urlPattern] = &http.Response{
		StatusCode: statusCode,
		Body:       io.NopCloser(bytes.NewReader(data)),
		Header:     http.Header{},
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
	return c.Configure(map[string]string{
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
		config  map[string]string
		wantErr string
	}{
		{
			name: "valid configuration",
			config: map[string]string{
				"oauth2_client_id":     "client-123",
				"oauth2_client_secret": "secret-456",
			},
		},
		{
			name: "missing client ID",
			config: map[string]string{
				"oauth2_client_secret": "secret-456",
			},
			wantErr: "oauth2_client_id is required",
		},
		{
			name: "empty client ID",
			config: map[string]string{
				"oauth2_client_id":     "",
				"oauth2_client_secret": "secret-456",
			},
			wantErr: "oauth2_client_id is required",
		},
		{
			name: "missing client secret",
			config: map[string]string{
				"oauth2_client_id": "client-123",
			},
			wantErr: "oauth2_client_secret is required",
		},
		{
			name: "with folder ID",
			config: map[string]string{
				"oauth2_client_id":     "client-123",
				"oauth2_client_secret": "secret-456",
				"folder_id":            "folder-789",
			},
		},
		{
			name: "with MIME type filter",
			config: map[string]string{
				"oauth2_client_id":     "client-123",
				"oauth2_client_secret": "secret-456",
				"mime_type_filter":     "application/pdf,text/plain",
			},
		},
		{
			name: "with shared drives",
			config: map[string]string{
				"oauth2_client_id":      "client-123",
				"oauth2_client_secret":  "secret-456",
				"include_shared_drives": "true",
			},
		},
		{
			name: "invalid max file size",
			config: map[string]string{
				"oauth2_client_id":     "client-123",
				"oauth2_client_secret": "secret-456",
				"max_file_size":        "not-a-number",
			},
			wantErr: "invalid max_file_size",
		},
		{
			name: "custom max file size",
			config: map[string]string{
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
	client.responses = nil
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

	err := c.Configure(map[string]string{
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

	rateLimitBody := `{"error":{"message":"Rate Limit Exceeded","errors":[{"reason":"rateLimitExceeded"}]}}`
	client.addResponse("drive/v3/files?", http.StatusForbidden, rateLimitBody)

	docCh, errCh := c.FetchAll(context.Background())
	_, err := collectDocuments(docCh, errCh)

	if err == nil {
		t.Fatal("FetchAll() succeeded, want rate limit error")
	}
	if !strings.Contains(err.Error(), "rate limit exceeded") {
		t.Errorf("error = %q, want containing %q", err, "rate limit exceeded")
	}
}

func TestGDriveConnector_FetchAll_FolderIDFiltering(t *testing.T) {
	client := newMockHTTPClient()
	c := newTestConnector(client)

	err := c.Configure(map[string]string{
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

	err := c.Configure(map[string]string{
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

	err := c.Configure(map[string]string{
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
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			client := newMockHTTPClient()
			c := newTestConnector(client)
			c.config = tc.config
			query := c.buildFileQuery()

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

// statefulMockHTTPClient allows per-request response customisation.
type statefulMockHTTPClient struct {
	inner   *mockHTTPClient
	handler func(req *http.Request) (*http.Response, error)
}

func (s *statefulMockHTTPClient) Do(req *http.Request) (*http.Response, error) {
	return s.handler(req)
}
