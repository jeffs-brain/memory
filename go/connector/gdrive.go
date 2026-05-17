// SPDX-License-Identifier: Apache-2.0

package connector

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"
)

// Google Drive API constants.
const (
	// driveFilesURL is the base endpoint for Drive files.list.
	driveFilesURL = "https://www.googleapis.com/drive/v3/files"

	// driveChangesURL is the base endpoint for Drive changes.list.
	driveChangesURL = "https://www.googleapis.com/drive/v3/changes"

	// driveStartPageTokenURL fetches the initial changes start token.
	driveStartPageTokenURL = "https://www.googleapis.com/drive/v3/changes/startPageToken"

	// driveFileFields are the fields requested for each file in list/changes.
	driveFileFields = "id,name,mimeType,modifiedTime,size,parents,webViewLink"

	// driveListFields specifies the top-level response fields for files.list.
	driveListFields = "nextPageToken,files(" + driveFileFields + ")"

	// driveChangesFields specifies the response fields for changes.list.
	driveChangesFields = "nextPageToken,newStartPageToken,changes(fileId,removed,file(" + driveFileFields + "))"

	// driveDefaultPageSize is the number of results per API page.
	driveDefaultPageSize = 100

	// driveExportMaxBytes is the Google-imposed limit on file exports (10 MB).
	driveExportMaxBytes = 10 * 1024 * 1024

	// defaultMaxFileSize is the default maximum file size to download (50 MB).
	defaultMaxFileSize int64 = 50 * 1024 * 1024

	// defaultPollInterval is the default continuous sync interval.
	defaultPollInterval = 15 * time.Minute

	// httpRequestTimeout is the per-request timeout for Google API calls.
	httpRequestTimeout = 30 * time.Second

	// httpDownloadTimeout is the timeout for file download/export requests.
	httpDownloadTimeout = 5 * time.Minute
)

// Google MIME types for native document formats.
const (
	mimeGoogleDoc          = "application/vnd.google-apps.document"
	mimeGoogleSheet        = "application/vnd.google-apps.spreadsheet"
	mimeGoogleSlides       = "application/vnd.google-apps.presentation"
	mimeGoogleDrawing      = "application/vnd.google-apps.drawing"
	mimeGoogleFolder       = "application/vnd.google-apps.folder"
	mimeTextMarkdown       = "text/markdown"
	mimeTextPlain          = "text/plain"
	mimeTextCSV            = "text/csv"
	mimeImagePNG           = "image/png"
	mimeApplicationOctet   = "application/octet-stream"
)

// defaultExportFormats maps Google-native MIME types to the export format
// used when downloading. Overridable via GDriveConfig.ExportFormats.
var defaultExportFormats = map[string]string{
	mimeGoogleDoc:     mimeTextMarkdown,
	mimeGoogleSheet:   mimeTextCSV,
	mimeGoogleSlides:  mimeTextPlain,
	mimeGoogleDrawing: mimeImagePNG,
}

// GDriveConfig holds Google Drive-specific connector settings.
type GDriveConfig struct {
	// OAuth2 holds the OAuth2 credentials for the Drive API.
	OAuth2 OAuth2Config

	// FolderID restricts sync to files within a specific folder (recursive).
	// Empty means sync the entire drive.
	FolderID string

	// MIMETypeFilter restricts sync to specific MIME types. Empty means all.
	MIMETypeFilter []string

	// IncludeSharedDrives enables shared/team drive access. Default: false.
	IncludeSharedDrives bool

	// MaxFileSize is the maximum file size in bytes to download.
	// Files larger than this are skipped with a warning. Default: 50 MB.
	MaxFileSize int64

	// ExportFormats overrides the default export format mapping for
	// Google-native document types.
	ExportFormats map[string]string
}

// driveFile mirrors the subset of the Google Drive API File resource
// needed for sync operations.
type driveFile struct {
	ID           string   `json:"id"`
	Name         string   `json:"name"`
	MIMEType     string   `json:"mimeType"`
	ModifiedTime string   `json:"modifiedTime"`
	Size         string   `json:"size"`
	Parents      []string `json:"parents"`
	WebViewLink  string   `json:"webViewLink"`
}

// driveChange mirrors a single entry from the Drive Changes API response.
type driveChange struct {
	FileID  string     `json:"fileId"`
	File    *driveFile `json:"file"`
	Removed bool       `json:"removed"`
}

// driveListResponse is the files.list API response envelope.
type driveListResponse struct {
	NextPageToken string      `json:"nextPageToken"`
	Files         []driveFile `json:"files"`
}

// driveChangesResponse is the changes.list API response envelope.
type driveChangesResponse struct {
	NextPageToken     string        `json:"nextPageToken"`
	NewStartPageToken string        `json:"newStartPageToken"`
	Changes           []driveChange `json:"changes"`
}

// driveStartPageTokenResponse is the changes.getStartPageToken response.
type driveStartPageTokenResponse struct {
	StartPageToken string `json:"startPageToken"`
}

// HTTPClient abstracts HTTP transport for testing.
type HTTPClient interface {
	Do(req *http.Request) (*http.Response, error)
}

// GDriveConnector implements the Connector interface for Google Drive.
// It supports full sync via files.list and incremental sync via the
// Changes API with startPageToken cursors.
type GDriveConnector struct {
	deps        ConnectorConfig
	config      GDriveConfig
	httpClient  HTTPClient
	logger      *slog.Logger
	accessToken string
	configured  bool
	stopFn      context.CancelFunc
}

// NewGDriveConnector creates a new Google Drive connector with the
// given shared dependencies. Call Configure before fetching.
func NewGDriveConnector(deps ConnectorConfig, logger *slog.Logger, httpClient HTTPClient) *GDriveConnector {
	return &GDriveConnector{
		deps:       deps,
		logger:     logger,
		httpClient: httpClient,
	}
}

// Name returns the connector identifier.
func (c *GDriveConnector) Name() string { return "gdrive" }

// Configure validates and stores the Google Drive-specific settings.
func (c *GDriveConnector) Configure(config map[string]string) error {
	clientID, ok := config["oauth2_client_id"]
	if !ok || clientID == "" {
		return fmt.Errorf("gdrive: oauth2_client_id is required")
	}

	clientSecret, ok := config["oauth2_client_secret"]
	if !ok || clientSecret == "" {
		return fmt.Errorf("gdrive: oauth2_client_secret is required")
	}

	c.config = GDriveConfig{
		OAuth2: OAuth2Config{
			ClientID:     clientID,
			ClientSecret: clientSecret,
			AuthURL:      "https://accounts.google.com/o/oauth2/v2/auth",
			TokenURL:     "https://oauth2.googleapis.com/token",
			Scopes:       []string{"https://www.googleapis.com/auth/drive.readonly"},
			RedirectURI:  config["oauth2_redirect_uri"],
		},
		FolderID:            config["folder_id"],
		IncludeSharedDrives: config["include_shared_drives"] == "true",
		MaxFileSize:         defaultMaxFileSize,
	}

	if maxSize, exists := config["max_file_size"]; exists {
		parsed, err := strconv.ParseInt(maxSize, 10, 64)
		if err != nil {
			return fmt.Errorf("gdrive: invalid max_file_size %q: %w", maxSize, err)
		}
		c.config.MaxFileSize = parsed
	}

	if mimeFilter, exists := config["mime_type_filter"]; exists && mimeFilter != "" {
		c.config.MIMETypeFilter = strings.Split(mimeFilter, ",")
	}

	if token, exists := config["access_token"]; exists {
		c.accessToken = token
	}

	c.config.ExportFormats = make(map[string]string, len(defaultExportFormats))
	for k, v := range defaultExportFormats {
		c.config.ExportFormats[k] = v
	}

	c.configured = true
	return nil
}

// SetAccessToken sets the OAuth2 access token directly. This is used
// when the caller manages token lifecycle externally.
func (c *GDriveConnector) SetAccessToken(token string) {
	c.accessToken = token
}

// FetchAll performs a full sync of all files from the configured Drive
// scope. It initialises the Changes API start page token for subsequent
// incremental syncs.
func (c *GDriveConnector) FetchAll(ctx context.Context) (<-chan ConnectorDocument, <-chan error) {
	docCh := make(chan ConnectorDocument, driveDefaultPageSize)
	errCh := make(chan error, 1)

	go func() {
		defer close(docCh)
		defer close(errCh)

		if !c.configured {
			errCh <- fmt.Errorf("gdrive: connector not configured")
			return
		}

		pageToken := ""
		for {
			resp, err := c.listFiles(ctx, pageToken)
			if err != nil {
				errCh <- fmt.Errorf("gdrive: listing files: %w", err)
				return
			}

			for i := range resp.Files {
				doc, err := c.fileToDocument(ctx, &resp.Files[i])
				if err != nil {
					c.logger.Warn("gdrive: skipping file",
						slog.String("file_id", resp.Files[i].ID),
						slog.String("name", resp.Files[i].Name),
						slog.String("error", err.Error()),
					)
					continue
				}
				if doc == nil {
					continue
				}

				select {
				case <-ctx.Done():
					errCh <- ctx.Err()
					return
				case docCh <- *doc:
				}
			}

			if resp.NextPageToken == "" {
				break
			}
			pageToken = resp.NextPageToken
		}
	}()

	return docCh, errCh
}

// FetchSince performs an incremental sync using the Drive Changes API.
// The cursor value must be a valid startPageToken from a previous sync.
func (c *GDriveConnector) FetchSince(ctx context.Context, cursor SyncCursor) (<-chan ConnectorDocument, <-chan error) {
	docCh := make(chan ConnectorDocument, driveDefaultPageSize)
	errCh := make(chan error, 1)

	go func() {
		defer close(docCh)
		defer close(errCh)

		if !c.configured {
			errCh <- fmt.Errorf("gdrive: connector not configured")
			return
		}

		pageToken := cursor.Value
		for {
			resp, err := c.listChanges(ctx, pageToken)
			if err != nil {
				errCh <- fmt.Errorf("gdrive: listing changes: %w", err)
				return
			}

			for i := range resp.Changes {
				change := &resp.Changes[i]
				doc, err := c.changeToDocument(ctx, change)
				if err != nil {
					c.logger.Warn("gdrive: skipping change",
						slog.String("file_id", change.FileID),
						slog.String("error", err.Error()),
					)
					continue
				}
				if doc == nil {
					continue
				}

				select {
				case <-ctx.Done():
					errCh <- ctx.Err()
					return
				case docCh <- *doc:
				}
			}

			if resp.NewStartPageToken != "" {
				// Emit a cursor-update document so callers can persist the new token.
				select {
				case <-ctx.Done():
					errCh <- ctx.Err()
					return
				case docCh <- ConnectorDocument{
					ExternalID: "__cursor_update__",
					Metadata: map[string]string{
						"new_start_page_token": resp.NewStartPageToken,
					},
				}:
				}
				break
			}

			if resp.NextPageToken == "" {
				break
			}
			pageToken = resp.NextPageToken
		}
	}()

	return docCh, errCh
}

// GetStartPageToken retrieves the current Changes API start page token.
// This token is used as the initial cursor for incremental sync.
func (c *GDriveConnector) GetStartPageToken(ctx context.Context) (string, error) {
	reqURL := driveStartPageTokenURL
	if c.config.IncludeSharedDrives {
		reqURL += "?supportsAllDrives=true"
	}

	body, err := c.doAPIGet(ctx, reqURL, httpRequestTimeout)
	if err != nil {
		return "", fmt.Errorf("gdrive: getting start page token: %w", err)
	}

	var resp driveStartPageTokenResponse
	if err := json.Unmarshal(body, &resp); err != nil {
		return "", fmt.Errorf("gdrive: parsing start page token response: %w", err)
	}
	return resp.StartPageToken, nil
}

// Start begins a continuous sync loop. Not implemented yet as it depends
// on the full P5-1 SyncStateManager integration.
func (c *GDriveConnector) Start(_ context.Context) error {
	return fmt.Errorf("gdrive: continuous sync not yet supported without P5-1 SyncStateManager")
}

// Stop halts the continuous sync loop.
func (c *GDriveConnector) Stop() error {
	if c.stopFn != nil {
		c.stopFn()
	}
	return nil
}

// listFiles calls the Drive files.list API with pagination and filtering.
func (c *GDriveConnector) listFiles(ctx context.Context, pageToken string) (*driveListResponse, error) {
	params := url.Values{
		"pageSize": {strconv.Itoa(driveDefaultPageSize)},
		"fields":   {driveListFields},
	}

	if pageToken != "" {
		params.Set("pageToken", pageToken)
	}

	query := c.buildFileQuery()
	if query != "" {
		params.Set("q", query)
	}

	if c.config.IncludeSharedDrives {
		params.Set("supportsAllDrives", "true")
		params.Set("includeItemsFromAllDrives", "true")
	}

	reqURL := driveFilesURL + "?" + params.Encode()
	body, err := c.doAPIGet(ctx, reqURL, httpRequestTimeout)
	if err != nil {
		return nil, err
	}

	var resp driveListResponse
	if err := json.Unmarshal(body, &resp); err != nil {
		return nil, fmt.Errorf("parsing files.list response: %w", err)
	}
	return &resp, nil
}

// listChanges calls the Drive changes.list API with the given page token.
func (c *GDriveConnector) listChanges(ctx context.Context, pageToken string) (*driveChangesResponse, error) {
	params := url.Values{
		"pageToken": {pageToken},
		"pageSize":  {strconv.Itoa(driveDefaultPageSize)},
		"fields":    {driveChangesFields},
	}

	if c.config.IncludeSharedDrives {
		params.Set("supportsAllDrives", "true")
		params.Set("includeItemsFromAllDrives", "true")
	}

	reqURL := driveChangesURL + "?" + params.Encode()
	body, err := c.doAPIGet(ctx, reqURL, httpRequestTimeout)
	if err != nil {
		return nil, err
	}

	var resp driveChangesResponse
	if err := json.Unmarshal(body, &resp); err != nil {
		return nil, fmt.Errorf("parsing changes.list response: %w", err)
	}
	return &resp, nil
}

// buildFileQuery constructs the Drive API query string from configuration.
func (c *GDriveConnector) buildFileQuery() string {
	var parts []string

	// Exclude folders from results.
	parts = append(parts, fmt.Sprintf("mimeType != '%s'", mimeGoogleFolder))

	// Filter to a specific folder.
	if c.config.FolderID != "" {
		parts = append(parts, fmt.Sprintf("'%s' in parents", c.config.FolderID))
	}

	// Filter by MIME types.
	if len(c.config.MIMETypeFilter) > 0 {
		mimeConditions := make([]string, 0, len(c.config.MIMETypeFilter))
		for _, mime := range c.config.MIMETypeFilter {
			mimeConditions = append(mimeConditions, fmt.Sprintf("mimeType = '%s'", mime))
		}
		parts = append(parts, "("+strings.Join(mimeConditions, " or ")+")")
	}

	// Exclude trashed files.
	parts = append(parts, "trashed = false")

	return strings.Join(parts, " and ")
}

// fileToDocument converts a Drive file resource into a ConnectorDocument.
// Returns nil for files that should be skipped (folders, too large).
func (c *GDriveConnector) fileToDocument(ctx context.Context, f *driveFile) (*ConnectorDocument, error) {
	if f.MIMEType == mimeGoogleFolder {
		return nil, nil
	}

	fileSize := parseFileSize(f.Size)
	if fileSize > c.config.MaxFileSize && !isGoogleNativeFormat(f.MIMEType) {
		c.logger.Warn("gdrive: file exceeds size limit, skipping",
			slog.String("file_id", f.ID),
			slog.String("name", f.Name),
			slog.Int64("size", fileSize),
			slog.Int64("limit", c.config.MaxFileSize),
		)
		return nil, nil
	}

	modifiedAt, err := time.Parse(time.RFC3339, f.ModifiedTime)
	if err != nil {
		return nil, fmt.Errorf("parsing modified time for %s: %w", f.ID, err)
	}

	content, mime, err := c.downloadOrExport(ctx, f)
	if err != nil {
		return nil, fmt.Errorf("downloading %s (%s): %w", f.ID, f.Name, err)
	}

	return &ConnectorDocument{
		ExternalID: f.ID,
		Content:    content,
		MIME:       mime,
		Title:      f.Name,
		URL:        f.WebViewLink,
		Metadata:   buildFileMetadata(f),
		ModifiedAt: modifiedAt,
	}, nil
}

// changeToDocument converts a Drive change entry into a ConnectorDocument.
func (c *GDriveConnector) changeToDocument(ctx context.Context, ch *driveChange) (*ConnectorDocument, error) {
	if ch.Removed || ch.File == nil {
		return &ConnectorDocument{
			ExternalID: ch.FileID,
			Deleted:    true,
			Metadata: map[string]string{
				"source": "gdrive",
			},
		}, nil
	}

	return c.fileToDocument(ctx, ch.File)
}

// downloadOrExport fetches file content. Google-native formats are
// exported via the export API; all others are downloaded directly.
func (c *GDriveConnector) downloadOrExport(ctx context.Context, f *driveFile) ([]byte, string, error) {
	exportMIME, isNative := c.resolveExportFormat(f.MIMEType)
	if isNative {
		return c.exportFile(ctx, f.ID, exportMIME)
	}
	return c.downloadFile(ctx, f.ID, f.MIMEType)
}

// resolveExportFormat returns the target MIME type for a Google-native
// format, checking custom overrides first.
func (c *GDriveConnector) resolveExportFormat(sourceMIME string) (string, bool) {
	if c.config.ExportFormats != nil {
		if target, exists := c.config.ExportFormats[sourceMIME]; exists {
			return target, true
		}
	}
	target, exists := defaultExportFormats[sourceMIME]
	return target, exists
}

// exportFile uses the Drive export API to download a Google-native file.
func (c *GDriveConnector) exportFile(ctx context.Context, fileID, targetMIME string) ([]byte, string, error) {
	exportURL := fmt.Sprintf("%s/%s/export?mimeType=%s",
		driveFilesURL, url.PathEscape(fileID), url.QueryEscape(targetMIME))

	body, err := c.doAPIGet(ctx, exportURL, httpDownloadTimeout)
	if err != nil {
		return nil, "", fmt.Errorf("exporting file %s: %w", fileID, err)
	}

	if int64(len(body)) > driveExportMaxBytes {
		return nil, "", fmt.Errorf("export of %s exceeds %d byte limit", fileID, driveExportMaxBytes)
	}

	return body, targetMIME, nil
}

// downloadFile fetches a binary file directly from Drive.
func (c *GDriveConnector) downloadFile(ctx context.Context, fileID, mime string) ([]byte, string, error) {
	downloadURL := fmt.Sprintf("%s/%s?alt=media", driveFilesURL, url.PathEscape(fileID))

	body, err := c.doAPIGet(ctx, downloadURL, httpDownloadTimeout)
	if err != nil {
		return nil, "", fmt.Errorf("downloading file %s: %w", fileID, err)
	}

	return body, mime, nil
}

// doAPIGet performs an authenticated GET request to the Google Drive API.
func (c *GDriveConnector) doAPIGet(ctx context.Context, reqURL string, timeout time.Duration) ([]byte, error) {
	if c.deps.RateLimiter != nil {
		// Rate limiter integration point; currently a no-op placeholder
		// until P5-1 is merged with the full RateLimiter.Acquire method.
	}

	reqCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	req, err := http.NewRequestWithContext(reqCtx, http.MethodGet, reqURL, nil)
	if err != nil {
		return nil, fmt.Errorf("creating request: %w", err)
	}
	req.Header.Set("Authorization", "Bearer "+c.accessToken)

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("executing request: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("reading response body: %w", err)
	}

	if resp.StatusCode == http.StatusForbidden {
		return nil, c.handleRateLimitError(body, resp.StatusCode)
	}

	if resp.StatusCode == http.StatusTooManyRequests {
		return nil, c.handleRateLimitError(body, resp.StatusCode)
	}

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf("API returned status %d: %s", resp.StatusCode, truncateBody(body))
	}

	return body, nil
}

// handleRateLimitError parses rate limit error responses.
func (c *GDriveConnector) handleRateLimitError(body []byte, statusCode int) error {
	var apiErr struct {
		Error struct {
			Message string `json:"message"`
			Errors  []struct {
				Reason string `json:"reason"`
			} `json:"errors"`
		} `json:"error"`
	}

	if err := json.Unmarshal(body, &apiErr); err != nil {
		return fmt.Errorf("rate limit error (status %d): %s", statusCode, truncateBody(body))
	}

	for _, e := range apiErr.Error.Errors {
		if e.Reason == "rateLimitExceeded" || e.Reason == "userRateLimitExceeded" {
			return fmt.Errorf("gdrive: rate limit exceeded: %s", apiErr.Error.Message)
		}
	}

	return fmt.Errorf("gdrive: API error (status %d): %s", statusCode, apiErr.Error.Message)
}

// isGoogleNativeFormat reports whether the MIME type is a Google-native
// document format that must be exported rather than downloaded.
func isGoogleNativeFormat(mime string) bool {
	switch mime {
	case mimeGoogleDoc, mimeGoogleSheet, mimeGoogleSlides, mimeGoogleDrawing:
		return true
	}
	return false
}

// parseFileSize parses the file size string from the Drive API. Returns
// 0 for empty or unparsable values (Google-native formats omit size).
func parseFileSize(sizeStr string) int64 {
	if sizeStr == "" {
		return 0
	}
	size, err := strconv.ParseInt(sizeStr, 10, 64)
	if err != nil {
		return 0
	}
	return size
}

// buildFileMetadata constructs the metadata map for a Drive file.
func buildFileMetadata(f *driveFile) map[string]string {
	meta := map[string]string{
		"source":    "gdrive",
		"mime_type": f.MIMEType,
		"file_id":   f.ID,
	}
	if len(f.Parents) > 0 {
		meta["parent_id"] = f.Parents[0]
	}
	if f.Size != "" {
		meta["size"] = f.Size
	}
	return meta
}

// truncateBody returns at most 200 bytes of a response body for error
// messages, avoiding excessively long error strings.
func truncateBody(body []byte) string {
	const maxLen = 200
	if len(body) <= maxLen {
		return string(body)
	}
	return string(body[:maxLen]) + "..."
}
