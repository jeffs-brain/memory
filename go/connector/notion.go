// SPDX-License-Identifier: Apache-2.0

package connector

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"sync"
	"time"
)

const (
	notionAPIVersion        = "2022-06-28"
	notionBaseURL           = "https://api.notion.com/v1"
	notionDefaultMaxDepth   = 10
	notionDefaultPollInterval = 15 * time.Minute
	notionDefaultTimeout    = 30 * time.Second
	notionDefaultPageSize   = 100
	notionDefaultRateLimit  = 3
	notionMaxRetryAttempts  = 5
	notionMaxResponseSize   = 10 * 1024 * 1024 // 10 MiB
)

// NotionConnectorConfig holds Notion-specific configuration.
type NotionConnectorConfig struct {
	APIToken          string
	RootPageIDs       []string
	DatabaseIDs       []string
	IncludeChildPages bool              // default: true
	IncludeDatabases  bool              // default: true
	MaxDepth          int               // default: 10
	BlockTypeFilter   map[string]struct{} // when non-empty, only these block types are rendered

	// OAuth2 fields: used when the connector authenticates via
	// Notion's public OAuth integration rather than an internal
	// integration token (which does not expire).
	OAuth2         *NotionOAuth2Config
	RefreshToken   string
	TokenExpiresAt time.Time
}

// NotionOAuth2Config holds Notion OAuth2-specific credentials.
type NotionOAuth2Config struct {
	ClientID     string
	ClientSecret string
}

// NotionPage represents a Notion page object from the API.
type NotionPage struct {
	ID             string
	Title          string
	URL            string
	LastEditedTime time.Time
	ParentID       string
}

// NotionHTTPClient abstracts HTTP calls so tests can inject a mock.
type NotionHTTPClient interface {
	Do(req *http.Request) (*http.Response, error)
}

// NotionConnector implements [Connector] for Notion workspaces.
type NotionConnector struct {
	deps         ConnectorConfig
	config       NotionConnectorConfig
	httpClient   NotionHTTPClient
	rateLimiter  *RateLimiter
	stopCh       chan struct{}
	stopOnce     sync.Once
	configured   bool
	oauth2Client *NotionOAuth2Client
	tokenStore   TokenStore
	mu           sync.Mutex
	lastSyncAt   time.Time
	errorCount   int64
}

// NewNotionConnector creates a Notion connector with the given base
// dependencies and an HTTP client for API calls. A default rate
// limiter of 3 req/sec is created if none is provided via deps.
func NewNotionConnector(deps ConnectorConfig, httpClient NotionHTTPClient) *NotionConnector {
	if httpClient == nil {
		httpClient = &http.Client{Timeout: notionDefaultTimeout}
	}

	rl := deps.RateLimiter
	if rl == nil {
		rl = NewRateLimiter(RateLimiterConfig{
			MaxTokens:  notionDefaultRateLimit,
			RefillRate: notionDefaultRateLimit,
		})
	}

	return &NotionConnector{
		deps:        deps,
		httpClient:  httpClient,
		rateLimiter: rl,
		stopCh:      make(chan struct{}),
	}
}

// Name returns "notion".
func (c *NotionConnector) Name() string { return "notion" }

// parseBlockTypeFilter converts a raw config value to a block type filter set.
func parseBlockTypeFilter(raw any) map[string]struct{} {
	if raw == nil {
		return nil
	}
	if strs, ok := raw.([]string); ok && len(strs) > 0 {
		m := make(map[string]struct{}, len(strs))
		for _, s := range strs {
			m[s] = struct{}{}
		}
		return m
	}
	if arr, ok := raw.([]any); ok && len(arr) > 0 {
		m := make(map[string]struct{}, len(arr))
		for _, v := range arr {
			if s, ok := v.(string); ok {
				m[s] = struct{}{}
			}
		}
		if len(m) > 0 {
			return m
		}
	}
	return nil
}

// Configure validates and stores Notion-specific configuration.
// Accepts map[string]any to match the P5-1 Connector interface.
func (c *NotionConnector) Configure(config map[string]any) error {
	token, _ := config["apiToken"].(string)
	if strings.TrimSpace(token) == "" {
		return fmt.Errorf("connector/notion: apiToken is required")
	}

	cfg := NotionConnectorConfig{
		APIToken:          token,
		IncludeChildPages: true,
		IncludeDatabases:  true,
		MaxDepth:          notionDefaultMaxDepth,
	}

	if ids, ok := config["rootPageIds"].([]string); ok {
		cfg.RootPageIDs = ids
	}
	if raw, ok := config["rootPageIds"].([]any); ok {
		for _, v := range raw {
			if s, ok := v.(string); ok {
				cfg.RootPageIDs = append(cfg.RootPageIDs, s)
			}
		}
	}

	if ids, ok := config["databaseIds"].([]string); ok {
		cfg.DatabaseIDs = ids
	}
	if raw, ok := config["databaseIds"].([]any); ok {
		for _, v := range raw {
			if s, ok := v.(string); ok {
				cfg.DatabaseIDs = append(cfg.DatabaseIDs, s)
			}
		}
	}

	if v, ok := config["includeChildPages"].(bool); ok {
		cfg.IncludeChildPages = v
	}
	if v, ok := config["includeDatabases"].(bool); ok {
		cfg.IncludeDatabases = v
	}
	if v, ok := config["maxDepth"].(int); ok && v > 0 {
		cfg.MaxDepth = v
	}
	if v, ok := config["maxDepth"].(float64); ok && v > 0 {
		cfg.MaxDepth = int(v)
	}

	cfg.BlockTypeFilter = parseBlockTypeFilter(config["blockTypeFilter"])

	// OAuth2: Notion public integrations use OAuth2 tokens that expire.
	// Internal integration tokens do not expire.
	if clientID, ok := config["oauth2_client_id"].(string); ok && clientID != "" {
		clientSecret, _ := config["oauth2_client_secret"].(string)
		cfg.OAuth2 = &NotionOAuth2Config{
			ClientID:     clientID,
			ClientSecret: clientSecret,
		}
	}
	if rt, ok := config["refresh_token"].(string); ok && rt != "" {
		cfg.RefreshToken = rt
	}
	if expiresAt, ok := config["token_expires_at"].(string); ok && expiresAt != "" {
		if parsed, parseErr := time.Parse(time.RFC3339, expiresAt); parseErr == nil {
			cfg.TokenExpiresAt = parsed
		}
	}

	c.config = cfg
	c.configured = true

	// Set up OAuth2Client for token refresh when credentials are complete.
	if cfg.OAuth2 != nil && cfg.RefreshToken != "" {
		exchanger, _ := config["token_exchanger"].(NotionTokenExchanger)
		if exchanger == nil {
			exchanger = &notionHTTPTokenExchanger{
				clientID:     cfg.OAuth2.ClientID,
				clientSecret: cfg.OAuth2.ClientSecret,
				httpClient:   c.httpClient,
			}
		}
		c.oauth2Client = NewNotionOAuth2Client(cfg.OAuth2.ClientID, cfg.OAuth2.ClientSecret, exchanger)
	}

	if ts, ok := config["token_store"].(TokenStore); ok {
		c.tokenStore = ts
	}

	return nil
}

// FetchAll streams all accessible pages and database entries.
func (c *NotionConnector) FetchAll(ctx context.Context) (<-chan ConnectorDocument, <-chan error) {
	docs := make(chan ConnectorDocument, 32)
	errs := make(chan error, 8)

	go func() {
		defer close(docs)
		defer close(errs)

		if !c.configured {
			errs <- fmt.Errorf("connector/notion: not configured, call Configure first")
			return
		}

		if err := c.ensureValidToken(ctx); err != nil {
			errs <- err
			return
		}

		if err := c.syncPages(ctx, "", docs, errs); err != nil {
			errs <- err
		}
	}()

	return docs, errs
}

// FetchSince streams documents modified since the cursor timestamp.
func (c *NotionConnector) FetchSince(ctx context.Context, cursor SyncCursor) (<-chan ConnectorDocument, <-chan error) {
	docs := make(chan ConnectorDocument, 32)
	errs := make(chan error, 8)

	go func() {
		defer close(docs)
		defer close(errs)

		if !c.configured {
			errs <- fmt.Errorf("connector/notion: not configured, call Configure first")
			return
		}

		if err := c.ensureValidToken(ctx); err != nil {
			errs <- err
			return
		}

		if err := c.syncPages(ctx, cursor.Value, docs, errs); err != nil {
			errs <- err
		}
	}()

	return docs, errs
}

// Start begins a continuous sync loop. Blocks until Stop is called
// or the context is cancelled.
func (c *NotionConnector) Start(ctx context.Context) error {
	if !c.configured {
		return fmt.Errorf("connector/notion: not configured, call Configure first")
	}

	interval := c.deps.PollInterval
	if interval == 0 {
		interval = notionDefaultPollInterval
	}

	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-c.stopCh:
			return nil
		case <-ticker.C:
			docsCh, errsCh := c.FetchAll(ctx)
			// drain channels
			for range docsCh {
			}
			for err := range errsCh {
				if err != nil {
					return err
				}
			}
		}
	}
}

// Stop signals the continuous sync loop to stop.
func (c *NotionConnector) Stop() error {
	c.stopOnce.Do(func() {
		close(c.stopCh)
	})
	return nil
}

// ensureValidToken checks whether the current OAuth2 token is expired
// (or within the 5-minute buffer) and refreshes it if an OAuth2Client
// is configured. For internal integration tokens (which do not expire),
// this is a no-op. Thread-safe: concurrent calls are deduplicated by
// the NotionOAuth2Client's pendingRefresh mechanism.
func (c *NotionConnector) ensureValidToken(ctx context.Context) error {
	c.mu.Lock()
	client := c.oauth2Client
	refreshToken := c.config.RefreshToken
	expiresAt := c.config.TokenExpiresAt
	c.mu.Unlock()

	// No OAuth2 client means internal integration token -- never expires.
	if client == nil {
		return nil
	}

	// Check if token is still valid (with 5-minute buffer).
	if time.Now().Add(tokenExpiryBuffer).Before(expiresAt) {
		return nil
	}

	// Token expired or within buffer -- refresh.
	refreshed, err := client.RefreshToken(ctx, refreshToken)
	if err != nil {
		c.mu.Lock()
		c.errorCount++
		c.mu.Unlock()
		return fmt.Errorf("connector/notion: token refresh failed: %w", err)
	}

	c.mu.Lock()
	c.config.APIToken = refreshed.AccessToken
	c.config.TokenExpiresAt = refreshed.ExpiresAt
	if refreshed.RefreshToken != "" {
		c.config.RefreshToken = refreshed.RefreshToken
	}
	c.mu.Unlock()

	// Persist refreshed token if a token store is available.
	if c.tokenStore != nil {
		tok := OAuth2Token{
			AccessToken:  refreshed.AccessToken,
			RefreshToken: refreshed.RefreshToken,
			ExpiresAt:    refreshed.ExpiresAt,
			TokenType:    "Bearer",
		}
		if storeErr := c.tokenStore.Save(ctx, "notion", c.deps.BrainID, tok); storeErr != nil {
			// Log-and-continue: failure to persist is not fatal.
			_ = storeErr
		}
	}

	return nil
}

// Health returns the current health status of the connector.
func (c *NotionConnector) Health() HealthStatus {
	c.mu.Lock()
	defer c.mu.Unlock()

	status := StatusConnected
	var message string

	if !c.configured {
		status = StatusDisconnected
		message = "connector not configured"
	} else if c.errorCount > 0 {
		status = StatusDegraded
		message = fmt.Sprintf("%d errors since last successful sync", c.errorCount)
	}

	return HealthStatus{
		Status:             status,
		LastSyncAt:         c.lastSyncAt,
		ErrorCount:         c.errorCount,
		RateLimitRemaining: -1,
		Message:            message,
	}
}

// syncPages fetches pages and databases, optionally filtering by
// sinceISO for incremental sync.
func (c *NotionConnector) syncPages(ctx context.Context, sinceISO string, docs chan<- ConnectorDocument, errs chan<- error) error {
	visited := make(map[string]struct{})

	// Fetch specific root pages if configured.
	for _, pageID := range c.config.RootPageIDs {
		if err := c.fetchPageTree(ctx, pageID, 0, sinceISO, visited, docs, errs); err != nil {
			return err
		}
	}

	// Fetch specific databases if configured.
	if c.config.IncludeDatabases {
		for _, dbID := range c.config.DatabaseIDs {
			if err := c.fetchDatabaseEntries(ctx, dbID, sinceISO, visited, docs, errs); err != nil {
				return err
			}
		}
	}

	// If no specific pages or databases, search the entire workspace.
	if len(c.config.RootPageIDs) == 0 && len(c.config.DatabaseIDs) == 0 {
		if err := c.searchWorkspace(ctx, sinceISO, visited, docs, errs); err != nil {
			return err
		}
	}

	return nil
}

// notionSearchRequestBody is the typed request body for the Notion
// search API endpoint.
type notionSearchRequestBody struct {
	PageSize    int               `json:"page_size"`
	Sort        *notionSearchSort `json:"sort,omitempty"`
	StartCursor string            `json:"start_cursor,omitempty"`
}

// notionSearchSort represents the sort parameter for search requests.
type notionSearchSort struct {
	Direction string `json:"direction"`
	Timestamp string `json:"timestamp"`
}

// notionDatabaseQueryBody is the typed request body for the Notion
// database query API endpoint.
type notionDatabaseQueryBody struct {
	PageSize    int                       `json:"page_size"`
	StartCursor string                    `json:"start_cursor,omitempty"`
	Filter      *notionDatabaseTimeFilter `json:"filter,omitempty"`
}

// notionDatabaseTimeFilter represents a last_edited_time filter.
type notionDatabaseTimeFilter struct {
	Timestamp      string                   `json:"timestamp"`
	LastEditedTime notionTimestampCondition  `json:"last_edited_time"`
}

// notionTimestampCondition holds the condition for timestamp filters.
type notionTimestampCondition struct {
	OnOrAfter string `json:"on_or_after,omitempty"`
}

// searchWorkspace uses the Notion search API to discover all
// accessible pages and databases.
func (c *NotionConnector) searchWorkspace(ctx context.Context, sinceISO string, visited map[string]struct{}, docs chan<- ConnectorDocument, errs chan<- error) error {
	var cursor string

	for {
		body := notionSearchRequestBody{
			PageSize: notionDefaultPageSize,
		}
		if sinceISO != "" {
			body.Sort = &notionSearchSort{
				Direction: "descending",
				Timestamp: "last_edited_time",
			}
		}
		if cursor != "" {
			body.StartCursor = cursor
		}

		respBody, err := c.doNotionRequest(ctx, http.MethodPost, "/search", body)
		if err != nil {
			return fmt.Errorf("connector/notion: search: %w", err)
		}

		var searchResp notionSearchResponse
		if err := json.Unmarshal(respBody, &searchResp); err != nil {
			return fmt.Errorf("connector/notion: parsing search response: %w", err)
		}

		for _, raw := range searchResp.Results {
			var obj struct {
				ID             string `json:"id"`
				Object         string `json:"object"`
				LastEditedTime string `json:"last_edited_time"`
			}
			if err := json.Unmarshal(raw, &obj); err != nil {
				errs <- fmt.Errorf("connector/notion: parsing search result: %w", err)
				continue
			}

			// For incremental sync, skip items older than cursor.
			if sinceISO != "" {
				editedTime, parseErr := time.Parse(time.RFC3339, obj.LastEditedTime)
				if parseErr == nil {
					cursorTime, cursorErr := time.Parse(time.RFC3339, sinceISO)
					if cursorErr == nil && editedTime.Before(cursorTime) {
						// Results sorted descending; all further results are older.
						return nil
					}
				}
			}

			switch obj.Object {
			case "page":
				if err := c.fetchPageTree(ctx, obj.ID, 0, sinceISO, visited, docs, errs); err != nil {
					return err
				}
			case "database":
				if c.config.IncludeDatabases {
					if err := c.fetchDatabaseEntries(ctx, obj.ID, sinceISO, visited, docs, errs); err != nil {
						return err
					}
				}
			}
		}

		if !searchResp.HasMore || searchResp.NextCursor == "" {
			break
		}
		cursor = searchResp.NextCursor
	}

	return nil
}

// fetchPageTree recursively fetches a page and its children up to
// maxDepth. Uses cycle detection via the visited set.
func (c *NotionConnector) fetchPageTree(ctx context.Context, pageID string, depth int, sinceISO string, visited map[string]struct{}, docs chan<- ConnectorDocument, errs chan<- error) error {
	if depth > c.config.MaxDepth {
		return nil
	}

	// Cycle detection.
	if _, seen := visited[pageID]; seen {
		return nil
	}
	visited[pageID] = struct{}{}

	// Fetch page metadata.
	respBody, err := c.doNotionRequest(ctx, http.MethodGet, "/pages/"+pageID, nil)
	if err != nil {
		errs <- fmt.Errorf("connector/notion: fetching page %s: %w", pageID, err)
		return nil
	}

	page, err := parsePageResponse(respBody)
	if err != nil {
		errs <- fmt.Errorf("connector/notion: parsing page %s: %w", pageID, err)
		return nil
	}

	// For incremental sync, check last_edited_time.
	if sinceISO != "" {
		cursorTime, parseErr := time.Parse(time.RFC3339, sinceISO)
		if parseErr == nil && page.LastEditedTime.Before(cursorTime) {
			return nil
		}
	}

	// Fetch page content: try Markdown API first, fall back to blocks.
	content, err := c.fetchPageContent(ctx, pageID)
	if err != nil {
		errs <- fmt.Errorf("connector/notion: fetching content for page %s: %w", pageID, err)
		return nil
	}

	doc := ConnectorDocument{
		ExternalID: pageID,
		Content:    []byte(content),
		MIME:       "text/markdown",
		Title:      page.Title,
		URL:        page.URL,
		Metadata: map[string]string{
			"source":           "notion",
			"type":             "page",
			"last_edited_time": page.LastEditedTime.Format(time.RFC3339),
		},
		ModifiedAt: page.LastEditedTime,
	}

	select {
	case <-ctx.Done():
		return ctx.Err()
	case docs <- doc:
	}

	// Recursively fetch child pages.
	if c.config.IncludeChildPages {
		childIDs, fetchErr := c.fetchChildPageIDs(ctx, pageID)
		if fetchErr != nil {
			errs <- fmt.Errorf("connector/notion: fetching children of %s: %w", pageID, fetchErr)
			return nil
		}
		for _, childID := range childIDs {
			if err := c.fetchPageTree(ctx, childID, depth+1, sinceISO, visited, docs, errs); err != nil {
				return err
			}
		}
	}

	return nil
}

// fetchPageContent retrieves a page's content. Tries the Markdown
// API first, falls back to block-by-block reconstruction.
func (c *NotionConnector) fetchPageContent(ctx context.Context, pageID string) (string, error) {
	// Try Markdown API.
	content, err := c.fetchMarkdownAPI(ctx, pageID)
	if err == nil {
		return content, nil
	}

	// Fall back to block retrieval.
	return c.fetchBlocksAsMarkdown(ctx, pageID, 0)
}

// fetchMarkdownAPI attempts to fetch page content via the Notion
// Markdown export API (February 2026).
func (c *NotionConnector) fetchMarkdownAPI(ctx context.Context, pageID string) (string, error) {
	respBody, err := c.doNotionRequest(ctx, http.MethodGet, "/pages/"+pageID+"/markdown", nil)
	if err != nil {
		return "", err
	}

	var mdResp struct {
		Markdown string `json:"markdown"`
	}
	if err := json.Unmarshal(respBody, &mdResp); err != nil {
		return "", fmt.Errorf("connector/notion: parsing markdown response: %w", err)
	}

	return mdResp.Markdown, nil
}

// fetchBlockChildren paginates through all block children of the
// given block ID. This is the shared pagination helper used by both
// fetchBlocksAsMarkdown and fetchChildPageIDs to avoid duplication.
func (c *NotionConnector) fetchBlockChildren(ctx context.Context, blockID string) ([]notionBlock, error) {
	var allBlocks []notionBlock
	var cursor string

	for {
		endpoint := fmt.Sprintf("/blocks/%s/children?page_size=%d", blockID, notionDefaultPageSize)
		if cursor != "" {
			endpoint += "&start_cursor=" + cursor
		}

		respBody, err := c.doNotionRequest(ctx, http.MethodGet, endpoint, nil)
		if err != nil {
			return nil, fmt.Errorf("connector/notion: fetching blocks for %s: %w", blockID, err)
		}

		var blockResp notionBlockChildrenResponse
		if err := json.Unmarshal(respBody, &blockResp); err != nil {
			return nil, fmt.Errorf("connector/notion: parsing blocks: %w", err)
		}

		allBlocks = append(allBlocks, blockResp.Results...)

		if !blockResp.HasMore || blockResp.NextCursor == "" {
			break
		}
		cursor = blockResp.NextCursor
	}

	return allBlocks, nil
}

// fetchBlocksAsMarkdown retrieves all block children and converts
// them to markdown recursively.
func (c *NotionConnector) fetchBlocksAsMarkdown(ctx context.Context, blockID string, depth int) (string, error) {
	if depth > c.config.MaxDepth {
		return "", nil
	}

	allBlocks, err := c.fetchBlockChildren(ctx, blockID)
	if err != nil {
		return "", err
	}

	var builder strings.Builder
	for _, block := range allBlocks {
		md := c.blockToMarkdown(block)
		if md != "" {
			if builder.Len() > 0 {
				builder.WriteString("\n")
			}
			builder.WriteString(md)
		}

		// Recursively fetch children of blocks that have them.
		if block.HasChildren {
			childContent, childErr := c.fetchBlocksAsMarkdown(ctx, block.ID, depth+1)
			if childErr != nil {
				continue
			}
			if childContent != "" {
				// Indent child content for nested blocks.
				lines := strings.Split(childContent, "\n")
				for _, line := range lines {
					builder.WriteString("\n")
					if isListBlock(block.Type) {
						builder.WriteString("  ")
					}
					builder.WriteString(line)
				}
			}
		}
	}

	return builder.String(), nil
}

// fetchChildPageIDs returns the IDs of child pages within a block.
// Uses the shared fetchBlockChildren helper to avoid duplicating
// the pagination loop.
func (c *NotionConnector) fetchChildPageIDs(ctx context.Context, blockID string) ([]string, error) {
	allBlocks, err := c.fetchBlockChildren(ctx, blockID)
	if err != nil {
		return nil, err
	}

	var childIDs []string
	for _, block := range allBlocks {
		if block.Type == "child_page" || block.Type == "child_database" {
			childIDs = append(childIDs, block.ID)
		}
	}
	return childIDs, nil
}

// fetchDatabaseEntries queries a database and yields each entry as
// a document with property listings.
func (c *NotionConnector) fetchDatabaseEntries(ctx context.Context, dbID string, sinceISO string, visited map[string]struct{}, docs chan<- ConnectorDocument, errs chan<- error) error {
	var cursor string

	for {
		body := notionDatabaseQueryBody{
			PageSize: notionDefaultPageSize,
		}
		if cursor != "" {
			body.StartCursor = cursor
		}

		// For incremental sync, filter by last_edited_time.
		if sinceISO != "" {
			body.Filter = &notionDatabaseTimeFilter{
				Timestamp:      "last_edited_time",
				LastEditedTime: notionTimestampCondition{OnOrAfter: sinceISO},
			}
		}

		respBody, err := c.doNotionRequest(ctx, http.MethodPost, "/databases/"+dbID+"/query", body)
		if err != nil {
			return fmt.Errorf("connector/notion: querying database %s: %w", dbID, err)
		}

		var queryResp notionDatabaseQueryResponse
		if err := json.Unmarshal(respBody, &queryResp); err != nil {
			return fmt.Errorf("connector/notion: parsing database query: %w", err)
		}

		for _, raw := range queryResp.Results {
			entry, parseErr := parseDatabaseEntry(raw)
			if parseErr != nil {
				errs <- fmt.Errorf("connector/notion: parsing database entry: %w", parseErr)
				continue
			}

			// Cycle detection for database entries that are also pages.
			if _, seen := visited[entry.pageID]; seen {
				continue
			}
			visited[entry.pageID] = struct{}{}

			// Optionally fetch page content for the entry.
			var pageContent string
			if entry.pageID != "" {
				content, fetchErr := c.fetchPageContent(ctx, entry.pageID)
				if fetchErr == nil && content != "" {
					pageContent = "\n\n" + content
				}
			}

			fullContent := entry.propertiesMarkdown + pageContent

			doc := ConnectorDocument{
				ExternalID: entry.pageID,
				Content:    []byte(fullContent),
				MIME:       "text/markdown",
				Title:      entry.title,
				URL:        entry.url,
				Metadata: map[string]string{
					"source":           "notion",
					"type":             "database_entry",
					"database_id":      dbID,
					"last_edited_time": entry.lastEditedTime.Format(time.RFC3339),
				},
				ModifiedAt: entry.lastEditedTime,
			}

			select {
			case <-ctx.Done():
				return ctx.Err()
			case docs <- doc:
			}
		}

		if !queryResp.HasMore || queryResp.NextCursor == "" {
			break
		}
		cursor = queryResp.NextCursor
	}

	return nil
}

