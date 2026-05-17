// SPDX-License-Identifier: Apache-2.0

package connector

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sort"
	"strings"
	"sync"
	"time"
)

// notionAPIVersion is the Notion API version header value.
const notionAPIVersion = "2022-06-28"

// notionBaseURL is the root endpoint for the Notion API.
const notionBaseURL = "https://api.notion.com/v1"

// notionDefaultMaxDepth caps recursive page traversal.
const notionDefaultMaxDepth = 10

// notionDefaultPollInterval is the default continuous sync interval.
const notionDefaultPollInterval = 15 * time.Minute

// notionDefaultTimeout is the per-request HTTP timeout for Notion
// API calls.
const notionDefaultTimeout = 30 * time.Second

// notionDefaultPageSize is the default page size for paginated
// Notion API requests.
const notionDefaultPageSize = 100

// NotionConnectorConfig holds Notion-specific configuration.
type NotionConnectorConfig struct {
	APIToken          string
	RootPageIDs       []string
	DatabaseIDs       []string
	IncludeChildPages bool // default: true
	IncludeDatabases  bool // default: true
	MaxDepth          int  // default: 10
}

// NotionPage represents a Notion page object from the API.
type NotionPage struct {
	ID             string
	Title          string
	URL            string
	LastEditedTime time.Time
	ParentID       string
}

// notionSearchResponse models the Notion search API response.
type notionSearchResponse struct {
	Results    []json.RawMessage `json:"results"`
	NextCursor string            `json:"next_cursor"`
	HasMore    bool              `json:"has_more"`
}

// notionDatabaseQueryResponse models the database query response.
type notionDatabaseQueryResponse struct {
	Results    []json.RawMessage `json:"results"`
	NextCursor string            `json:"next_cursor"`
	HasMore    bool              `json:"has_more"`
}

// notionBlockChildrenResponse models the block children response.
type notionBlockChildrenResponse struct {
	Results    []notionBlock `json:"results"`
	NextCursor string       `json:"next_cursor"`
	HasMore    bool         `json:"has_more"`
}

// notionBlock represents a single Notion block.
type notionBlock struct {
	ID          string          `json:"id"`
	Type        string          `json:"type"`
	HasChildren bool            `json:"has_children"`
	RawData     json.RawMessage `json:"-"`
}

// UnmarshalJSON implements custom unmarshalling for notionBlock to
// capture the full raw data alongside parsed fields.
func (b *notionBlock) UnmarshalJSON(data []byte) error {
	type alias notionBlock
	var a alias
	if err := json.Unmarshal(data, &a); err != nil {
		return err
	}
	a.RawData = data
	*b = notionBlock(a)
	return nil
}

// notionRichText represents a Notion rich text object.
type notionRichText struct {
	Type      string                 `json:"type"`
	PlainText string                 `json:"plain_text"`
	Href      string                 `json:"href"`
	Annotations notionAnnotations    `json:"annotations"`
}

// notionAnnotations contains formatting flags for rich text.
type notionAnnotations struct {
	Bold          bool   `json:"bold"`
	Italic        bool   `json:"italic"`
	Strikethrough bool   `json:"strikethrough"`
	Underline     bool   `json:"underline"`
	Code          bool   `json:"code"`
	Colour        string `json:"color"`
}

// NotionHTTPClient abstracts HTTP calls so tests can inject a mock.
type NotionHTTPClient interface {
	Do(req *http.Request) (*http.Response, error)
}

// NotionConnector implements [Connector] for Notion workspaces.
type NotionConnector struct {
	deps        ConnectorConfig
	config      NotionConnectorConfig
	httpClient  NotionHTTPClient
	stopCh      chan struct{}
	stopOnce    sync.Once
	configured  bool
}

// NewNotionConnector creates a Notion connector with the given base
// dependencies and an HTTP client for API calls.
func NewNotionConnector(deps ConnectorConfig, httpClient NotionHTTPClient) *NotionConnector {
	if httpClient == nil {
		httpClient = &http.Client{Timeout: notionDefaultTimeout}
	}
	return &NotionConnector{
		deps:       deps,
		httpClient: httpClient,
		stopCh:     make(chan struct{}),
	}
}

// Name returns "notion".
func (c *NotionConnector) Name() string { return "notion" }

// Configure validates and stores Notion-specific configuration.
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

	c.config = cfg
	c.configured = true
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

// searchWorkspace uses the Notion search API to discover all
// accessible pages and databases.
func (c *NotionConnector) searchWorkspace(ctx context.Context, sinceISO string, visited map[string]struct{}, docs chan<- ConnectorDocument, errs chan<- error) error {
	var cursor string

	for {
		if err := c.acquireRateLimit(ctx); err != nil {
			return err
		}

		body := map[string]any{
			"page_size": notionDefaultPageSize,
		}
		if sinceISO != "" {
			body["sort"] = map[string]string{
				"direction": "descending",
				"timestamp": "last_edited_time",
			}
		}
		if cursor != "" {
			body["start_cursor"] = cursor
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
	if err := c.acquireRateLimit(ctx); err != nil {
		return err
	}

	respBody, err := c.doNotionRequest(ctx, http.MethodGet, "/pages/"+pageID, nil)
	if err != nil {
		errs <- fmt.Errorf("connector/notion: fetching page %s: %w", pageID, err)
		return nil
	}

	page, err := c.parsePageResponse(respBody)
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
	if err := c.acquireRateLimit(ctx); err != nil {
		return "", err
	}

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

// fetchBlocksAsMarkdown retrieves all block children and converts
// them to markdown recursively.
func (c *NotionConnector) fetchBlocksAsMarkdown(ctx context.Context, blockID string, depth int) (string, error) {
	if depth > c.config.MaxDepth {
		return "", nil
	}

	var allBlocks []notionBlock
	var cursor string

	for {
		if err := c.acquireRateLimit(ctx); err != nil {
			return "", err
		}

		endpoint := fmt.Sprintf("/blocks/%s/children?page_size=%d", blockID, notionDefaultPageSize)
		if cursor != "" {
			endpoint += "&start_cursor=" + cursor
		}

		respBody, err := c.doNotionRequest(ctx, http.MethodGet, endpoint, nil)
		if err != nil {
			return "", fmt.Errorf("connector/notion: fetching blocks for %s: %w", blockID, err)
		}

		var blockResp notionBlockChildrenResponse
		if err := json.Unmarshal(respBody, &blockResp); err != nil {
			return "", fmt.Errorf("connector/notion: parsing blocks: %w", err)
		}

		allBlocks = append(allBlocks, blockResp.Results...)

		if !blockResp.HasMore || blockResp.NextCursor == "" {
			break
		}
		cursor = blockResp.NextCursor
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
			childContent, err := c.fetchBlocksAsMarkdown(ctx, block.ID, depth+1)
			if err != nil {
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

// blockToMarkdown converts a single Notion block to its markdown
// representation.
func (c *NotionConnector) blockToMarkdown(block notionBlock) string {
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(block.RawData, &raw); err != nil {
		return ""
	}

	blockData, ok := raw[block.Type]
	if !ok {
		return ""
	}

	var typed struct {
		RichText []notionRichText `json:"rich_text"`
		Caption  []notionRichText `json:"caption"`
		Language string           `json:"language"`
		URL      string           `json:"url"`
		Checked  bool             `json:"checked"`
		Expression string         `json:"expression"`
	}
	if err := json.Unmarshal(blockData, &typed); err != nil {
		return ""
	}

	text := renderRichText(typed.RichText)

	blockRenderers := map[string]func() string{
		"paragraph":          func() string { return text + "\n" },
		"heading_1":          func() string { return "# " + text + "\n" },
		"heading_2":          func() string { return "## " + text + "\n" },
		"heading_3":          func() string { return "### " + text + "\n" },
		"bulleted_list_item": func() string { return "- " + text },
		"numbered_list_item": func() string { return "1. " + text },
		"to_do": func() string {
			marker := "[ ]"
			if typed.Checked {
				marker = "[x]"
			}
			return "- " + marker + " " + text
		},
		"toggle":    func() string { return "- " + text },
		"quote":     func() string { return "> " + text + "\n" },
		"callout":   func() string { return "> " + text + "\n" },
		"divider":   func() string { return "---\n" },
		"code": func() string {
			lang := typed.Language
			return "```" + lang + "\n" + text + "\n```\n"
		},
		"equation": func() string {
			return "$$\n" + typed.Expression + "\n$$\n"
		},
		"image": func() string {
			caption := renderRichText(typed.Caption)
			url := extractFileURL(blockData)
			if caption != "" {
				return fmt.Sprintf("![%s](%s)\n", caption, url)
			}
			return fmt.Sprintf("![](%s)\n", url)
		},
		"file": func() string {
			url := extractFileURL(blockData)
			caption := renderRichText(typed.Caption)
			if caption == "" {
				caption = "file"
			}
			return fmt.Sprintf("[%s](%s)\n", caption, url)
		},
		"bookmark": func() string {
			caption := renderRichText(typed.Caption)
			if typed.URL != "" {
				if caption != "" {
					return fmt.Sprintf("[%s](%s)\n", caption, typed.URL)
				}
				return typed.URL + "\n"
			}
			return ""
		},
		"embed": func() string {
			if typed.URL != "" {
				return typed.URL + "\n"
			}
			return ""
		},
		"table_of_contents": func() string { return "" },
		"breadcrumb":        func() string { return "" },
		"column_list":       func() string { return "" },
		"column":            func() string { return "" },
		"child_page":        func() string { return "" },
		"child_database":    func() string { return "" },
		"synced_block":      func() string { return "" },
		"link_preview": func() string {
			if typed.URL != "" {
				return typed.URL + "\n"
			}
			return ""
		},
		"table":     func() string { return "" },
		"table_row": func() string { return renderTableRow(blockData) },
	}

	renderer, found := blockRenderers[block.Type]
	if !found {
		return text
	}
	return renderer()
}

// fetchDatabaseEntries queries a database and yields each entry as
// a document with property listings.
func (c *NotionConnector) fetchDatabaseEntries(ctx context.Context, dbID string, sinceISO string, visited map[string]struct{}, docs chan<- ConnectorDocument, errs chan<- error) error {
	var cursor string

	for {
		if err := c.acquireRateLimit(ctx); err != nil {
			return err
		}

		body := map[string]any{
			"page_size": notionDefaultPageSize,
		}
		if cursor != "" {
			body["start_cursor"] = cursor
		}

		// For incremental sync, filter by last_edited_time.
		if sinceISO != "" {
			body["filter"] = map[string]any{
				"timestamp":        "last_edited_time",
				"last_edited_time": map[string]string{"on_or_after": sinceISO},
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
			entry, parseErr := c.parseDatabaseEntry(raw)
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

// databaseEntryParsed holds parsed fields from a database entry.
type databaseEntryParsed struct {
	pageID             string
	title              string
	url                string
	lastEditedTime     time.Time
	propertiesMarkdown string
}

// parseDatabaseEntry extracts structured data from a raw database
// entry JSON.
func (c *NotionConnector) parseDatabaseEntry(raw json.RawMessage) (databaseEntryParsed, error) {
	var entry struct {
		ID             string                     `json:"id"`
		URL            string                     `json:"url"`
		LastEditedTime string                     `json:"last_edited_time"`
		Properties     map[string]json.RawMessage `json:"properties"`
	}
	if err := json.Unmarshal(raw, &entry); err != nil {
		return databaseEntryParsed{}, err
	}

	editedTime, _ := time.Parse(time.RFC3339, entry.LastEditedTime)

	title := ""
	var propLines []string

	// Sort property keys for deterministic output.
	keys := make([]string, 0, len(entry.Properties))
	for k := range entry.Properties {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	for _, key := range keys {
		propRaw := entry.Properties[key]
		val := extractPropertyValue(propRaw)

		// Detect title property.
		var propType struct {
			Type string `json:"type"`
		}
		if err := json.Unmarshal(propRaw, &propType); err == nil && propType.Type == "title" {
			title = val
			continue
		}

		if val != "" {
			propLines = append(propLines, fmt.Sprintf("- **%s**: %s", key, val))
		}
	}

	var builder strings.Builder
	if title != "" {
		builder.WriteString("## " + title + "\n\n")
	}
	if len(propLines) > 0 {
		builder.WriteString(strings.Join(propLines, "\n"))
		builder.WriteString("\n")
	}

	return databaseEntryParsed{
		pageID:             entry.ID,
		title:              title,
		url:                entry.URL,
		lastEditedTime:     editedTime,
		propertiesMarkdown: builder.String(),
	}, nil
}

// fetchChildPageIDs returns the IDs of child pages within a block.
func (c *NotionConnector) fetchChildPageIDs(ctx context.Context, blockID string) ([]string, error) {
	var childIDs []string
	var cursor string

	for {
		if err := c.acquireRateLimit(ctx); err != nil {
			return nil, err
		}

		endpoint := fmt.Sprintf("/blocks/%s/children?page_size=%d", blockID, notionDefaultPageSize)
		if cursor != "" {
			endpoint += "&start_cursor=" + cursor
		}

		respBody, err := c.doNotionRequest(ctx, http.MethodGet, endpoint, nil)
		if err != nil {
			return nil, err
		}

		var blockResp notionBlockChildrenResponse
		if err := json.Unmarshal(respBody, &blockResp); err != nil {
			return nil, fmt.Errorf("connector/notion: parsing block children: %w", err)
		}

		for _, block := range blockResp.Results {
			if block.Type == "child_page" || block.Type == "child_database" {
				childIDs = append(childIDs, block.ID)
			}
		}

		if !blockResp.HasMore || blockResp.NextCursor == "" {
			break
		}
		cursor = blockResp.NextCursor
	}

	return childIDs, nil
}

// parsePageResponse extracts page metadata from a Notion page API
// response.
func (c *NotionConnector) parsePageResponse(data []byte) (NotionPage, error) {
	var resp struct {
		ID             string                     `json:"id"`
		URL            string                     `json:"url"`
		LastEditedTime string                     `json:"last_edited_time"`
		Properties     map[string]json.RawMessage `json:"properties"`
		Parent         struct {
			Type   string `json:"type"`
			PageID string `json:"page_id"`
		} `json:"parent"`
	}
	if err := json.Unmarshal(data, &resp); err != nil {
		return NotionPage{}, fmt.Errorf("parsing page: %w", err)
	}

	editedTime, _ := time.Parse(time.RFC3339, resp.LastEditedTime)

	// Extract title from properties.
	title := extractTitleFromProperties(resp.Properties)

	return NotionPage{
		ID:             resp.ID,
		Title:          title,
		URL:            resp.URL,
		LastEditedTime: editedTime,
		ParentID:       resp.Parent.PageID,
	}, nil
}

// doNotionRequest performs a single Notion API request with proper
// headers and timeout. Returns the response body or an error.
func (c *NotionConnector) doNotionRequest(ctx context.Context, method, path string, body any) ([]byte, error) {
	var bodyReader io.Reader
	if body != nil {
		data, err := json.Marshal(body)
		if err != nil {
			return nil, fmt.Errorf("connector/notion: marshalling request body: %w", err)
		}
		bodyReader = strings.NewReader(string(data))
	}

	url := notionBaseURL + path
	// Override base URL if set via context (for tests).
	if baseURL, ok := ctx.Value(notionBaseURLKey{}).(string); ok {
		url = baseURL + path
	}

	reqCtx, cancel := context.WithTimeout(ctx, notionDefaultTimeout)
	defer cancel()

	req, err := http.NewRequestWithContext(reqCtx, method, url, bodyReader)
	if err != nil {
		return nil, fmt.Errorf("connector/notion: creating request: %w", err)
	}

	req.Header.Set("Authorization", "Bearer "+c.config.APIToken)
	req.Header.Set("Notion-Version", notionAPIVersion)
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("connector/notion: request failed: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(io.LimitReader(resp.Body, 10*1024*1024))
	if err != nil {
		return nil, fmt.Errorf("connector/notion: reading response: %w", err)
	}

	if resp.StatusCode == http.StatusTooManyRequests {
		return nil, fmt.Errorf("connector/notion: rate limited (429)")
	}

	if resp.StatusCode == http.StatusNotFound {
		return nil, fmt.Errorf("connector/notion: not found (404)")
	}

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf("connector/notion: HTTP %d: %s", resp.StatusCode, string(respBody))
	}

	return respBody, nil
}

// acquireRateLimit acquires a single rate limit token, or returns
// immediately if no rate limiter is configured.
func (c *NotionConnector) acquireRateLimit(ctx context.Context) error {
	if c.deps.RateLimiter == nil {
		return nil
	}
	return c.deps.RateLimiter.Acquire(ctx, 1)
}

// notionBaseURLKey is a context key for overriding the Notion API
// base URL in tests.
type notionBaseURLKey struct{}

// WithNotionBaseURL returns a context with an overridden Notion API
// base URL.
func WithNotionBaseURL(ctx context.Context, baseURL string) context.Context {
	return context.WithValue(ctx, notionBaseURLKey{}, baseURL)
}

// ---------- Helper functions ----------

// renderRichText converts a slice of Notion rich text objects to a
// markdown string with formatting annotations.
func renderRichText(texts []notionRichText) string {
	var builder strings.Builder
	for _, t := range texts {
		text := t.PlainText

		if t.Annotations.Code {
			text = "`" + text + "`"
		}
		if t.Annotations.Bold {
			text = "**" + text + "**"
		}
		if t.Annotations.Italic {
			text = "*" + text + "*"
		}
		if t.Annotations.Strikethrough {
			text = "~~" + text + "~~"
		}

		if t.Href != "" {
			text = "[" + text + "](" + t.Href + ")"
		}

		builder.WriteString(text)
	}
	return builder.String()
}

// extractFileURL pulls the URL from a file-type block's JSON data.
func extractFileURL(data json.RawMessage) string {
	var fileBlock struct {
		External struct {
			URL string `json:"url"`
		} `json:"external"`
		File struct {
			URL string `json:"url"`
		} `json:"file"`
	}
	if err := json.Unmarshal(data, &fileBlock); err != nil {
		return ""
	}
	if fileBlock.File.URL != "" {
		return fileBlock.File.URL
	}
	return fileBlock.External.URL
}

// extractTitleFromProperties finds the title property in a page's
// properties map and returns its plain text value.
func extractTitleFromProperties(properties map[string]json.RawMessage) string {
	for _, propRaw := range properties {
		var prop struct {
			Type  string `json:"type"`
			Title []struct {
				PlainText string `json:"plain_text"`
			} `json:"title"`
		}
		if err := json.Unmarshal(propRaw, &prop); err != nil {
			continue
		}
		if prop.Type == "title" && len(prop.Title) > 0 {
			var parts []string
			for _, t := range prop.Title {
				parts = append(parts, t.PlainText)
			}
			return strings.Join(parts, "")
		}
	}
	return ""
}

// extractPropertyValue converts a Notion property value to its
// string representation.
func extractPropertyValue(raw json.RawMessage) string {
	var prop struct {
		Type           string           `json:"type"`
		Title          []notionRichText `json:"title"`
		RichText       []notionRichText `json:"rich_text"`
		Number         *float64         `json:"number"`
		Select         *struct {
			Name string `json:"name"`
		} `json:"select"`
		MultiSelect []struct {
			Name string `json:"name"`
		} `json:"multi_select"`
		Date *struct {
			Start string `json:"start"`
			End   string `json:"end"`
		} `json:"date"`
		Checkbox bool `json:"checkbox"`
		URL      string `json:"url"`
		Email    string `json:"email"`
		Phone    string `json:"phone_number"`
		Status   *struct {
			Name string `json:"name"`
		} `json:"status"`
		People []struct {
			Name string `json:"name"`
		} `json:"people"`
		Relation []struct {
			ID string `json:"id"`
		} `json:"relation"`
	}
	if err := json.Unmarshal(raw, &prop); err != nil {
		return ""
	}

	renderers := map[string]func() string{
		"title":    func() string { return renderPlainRichText(prop.Title) },
		"rich_text": func() string { return renderPlainRichText(prop.RichText) },
		"number": func() string {
			if prop.Number == nil {
				return ""
			}
			return fmt.Sprintf("%g", *prop.Number)
		},
		"select": func() string {
			if prop.Select == nil {
				return ""
			}
			return prop.Select.Name
		},
		"multi_select": func() string {
			names := make([]string, 0, len(prop.MultiSelect))
			for _, ms := range prop.MultiSelect {
				names = append(names, ms.Name)
			}
			return strings.Join(names, ", ")
		},
		"date": func() string {
			if prop.Date == nil {
				return ""
			}
			if prop.Date.End != "" {
				return prop.Date.Start + " to " + prop.Date.End
			}
			return prop.Date.Start
		},
		"checkbox": func() string {
			if prop.Checkbox {
				return "true"
			}
			return "false"
		},
		"url":          func() string { return prop.URL },
		"email":        func() string { return prop.Email },
		"phone_number": func() string { return prop.Phone },
		"status": func() string {
			if prop.Status == nil {
				return ""
			}
			return prop.Status.Name
		},
		"people": func() string {
			names := make([]string, 0, len(prop.People))
			for _, p := range prop.People {
				names = append(names, p.Name)
			}
			return strings.Join(names, ", ")
		},
		"relation": func() string {
			ids := make([]string, 0, len(prop.Relation))
			for _, r := range prop.Relation {
				ids = append(ids, r.ID)
			}
			return strings.Join(ids, ", ")
		},
	}

	renderer, found := renderers[prop.Type]
	if !found {
		return ""
	}
	return renderer()
}

// renderPlainRichText extracts plain text from a rich text array
// without formatting.
func renderPlainRichText(texts []notionRichText) string {
	var parts []string
	for _, t := range texts {
		parts = append(parts, t.PlainText)
	}
	return strings.Join(parts, "")
}

// isListBlock returns true if the block type is a list item.
func isListBlock(blockType string) bool {
	listTypes := map[string]struct{}{
		"bulleted_list_item": {},
		"numbered_list_item": {},
		"to_do":              {},
		"toggle":             {},
	}
	_, ok := listTypes[blockType]
	return ok
}

// renderTableRow converts a table_row block to a markdown table
// row.
func renderTableRow(data json.RawMessage) string {
	var row struct {
		Cells [][]notionRichText `json:"cells"`
	}
	if err := json.Unmarshal(data, &row); err != nil {
		return ""
	}

	cells := make([]string, 0, len(row.Cells))
	for _, cell := range row.Cells {
		cells = append(cells, renderRichText(cell))
	}

	return "| " + strings.Join(cells, " | ") + " |"
}
