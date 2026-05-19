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
	"sync"
	"time"
)

// maxAPIRetries is the maximum number of retry attempts for rate-limited
// (HTTP 429) API responses before returning an error.
const maxAPIRetries = 5

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

// SlackConnectorConfig holds the configuration for the Slack connector.
type SlackConnectorConfig struct {
	BotToken        string
	Channels        []string
	IncludeThreads  bool
	IncludeFiles    bool
	MaxFileSize     int64
	OldestTimestamp string
	PollInterval    time.Duration
	Logger          *slog.Logger
	HTTPClient      HTTPDoer
}

// defaultSlackConfig returns a SlackConnectorConfig with sensible defaults
// applied for unset fields.
func defaultSlackConfig() SlackConnectorConfig {
	return SlackConnectorConfig{
		IncludeThreads: true,
		IncludeFiles:   true,
		MaxFileSize:    50 * 1024 * 1024, // 50 MB
		PollInterval:   5 * time.Minute,
	}
}

// ---------------------------------------------------------------------------
// HTTP abstraction (for testability)
// ---------------------------------------------------------------------------

// HTTPDoer is the minimal interface for issuing HTTP requests. Both
// *http.Client and test doubles satisfy it.
type HTTPDoer interface {
	Do(req *http.Request) (*http.Response, error)
}

// ---------------------------------------------------------------------------
// Slack API response types
// ---------------------------------------------------------------------------

type slackResponse struct {
	OK               bool           `json:"ok"`
	Error            string         `json:"error"`
	Messages         []slackMessage `json:"messages"`
	ResponseMetadata slackRespMeta  `json:"response_metadata"`
}

type slackRespMeta struct {
	NextCursor string `json:"next_cursor"`
}

type slackMessage struct {
	Type       string      `json:"type"`
	User       string      `json:"user"`
	Text       string      `json:"text"`
	TS         string      `json:"ts"`
	ThreadTS   string      `json:"thread_ts"`
	ReplyCount int         `json:"reply_count"`
	Files      []slackFile `json:"files"`
}

type slackFile struct {
	ID                 string `json:"id"`
	Name               string `json:"name"`
	MIMEType           string `json:"mimetype"`
	Size               int64  `json:"size"`
	URLPrivateDownload string `json:"url_private_download"`
}

type slackUserResponse struct {
	OK   bool      `json:"ok"`
	User slackUser `json:"user"`
}

type slackUser struct {
	RealName string `json:"real_name"`
	Name     string `json:"name"`
}

// ---------------------------------------------------------------------------
// Slack connector
// ---------------------------------------------------------------------------

// SlackConnector implements the Connector interface for Slack workspace
// integration. It fetches channel messages, reconstructs threads,
// downloads file attachments, and converts Slack mrkdwn to standard
// markdown.
type SlackConnector struct {
	config       SlackConnectorConfig
	rateLimiter  *RateLimiter
	userCache    map[string]string
	userMu       sync.RWMutex
	stopCh       chan struct{}
	stopped      bool
	mu           sync.Mutex
	logger       *slog.Logger
	httpClient   HTTPDoer
	urlValidator func(string) error // SSRF validation; defaults to validateDownloadURL
}

// NewSlackConnector creates a new Slack connector with the given config.
// Missing config fields are filled with defaults.
func NewSlackConnector(cfg SlackConnectorConfig) *SlackConnector {
	defaults := defaultSlackConfig()

	if cfg.MaxFileSize <= 0 {
		cfg.MaxFileSize = defaults.MaxFileSize
	}
	if cfg.PollInterval <= 0 {
		cfg.PollInterval = defaults.PollInterval
	}
	if cfg.Logger == nil {
		cfg.Logger = slog.Default()
	}
	if cfg.HTTPClient == nil {
		cfg.HTTPClient = &http.Client{Timeout: 30 * time.Second}
	}

	return &SlackConnector{
		config: cfg,
		// Slack Tier 3: ~50 requests per minute = ~0.833 per second
		rateLimiter:  NewRateLimiter(50, 50.0/60.0),
		userCache:    make(map[string]string),
		stopCh:       make(chan struct{}),
		logger:       cfg.Logger,
		httpClient:   cfg.HTTPClient,
		urlValidator: validateDownloadURL,
	}
}

// Name returns "slack".
func (c *SlackConnector) Name() string { return "slack" }

// Configure validates and applies connector-specific configuration.
// Accepts map[string]any to match the P5-1 Connector interface.
func (c *SlackConnector) Configure(config map[string]any) error {
	token, _ := config["botToken"].(string)
	if token == "" {
		return fmt.Errorf("slack: botToken is required")
	}
	c.config.BotToken = token

	channelsRaw, _ := config["channels"].(string)
	if channelsRaw == "" {
		return fmt.Errorf("slack: at least one channel is required")
	}
	c.config.Channels = strings.Split(channelsRaw, ",")
	for i, ch := range c.config.Channels {
		c.config.Channels[i] = strings.TrimSpace(ch)
	}

	if v, ok := config["includeThreads"].(bool); ok {
		c.config.IncludeThreads = v
	} else if v, ok := config["includeThreads"].(string); ok {
		c.config.IncludeThreads = v != "false"
	}

	if v, ok := config["includeFiles"].(bool); ok {
		c.config.IncludeFiles = v
	} else if v, ok := config["includeFiles"].(string); ok {
		c.config.IncludeFiles = v != "false"
	}

	if v, ok := config["maxFileSize"].(string); ok {
		size, err := strconv.ParseInt(v, 10, 64)
		if err != nil {
			return fmt.Errorf("slack: invalid maxFileSize: %w", err)
		}
		c.config.MaxFileSize = size
	} else if v, ok := config["maxFileSize"].(float64); ok {
		c.config.MaxFileSize = int64(v)
	} else if v, ok := config["maxFileSize"].(int); ok {
		c.config.MaxFileSize = int64(v)
	}

	if v, ok := config["oldestTimestamp"].(string); ok {
		c.config.OldestTimestamp = v
	}

	return nil
}

// FetchAll performs a full sync of all configured channels.
func (c *SlackConnector) FetchAll(ctx context.Context) (<-chan ConnectorDocument, <-chan error) {
	return c.fetchMessages(ctx, "")
}

// FetchSince performs an incremental sync starting from the cursor value
// (a Slack message timestamp).
func (c *SlackConnector) FetchSince(ctx context.Context, cursor SyncCursor) (<-chan ConnectorDocument, <-chan error) {
	return c.fetchMessages(ctx, cursor.Value)
}

// Start begins a continuous sync loop that polls at the configured
// interval. Blocks until the context is cancelled or Stop is called.
// Returns the first fatal error encountered, or nil on clean shutdown.
func (c *SlackConnector) Start(ctx context.Context) error {
	var lastCursor string
	if c.config.OldestTimestamp != "" {
		lastCursor = c.config.OldestTimestamp
	}

	ticker := time.NewTicker(c.config.PollInterval)
	defer ticker.Stop()

	// Run once immediately, then on each tick.
	for {
		docs, errs := c.fetchMessages(ctx, lastCursor)
		var latestTS string

		for doc := range docs {
			ts, tsOK := doc.Metadata["ts"]
			if tsOK && ts > latestTS {
				latestTS = ts
			}
		}

		if err, ok := <-errs; ok && err != nil {
			return fmt.Errorf("slack: sync error: %w", err)
		}

		if latestTS != "" {
			lastCursor = latestTS
		}

		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-c.stopCh:
			return nil
		case <-ticker.C:
		}
	}
}

// Stop gracefully stops the continuous sync loop.
func (c *SlackConnector) Stop() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if !c.stopped {
		c.stopped = true
		close(c.stopCh)
	}
	return nil
}

// ---------------------------------------------------------------------------
// Internal: message fetching
// ---------------------------------------------------------------------------

func (c *SlackConnector) fetchMessages(ctx context.Context, oldest string) (<-chan ConnectorDocument, <-chan error) {
	docCh := make(chan ConnectorDocument, 100)
	errCh := make(chan error, 1)

	go func() {
		defer close(docCh)
		defer close(errCh)

		for _, channelID := range c.config.Channels {
			if err := c.fetchChannelMessages(ctx, channelID, oldest, docCh); err != nil {
				select {
				case errCh <- fmt.Errorf("slack: channel %s: %w", channelID, err):
				default:
				}
				return
			}
		}
	}()

	return docCh, errCh
}

func (c *SlackConnector) fetchChannelMessages(
	ctx context.Context,
	channelID string,
	oldest string,
	docCh chan<- ConnectorDocument,
) error {
	cursor := ""
	for {
		if err := c.rateLimiter.Acquire(ctx, 1); err != nil {
			return err
		}

		resp, err := c.callConversationsHistory(ctx, channelID, oldest, cursor)
		if err != nil {
			return fmt.Errorf("conversations.history: %w", err)
		}

		for _, msg := range resp.Messages {
			doc := c.messageToDocument(channelID, msg)
			select {
			case docCh <- doc:
			case <-ctx.Done():
				return ctx.Err()
			}

			// Fetch thread replies if enabled and the message is a thread parent.
			if c.config.IncludeThreads && msg.ReplyCount > 0 && msg.ThreadTS != "" {
				threadDoc, err := c.fetchThread(ctx, channelID, msg)
				if err != nil {
					c.logger.Warn("slack: failed to fetch thread",
						"channel", channelID,
						"thread_ts", msg.ThreadTS,
						"error", err,
					)
					continue
				}
				select {
				case docCh <- threadDoc:
				case <-ctx.Done():
					return ctx.Err()
				}
			}

			// Fetch file attachments if enabled.
			if c.config.IncludeFiles && len(msg.Files) > 0 {
				for _, file := range msg.Files {
					if file.Size > c.config.MaxFileSize {
						c.logger.Warn("slack: skipping file exceeding size limit",
							"file_id", file.ID,
							"file_name", file.Name,
							"size", file.Size,
							"max_size", c.config.MaxFileSize,
						)
						continue
					}
					fileDoc, err := c.downloadFile(ctx, channelID, file)
					if err != nil {
						c.logger.Warn("slack: failed to download file",
							"file_id", file.ID,
							"error", err,
						)
						continue
					}
					select {
					case docCh <- fileDoc:
					case <-ctx.Done():
						return ctx.Err()
					}
				}
			}
		}

		if resp.ResponseMetadata.NextCursor == "" {
			break
		}
		cursor = resp.ResponseMetadata.NextCursor
	}
	return nil
}

// ---------------------------------------------------------------------------
// Internal: thread fetching
// ---------------------------------------------------------------------------

func (c *SlackConnector) fetchThread(
	ctx context.Context,
	channelID string,
	parent slackMessage,
) (ConnectorDocument, error) {
	var allReplies []slackMessage
	cursor := ""

	for {
		if err := c.rateLimiter.Acquire(ctx, 1); err != nil {
			return ConnectorDocument{}, err
		}

		resp, err := c.callConversationsReplies(ctx, channelID, parent.ThreadTS, cursor)
		if err != nil {
			return ConnectorDocument{}, fmt.Errorf("conversations.replies: %w", err)
		}

		allReplies = append(allReplies, resp.Messages...)

		if resp.ResponseMetadata.NextCursor == "" {
			break
		}
		cursor = resp.ResponseMetadata.NextCursor
	}

	content := c.buildThreadDocument(parent, allReplies)
	ts := parseSlackTimestamp(parent.TS)

	return ConnectorDocument{
		ExternalID: fmt.Sprintf("%s:thread:%s", channelID, parent.ThreadTS),
		Content:    []byte(content),
		MIME:       "text/markdown",
		Title:      fmt.Sprintf("Thread in %s", channelID),
		Metadata: map[string]string{
			"channel":     channelID,
			"user":        parent.User,
			"ts":          parent.TS,
			"thread_ts":   parent.ThreadTS,
			"reply_count": strconv.Itoa(parent.ReplyCount),
			"source":      "slack",
			"type":        "thread",
		},
		ModifiedAt: ts,
	}, nil
}

func (c *SlackConnector) buildThreadDocument(parent slackMessage, replies []slackMessage) string {
	var sb strings.Builder

	parentUser := c.resolveUserName(parent.User)
	parentTS := parseSlackTimestamp(parent.TS)
	sb.WriteString("## Thread: ")
	sb.WriteString(ConvertMrkdwn(parent.Text))
	sb.WriteString("\n\n")

	// Include the parent message as the first entry.
	sb.WriteString(fmt.Sprintf("**%s** (%s):\n", parentUser, formatTimestamp(parentTS)))
	sb.WriteString(ConvertMrkdwn(parent.Text))
	sb.WriteString("\n\n")

	for _, reply := range replies {
		// Skip the parent message itself (Slack includes it in replies).
		if reply.TS == parent.TS {
			continue
		}
		replyUser := c.resolveUserName(reply.User)
		ts := parseSlackTimestamp(reply.TS)
		sb.WriteString(fmt.Sprintf("**%s** (%s):\n", replyUser, formatTimestamp(ts)))
		sb.WriteString(ConvertMrkdwn(reply.Text))
		sb.WriteString("\n\n")
	}

	return strings.TrimRight(sb.String(), "\n")
}

// ---------------------------------------------------------------------------
// Internal: file download
// ---------------------------------------------------------------------------

func (c *SlackConnector) downloadFile(
	ctx context.Context,
	channelID string,
	file slackFile,
) (ConnectorDocument, error) {
	// Validate the download URL against SSRF before fetching.
	if err := c.urlValidator(file.URLPrivateDownload); err != nil {
		return ConnectorDocument{}, err
	}

	if err := c.rateLimiter.Acquire(ctx, 1); err != nil {
		return ConnectorDocument{}, err
	}

	reqCtx, cancel := context.WithTimeout(ctx, 60*time.Second)
	defer cancel()

	req, err := http.NewRequestWithContext(reqCtx, http.MethodGet, file.URLPrivateDownload, nil)
	if err != nil {
		return ConnectorDocument{}, fmt.Errorf("create file request: %w", err)
	}
	req.Header.Set("Authorization", "Bearer "+c.config.BotToken)

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return ConnectorDocument{}, fmt.Errorf("download file: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		return ConnectorDocument{}, fmt.Errorf("download file: HTTP %d", resp.StatusCode)
	}

	body, err := io.ReadAll(io.LimitReader(resp.Body, c.config.MaxFileSize+1))
	if err != nil {
		return ConnectorDocument{}, fmt.Errorf("read file body: %w", err)
	}
	if int64(len(body)) > c.config.MaxFileSize {
		return ConnectorDocument{}, fmt.Errorf("file %s exceeds maximum size of %d bytes", file.Name, c.config.MaxFileSize)
	}

	return ConnectorDocument{
		ExternalID: fmt.Sprintf("%s:file:%s", channelID, file.ID),
		Content:    body,
		MIME:       file.MIMEType,
		Title:      file.Name,
		Metadata: map[string]string{
			"channel":   channelID,
			"file_id":   file.ID,
			"filename":  file.Name,
			"filetype":  file.MIMEType,
			"file_size": strconv.FormatInt(file.Size, 10),
			"source":    "slack",
			"type":      "file",
		},
		ModifiedAt: time.Now(),
	}, nil
}

// ---------------------------------------------------------------------------
// Internal: Slack API calls
// ---------------------------------------------------------------------------

func (c *SlackConnector) callConversationsHistory(
	ctx context.Context,
	channelID string,
	oldest string,
	cursor string,
) (slackResponse, error) {
	params := url.Values{
		"channel": {channelID},
		"limit":   {"200"},
	}
	if oldest != "" {
		params.Set("oldest", oldest)
	}
	if cursor != "" {
		params.Set("cursor", cursor)
	}

	return c.callSlackAPI(ctx, "https://slack.com/api/conversations.history", params)
}

func (c *SlackConnector) callConversationsReplies(
	ctx context.Context,
	channelID string,
	threadTS string,
	cursor string,
) (slackResponse, error) {
	params := url.Values{
		"channel": {channelID},
		"ts":      {threadTS},
		"limit":   {"200"},
	}
	if cursor != "" {
		params.Set("cursor", cursor)
	}

	return c.callSlackAPI(ctx, "https://slack.com/api/conversations.replies", params)
}

func (c *SlackConnector) callSlackAPI(
	ctx context.Context,
	endpoint string,
	params url.Values,
) (slackResponse, error) {
	reqURL := endpoint + "?" + params.Encode()
	backoff := time.Duration(0)

	for attempt := range maxAPIRetries {
		if backoff > 0 {
			select {
			case <-ctx.Done():
				return slackResponse{}, ctx.Err()
			case <-time.After(backoff):
			}
		}

		reqCtx, cancel := context.WithTimeout(ctx, 30*time.Second)
		req, err := http.NewRequestWithContext(reqCtx, http.MethodGet, reqURL, nil)
		if err != nil {
			cancel()
			return slackResponse{}, fmt.Errorf("create request: %w", err)
		}
		req.Header.Set("Authorization", "Bearer "+c.config.BotToken)
		req.Header.Set("Content-Type", "application/x-www-form-urlencoded")

		resp, err := c.httpClient.Do(req)
		if err != nil {
			cancel()
			return slackResponse{}, fmt.Errorf("http request: %w", err)
		}

		// Handle rate limiting (HTTP 429) with exponential backoff.
		if resp.StatusCode == http.StatusTooManyRequests {
			retryAfter := resp.Header.Get("Retry-After")
			waitSecs, parseErr := strconv.Atoi(retryAfter)
			if parseErr != nil || waitSecs <= 0 {
				waitSecs = 5
			}
			resp.Body.Close()
			cancel()

			if attempt == maxAPIRetries-1 {
				return slackResponse{}, fmt.Errorf("slack API rate limited after %d retries", maxAPIRetries)
			}

			// Exponential backoff: use Retry-After as base, double on each subsequent retry.
			backoff = time.Duration(waitSecs) * time.Second * (1 << attempt)
			continue
		}

		if resp.StatusCode != http.StatusOK {
			resp.Body.Close()
			cancel()
			return slackResponse{}, fmt.Errorf("slack API returned HTTP %d", resp.StatusCode)
		}

		body, readErr := io.ReadAll(io.LimitReader(resp.Body, 10*1024*1024)) // 10 MB limit
		resp.Body.Close()
		cancel()
		if readErr != nil {
			return slackResponse{}, fmt.Errorf("read response: %w", readErr)
		}

		var result slackResponse
		if err := json.Unmarshal(body, &result); err != nil {
			return slackResponse{}, fmt.Errorf("decode response: %w", err)
		}

		if !result.OK {
			return slackResponse{}, fmt.Errorf("slack API error: %s", result.Error)
		}

		return result, nil
	}

	return slackResponse{}, fmt.Errorf("slack API: exhausted all %d retry attempts", maxAPIRetries)
}
