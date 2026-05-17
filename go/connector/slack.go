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
	OK               bool             `json:"ok"`
	Error            string           `json:"error"`
	Messages         []slackMessage   `json:"messages"`
	ResponseMetadata slackRespMeta    `json:"response_metadata"`
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
	config      SlackConnectorConfig
	rateLimiter *RateLimiter
	userCache   map[string]string
	userMu      sync.RWMutex
	stopCh      chan struct{}
	stopped     bool
	mu          sync.Mutex
	logger      *slog.Logger
	httpClient  HTTPDoer
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
		rateLimiter: NewRateLimiter(50, 50.0/60.0),
		userCache:   make(map[string]string),
		stopCh:      make(chan struct{}),
		logger:      cfg.Logger,
		httpClient:  cfg.HTTPClient,
	}
}

// Name returns "slack".
func (c *SlackConnector) Name() string { return "slack" }

// Configure validates and applies configuration from a string map.
func (c *SlackConnector) Configure(config map[string]string) error {
	token, ok := config["botToken"]
	if !ok || token == "" {
		return fmt.Errorf("slack: botToken is required")
	}
	c.config.BotToken = token

	channels, ok := config["channels"]
	if !ok || channels == "" {
		return fmt.Errorf("slack: at least one channel is required")
	}
	c.config.Channels = strings.Split(channels, ",")
	for i, ch := range c.config.Channels {
		c.config.Channels[i] = strings.TrimSpace(ch)
	}

	if v, ok := config["includeThreads"]; ok {
		c.config.IncludeThreads = v != "false"
	}
	if v, ok := config["includeFiles"]; ok {
		c.config.IncludeFiles = v != "false"
	}
	if v, ok := config["maxFileSize"]; ok {
		size, err := strconv.ParseInt(v, 10, 64)
		if err != nil {
			return fmt.Errorf("slack: invalid maxFileSize: %w", err)
		}
		c.config.MaxFileSize = size
	}
	if v, ok := config["oldestTimestamp"]; ok {
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
// interval. The first iteration performs a full sync if no cursor
// exists, then subsequent iterations sync incrementally.
func (c *SlackConnector) Start(ctx context.Context) (<-chan ConnectorDocument, <-chan error) {
	docCh := make(chan ConnectorDocument, 100)
	errCh := make(chan error, 1)

	go func() {
		defer close(docCh)
		defer close(errCh)

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
				select {
				case docCh <- doc:
				case <-ctx.Done():
					return
				}
			}

			if err, ok := <-errs; ok && err != nil {
				select {
				case errCh <- err:
				default:
				}
			}

			if latestTS != "" {
				lastCursor = latestTS
			}

			select {
			case <-ctx.Done():
				return
			case <-c.stopCh:
				return
			case <-ticker.C:
			}
		}
	}()

	return docCh, errCh
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
	sb.WriteString(fmt.Sprintf("**%s** (%s):\n", parentUser, parentTS.Format("2006-01-02 15:04")))
	sb.WriteString(ConvertMrkdwn(parent.Text))
	sb.WriteString("\n\n")

	for _, reply := range replies {
		// Skip the parent message itself (Slack includes it in replies).
		if reply.TS == parent.TS {
			continue
		}
		replyUser := c.resolveUserName(reply.User)
		ts := parseSlackTimestamp(reply.TS)
		sb.WriteString(fmt.Sprintf("**%s** (%s):\n", replyUser, ts.Format("2006-01-02 15:04")))
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
	defer resp.Body.Close()

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
	reqCtx, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()

	reqURL := endpoint + "?" + params.Encode()
	req, err := http.NewRequestWithContext(reqCtx, http.MethodGet, reqURL, nil)
	if err != nil {
		return slackResponse{}, fmt.Errorf("create request: %w", err)
	}
	req.Header.Set("Authorization", "Bearer "+c.config.BotToken)
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return slackResponse{}, fmt.Errorf("http request: %w", err)
	}
	defer resp.Body.Close()

	// Handle rate limiting (HTTP 429).
	if resp.StatusCode == http.StatusTooManyRequests {
		retryAfter := resp.Header.Get("Retry-After")
		waitSecs, parseErr := strconv.Atoi(retryAfter)
		if parseErr != nil || waitSecs <= 0 {
			waitSecs = 5
		}
		select {
		case <-ctx.Done():
			return slackResponse{}, ctx.Err()
		case <-time.After(time.Duration(waitSecs) * time.Second):
		}
		// Retry once after waiting.
		return c.callSlackAPI(ctx, endpoint, params)
	}

	if resp.StatusCode != http.StatusOK {
		return slackResponse{}, fmt.Errorf("slack API returned HTTP %d", resp.StatusCode)
	}

	body, err := io.ReadAll(io.LimitReader(resp.Body, 10*1024*1024)) // 10 MB limit
	if err != nil {
		return slackResponse{}, fmt.Errorf("read response: %w", err)
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

// ---------------------------------------------------------------------------
// Internal: user resolution
// ---------------------------------------------------------------------------

func (c *SlackConnector) resolveUserName(userID string) string {
	if userID == "" {
		return "unknown"
	}

	c.userMu.RLock()
	name, ok := c.userCache[userID]
	c.userMu.RUnlock()
	if ok {
		return name
	}

	// Attempt to fetch the user name via users.info.
	resolved := c.fetchUserName(userID)

	c.userMu.Lock()
	c.userCache[userID] = resolved
	c.userMu.Unlock()

	return resolved
}

func (c *SlackConnector) fetchUserName(userID string) string {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	params := url.Values{"user": {userID}}
	reqURL := "https://slack.com/api/users.info?" + params.Encode()

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, reqURL, nil)
	if err != nil {
		return userID
	}
	req.Header.Set("Authorization", "Bearer "+c.config.BotToken)

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return userID
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return userID
	}

	body, err := io.ReadAll(io.LimitReader(resp.Body, 1*1024*1024))
	if err != nil {
		return userID
	}

	var result slackUserResponse
	if err := json.Unmarshal(body, &result); err != nil {
		return userID
	}

	if !result.OK {
		return userID
	}

	if result.User.RealName != "" {
		return result.User.RealName
	}
	if result.User.Name != "" {
		return result.User.Name
	}
	return userID
}

// ---------------------------------------------------------------------------
// Internal: message -> document conversion
// ---------------------------------------------------------------------------

func (c *SlackConnector) messageToDocument(channelID string, msg slackMessage) ConnectorDocument {
	ts := parseSlackTimestamp(msg.TS)
	content := ConvertMrkdwn(msg.Text)

	metadata := map[string]string{
		"channel": channelID,
		"user":    msg.User,
		"ts":      msg.TS,
		"source":  "slack",
		"type":    "message",
	}
	if msg.ThreadTS != "" {
		metadata["thread_ts"] = msg.ThreadTS
	}
	if msg.ReplyCount > 0 {
		metadata["reply_count"] = strconv.Itoa(msg.ReplyCount)
	}

	return ConnectorDocument{
		ExternalID: fmt.Sprintf("%s:%s", channelID, msg.TS),
		Content:    []byte(content),
		MIME:       "text/markdown",
		Title:      fmt.Sprintf("Message in %s", channelID),
		Metadata:   metadata,
		ModifiedAt: ts,
	}
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// parseSlackTimestamp converts a Slack epoch timestamp string (e.g.
// "1234567890.123456") to a time.Time.
func parseSlackTimestamp(ts string) time.Time {
	parts := strings.SplitN(ts, ".", 2)
	secs, err := strconv.ParseInt(parts[0], 10, 64)
	if err != nil {
		return time.Time{}
	}
	var micros int64
	if len(parts) > 1 {
		micros, _ = strconv.ParseInt(parts[1], 10, 64)
	}
	return time.Unix(secs, micros*1000)
}
