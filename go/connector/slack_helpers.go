// SPDX-License-Identifier: Apache-2.0

package connector

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"
	"time"
)

// ---------------------------------------------------------------------------
// User resolution
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

	reqURL := "https://slack.com/api/users.info?user=" + userID

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
// Message -> document conversion
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

// formatTimestamp formats a time.Time for display in thread documents
// using UTC for consistency across environments.
func formatTimestamp(t time.Time) string {
	return t.UTC().Format("2006-01-02 15:04")
}
