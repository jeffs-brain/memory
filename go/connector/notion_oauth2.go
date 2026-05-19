// SPDX-License-Identifier: Apache-2.0

package connector

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"time"
)

// Notion OAuth2 sentinel errors.
var (
	// ErrNotionTokenRefreshFailed is returned when the Notion token
	// refresh request fails.
	ErrNotionTokenRefreshFailed = errors.New("connector/notion: token refresh failed")
)

// notionTokenEndpoint is the Notion OAuth2 token endpoint.
const notionTokenEndpoint = "https://api.notion.com/v1/oauth/token"

// NotionRefreshedToken holds the result of a Notion token refresh.
type NotionRefreshedToken struct {
	AccessToken  string
	RefreshToken string
	ExpiresAt    time.Time
}

// NotionTokenExchanger abstracts the HTTP calls for Notion OAuth2
// token operations, allowing tests to inject mock implementations.
type NotionTokenExchanger interface {
	// Refresh uses a refresh token to obtain a new access token from
	// Notion's token endpoint.
	Refresh(ctx context.Context, refreshToken string) (NotionRefreshedToken, error)
}

// notionPendingRefresh holds the in-flight refresh state for
// deduplication.
type notionPendingRefresh struct {
	done  chan struct{}
	token NotionRefreshedToken
	err   error
}

// NotionOAuth2Client manages Notion OAuth2 token refresh with
// concurrent deduplication.
type NotionOAuth2Client struct {
	clientID     string
	clientSecret string
	exchanger    NotionTokenExchanger

	mu      sync.Mutex
	pending *notionPendingRefresh
}

// NewNotionOAuth2Client creates a new NotionOAuth2Client.
func NewNotionOAuth2Client(clientID, clientSecret string, exchanger NotionTokenExchanger) *NotionOAuth2Client {
	return &NotionOAuth2Client{
		clientID:     clientID,
		clientSecret: clientSecret,
		exchanger:    exchanger,
	}
}

// RefreshToken refreshes an expired access token. Concurrent calls
// are deduplicated: only the first caller triggers the actual HTTP
// refresh; subsequent callers await the same result.
func (c *NotionOAuth2Client) RefreshToken(ctx context.Context, refreshToken string) (NotionRefreshedToken, error) {
	if refreshToken == "" {
		return NotionRefreshedToken{}, fmt.Errorf("%w: no refresh token available", ErrNotionTokenRefreshFailed)
	}

	c.mu.Lock()
	if c.pending != nil {
		p := c.pending
		c.mu.Unlock()
		<-p.done
		return p.token, p.err
	}

	p := &notionPendingRefresh{done: make(chan struct{})}
	c.pending = p
	c.mu.Unlock()

	refreshed, err := c.exchanger.Refresh(ctx, refreshToken)
	if err != nil {
		p.err = fmt.Errorf("%w: %v", ErrNotionTokenRefreshFailed, err)
	} else {
		// Preserve the refresh token if Notion did not issue a new one.
		if refreshed.RefreshToken == "" {
			refreshed.RefreshToken = refreshToken
		}
		p.token = refreshed
	}

	close(p.done)
	c.mu.Lock()
	c.pending = nil
	c.mu.Unlock()

	return p.token, p.err
}

// notionHTTPTokenExchanger implements NotionTokenExchanger using real
// HTTP calls to Notion's token endpoint. Notion uses HTTP Basic Auth
// (clientID:clientSecret) for the token exchange, unlike standard
// OAuth2 which sends credentials in the body.
type notionHTTPTokenExchanger struct {
	clientID     string
	clientSecret string
	httpClient   NotionHTTPClient
}

// Refresh sends a refresh request to Notion's token endpoint.
func (e *notionHTTPTokenExchanger) Refresh(ctx context.Context, refreshToken string) (NotionRefreshedToken, error) {
	reqBody := map[string]string{
		"grant_type":    "refresh_token",
		"refresh_token": refreshToken,
	}
	bodyJSON, err := json.Marshal(reqBody)
	if err != nil {
		return NotionRefreshedToken{}, fmt.Errorf("marshalling refresh request: %w", err)
	}

	reqCtx, cancel := context.WithTimeout(ctx, notionDefaultTimeout)
	defer cancel()

	req, err := http.NewRequestWithContext(reqCtx, http.MethodPost, notionTokenEndpoint, strings.NewReader(string(bodyJSON)))
	if err != nil {
		return NotionRefreshedToken{}, fmt.Errorf("creating refresh request: %w", err)
	}

	// Notion requires Basic Auth with clientID:clientSecret for token
	// exchange.
	basicAuth := base64.StdEncoding.EncodeToString([]byte(e.clientID + ":" + e.clientSecret))
	req.Header.Set("Authorization", "Basic "+basicAuth)
	req.Header.Set("Content-Type", "application/json")

	resp, err := e.httpClient.Do(req)
	if err != nil {
		return NotionRefreshedToken{}, fmt.Errorf("executing refresh request: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	body, err := io.ReadAll(io.LimitReader(resp.Body, notionMaxResponseSize))
	if err != nil {
		return NotionRefreshedToken{}, fmt.Errorf("reading refresh response: %w", err)
	}

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return NotionRefreshedToken{}, fmt.Errorf("token endpoint returned status %d: %s", resp.StatusCode, string(body))
	}

	var tokenResp struct {
		AccessToken  string `json:"access_token"`
		RefreshToken string `json:"refresh_token"`
		ExpiresIn    int64  `json:"expires_in"`
	}
	if err := json.Unmarshal(body, &tokenResp); err != nil {
		return NotionRefreshedToken{}, fmt.Errorf("parsing refresh response: %w", err)
	}

	return NotionRefreshedToken{
		AccessToken:  tokenResp.AccessToken,
		RefreshToken: tokenResp.RefreshToken,
		ExpiresAt:    time.Now().Add(time.Duration(tokenResp.ExpiresIn) * time.Second),
	}, nil
}
