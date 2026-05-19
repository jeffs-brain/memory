// SPDX-License-Identifier: Apache-2.0

package connector

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"time"
)

// OAuth2 sentinel errors.
var (
	// ErrTokenExpired is returned when a token has expired and no refresh
	// token is available.
	ErrTokenExpired = errors.New("connector: oauth2 token expired")

	// ErrTokenRefreshFailed is returned when the token refresh request fails.
	ErrTokenRefreshFailed = errors.New("connector: oauth2 token refresh failed")

	// ErrInvalidOAuth2Config is returned when the OAuth2 configuration is
	// incomplete or invalid.
	ErrInvalidOAuth2Config = errors.New("connector: invalid oauth2 config")
)

// tokenExpiryBuffer is the duration before actual expiry at which we
// proactively refresh the token.
const tokenExpiryBuffer = 5 * time.Minute

// OAuth2Config holds the configuration for an OAuth2 authorization code
// flow.
type OAuth2Config struct {
	ClientID     string
	ClientSecret string
	AuthURL      string
	TokenURL     string
	Scopes       []string
	RedirectURI  string
}

// Validate checks that all required fields are set.
func (c OAuth2Config) Validate() error {
	missing := make([]string, 0, 4)
	if c.ClientID == "" {
		missing = append(missing, "ClientID")
	}
	if c.ClientSecret == "" {
		missing = append(missing, "ClientSecret")
	}
	if c.AuthURL == "" {
		missing = append(missing, "AuthURL")
	}
	if c.TokenURL == "" {
		missing = append(missing, "TokenURL")
	}
	// RedirectURI is only required for the authorization code flow,
	// not for token refresh. Validated at AuthorisationURL call-site.
	if len(missing) > 0 {
		return fmt.Errorf("%w: missing fields: %s", ErrInvalidOAuth2Config, strings.Join(missing, ", "))
	}
	return nil
}

// OAuth2Token holds the result of an OAuth2 token exchange or refresh.
type OAuth2Token struct {
	AccessToken  string    `json:"access_token"`
	RefreshToken string    `json:"refresh_token"`
	ExpiresAt    time.Time `json:"expires_at"`
	TokenType    string    `json:"token_type"`
	Scopes       []string  `json:"scopes"`
}

// IsExpired reports whether the token has expired, accounting for the
// proactive refresh buffer.
func (t OAuth2Token) IsExpired() bool {
	return time.Now().Add(tokenExpiryBuffer).After(t.ExpiresAt)
}

// TokenExchanger abstracts the HTTP calls for OAuth2 token operations,
// allowing tests to inject mock implementations without real HTTP.
type TokenExchanger interface {
	// Exchange trades an authorization code for a token pair.
	Exchange(ctx context.Context, code string) (OAuth2Token, error)

	// Refresh uses a refresh token to obtain a new access token.
	Refresh(ctx context.Context, refreshToken string) (OAuth2Token, error)
}

// pendingRefresh holds the in-flight refresh state, shared across
// concurrent callers to prevent duplicate refreshes.
type pendingRefresh struct {
	done  chan struct{}
	token OAuth2Token
	err   error
}

// OAuth2Client manages the OAuth2 authorization code flow including
// authorization URL generation, code exchange, and token refresh.
// Concurrent refresh calls are deduplicated: only the first caller
// triggers the actual HTTP refresh; subsequent callers receive the same
// result.
type OAuth2Client struct {
	config    OAuth2Config
	exchanger TokenExchanger

	mu      sync.Mutex
	pending *pendingRefresh
}

// NewOAuth2Client creates a new OAuth2Client. The exchanger handles
// the actual HTTP token requests.
func NewOAuth2Client(config OAuth2Config, exchanger TokenExchanger) (*OAuth2Client, error) {
	if err := config.Validate(); err != nil {
		return nil, err
	}
	return &OAuth2Client{
		config:    config,
		exchanger: exchanger,
	}, nil
}

// AuthorisationURL generates the URL the user must visit to grant
// access. The state parameter should be a cryptographically random
// value to prevent CSRF attacks.
func (c *OAuth2Client) AuthorisationURL(state string) string {
	params := url.Values{
		"client_id":     {c.config.ClientID},
		"redirect_uri":  {c.config.RedirectURI},
		"response_type": {"code"},
		"state":         {state},
	}
	if len(c.config.Scopes) > 0 {
		params.Set("scope", strings.Join(c.config.Scopes, " "))
	}
	return c.config.AuthURL + "?" + params.Encode()
}

// ExchangeCode trades an authorization code for an access/refresh token
// pair by delegating to the configured TokenExchanger.
func (c *OAuth2Client) ExchangeCode(ctx context.Context, code string) (OAuth2Token, error) {
	token, err := c.exchanger.Exchange(ctx, code)
	if err != nil {
		return OAuth2Token{}, fmt.Errorf("connector: oauth2 code exchange: %w", err)
	}
	return token, nil
}

// RefreshToken refreshes an expired access token using the refresh
// token. Concurrent calls are deduplicated: only the first caller
// triggers the actual HTTP refresh; subsequent callers await the same
// result.
func (c *OAuth2Client) RefreshToken(ctx context.Context, token OAuth2Token) (OAuth2Token, error) {
	if token.RefreshToken == "" {
		return OAuth2Token{}, fmt.Errorf("%w: no refresh token available", ErrTokenRefreshFailed)
	}

	c.mu.Lock()
	if c.pending != nil {
		// Another goroutine is already refreshing. Wait for its result.
		p := c.pending
		c.mu.Unlock()
		<-p.done
		return p.token, p.err
	}

	// We are the first caller: set up the shared pending state.
	p := &pendingRefresh{done: make(chan struct{})}
	c.pending = p
	c.mu.Unlock()

	refreshed, err := c.exchanger.Refresh(ctx, token.RefreshToken)
	if err != nil {
		p.err = fmt.Errorf("%w: %v", ErrTokenRefreshFailed, err)
	} else {
		// Preserve the refresh token if the provider did not issue a new one.
		if refreshed.RefreshToken == "" {
			refreshed.RefreshToken = token.RefreshToken
		}
		p.token = refreshed
	}

	// Broadcast result to all waiters and clear the pending state.
	close(p.done)
	c.mu.Lock()
	c.pending = nil
	c.mu.Unlock()

	return p.token, p.err
}

// ValidToken returns the current token if it is still valid, or
// refreshes it if it has expired (or is within the 5-minute buffer).
func (c *OAuth2Client) ValidToken(ctx context.Context, token OAuth2Token) (OAuth2Token, error) {
	if !token.IsExpired() {
		return token, nil
	}
	return c.RefreshToken(ctx, token)
}

// Config returns the OAuth2 configuration for inspection (read-only).
func (c *OAuth2Client) Config() OAuth2Config {
	return c.config
}

// httpTokenExchanger implements TokenExchanger using real HTTP calls
// via an HTTPClient. Used as the default exchanger when no mock is
// injected via the config map.
type httpTokenExchanger struct {
	tokenURL     string
	clientID     string
	clientSecret string
	httpClient   HTTPClient
}

// Exchange trades an authorization code for a token pair via the token
// endpoint.
func (e *httpTokenExchanger) Exchange(ctx context.Context, code string) (OAuth2Token, error) {
	form := url.Values{
		"grant_type":    {"authorization_code"},
		"code":          {code},
		"client_id":     {e.clientID},
		"client_secret": {e.clientSecret},
	}
	return e.post(ctx, form)
}

// Refresh uses a refresh token to obtain a new access token from the
// token endpoint.
func (e *httpTokenExchanger) Refresh(ctx context.Context, refreshToken string) (OAuth2Token, error) {
	form := url.Values{
		"grant_type":    {"refresh_token"},
		"refresh_token": {refreshToken},
		"client_id":     {e.clientID},
		"client_secret": {e.clientSecret},
	}
	return e.post(ctx, form)
}

func (e *httpTokenExchanger) post(ctx context.Context, form url.Values) (OAuth2Token, error) {
	body := form.Encode()
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, e.tokenURL, bytes.NewBufferString(body))
	if err != nil {
		return OAuth2Token{}, fmt.Errorf("oauth2: build request: %w", err)
	}
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")

	resp, err := e.httpClient.Do(req)
	if err != nil {
		return OAuth2Token{}, fmt.Errorf("oauth2: token request: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return OAuth2Token{}, fmt.Errorf("oauth2: read response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return OAuth2Token{}, fmt.Errorf("oauth2: token endpoint returned %d: %s", resp.StatusCode, string(respBody))
	}

	var tokenResp struct {
		AccessToken  string `json:"access_token"`
		RefreshToken string `json:"refresh_token"`
		ExpiresIn    int64  `json:"expires_in"`
		TokenType    string `json:"token_type"`
	}
	if err := json.Unmarshal(respBody, &tokenResp); err != nil {
		return OAuth2Token{}, fmt.Errorf("oauth2: parse token response: %w", err)
	}

	return OAuth2Token{
		AccessToken:  tokenResp.AccessToken,
		RefreshToken: tokenResp.RefreshToken,
		ExpiresAt:    time.Now().Add(time.Duration(tokenResp.ExpiresIn) * time.Second),
		TokenType:    tokenResp.TokenType,
	}, nil
}
