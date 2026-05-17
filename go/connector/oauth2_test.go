// SPDX-License-Identifier: Apache-2.0

package connector_test

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/jeffs-brain/memory/go/connector"
)

// mockExchanger implements TokenExchanger for testing.
type mockExchanger struct {
	exchangeFn func(ctx context.Context, code string) (connector.OAuth2Token, error)
	refreshFn  func(ctx context.Context, refreshToken string) (connector.OAuth2Token, error)
}

func (m *mockExchanger) Exchange(ctx context.Context, code string) (connector.OAuth2Token, error) {
	return m.exchangeFn(ctx, code)
}

func (m *mockExchanger) Refresh(ctx context.Context, refreshToken string) (connector.OAuth2Token, error) {
	return m.refreshFn(ctx, refreshToken)
}

func validOAuth2Config() connector.OAuth2Config {
	return connector.OAuth2Config{
		ClientID:     "test-client-id",
		ClientSecret: "test-client-secret",
		AuthURL:      "https://provider.example.com/auth",
		TokenURL:     "https://provider.example.com/token",
		Scopes:       []string{"read", "write"},
		RedirectURI:  "https://app.example.com/callback",
	}
}

func TestOAuth2Config_Validate_MissingFields(t *testing.T) {
	cfg := connector.OAuth2Config{}
	err := cfg.Validate()
	if err == nil {
		t.Fatal("expected validation error for empty config")
	}
	if !errors.Is(err, connector.ErrInvalidOAuth2Config) {
		t.Errorf("error should wrap ErrInvalidOAuth2Config: %v", err)
	}
}

func TestOAuth2Config_Validate_Valid(t *testing.T) {
	if err := validOAuth2Config().Validate(); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestOAuth2Client_AuthorisationURL(t *testing.T) {
	cfg := validOAuth2Config()
	exchanger := &mockExchanger{}
	client, err := connector.NewOAuth2Client(cfg, exchanger)
	if err != nil {
		t.Fatalf("NewOAuth2Client: %v", err)
	}

	url := client.AuthorisationURL("random-state")

	if !strings.HasPrefix(url, cfg.AuthURL+"?") {
		t.Errorf("URL does not start with auth URL: %s", url)
	}
	if !strings.Contains(url, "client_id="+cfg.ClientID) {
		t.Error("URL missing client_id")
	}
	if !strings.Contains(url, "response_type=code") {
		t.Error("URL missing response_type=code")
	}
	if !strings.Contains(url, "state=random-state") {
		t.Error("URL missing state parameter")
	}
	if !strings.Contains(url, "redirect_uri=") {
		t.Error("URL missing redirect_uri")
	}
	if !strings.Contains(url, "scope=read+write") {
		t.Error("URL missing scopes")
	}
}

func TestOAuth2Client_ExchangeCode(t *testing.T) {
	expectedToken := connector.OAuth2Token{
		AccessToken:  "access-123",
		RefreshToken: "refresh-456",
		ExpiresAt:    time.Now().Add(time.Hour),
		TokenType:    "Bearer",
		Scopes:       []string{"read"},
	}

	exchanger := &mockExchanger{
		exchangeFn: func(_ context.Context, code string) (connector.OAuth2Token, error) {
			if code != "auth-code-789" {
				t.Errorf("unexpected code: %s", code)
			}
			return expectedToken, nil
		},
	}

	client, err := connector.NewOAuth2Client(validOAuth2Config(), exchanger)
	if err != nil {
		t.Fatalf("NewOAuth2Client: %v", err)
	}

	token, err := client.ExchangeCode(context.Background(), "auth-code-789")
	if err != nil {
		t.Fatalf("ExchangeCode: %v", err)
	}
	if token.AccessToken != "access-123" {
		t.Errorf("AccessToken = %q, want %q", token.AccessToken, "access-123")
	}
	if token.RefreshToken != "refresh-456" {
		t.Errorf("RefreshToken = %q, want %q", token.RefreshToken, "refresh-456")
	}
}

func TestOAuth2Client_RefreshToken(t *testing.T) {
	refreshed := connector.OAuth2Token{
		AccessToken:  "new-access",
		RefreshToken: "new-refresh",
		ExpiresAt:    time.Now().Add(time.Hour),
		TokenType:    "Bearer",
	}

	exchanger := &mockExchanger{
		refreshFn: func(_ context.Context, rt string) (connector.OAuth2Token, error) {
			if rt != "old-refresh" {
				t.Errorf("unexpected refresh token: %s", rt)
			}
			return refreshed, nil
		},
	}

	client, err := connector.NewOAuth2Client(validOAuth2Config(), exchanger)
	if err != nil {
		t.Fatalf("NewOAuth2Client: %v", err)
	}

	old := connector.OAuth2Token{
		AccessToken:  "old-access",
		RefreshToken: "old-refresh",
		ExpiresAt:    time.Now().Add(-time.Hour),
	}

	token, err := client.RefreshToken(context.Background(), old)
	if err != nil {
		t.Fatalf("RefreshToken: %v", err)
	}
	if token.AccessToken != "new-access" {
		t.Errorf("AccessToken = %q, want %q", token.AccessToken, "new-access")
	}
}

func TestOAuth2Client_RefreshToken_PreservesRefreshToken(t *testing.T) {
	exchanger := &mockExchanger{
		refreshFn: func(_ context.Context, _ string) (connector.OAuth2Token, error) {
			return connector.OAuth2Token{
				AccessToken: "new-access",
				ExpiresAt:   time.Now().Add(time.Hour),
				TokenType:   "Bearer",
				// No refresh token returned by provider.
			}, nil
		},
	}

	client, err := connector.NewOAuth2Client(validOAuth2Config(), exchanger)
	if err != nil {
		t.Fatalf("NewOAuth2Client: %v", err)
	}

	old := connector.OAuth2Token{
		AccessToken:  "old-access",
		RefreshToken: "old-refresh",
		ExpiresAt:    time.Now().Add(-time.Hour),
	}

	token, err := client.RefreshToken(context.Background(), old)
	if err != nil {
		t.Fatalf("RefreshToken: %v", err)
	}
	if token.RefreshToken != "old-refresh" {
		t.Errorf("RefreshToken not preserved: got %q, want %q", token.RefreshToken, "old-refresh")
	}
}

func TestOAuth2Client_RefreshToken_NoRefreshToken(t *testing.T) {
	exchanger := &mockExchanger{}
	client, err := connector.NewOAuth2Client(validOAuth2Config(), exchanger)
	if err != nil {
		t.Fatalf("NewOAuth2Client: %v", err)
	}

	old := connector.OAuth2Token{
		AccessToken: "access",
		ExpiresAt:   time.Now().Add(-time.Hour),
		// No refresh token.
	}

	_, err = client.RefreshToken(context.Background(), old)
	if err == nil {
		t.Fatal("expected error when no refresh token")
	}
	if !errors.Is(err, connector.ErrTokenRefreshFailed) {
		t.Errorf("error should wrap ErrTokenRefreshFailed: %v", err)
	}
}

func TestOAuth2Client_ValidToken_NotExpired(t *testing.T) {
	refreshCalled := false
	exchanger := &mockExchanger{
		refreshFn: func(_ context.Context, _ string) (connector.OAuth2Token, error) {
			refreshCalled = true
			return connector.OAuth2Token{}, nil
		},
	}

	client, err := connector.NewOAuth2Client(validOAuth2Config(), exchanger)
	if err != nil {
		t.Fatalf("NewOAuth2Client: %v", err)
	}

	valid := connector.OAuth2Token{
		AccessToken:  "access",
		RefreshToken: "refresh",
		ExpiresAt:    time.Now().Add(30 * time.Minute),
	}

	token, err := client.ValidToken(context.Background(), valid)
	if err != nil {
		t.Fatalf("ValidToken: %v", err)
	}
	if refreshCalled {
		t.Error("refresh should not be called for non-expired token")
	}
	if token.AccessToken != "access" {
		t.Errorf("AccessToken = %q, want %q", token.AccessToken, "access")
	}
}

func TestOAuth2Client_ValidToken_ExpiringWithinBuffer(t *testing.T) {
	exchanger := &mockExchanger{
		refreshFn: func(_ context.Context, _ string) (connector.OAuth2Token, error) {
			return connector.OAuth2Token{
				AccessToken:  "refreshed",
				RefreshToken: "new-refresh",
				ExpiresAt:    time.Now().Add(time.Hour),
				TokenType:    "Bearer",
			}, nil
		},
	}

	client, err := connector.NewOAuth2Client(validOAuth2Config(), exchanger)
	if err != nil {
		t.Fatalf("NewOAuth2Client: %v", err)
	}

	// Token expiring in 3 minutes (within the 5-minute buffer).
	expiring := connector.OAuth2Token{
		AccessToken:  "old-access",
		RefreshToken: "old-refresh",
		ExpiresAt:    time.Now().Add(3 * time.Minute),
	}

	token, err := client.ValidToken(context.Background(), expiring)
	if err != nil {
		t.Fatalf("ValidToken: %v", err)
	}
	if token.AccessToken != "refreshed" {
		t.Errorf("AccessToken = %q, want %q", token.AccessToken, "refreshed")
	}
}

func TestOAuth2Client_RefreshFailure(t *testing.T) {
	exchanger := &mockExchanger{
		refreshFn: func(_ context.Context, _ string) (connector.OAuth2Token, error) {
			return connector.OAuth2Token{}, errors.New("401 unauthorized")
		},
	}

	client, err := connector.NewOAuth2Client(validOAuth2Config(), exchanger)
	if err != nil {
		t.Fatalf("NewOAuth2Client: %v", err)
	}

	old := connector.OAuth2Token{
		AccessToken:  "access",
		RefreshToken: "refresh",
		ExpiresAt:    time.Now().Add(-time.Hour),
	}

	_, err = client.RefreshToken(context.Background(), old)
	if err == nil {
		t.Fatal("expected error on refresh failure")
	}
	if !errors.Is(err, connector.ErrTokenRefreshFailed) {
		t.Errorf("error should wrap ErrTokenRefreshFailed: %v", err)
	}
}

func TestOAuth2Token_IsExpired(t *testing.T) {
	tests := []struct {
		name    string
		expires time.Time
		want    bool
	}{
		{"expired", time.Now().Add(-time.Hour), true},
		{"within buffer (3min left)", time.Now().Add(3 * time.Minute), true},
		{"not expired (30min left)", time.Now().Add(30 * time.Minute), false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			token := connector.OAuth2Token{ExpiresAt: tt.expires}
			if got := token.IsExpired(); got != tt.want {
				t.Errorf("IsExpired() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestOAuth2Client_InvalidConfig(t *testing.T) {
	_, err := connector.NewOAuth2Client(connector.OAuth2Config{}, &mockExchanger{})
	if err == nil {
		t.Fatal("expected error for invalid config")
	}
	if !errors.Is(err, connector.ErrInvalidOAuth2Config) {
		t.Errorf("error should wrap ErrInvalidOAuth2Config: %v", err)
	}
}
