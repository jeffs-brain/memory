// SPDX-License-Identifier: Apache-2.0

package httpd

import (
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestValidBearerToken_valid(t *testing.T) {
	tests := []struct {
		name   string
		header string
		token  string
	}{
		{
			name:   "standard bearer prefix",
			header: "Bearer my-secret-token",
			token:  "my-secret-token",
		},
		{
			name:   "lowercase bearer prefix",
			header: "bearer my-secret-token",
			token:  "my-secret-token",
		},
		{
			name:   "uppercase bearer prefix",
			header: "BEARER my-secret-token",
			token:  "my-secret-token",
		},
		{
			name:   "mixed case bearer prefix",
			header: "bEaReR my-secret-token",
			token:  "my-secret-token",
		},
		{
			name:   "token with special characters",
			header: "Bearer abc!@#$%^&*()_+-=[]{}|;':\",./<>?",
			token:  "abc!@#$%^&*()_+-=[]{}|;':\",./<>?",
		},
		{
			name:   "very long token",
			header: "Bearer " + string(make([]byte, 4096)),
			token:  string(make([]byte, 4096)),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if !validBearerToken(tt.header, tt.token) {
				t.Errorf("validBearerToken(%q, %q) = false, want true", tt.header, tt.token)
			}
		})
	}
}

func TestValidBearerToken_invalid(t *testing.T) {
	tests := []struct {
		name   string
		header string
		token  string
	}{
		{
			name:   "wrong token",
			header: "Bearer wrong-token",
			token:  "correct-token",
		},
		{
			name:   "empty header",
			header: "",
			token:  "my-token",
		},
		{
			name:   "non-bearer scheme",
			header: "Basic dXNlcjpwYXNz",
			token:  "dXNlcjpwYXNz",
		},
		{
			name:   "empty token after bearer prefix",
			header: "Bearer ",
			token:  "my-token",
		},
		{
			name:   "bearer prefix only no space",
			header: "Bearer",
			token:  "my-token",
		},
		{
			name:   "token case mismatch is rejected",
			header: "Bearer MyToken",
			token:  "mytoken",
		},
		{
			name:   "partial prefix",
			header: "Bear my-token",
			token:  "my-token",
		},
		{
			name:   "token with trailing space differs",
			header: "Bearer my-token ",
			token:  "my-token",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if validBearerToken(tt.header, tt.token) {
				t.Errorf("validBearerToken(%q, %q) = true, want false", tt.header, tt.token)
			}
		})
	}
}

func TestAuthMiddleware_authorized(t *testing.T) {
	handler := AuthMiddleware("secret", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))

	tests := []struct {
		name   string
		header string
	}{
		{name: "standard bearer", header: "Bearer secret"},
		{name: "lowercase bearer", header: "bearer secret"},
		{name: "uppercase bearer", header: "BEARER secret"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := httptest.NewRequest(http.MethodGet, "/api/test", nil)
			req.Header.Set("Authorization", tt.header)
			rec := httptest.NewRecorder()

			handler.ServeHTTP(rec, req)

			if rec.Code != http.StatusOK {
				t.Errorf("got status %d, want %d", rec.Code, http.StatusOK)
			}
		})
	}
}

func TestAuthMiddleware_unauthorized(t *testing.T) {
	handler := AuthMiddleware("secret", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))

	tests := []struct {
		name       string
		header     string
		wantStatus int
	}{
		{
			name:       "missing authorization header",
			header:     "",
			wantStatus: http.StatusUnauthorized,
		},
		{
			name:       "wrong token",
			header:     "Bearer wrong",
			wantStatus: http.StatusForbidden,
		},
		{
			name:       "non-bearer scheme",
			header:     "Basic c2VjcmV0",
			wantStatus: http.StatusForbidden,
		},
		{
			name:       "empty token value",
			header:     "Bearer ",
			wantStatus: http.StatusForbidden,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := httptest.NewRequest(http.MethodGet, "/api/test", nil)
			if tt.header != "" {
				req.Header.Set("Authorization", tt.header)
			}
			rec := httptest.NewRecorder()

			handler.ServeHTTP(rec, req)

			if rec.Code != tt.wantStatus {
				t.Errorf("got status %d, want %d", rec.Code, tt.wantStatus)
			}
		})
	}
}

func TestAuthMiddleware_healthzBypass(t *testing.T) {
	handler := AuthMiddleware("secret", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))

	req := httptest.NewRequest(http.MethodGet, "/healthz", nil)
	rec := httptest.NewRecorder()

	handler.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Errorf("got status %d, want %d for /healthz bypass", rec.Code, http.StatusOK)
	}
}

func TestAuthMiddleware_emptyTokenPassthrough(t *testing.T) {
	called := false
	handler := AuthMiddleware("", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		called = true
		w.WriteHeader(http.StatusOK)
	}))

	req := httptest.NewRequest(http.MethodGet, "/api/test", nil)
	rec := httptest.NewRecorder()

	handler.ServeHTTP(rec, req)

	if !called {
		t.Error("handler was not called when token is empty (should be passthrough)")
	}
	if rec.Code != http.StatusOK {
		t.Errorf("got status %d, want %d for empty-token passthrough", rec.Code, http.StatusOK)
	}
}
