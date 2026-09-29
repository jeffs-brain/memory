// SPDX-License-Identifier: Apache-2.0

package httpd

import (
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestResolveBindAddr(t *testing.T) {
	tests := []struct {
		name    string
		addr    string
		token   string
		want    string
		wantErr bool
	}{
		{name: "empty defaults to loopback", addr: "", want: "127.0.0.1:8080"},
		{name: "bare port binds loopback", addr: ":9000", want: "127.0.0.1:9000"},
		{name: "explicit loopback", addr: "127.0.0.1:18841", want: "127.0.0.1:18841"},
		{name: "localhost", addr: "localhost:8080", want: "localhost:8080"},
		{name: "ipv6 loopback", addr: "[::1]:8080", want: "[::1]:8080"},
		{name: "all interfaces without token", addr: "0.0.0.0:8080", wantErr: true},
		{name: "lan address without token", addr: "192.0.2.10:8080", wantErr: true},
		{name: "ipv6 any without token", addr: "[::]:8080", wantErr: true},
		{name: "all interfaces with token", addr: "0.0.0.0:8080", token: "secret", want: "0.0.0.0:8080"},
		{name: "missing port", addr: "127.0.0.1", wantErr: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := ResolveBindAddr(tt.addr, tt.token)
			if tt.wantErr {
				if err == nil {
					t.Fatalf("ResolveBindAddr(%q) = %q, want error", tt.addr, got)
				}
				return
			}
			if err != nil {
				t.Fatalf("ResolveBindAddr(%q) error: %v", tt.addr, err)
			}
			if got != tt.want {
				t.Fatalf("ResolveBindAddr(%q) = %q, want %q", tt.addr, got, tt.want)
			}
		})
	}
	if _, err := ResolveBindAddr("0.0.0.0:8080", ""); !errors.Is(err, ErrUnauthenticatedBind) {
		t.Fatalf("error = %v, want ErrUnauthenticatedBind", err)
	}
}

func TestLoopbackGuard(t *testing.T) {
	ok := http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusNoContent)
	})
	guard := LoopbackGuard(ok)
	tests := []struct {
		name   string
		path   string
		host   string
		origin string
		want   int
	}{
		{name: "loopback ip", path: "/v1/brains", host: "127.0.0.1:8080", want: http.StatusNoContent},
		{name: "localhost", path: "/v1/brains", host: "localhost:8080", want: http.StatusNoContent},
		{name: "ipv6 loopback", path: "/v1/brains", host: "[::1]:8080", want: http.StatusNoContent},
		{name: "rebound hostname", path: "/v1/brains", host: "attacker.example:8080", want: http.StatusMisdirectedRequest},
		{name: "lan ip", path: "/v1/brains", host: "192.0.2.10:8080", want: http.StatusMisdirectedRequest},
		{name: "loopback origin", path: "/v1/brains", host: "127.0.0.1:8080", origin: "http://localhost:3000", want: http.StatusNoContent},
		{name: "foreign origin", path: "/v1/brains", host: "127.0.0.1:8080", origin: "https://attacker.example", want: http.StatusForbidden},
		{name: "null origin", path: "/v1/brains", host: "127.0.0.1:8080", origin: "null", want: http.StatusForbidden},
		{name: "healthz bypass", path: "/healthz", host: "attacker.example", want: http.StatusNoContent},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := httptest.NewRequest(http.MethodGet, tt.path, nil)
			req.Host = tt.host
			if tt.origin != "" {
				req.Header.Set("Origin", tt.origin)
			}
			rec := httptest.NewRecorder()
			guard.ServeHTTP(rec, req)
			if rec.Code != tt.want {
				t.Fatalf("status = %d, want %d", rec.Code, tt.want)
			}
		})
	}
}

func TestInternalError_HidesDetail(t *testing.T) {
	rec := httptest.NewRecorder()
	InternalError(rec, "open /srv/secret/brains/x: permission denied")
	if rec.Code != http.StatusInternalServerError {
		t.Fatalf("status = %d", rec.Code)
	}
	if strings.Contains(rec.Body.String(), "/srv/secret") {
		t.Fatalf("500 body leaked detail: %s", rec.Body.String())
	}
	var p Problem
	if err := json.Unmarshal(rec.Body.Bytes(), &p); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if p.Code != "internal_error" || p.Detail != "internal error" {
		t.Fatalf("problem = %+v", p)
	}
}
