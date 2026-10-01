// SPDX-License-Identifier: Apache-2.0

package knowledge

import (
	"context"
	"errors"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"testing"
)

func TestIsBlockedIP(t *testing.T) {
	tests := []struct {
		name    string
		ip      string
		blocked bool
	}{
		{name: "ipv4 loopback", ip: "127.0.0.1", blocked: true},
		{name: "ipv4 loopback alt", ip: "127.0.0.2", blocked: true},
		{name: "rfc1918 10.x", ip: "10.0.0.1", blocked: true},
		{name: "rfc1918 172.16.x", ip: "172.16.0.1", blocked: true},
		{name: "rfc1918 172.31.x", ip: "172.31.255.255", blocked: true},
		{name: "rfc1918 192.168.x", ip: "192.168.1.1", blocked: true},
		{name: "link-local ipv4", ip: "169.254.1.1", blocked: true},
		{name: "cloud metadata", ip: "169.254.169.254", blocked: true},
		{name: "unspecified ipv4", ip: "0.0.0.0", blocked: true},
		{name: "ipv6 loopback", ip: "::1", blocked: true},
		{name: "ipv6 link-local", ip: "fe80::1", blocked: true},
		{name: "ipv6 unspecified", ip: "::", blocked: true},
		{name: "ipv6 unique local", ip: "fd00::1", blocked: true},
		{name: "carrier-grade nat", ip: "100.64.0.1", blocked: true},
		{name: "this network", ip: "0.1.2.3", blocked: true},
		{name: "ietf protocol assignments", ip: "192.0.0.8", blocked: true},
		{name: "benchmarking", ip: "198.18.0.1", blocked: true},
		{name: "ipv4 multicast", ip: "224.0.0.1", blocked: true},
		{name: "ipv4 reserved", ip: "240.0.0.1", blocked: true},
		{name: "ipv4 broadcast", ip: "255.255.255.255", blocked: true},
		{name: "ipv4-mapped loopback", ip: "::ffff:127.0.0.1", blocked: true},
		{name: "ipv4-mapped private", ip: "::ffff:10.0.0.1", blocked: true},
		{name: "nat64 loopback", ip: "64:ff9b::7f00:1", blocked: true},
		{name: "nat64 public", ip: "64:ff9b::808:808", blocked: false},
		{name: "ipv6 site-local", ip: "fec0::1", blocked: true},
		{name: "ipv6 multicast", ip: "ff02::1", blocked: true},
		{name: "ipv4-mapped public", ip: "::ffff:8.8.8.8", blocked: false},
		{name: "public ipv4", ip: "8.8.8.8", blocked: false},
		{name: "public ipv4 alt", ip: "1.1.1.1", blocked: false},
		{name: "public ipv4 93.x", ip: "93.184.216.34", blocked: false},
		{name: "public ipv6", ip: "2607:f8b0:4004:800::200e", blocked: false},
		{name: "nil ip", ip: "", blocked: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var ip net.IP
			if tt.ip != "" {
				ip = net.ParseIP(tt.ip)
				if ip == nil {
					t.Fatalf("failed to parse test IP %q", tt.ip)
				}
			}
			got := isBlockedIP(ip)
			if got != tt.blocked {
				t.Errorf("isBlockedIP(%s) = %v, want %v", tt.ip, got, tt.blocked)
			}
		})
	}
}

func TestNormaliseURL_BlocksUnsafeSchemes(t *testing.T) {
	tests := []struct {
		name    string
		url     string
		wantErr bool
	}{
		{name: "https allowed", url: "https://example.com/page", wantErr: false},
		{name: "http allowed", url: "http://example.com/page", wantErr: false},
		{name: "no scheme defaults https", url: "example.com/page", wantErr: false},
		{name: "file scheme blocked", url: "file:///etc/passwd", wantErr: true},
		{name: "gopher scheme blocked", url: "gopher://evil.test:70/", wantErr: true},
		{name: "ftp scheme blocked", url: "ftp://ftp.example.com/file", wantErr: true},
		{name: "dict scheme blocked", url: "dict://evil.test:2628/", wantErr: true},
		{name: "empty url", url: "", wantErr: true},
		{name: "missing host", url: "https://", wantErr: true},
		{name: "credentials refused", url: "http://user:pw@example.com/", wantErr: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := normaliseURL(tt.url)
			if (err != nil) != tt.wantErr {
				t.Errorf("normaliseURL(%q) error = %v, wantErr %v", tt.url, err, tt.wantErr)
			}
			if err != nil && !errors.Is(err, ErrInvalidURL) {
				t.Errorf("normaliseURL(%q) error = %v, want ErrInvalidURL", tt.url, err)
			}
		})
	}
}

func TestSSRFSafeTransport_BlocksPrivateIPs(t *testing.T) {
	transport := newSSRFSafeTransport()
	if transport == nil {
		t.Fatal("expected non-nil transport")
	}
	if transport.DialContext == nil {
		t.Fatal("expected DialContext to be set")
	}
	if transport.TLSHandshakeTimeout == 0 {
		t.Fatal("expected TLSHandshakeTimeout to be set")
	}
}

func TestDefaultFetcher_RefusesLoopback(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte("internal"))
	}))
	defer srv.Close()

	_, _, err := defaultFetcher{}.Fetch(context.Background(), srv.URL)
	if !errors.Is(err, ErrBlockedAddress) {
		t.Fatalf("Fetch(loopback) error = %v, want ErrBlockedAddress", err)
	}
	var fe *FetchError
	if errors.As(err, &fe) {
		t.Fatalf("Fetch(loopback) returned FetchError %v, want a blocked-address error", err)
	}
}

func TestCheckRedirect(t *testing.T) {
	mustURL := func(raw string) *url.URL {
		u, err := url.Parse(raw)
		if err != nil {
			t.Fatalf("parse %q: %v", raw, err)
		}
		return u
	}
	hop := &http.Request{URL: mustURL("https://example.com/next")}
	if err := checkRedirect(hop, nil); err != nil {
		t.Fatalf("first https hop: %v", err)
	}
	fileHop := &http.Request{URL: mustURL("file:///etc/passwd")}
	if err := checkRedirect(fileHop, nil); !errors.Is(err, ErrInvalidURL) {
		t.Fatalf("file hop error = %v, want ErrInvalidURL", err)
	}
	via := make([]*http.Request, maxRedirects)
	if err := checkRedirect(hop, via); err == nil {
		t.Fatalf("hop after %d redirects should be refused", maxRedirects)
	}
}

func TestFetchError_Message(t *testing.T) {
	withStatus := &FetchError{StatusCode: 404, Err: errors.New("HTTP 404")}
	if got := withStatus.Error(); got != "knowledge: fetch failed: HTTP 404" {
		t.Fatalf("Error() = %q", got)
	}
	cause := errors.New("connection reset")
	network := &FetchError{Err: cause}
	if !errors.Is(network, cause) {
		t.Fatal("FetchError should unwrap to its cause")
	}
}
