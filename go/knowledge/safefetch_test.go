// SPDX-License-Identifier: Apache-2.0

package knowledge

import (
	"net"
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
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := normaliseURL(tt.url)
			if (err != nil) != tt.wantErr {
				t.Errorf("normaliseURL(%q) error = %v, wantErr %v", tt.url, err, tt.wantErr)
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
