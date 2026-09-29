// SPDX-License-Identifier: Apache-2.0

package knowledge

import (
	"context"
	"errors"
	"fmt"
	"net"
	"net/http"
	"net/netip"
	"time"
)

// maxRedirects caps how many redirects a URL fetch follows. Every hop is
// re-dialled through the SSRF-safe transport.
const maxRedirects = 5

// ErrInvalidURL reports a URL that is malformed or uses a scheme other
// than http or https.
var ErrInvalidURL = errors.New("knowledge: invalid URL")

// ErrInvalidContent reports an ingest request whose body cannot be
// ingested: missing, not valid UTF-8, an unsupported content type, an
// unreadable PDF, a directory or an oversized file.
var ErrInvalidContent = errors.New("knowledge: invalid content")

// ErrBlockedAddress reports a URL whose host resolves to a loopback,
// private, link-local, multicast or otherwise non-public address.
var ErrBlockedAddress = errors.New("knowledge: URL resolves to a non-public address")

// FetchError reports an upstream failure while fetching a URL: a
// non-2xx status, a network error or an oversized body. StatusCode is
// zero when no HTTP response was received.
type FetchError struct {
	StatusCode int
	Err        error
}

func (e *FetchError) Error() string {
	if e.StatusCode > 0 {
		return fmt.Sprintf("knowledge: fetch failed: HTTP %d", e.StatusCode)
	}
	return fmt.Sprintf("knowledge: fetch failed: %v", e.Err)
}

func (e *FetchError) Unwrap() error { return e.Err }

// newSSRFSafeTransport returns an [http.Transport] whose DialContext
// resolves DNS then validates all returned IPs against a blocklist
// before establishing a connection, and dials the validated IP so a
// second resolution cannot swap in an internal address. This prevents
// SSRF attacks via user-controlled URLs that resolve to internal
// infrastructure, including through redirects and DNS rebinding.
func newSSRFSafeTransport() *http.Transport {
	return &http.Transport{
		DialContext: func(ctx context.Context, network, addr string) (net.Conn, error) {
			host, port, err := net.SplitHostPort(addr)
			if err != nil {
				return nil, fmt.Errorf("%w: %v", ErrInvalidURL, err)
			}
			ips, err := net.DefaultResolver.LookupIPAddr(ctx, host)
			if err != nil {
				return nil, fmt.Errorf("knowledge: dns lookup failed: %w", err)
			}
			if len(ips) == 0 {
				return nil, fmt.Errorf("knowledge: no addresses resolved for %s", host)
			}
			for _, ip := range ips {
				if isBlockedIP(ip.IP) {
					return nil, fmt.Errorf("%w: %s", ErrBlockedAddress, host)
				}
			}
			dialer := &net.Dialer{Timeout: 10 * time.Second}
			return dialer.DialContext(ctx, network, net.JoinHostPort(ips[0].IP.String(), port))
		},
		TLSHandshakeTimeout: 10 * time.Second,
	}
}

// checkRedirect limits the redirect chain and refuses a hop to any
// scheme other than http or https, or to a URL carrying credentials.
func checkRedirect(req *http.Request, via []*http.Request) error {
	if len(via) >= maxRedirects {
		return fmt.Errorf("knowledge: stopped after %d redirects", maxRedirects)
	}
	if req.URL.Scheme != "http" && req.URL.Scheme != "https" {
		return fmt.Errorf("%w: redirect to scheme %q", ErrInvalidURL, req.URL.Scheme)
	}
	if req.URL.User != nil {
		return fmt.Errorf("%w: redirect to a URL carrying credentials", ErrInvalidURL)
	}
	return nil
}

// blockedPrefixes lists every non-public range a URL fetch refuses. The
// same list is implemented by the TypeScript and Python SDKs.
var blockedPrefixes = []netip.Prefix{
	netip.MustParsePrefix("0.0.0.0/8"),
	netip.MustParsePrefix("10.0.0.0/8"),
	netip.MustParsePrefix("100.64.0.0/10"),
	netip.MustParsePrefix("127.0.0.0/8"),
	netip.MustParsePrefix("169.254.0.0/16"),
	netip.MustParsePrefix("172.16.0.0/12"),
	netip.MustParsePrefix("192.0.0.0/24"),
	netip.MustParsePrefix("192.168.0.0/16"),
	netip.MustParsePrefix("198.18.0.0/15"),
	netip.MustParsePrefix("224.0.0.0/4"),
	netip.MustParsePrefix("240.0.0.0/4"),
	netip.MustParsePrefix("::/128"),
	netip.MustParsePrefix("::1/128"),
	netip.MustParsePrefix("fc00::/7"),
	netip.MustParsePrefix("fe80::/10"),
	netip.MustParsePrefix("fec0::/10"),
	netip.MustParsePrefix("ff00::/8"),
}

// nat64Prefix is the well-known NAT64 range; its low 32 bits embed an
// IPv4 address that is checked against the IPv4 blocklist.
var nat64Prefix = netip.MustParsePrefix("64:ff9b::/96")

// isBlockedIP returns true when the IP belongs to a loopback, private,
// carrier-grade NAT, link-local, multicast, reserved or otherwise
// non-routable range. IPv4-mapped and NAT64 IPv6 addresses are checked
// against the embedded IPv4 address. The cloud metadata endpoint at
// 169.254.169.254 falls inside the link-local range.
func isBlockedIP(ip net.IP) bool {
	addr, ok := netip.AddrFromSlice(ip)
	if !ok {
		return true
	}
	addr = addr.Unmap()
	if addr.Is6() && nat64Prefix.Contains(addr) {
		raw := addr.As16()
		addr = netip.AddrFrom4([4]byte{raw[12], raw[13], raw[14], raw[15]})
	}
	for _, prefix := range blockedPrefixes {
		if prefix.Contains(addr) {
			return true
		}
	}
	return false
}
