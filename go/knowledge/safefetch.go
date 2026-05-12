// SPDX-License-Identifier: Apache-2.0

package knowledge

import (
	"context"
	"fmt"
	"net"
	"net/http"
	"time"
)

// newSSRFSafeTransport returns an [http.Transport] whose DialContext
// resolves DNS then validates all returned IPs against a blocklist
// before establishing a connection. This prevents SSRF attacks via
// user-controlled URLs that resolve to internal infrastructure.
func newSSRFSafeTransport() *http.Transport {
	return &http.Transport{
		DialContext: func(ctx context.Context, network, addr string) (net.Conn, error) {
			host, port, err := net.SplitHostPort(addr)
			if err != nil {
				return nil, fmt.Errorf("ingest: invalid address: %w", err)
			}
			ips, err := net.DefaultResolver.LookupIPAddr(ctx, host)
			if err != nil {
				return nil, fmt.Errorf("ingest: dns lookup failed: %w", err)
			}
			if len(ips) == 0 {
				return nil, fmt.Errorf("ingest: no addresses resolved for %s", host)
			}
			for _, ip := range ips {
				if isBlockedIP(ip.IP) {
					return nil, fmt.Errorf("ingest: request to private network blocked")
				}
			}
			dialer := &net.Dialer{Timeout: 10 * time.Second}
			return dialer.DialContext(ctx, network, net.JoinHostPort(ips[0].IP.String(), port))
		},
		TLSHandshakeTimeout: 10 * time.Second,
	}
}

// isBlockedIP returns true when the IP belongs to a private, loopback,
// link-local, or otherwise non-routable address range. Also blocks the
// cloud metadata endpoint at 169.254.169.254.
func isBlockedIP(ip net.IP) bool {
	if ip == nil {
		return true
	}
	return ip.IsLoopback() ||
		ip.IsPrivate() ||
		ip.IsLinkLocalUnicast() ||
		ip.IsLinkLocalMulticast() ||
		ip.IsUnspecified() ||
		ip.Equal(net.IPv4(169, 254, 169, 254))
}
