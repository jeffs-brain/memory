// SPDX-License-Identifier: Apache-2.0

package connector

import (
	"fmt"
	"net"
	"net/url"
)

// validateDownloadURL checks that a URL is safe to fetch by resolving
// its hostname and verifying none of the resulting IPs belong to
// private, loopback, link-local, or cloud-metadata address ranges.
// This prevents SSRF attacks via crafted download URLs.
func validateDownloadURL(rawURL string) error {
	parsed, err := url.Parse(rawURL)
	if err != nil {
		return fmt.Errorf("slack: invalid download URL: %w", err)
	}

	scheme := parsed.Scheme
	if scheme != "https" && scheme != "http" {
		return fmt.Errorf("slack: blocked URL scheme %q (only http/https allowed)", scheme)
	}

	host := parsed.Hostname()
	if host == "" {
		return fmt.Errorf("slack: download URL has no hostname")
	}

	// Resolve the hostname to IP addresses.
	ips, err := net.LookupIP(host)
	if err != nil {
		return fmt.Errorf("slack: dns lookup for %s failed: %w", host, err)
	}
	if len(ips) == 0 {
		return fmt.Errorf("slack: no addresses resolved for %s", host)
	}

	for _, ip := range ips {
		if isBlockedIP(ip) {
			return fmt.Errorf("slack: request to private/internal network blocked for %s", host)
		}
	}

	return nil
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
