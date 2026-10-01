// SPDX-License-Identifier: Apache-2.0

package httpd

import (
	"errors"
	"fmt"
	"net"
	"net/http"
	"net/url"
	"strings"
)

// DefaultAddr is the bind address used when none is configured. The
// daemon listens on loopback only unless told otherwise.
const DefaultAddr = "127.0.0.1:8080"

// ErrUnauthenticatedBind reports an attempt to listen on a non-loopback
// interface without a bearer token.
var ErrUnauthenticatedBind = errors.New("httpd: refusing to listen on a non-loopback address without an auth token")

// ResolveBindAddr normalises addr and enforces the bind policy from
// spec/PROTOCOL.md: an empty host binds loopback, and a non-loopback
// host requires a non-empty token.
func ResolveBindAddr(addr, token string) (string, error) {
	addr = strings.TrimSpace(addr)
	if addr == "" {
		addr = DefaultAddr
	}
	host, port, err := net.SplitHostPort(addr)
	if err != nil {
		return "", fmt.Errorf("httpd: invalid address %q: %w", addr, err)
	}
	if host == "" {
		host = "127.0.0.1"
	}
	resolved := net.JoinHostPort(host, port)
	if token == "" && !IsLoopbackHost(host) {
		return "", fmt.Errorf("%w: %s (set an auth token or bind to 127.0.0.1)", ErrUnauthenticatedBind, resolved)
	}
	return resolved, nil
}

// IsLoopbackHost reports whether host is "localhost" or a loopback IP
// literal. Brackets around an IPv6 literal are tolerated.
func IsLoopbackHost(host string) bool {
	host = strings.TrimSuffix(strings.TrimPrefix(host, "["), "]")
	if strings.EqualFold(host, "localhost") {
		return true
	}
	ip := net.ParseIP(host)
	return ip != nil && ip.IsLoopback()
}

// LoopbackGuard rejects requests that did not come from a local client
// addressing the daemon by a loopback name. It closes the DNS
// rebinding and cross-site request routes into an unauthenticated
// daemon: the Host header must name a loopback host (421 otherwise) and
// a browser Origin, when present, must be a loopback origin (403
// otherwise). /healthz is always allowed through.
func LoopbackGuard(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/healthz" {
			next.ServeHTTP(w, r)
			return
		}
		if !IsLoopbackHost(hostOnly(r.Host)) {
			MisdirectedRequest(w, "unauthenticated daemon only serves loopback hosts")
			return
		}
		if origin := r.Header.Get("Origin"); origin != "" && !isLoopbackOrigin(origin) {
			Forbidden(w, "cross-origin requests are refused by an unauthenticated daemon")
			return
		}
		next.ServeHTTP(w, r)
	})
}

func hostOnly(hostport string) string {
	if host, _, err := net.SplitHostPort(hostport); err == nil {
		return host
	}
	return hostport
}

func isLoopbackOrigin(origin string) bool {
	u, err := url.Parse(origin)
	if err != nil || (u.Scheme != "http" && u.Scheme != "https") {
		return false
	}
	return IsLoopbackHost(u.Hostname())
}
