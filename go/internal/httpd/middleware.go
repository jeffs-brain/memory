// SPDX-License-Identifier: Apache-2.0

package httpd

import (
	"crypto/sha256"
	"crypto/subtle"
	"log/slog"
	"net/http"
	"strings"
	"time"
)

// LogMiddleware returns a handler that logs each request to log.
func LogMiddleware(log *slog.Logger, next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		wrapped := &statusRecorder{ResponseWriter: w, status: http.StatusOK}
		next.ServeHTTP(wrapped, r)
		log.Info("http",
			"method", r.Method,
			"path", r.URL.Path,
			"status", wrapped.status,
			"elapsed_ms", time.Since(start).Milliseconds(),
		)
	})
}

// AuthMiddleware enforces a shared bearer token on every request when
// the token is non-empty. /healthz is always allowed through so
// liveness probes never need credentials. When the token is empty
// the middleware is a no-op pass-through.
func AuthMiddleware(token string, next http.Handler) http.Handler {
	if token == "" {
		return next
	}
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/healthz" {
			next.ServeHTTP(w, r)
			return
		}
		got := r.Header.Get("Authorization")
		if got == "" {
			Unauthorized(w, "missing Authorization header")
			return
		}
		if !validBearerToken(got, token) {
			Forbidden(w, "invalid bearer token")
			return
		}
		next.ServeHTTP(w, r)
	})
}

const bearerScheme = "bearer"

// validBearerToken checks that header is a Bearer token matching expected.
// The scheme comparison is case-insensitive per RFC 7235 and accepts
// one-or-more spaces between scheme and token per the RFC grammar.
// Token comparison hashes both sides with SHA-256 before calling
// ConstantTimeCompare so that differing lengths do not leak timing.
func validBearerToken(header, expected string) bool {
	if len(header) < len(bearerScheme)+1 {
		return false
	}
	if !strings.EqualFold(header[:len(bearerScheme)], bearerScheme) {
		return false
	}
	rest := header[len(bearerScheme):]
	if len(rest) == 0 || rest[0] != ' ' {
		return false
	}
	actual := strings.TrimLeft(rest, " ")
	if len(actual) == 0 {
		return false
	}
	actualHash := sha256.Sum256([]byte(actual))
	expectedHash := sha256.Sum256([]byte(expected))
	return subtle.ConstantTimeCompare(actualHash[:], expectedHash[:]) == 1
}

// Implements http.ResponseWriter / http.Flusher passthrough so SSE
// handlers can flush through the wrapper.
type statusRecorder struct {
	http.ResponseWriter
	status int
}

func (r *statusRecorder) WriteHeader(code int) {
	r.status = code
	r.ResponseWriter.WriteHeader(code)
}

// Flush forwards to the wrapped writer when it implements Flusher.
func (r *statusRecorder) Flush() {
	if f, ok := r.ResponseWriter.(http.Flusher); ok {
		f.Flush()
	}
}
