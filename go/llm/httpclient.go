// SPDX-License-Identifier: Apache-2.0

package llm

import (
	"net/http"
	"time"
)

// Transport-level timeouts for the default HTTP client shared by all LLM
// providers. ResponseHeaderTimeout governs only the wait for the first
// response headers, so long-lived SSE streams are not killed once data
// starts flowing.
const (
	defaultResponseHeaderTimeout = 30 * time.Second
	defaultTLSHandshakeTimeout   = 10 * time.Second
	defaultIdleConnTimeout       = 90 * time.Second
	defaultMaxIdleConns          = 100
	defaultMaxIdleConnsPerHost   = 10
)

// newDefaultClient returns an *http.Client with sensible transport-level
// timeouts. It intentionally avoids setting http.Client.Timeout because
// that applies to the entire request lifecycle including body reads, which
// would kill long-running SSE streams.
func newDefaultClient() *http.Client {
	return &http.Client{
		Transport: &http.Transport{
			ResponseHeaderTimeout: defaultResponseHeaderTimeout,
			TLSHandshakeTimeout:   defaultTLSHandshakeTimeout,
			MaxIdleConns:          defaultMaxIdleConns,
			MaxIdleConnsPerHost:   defaultMaxIdleConnsPerHost,
			IdleConnTimeout:       defaultIdleConnTimeout,
		},
	}
}
