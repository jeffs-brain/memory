// SPDX-License-Identifier: Apache-2.0

package connector

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"
	"time"
)

// doNotionRequest performs a single Notion API request with proper
// headers and timeout. Returns the response body or an error.
// Retries automatically on 429 responses using the Retry-After
// header with exponential backoff.
func (c *NotionConnector) doNotionRequest(ctx context.Context, method, path string, body any) ([]byte, error) {
	for attempt := 0; attempt <= notionMaxRetryAttempts; attempt++ {
		// Ensure token is valid before each request attempt.
		if tokenErr := c.ensureValidToken(ctx); tokenErr != nil {
			return nil, tokenErr
		}

		if err := c.rateLimiter.Acquire(ctx, 1); err != nil {
			return nil, err
		}

		respBody, retryAfter, err := c.doSingleRequest(ctx, method, path, body)
		if err == nil {
			return respBody, nil
		}

		// Only retry on 429 (rate limited).
		if retryAfter == notionNoRetry {
			return nil, err
		}

		if attempt >= notionMaxRetryAttempts {
			return nil, err
		}

		// Use Retry-After header value, or fall back to exponential
		// backoff with jitter.
		var wait time.Duration
		if retryAfter >= 0 {
			wait = time.Duration(retryAfter) * time.Second
		} else {
			wait = c.rateLimiter.BackoffDuration(attempt)
		}

		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-time.After(wait):
			// continue to retry
		}
	}

	return nil, fmt.Errorf("connector/notion: exhausted retries for %s %s", method, path)
}

// notionNoRetry is the sentinel value for doSingleRequest's
// retryAfter return, indicating the error is not retryable.
const notionNoRetry = -2

// doSingleRequest executes one HTTP request. Returns the response body
// on success, or an error. The retryAfter return value is:
//   - >= 0  if a 429 was received and Retry-After header had a value
//   - -1    if a 429 was received but no Retry-After header (use backoff)
//   - notionNoRetry (-2) for all other errors (do not retry)
func (c *NotionConnector) doSingleRequest(ctx context.Context, method, path string, body any) ([]byte, int, error) {
	var bodyReader io.Reader
	if body != nil {
		data, err := json.Marshal(body)
		if err != nil {
			return nil, notionNoRetry, fmt.Errorf("connector/notion: marshalling request body: %w", err)
		}
		bodyReader = strings.NewReader(string(data))
	}

	url := notionBaseURL + path
	// Override base URL if set via context (for tests).
	if baseURL, ok := ctx.Value(notionBaseURLKey{}).(string); ok {
		url = baseURL + path
	}

	reqCtx, cancel := context.WithTimeout(ctx, notionDefaultTimeout)
	defer cancel()

	req, err := http.NewRequestWithContext(reqCtx, method, url, bodyReader)
	if err != nil {
		return nil, notionNoRetry, fmt.Errorf("connector/notion: creating request: %w", err)
	}

	c.mu.Lock()
	apiToken := c.config.APIToken
	c.mu.Unlock()
	req.Header.Set("Authorization", "Bearer "+apiToken)
	req.Header.Set("Notion-Version", notionAPIVersion)
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, notionNoRetry, fmt.Errorf("connector/notion: request failed: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(io.LimitReader(resp.Body, notionMaxResponseSize))
	if err != nil {
		return nil, notionNoRetry, fmt.Errorf("connector/notion: reading response: %w", err)
	}

	if resp.StatusCode == http.StatusTooManyRequests {
		retryAfter := parseRetryAfterHeader(resp.Header.Get("Retry-After"))
		return nil, retryAfter, fmt.Errorf("connector/notion: rate limited (429)")
	}

	if resp.StatusCode == http.StatusNotFound {
		return nil, notionNoRetry, fmt.Errorf("connector/notion: not found (404)")
	}

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, notionNoRetry, fmt.Errorf("connector/notion: HTTP %d: %s", resp.StatusCode, string(respBody))
	}

	return respBody, notionNoRetry, nil
}

// parseRetryAfterHeader parses the Retry-After header value as an
// integer number of seconds. Returns -1 if the header is absent or
// unparseable (indicating exponential backoff should be used).
func parseRetryAfterHeader(value string) int {
	if value == "" {
		return -1
	}
	seconds, err := strconv.Atoi(value)
	if err != nil {
		return -1
	}
	return seconds
}

// notionBaseURLKey is a context key for overriding the Notion API
// base URL in tests.
type notionBaseURLKey struct{}

// WithNotionBaseURL returns a context with an overridden Notion API
// base URL.
func WithNotionBaseURL(ctx context.Context, baseURL string) context.Context {
	return context.WithValue(ctx, notionBaseURLKey{}, baseURL)
}
