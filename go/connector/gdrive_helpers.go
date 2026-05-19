// SPDX-License-Identifier: Apache-2.0

package connector

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand/v2"
	"net/http"
	"regexp"
	"strconv"
	"time"
)

// folderIDPattern matches valid Google Drive folder IDs (alphanumeric, hyphens, underscores).
var folderIDPattern = regexp.MustCompile(`^[a-zA-Z0-9_-]+$`)

// mimeTypePattern matches valid MIME type strings.
var mimeTypePattern = regexp.MustCompile(`^[a-zA-Z0-9][a-zA-Z0-9!#$&\-.^_+]+/[a-zA-Z0-9][a-zA-Z0-9!#$&\-.^_+]+$`)

// isGoogleNativeFormat reports whether the MIME type is a Google-native
// document format that must be exported rather than downloaded.
func isGoogleNativeFormat(mime string) bool {
	switch mime {
	case mimeGoogleDoc, mimeGoogleSheet, mimeGoogleSlides, mimeGoogleDrawing:
		return true
	}
	return false
}

// parseFileSize parses the file size string from the Drive API. Returns
// 0 for empty or unparsable values (Google-native formats omit size).
func parseFileSize(sizeStr string) int64 {
	if sizeStr == "" {
		return 0
	}
	size, err := strconv.ParseInt(sizeStr, 10, 64)
	if err != nil {
		return 0
	}
	return size
}

// buildFileMetadata constructs the metadata map for a Drive file.
func buildFileMetadata(f *driveFile) map[string]string {
	meta := map[string]string{
		"source":    "gdrive",
		"mime_type": f.MIMEType,
		"file_id":   f.ID,
	}
	if len(f.Parents) > 0 {
		meta["parent_id"] = f.Parents[0]
	}
	if f.Size != "" {
		meta["size"] = f.Size
	}
	return meta
}

// truncateBody returns at most 200 bytes of a response body for error
// messages, avoiding excessively long error strings.
func truncateBody(body []byte) string {
	const maxLen = 200
	if len(body) <= maxLen {
		return string(body)
	}
	return string(body[:maxLen]) + "..."
}

// isRateLimitError checks whether the error response body indicates a
// rate-limit error that should be retried.
func isRateLimitError(body []byte, statusCode int) bool {
	if statusCode == http.StatusTooManyRequests {
		return true
	}

	var apiErr struct {
		Error struct {
			Errors []struct {
				Reason string `json:"reason"`
			} `json:"errors"`
		} `json:"error"`
	}
	if err := json.Unmarshal(body, &apiErr); err != nil {
		return false
	}

	rateLimitReasons := map[string]bool{
		"rateLimitExceeded":     true,
		"userRateLimitExceeded": true,
	}
	for _, e := range apiErr.Error.Errors {
		if rateLimitReasons[e.Reason] {
			return true
		}
	}
	return false
}

// calculateGoBackoff computes the backoff duration with exponential
// increase and jitter, respecting the Retry-After header when present.
func calculateGoBackoff(attempt int, retryAfter string) time.Duration {
	if retryAfter != "" {
		seconds, err := strconv.ParseFloat(retryAfter, 64)
		if err == nil && seconds > 0 {
			d := time.Duration(seconds * float64(time.Second))
			if d > rateLimitMaxBackoff {
				return rateLimitMaxBackoff
			}
			return d
		}
	}

	exponential := float64(rateLimitBaseBackoff) * math.Pow(2, float64(attempt))
	jitter := rand.Float64() * float64(rateLimitBaseBackoff)
	backoff := time.Duration(exponential + jitter)

	if backoff > rateLimitMaxBackoff {
		return rateLimitMaxBackoff
	}
	return backoff
}

// bearerTokenPattern matches Bearer tokens in error messages.
var bearerTokenPattern = regexp.MustCompile(`Bearer\s+[A-Za-z0-9\-._~+/]+=*`)

// sanitiseGoError strips access tokens from error messages to prevent
// credential leakage through error wrapping.
func sanitiseGoError(err error) error {
	msg := err.Error()
	sanitised := bearerTokenPattern.ReplaceAllString(msg, "Bearer [REDACTED]")
	if sanitised != msg {
		return fmt.Errorf("%s", sanitised)
	}
	return err
}
