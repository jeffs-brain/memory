// SPDX-License-Identifier: Apache-2.0

package search

import "strings"

// trailingSegment returns the final slash-delimited segment of a logical
// path, or the path itself if it has no slashes.
func trailingSegment(p string) string {
	if idx := strings.LastIndex(p, "/"); idx >= 0 {
		return p[idx+1:]
	}
	return p
}
