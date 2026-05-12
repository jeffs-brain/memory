// SPDX-License-Identifier: Apache-2.0

package memory

// stemToken applies simple English suffix stripping to improve recall
// in text-overlap comparisons.
func stemToken(token string) string {
	n := len(token)
	if n > 5 && len(token) >= 4 && token[n-3:] == "ies" {
		return token[:n-3] + "y"
	}
	if n > 5 && token[n-2:] == "es" {
		return token[:n-2]
	}
	if n > 4 && token[n-1] == 's' && (n < 2 || token[n-2] != 's') {
		return token[:n-1]
	}
	return token
}

// countOverlap counts how many tokens from left appear in right.
func countOverlap(left, right []string) int {
	if len(left) == 0 || len(right) == 0 {
		return 0
	}
	rightSet := make(map[string]struct{}, len(right))
	for _, t := range right {
		rightSet[t] = struct{}{}
	}
	count := 0
	for _, t := range left {
		if _, ok := rightSet[t]; ok {
			count++
		}
	}
	return count
}
