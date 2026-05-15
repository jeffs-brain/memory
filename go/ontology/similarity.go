// SPDX-License-Identifier: Apache-2.0
package ontology

import (
	"fmt"
	"math"
	"strings"
)

// JaroWinklerDistance computes case-insensitive Jaro-Winkler similarity
// between two strings. Returns a value in [0, 1] where 1 indicates
// identical strings and 0 indicates completely dissimilar strings.
//
// This is a native implementation with no external dependencies.
//
// Time: O(max(len(s1), len(s2)))
// Space: O(max(len(s1), len(s2)))
func JaroWinklerDistance(s1, s2 string) float64 {
	a := strings.TrimSpace(strings.ToLower(s1))
	b := strings.TrimSpace(strings.ToLower(s2))
	if a == "" && b == "" {
		return 1.0
	}
	if a == "" || b == "" {
		return 0.0
	}
	if a == b {
		return 1.0
	}

	jaro := jaroDistance(a, b)

	ra := []rune(a)
	rb := []rune(b)
	commonPrefix := 0
	limit := 4
	if len(ra) < limit {
		limit = len(ra)
	}
	if len(rb) < limit {
		limit = len(rb)
	}
	for i := 0; i < limit; i++ {
		if ra[i] != rb[i] {
			break
		}
		commonPrefix++
	}

	return jaro + float64(commonPrefix)*0.1*(1.0-jaro)
}

// jaroDistance computes the Jaro similarity between two strings.
// Both strings must be non-empty and already lowercased/trimmed.
// Uses []rune internally for correct multi-byte character handling.
func jaroDistance(s1, s2 string) float64 {
	r1 := []rune(s1)
	r2 := []rune(s2)

	maxLen := len(r1)
	if len(r2) > maxLen {
		maxLen = len(r2)
	}
	matchWindow := maxLen/2 - 1
	if matchWindow < 0 {
		matchWindow = 0
	}

	r1Matches := make([]bool, len(r1))
	r2Matches := make([]bool, len(r2))

	matches := 0
	transpositions := 0

	for i := 0; i < len(r1); i++ {
		start := i - matchWindow
		if start < 0 {
			start = 0
		}
		end := i + matchWindow + 1
		if end > len(r2) {
			end = len(r2)
		}
		for j := start; j < end; j++ {
			if r2Matches[j] || r1[i] != r2[j] {
				continue
			}
			r1Matches[i] = true
			r2Matches[j] = true
			matches++
			break
		}
	}

	if matches == 0 {
		return 0.0
	}

	k := 0
	for i := 0; i < len(r1); i++ {
		if !r1Matches[i] {
			continue
		}
		for !r2Matches[k] {
			k++
		}
		if r1[i] != r2[k] {
			transpositions++
		}
		k++
	}

	return (float64(matches)/float64(len(r1)) +
		float64(matches)/float64(len(r2)) +
		float64(matches-transpositions/2)/float64(matches)) / 3.0
}

// CosineSimilarity computes the cosine similarity between two float32
// vectors. Returns a value in [-1, 1] where 1 indicates identical
// direction, 0 indicates orthogonal vectors, and -1 indicates opposite
// direction. Returns 0 when either vector has zero magnitude.
//
// Returns an error if the vectors have different lengths.
//
// Time: O(n) where n = len(a)
// Space: O(1)
func CosineSimilarity(a, b []float32) (float64, error) {
	if len(a) != len(b) {
		return 0, fmt.Errorf("ontology: cosine similarity requires equal-length vectors, got %d and %d", len(a), len(b))
	}
	if len(a) == 0 {
		return 0, nil
	}

	var dot, normA, normB float64
	for i := range a {
		ai := float64(a[i])
		bi := float64(b[i])
		dot += ai * bi
		normA += ai * ai
		normB += bi * bi
	}

	magA := math.Sqrt(normA)
	magB := math.Sqrt(normB)
	if magA == 0 || magB == 0 {
		return 0, nil
	}
	return dot / (magA * magB), nil
}
