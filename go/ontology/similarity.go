// SPDX-License-Identifier: Apache-2.0
package ontology

import (
	"fmt"
	"math"
	"strings"

	"github.com/xrash/smetrics"
)

// JaroWinklerDistance computes case-insensitive Jaro-Winkler similarity
// between two strings. Returns a value in [0, 1] where 1 indicates
// identical strings and 0 indicates completely dissimilar strings.
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
	return smetrics.JaroWinkler(a, b, 0.7, 4)
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
