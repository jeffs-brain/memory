// SPDX-License-Identifier: Apache-2.0
package ontology

import (
	"math"
	"strings"
	"testing"
)

func TestJaroWinklerDistance(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name     string
		s1       string
		s2       string
		wantMin  float64
		wantMax  float64
		wantExac float64
	}{
		{"identical strings", "Customer", "Customer", 1.0, 1.0, 1.0},
		{"case insensitive", "ABC", "abc", 1.0, 1.0, 1.0},
		{"similar strings", "Customer", "Customers", 0.9, 1.0, -1},
		{"different strings", "Customer", "Product", 0.0, 0.7, -1},
		{"empty both", "", "", 1.0, 1.0, 1.0},
		{"empty first", "", "hello", 0.0, 0.0, 0.0},
		{"empty second", "hello", "", 0.0, 0.0, 0.0},
		{"whitespace trimmed", "  hello  ", "hello", 1.0, 1.0, 1.0},
		{"singular vs plural", "customer_record", "customer_records", 0.9, 1.0, -1},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			got := JaroWinklerDistance(tc.s1, tc.s2)
			if tc.wantExac >= 0 {
				if math.Abs(got-tc.wantExac) > 1e-10 {
					t.Fatalf("JaroWinklerDistance(%q, %q) = %f, want %f", tc.s1, tc.s2, got, tc.wantExac)
				}
				return
			}
			if got < tc.wantMin || got > tc.wantMax {
				t.Fatalf("JaroWinklerDistance(%q, %q) = %f, want [%f, %f]", tc.s1, tc.s2, got, tc.wantMin, tc.wantMax)
			}
		})
	}
}

func TestCosineSimilarity(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name string
		a    []float32
		b    []float32
		want float64
	}{
		{"identical vectors", []float32{1, 0, 0}, []float32{1, 0, 0}, 1.0},
		{"orthogonal vectors", []float32{1, 0, 0}, []float32{0, 1, 0}, 0.0},
		{"opposite vectors", []float32{1, 0, 0}, []float32{-1, 0, 0}, -1.0},
		{"zero first vector", []float32{0, 0, 0}, []float32{1, 0, 0}, 0.0},
		{"zero second vector", []float32{1, 0, 0}, []float32{0, 0, 0}, 0.0},
		{"both zero vectors", []float32{0, 0, 0}, []float32{0, 0, 0}, 0.0},
		{"45 degrees", []float32{1, 1, 0}, []float32{1, 0, 0}, 1.0 / math.Sqrt(2)},
		{"empty vectors", []float32{}, []float32{}, 0.0},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			got, err := CosineSimilarity(tc.a, tc.b)
			if err != nil {
				t.Fatalf("CosineSimilarity(%v, %v) unexpected error: %v", tc.a, tc.b, err)
			}
			if math.Abs(got-tc.want) > 1e-7 {
				t.Fatalf("CosineSimilarity(%v, %v) = %f, want %f", tc.a, tc.b, got, tc.want)
			}
		})
	}
}

func TestCosineSimilarity_ErrorOnMismatch(t *testing.T) {
	t.Parallel()
	_, err := CosineSimilarity([]float32{1, 0}, []float32{1, 0, 0})
	if err == nil {
		t.Fatal("expected error on mismatched vector lengths, got nil")
	}
	if !strings.Contains(err.Error(), "equal-length vectors") {
		t.Fatalf("expected error about equal-length vectors, got: %v", err)
	}
}
