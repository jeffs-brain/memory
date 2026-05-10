// SPDX-License-Identifier: Apache-2.0
package ingest

import (
	"crypto/sha256"
	"encoding/hex"
	"strings"
	"testing"
)

func TestHashDocument_Deterministic(t *testing.T) {
	t.Parallel()
	input := []byte("the quick brown fox jumps over the lazy dog")
	first := HashDocument(input)
	second := HashDocument(input)
	if first != second {
		t.Fatalf("expected deterministic output, got %q and %q", first, second)
	}
}

func TestHashDocument_DifferentInputs(t *testing.T) {
	t.Parallel()
	a := HashDocument([]byte("input alpha"))
	b := HashDocument([]byte("input beta"))
	if a == b {
		t.Fatalf("different inputs produced identical hash: %q", a)
	}
}

func TestHashDocument_OutputFormat(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name  string
		input []byte
	}{
		{"empty input", []byte{}},
		{"single byte", []byte{0x42}},
		{"short string", []byte("hello")},
		{"longer content", []byte(strings.Repeat("a", 4096))},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			hash := HashDocument(tc.input)
			if len(hash) != 64 {
				t.Fatalf("expected 64-char hex string, got %d chars: %q", len(hash), hash)
			}
			if _, err := hex.DecodeString(hash); err != nil {
				t.Fatalf("output is not valid hex: %q, err: %v", hash, err)
			}
		})
	}
}

func TestHashChunk_MatchesHashDocument(t *testing.T) {
	t.Parallel()
	input := []byte("chunk content for verification")
	docHash := HashDocument(input)
	chunkHash := HashChunk(input)
	if docHash != chunkHash {
		t.Fatalf("HashChunk and HashDocument diverge for same input: %q vs %q", chunkHash, docHash)
	}
}

func TestHashString_MatchesHashDocument(t *testing.T) {
	t.Parallel()
	s := "string input for verification"
	fromString := HashString(s)
	fromBytes := HashDocument([]byte(s))
	if fromString != fromBytes {
		t.Fatalf("HashString and HashDocument diverge: %q vs %q", fromString, fromBytes)
	}
}

func TestHashDocument_EmptyInput(t *testing.T) {
	t.Parallel()
	hash := HashDocument([]byte{})
	if len(hash) != 64 {
		t.Fatalf("expected 64-char hex for empty input, got %d chars", len(hash))
	}
	if _, err := hex.DecodeString(hash); err != nil {
		t.Fatalf("empty input hash is not valid hex: %v", err)
	}
}

func TestHashDocument_CrossSDKConformance(t *testing.T) {
	t.Parallel()
	// Known input/output pair for cross-SDK verification.
	// Input: "jeff's brain memory system"
	// Expected: BLAKE3 256-bit hex of that exact UTF-8 byte string.
	// The TS SDK test hardcodes the same expected value.
	input := []byte("jeff's brain memory system")
	hash := HashDocument(input)
	expected := "e311e54b56b26bfef4e5c8501f04c708f1e02233106022f58a7e94b728b7265c"
	if hash != expected {
		t.Fatalf("cross-SDK conformance failed:\n  got:  %s\n  want: %s", hash, expected)
	}
}

func BenchmarkBLAKE3_1KB(b *testing.B) {
	data := []byte(strings.Repeat("x", 1024))
	b.SetBytes(1024)
	b.ResetTimer()
	for b.Loop() {
		HashDocument(data)
	}
}

func BenchmarkBLAKE3_10KB(b *testing.B) {
	data := []byte(strings.Repeat("x", 10*1024))
	b.SetBytes(10 * 1024)
	b.ResetTimer()
	for b.Loop() {
		HashDocument(data)
	}
}

func BenchmarkBLAKE3_100KB(b *testing.B) {
	data := []byte(strings.Repeat("x", 100*1024))
	b.SetBytes(100 * 1024)
	b.ResetTimer()
	for b.Loop() {
		HashDocument(data)
	}
}

func BenchmarkSHA256_1KB(b *testing.B) {
	data := []byte(strings.Repeat("x", 1024))
	b.SetBytes(1024)
	b.ResetTimer()
	for b.Loop() {
		sum := sha256.Sum256(data)
		_ = hex.EncodeToString(sum[:])
	}
}

func BenchmarkSHA256_10KB(b *testing.B) {
	data := []byte(strings.Repeat("x", 10*1024))
	b.SetBytes(10 * 1024)
	b.ResetTimer()
	for b.Loop() {
		sum := sha256.Sum256(data)
		_ = hex.EncodeToString(sum[:])
	}
}

func BenchmarkSHA256_100KB(b *testing.B) {
	data := []byte(strings.Repeat("x", 100*1024))
	b.SetBytes(100 * 1024)
	b.ResetTimer()
	for b.Loop() {
		sum := sha256.Sum256(data)
		_ = hex.EncodeToString(sum[:])
	}
}
