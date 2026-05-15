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

func TestHashSlug_Returns12HexChars(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name  string
		input []byte
	}{
		{"empty", []byte{}},
		{"short", []byte("hello")},
		{"long", []byte(strings.Repeat("a", 4096))},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			slug := HashSlug(tc.input)
			if len(slug) != 12 {
				t.Fatalf("expected 12 chars, got %d: %q", len(slug), slug)
			}
			if _, err := hex.DecodeString(slug + "0000"); err != nil {
				// Validate it's valid hex by padding to even length check.
				for _, c := range slug {
					if !((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f')) {
						t.Fatalf("slug contains non-hex char: %q", slug)
					}
				}
			}
		})
	}
}

func TestHashSlug_IsPrefixOfHashDocument(t *testing.T) {
	t.Parallel()
	input := []byte("consistency check between HashSlug and HashDocument")
	full := HashDocument(input)
	slug := HashSlug(input)
	if full[:12] != slug {
		t.Fatalf("HashSlug is not a prefix of HashDocument: %q vs %q", slug, full[:12])
	}
}

func TestHashDocumentID_Returns16HexChars(t *testing.T) {
	t.Parallel()
	id := HashDocumentID("brain-1", "abc123")
	if len(id) != 16 {
		t.Fatalf("expected 16 chars, got %d: %q", len(id), id)
	}
	if _, err := hex.DecodeString(id); err != nil {
		t.Fatalf("output is not valid hex: %q, err: %v", id, err)
	}
}

func TestHashDocumentID_IsBrainScoped(t *testing.T) {
	t.Parallel()
	contentHash := "deadbeefcafebabe"
	idA := HashDocumentID("brain-alpha", contentHash)
	idB := HashDocumentID("brain-beta", contentHash)
	if idA == idB {
		t.Fatalf("different brains produced identical document IDs: %q", idA)
	}
}

func TestHashDocumentID_DifferentBrainsDifferentIDs(t *testing.T) {
	t.Parallel()
	contentHash := HashDocument([]byte("shared document content"))
	ids := make(map[string]string, 5)
	brains := []string{"brain-a", "brain-b", "brain-c", "brain-d", "brain-e"}
	for _, b := range brains {
		id := HashDocumentID(b, contentHash)
		if existing, ok := ids[id]; ok {
			t.Fatalf("collision: brain %q and %q both produced ID %q", existing, b, id)
		}
		ids[id] = b
	}
}

func TestHasher_BLAKE3HasherConformance(t *testing.T) {
	t.Parallel()
	var h Hasher = BLAKE3Hasher{}
	if h.Name() != "blake3" {
		t.Fatalf("expected name 'blake3', got %q", h.Name())
	}
	input := []byte("hasher interface test")
	got := h.Hash(input)
	want := HashDocument(input)
	if got != want {
		t.Fatalf("BLAKE3Hasher.Hash differs from HashDocument: %q vs %q", got, want)
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
