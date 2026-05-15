// SPDX-License-Identifier: Apache-2.0
package ingest

import (
	"crypto/sha256"
	"encoding/hex"

	"github.com/zeebo/blake3"
	lukechampineblake3 "lukechampine.com/blake3"
)

// HashContent computes the BLAKE3 hash of data and returns the full
// hex-encoded digest (64 characters).
func HashContent(data []byte) string {
	sum := lukechampineblake3.Sum256(data)
	return hex.EncodeToString(sum[:])
}

// HashContentSHA256 computes the SHA-256 hash of data and returns the
// full hex-encoded digest. Used during migration to look up documents
// stored under the legacy SHA-256 scheme.
func HashContentSHA256(data []byte) string {
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:])
}

// HashSlugSHA256 computes the SHA-256 hash of data and returns the
// first 12 hex characters. Matches the legacy hashSlug function from
// knowledge/ingest.go.
func HashSlugSHA256(data []byte) string {
	return HashContentSHA256(data)[:12]
}

// Hasher abstracts a content-hashing algorithm so callers can swap
// implementations (BLAKE3, SHA-256, etc.) without changing call sites.
type Hasher interface {
	// Hash computes a hex-encoded digest of content.
	Hash(content []byte) string
	// Name returns a short identifier for the algorithm (e.g. "blake3").
	Name() string
}

// BLAKE3Hasher implements [Hasher] using the BLAKE3 algorithm with a
// 256-bit output, hex-encoded to 64 lowercase characters.
type BLAKE3Hasher struct{}

// Hash implements [Hasher].
func (BLAKE3Hasher) Hash(content []byte) string {
	sum := blake3.Sum256(content)
	return hex.EncodeToString(sum[:])
}

// Name implements [Hasher].
func (BLAKE3Hasher) Name() string { return "blake3" }

// DefaultHasher is the package-level Hasher used by all public hash
// functions. Defaults to BLAKE3.
var DefaultHasher Hasher = BLAKE3Hasher{}

// HashDocument computes the BLAKE3 hash of document content and returns
// the full hex-encoded 256-bit digest (64 characters). Used for document
// deduplication in the ingest pipeline.
func HashDocument(content []byte) string {
	sum := blake3.Sum256(content)
	return hex.EncodeToString(sum[:])
}

// HashChunk computes the BLAKE3 hash of chunk content and returns the
// full hex-encoded 256-bit digest (64 characters). Semantically identical
// to HashDocument but named separately for clarity at call sites where
// chunk-level change detection is the intent.
func HashChunk(content []byte) string {
	return HashDocument(content)
}

// HashString computes the BLAKE3 hash of a string input and returns the
// full hex-encoded 256-bit digest (64 characters). Convenience wrapper
// that avoids a []byte conversion at call sites.
func HashString(s string) string {
	sum := blake3.Sum256([]byte(s))
	return hex.EncodeToString(sum[:])
}

// HashSlug computes the BLAKE3 hash of data and returns the first 12
// hex characters. Used as a deterministic fallback slug when a
// human-readable slug is unavailable.
func HashSlug(data []byte) string {
	return HashDocument(data)[:12]
}

// HashDocumentID computes a stable, brain-scoped document identifier
// from a brain ID and a content hash. Returns the first 16 hex
// characters of BLAKE3(brainID + ":" + contentHash).
func HashDocumentID(brainID, contentHash string) string {
	input := []byte(brainID + ":" + contentHash)
	return HashDocument(input)[:16]
}
