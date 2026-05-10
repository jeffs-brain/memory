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

// HashSlug computes the BLAKE3 hash of data and returns the first 12
// hex characters. Used as a deterministic fallback slug.
func HashSlug(data []byte) string {
	return HashContent(data)[:12]
}

// HashDocumentID computes a stable document identifier from brain ID
// and content hash. Returns the first 16 hex characters of
// BLAKE3(brainID + ":" + contentHash).
func HashDocumentID(brainID, contentHash string) string {
	combined := []byte(brainID + ":" + contentHash)
	sum := lukechampineblake3.Sum256(combined)
	return hex.EncodeToString(sum[:])[:16]
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
	sum := blake3.Sum256(content)
	return hex.EncodeToString(sum[:])
}

// HashString computes the BLAKE3 hash of a string input and returns the
// full hex-encoded 256-bit digest (64 characters). Convenience wrapper
// that avoids a []byte conversion at call sites.
func HashString(s string) string {
	sum := blake3.Sum256([]byte(s))
	return hex.EncodeToString(sum[:])
}
