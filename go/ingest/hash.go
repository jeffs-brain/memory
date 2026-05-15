// SPDX-License-Identifier: Apache-2.0
package ingest

import (
	"crypto/sha256"
	"encoding/hex"

	"lukechampine.com/blake3"
)

// HashContent computes the BLAKE3 hash of data and returns the full
// hex-encoded digest (64 characters).
func HashContent(data []byte) string {
	sum := blake3.Sum256(data)
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
	sum := blake3.Sum256(combined)
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
