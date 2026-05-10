// SPDX-License-Identifier: Apache-2.0
package knowledge

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"regexp"
	"time"

	"github.com/jeffs-brain/memory/go/brain"
)

// validDocumentHashRe validates that a document hash is a lowercase hex
// string (SHA-256 or BLAKE3 output). Prevents path traversal via
// crafted hash values.
var validDocumentHashRe = regexp.MustCompile(`^[a-f0-9]+$`)

// chunkManifestsPrefix is the logical prefix where chunk manifests are
// persisted. Each document gets one manifest keyed by its content hash.
const chunkManifestsPrefix = "raw/.chunk-manifests"

// ChunkManifestEntry records the content hash of a single chunk and its
// positional identifier within the parent document. Matching during delta
// detection is by Hash (content-based), not by ChunkID (ordinal-based).
type ChunkManifestEntry struct {
	Hash    string `json:"hash"`
	ChunkID string `json:"chunkId"`
}

// ChunkManifest records all chunk hashes for a document. Used by delta
// detection to skip unchanged chunks on re-ingest. The Generation counter
// increments on every successful write, enabling external consumers to
// detect stale manifests.
type ChunkManifest struct {
	DocumentHash string               `json:"documentHash"`
	Generation   int                  `json:"generation"`
	Chunks       []ChunkManifestEntry `json:"chunks"`
	UpdatedAt    time.Time            `json:"updatedAt"`
}

// DeltaCategory classifies a chunk's change status during re-ingest.
// Matching is by BLAKE3/SHA-256 content hash regardless of position.
type DeltaCategory int

const (
	// DeltaUnchanged means the chunk's content hash exists in both the
	// prior manifest and the current chunk set.
	DeltaUnchanged DeltaCategory = iota
	// DeltaNew means the chunk's content hash exists in the current set
	// but not in the prior manifest.
	DeltaNew
	// DeltaRemoved means the chunk's content hash exists in the prior
	// manifest but not in the current set.
	DeltaRemoved
)

// ChunkDelta pairs a chunk with its change category and content hash.
type ChunkDelta struct {
	Chunk    Chunk
	Category DeltaCategory
	Hash     string
}

// HashChunk computes a SHA-256 content hash of a chunk's text. This is
// the canonical hashing function for chunk-level change detection. When
// P1-2 lands BLAKE3 support, this function becomes the single point of
// swap.
func HashChunk(text string) string {
	sum := sha256.Sum256([]byte(text))
	return hex.EncodeToString(sum[:])
}

// ComputeChunkDeltas compares new chunks against a stored manifest and
// returns the delta set. Matching is by content hash regardless of
// position: hash in old manifest but not in new = DeltaRemoved, hash in
// new but not in old = DeltaNew, hash in both = DeltaUnchanged.
//
// When manifest is nil (first ingest), all chunks are categorized as
// DeltaNew. Removed chunks carry a zero-value Chunk with only the
// DocumentID and Ordinal set from the prior manifest entry.
//
// Time: O(N + M) where N = len(newChunks), M = len(manifest.Chunks).
// Space: O(N + M) for the hash lookup maps.
func ComputeChunkDeltas(newChunks []Chunk, manifest *ChunkManifest) []ChunkDelta {
	if len(newChunks) == 0 && manifest == nil {
		return nil
	}

	newHashes := make(map[string][]int, len(newChunks))
	for i, chunk := range newChunks {
		h := HashChunk(chunk.Text)
		newHashes[h] = append(newHashes[h], i)
	}

	if manifest == nil {
		out := make([]ChunkDelta, 0, len(newChunks))
		for i, chunk := range newChunks {
			out = append(out, ChunkDelta{
				Chunk:    chunk,
				Category: DeltaNew,
				Hash:     HashChunk(newChunks[i].Text),
			})
			_ = chunk // silence vet
		}
		return out
	}

	oldHashes := make(map[string][]int, len(manifest.Chunks))
	for i, entry := range manifest.Chunks {
		oldHashes[entry.Hash] = append(oldHashes[entry.Hash], i)
	}

	// Track which old entries have been matched so we can identify removals.
	matchedOld := make(map[int]bool, len(manifest.Chunks))
	out := make([]ChunkDelta, 0, len(newChunks)+len(manifest.Chunks))

	for i, chunk := range newChunks {
		h := HashChunk(chunk.Text)
		oldIndices := oldHashes[h]
		matched := false
		for _, oldIdx := range oldIndices {
			if !matchedOld[oldIdx] {
				matchedOld[oldIdx] = true
				matched = true
				break
			}
		}
		category := DeltaNew
		if matched {
			category = DeltaUnchanged
		}
		out = append(out, ChunkDelta{
			Chunk:    newChunks[i],
			Category: category,
			Hash:     h,
		})
	}

	// Identify removed chunks: old entries with no match in new set.
	for i, entry := range manifest.Chunks {
		if matchedOld[i] {
			continue
		}
		out = append(out, ChunkDelta{
			Chunk: Chunk{
				DocumentID: manifest.DocumentHash,
				Ordinal:    i,
			},
			Category: DeltaRemoved,
			Hash:     entry.Hash,
		})
	}

	return out
}

// validateDocumentHash returns an error if the hash is not a valid
// lowercase hex string. This prevents path traversal attacks via crafted
// document hash values containing slashes or relative paths.
func validateDocumentHash(documentHash string) error {
	if !validDocumentHashRe.MatchString(documentHash) {
		return fmt.Errorf("knowledge: invalid document hash %q: must match ^[a-f0-9]+$", documentHash)
	}
	return nil
}

// manifestPath returns the brain.Path where a document's chunk manifest
// is persisted. Caller must validate documentHash before calling.
func manifestPath(documentHash string) brain.Path {
	return brain.Path(chunkManifestsPrefix + "/" + documentHash + ".json")
}

// ReadChunkManifest loads the chunk manifest for a document from the
// store. Returns nil and no error when the manifest does not exist (first
// ingest scenario).
func ReadChunkManifest(ctx context.Context, store brain.Store, documentHash string) (*ChunkManifest, error) {
	if err := validateDocumentHash(documentHash); err != nil {
		return nil, err
	}
	p := manifestPath(documentHash)
	data, err := store.Read(ctx, p)
	if err != nil {
		if errors.Is(err, brain.ErrNotFound) {
			return nil, nil
		}
		return nil, fmt.Errorf("knowledge: reading chunk manifest %s: %w", p, err)
	}
	var manifest ChunkManifest
	if err := json.Unmarshal(data, &manifest); err != nil {
		return nil, fmt.Errorf("knowledge: parsing chunk manifest %s: %w", p, err)
	}
	return &manifest, nil
}

// WriteChunkManifest persists the chunk manifest for a document. Reads
// the prior manifest to increment the generation counter. When no prior
// manifest exists, generation starts at 1.
func WriteChunkManifest(ctx context.Context, store brain.Store, manifest ChunkManifest) error {
	if err := validateDocumentHash(manifest.DocumentHash); err != nil {
		return err
	}
	p := manifestPath(manifest.DocumentHash)

	prior, err := ReadChunkManifest(ctx, store, manifest.DocumentHash)
	if err != nil {
		return err
	}
	if prior != nil {
		manifest.Generation = prior.Generation + 1
	} else {
		manifest.Generation = 1
	}
	manifest.UpdatedAt = time.Now().UTC()

	data, err := json.Marshal(manifest)
	if err != nil {
		return fmt.Errorf("knowledge: marshalling chunk manifest: %w", err)
	}
	if err := store.Write(ctx, p, data); err != nil {
		return fmt.Errorf("knowledge: writing chunk manifest %s: %w", p, err)
	}
	return nil
}

// BuildChunkManifest constructs a ChunkManifest from a set of chunks and
// their parent document hash. This is the entry point for creating a new
// manifest after chunking.
func BuildChunkManifest(documentHash string, chunks []Chunk) ChunkManifest {
	entries := make([]ChunkManifestEntry, 0, len(chunks))
	for i, chunk := range chunks {
		entries = append(entries, ChunkManifestEntry{
			Hash:    HashChunk(chunk.Text),
			ChunkID: fmt.Sprintf("%s_%d", documentHash, i),
		})
	}
	return ChunkManifest{
		DocumentHash: documentHash,
		Chunks:       entries,
	}
}
