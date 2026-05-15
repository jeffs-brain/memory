// SPDX-License-Identifier: Apache-2.0
package ingest

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"regexp"
	"time"

	"github.com/jeffs-brain/memory/go/brain"
)

// PipelineStage represents the ordered stages a document passes through
// during ingestion. Terminal stages are "completed" and "failed".
type PipelineStage string

const (
	StageReceived  PipelineStage = "received"
	StageStored    PipelineStage = "stored"
	StageChunked   PipelineStage = "chunked"
	StageEmbedded  PipelineStage = "embedded"
	StageIndexed   PipelineStage = "indexed"
	StageCompleted PipelineStage = "completed"
	StageFailed    PipelineStage = "failed"
)

// IsTerminal reports whether the stage represents a final state that
// should not appear in ListIncomplete results.
func (s PipelineStage) IsTerminal() bool {
	return s == StageCompleted || s == StageFailed
}

// PipelineStateEntry tracks a single document's progress through the
// ingest pipeline.
type PipelineStateEntry struct {
	DocumentHash string        `json:"documentHash"`
	BrainID      string        `json:"brainId"`
	Stage        PipelineStage `json:"stage"`
	RetryCount   int           `json:"retryCount"`
	LastError    string        `json:"lastError,omitempty"`
	CreatedAt    time.Time     `json:"createdAt"`
	UpdatedAt    time.Time     `json:"updatedAt"`
	CompletedAt  *time.Time    `json:"completedAt,omitempty"`
}

// PipelineStateStore persists pipeline state for crash recovery. Both
// file-based and PostgreSQL implementations satisfy this interface.
type PipelineStateStore interface {
	// Get retrieves the state entry for the given document hash. Returns
	// nil without error when no record exists.
	Get(ctx context.Context, documentHash string) (*PipelineStateEntry, error)

	// Set creates or overwrites the state entry for a document.
	Set(ctx context.Context, entry PipelineStateEntry) error

	// ListIncomplete returns all entries for the given brainID whose stage
	// is not terminal (not "completed" or "failed").
	ListIncomplete(ctx context.Context, brainID string) ([]PipelineStateEntry, error)

	// Delete removes the state entry for the given document hash. Does not
	// return an error if the entry does not exist.
	Delete(ctx context.Context, documentHash string) error
}

const defaultStatePrefix = "raw/.pipeline-state"

// documentHashPattern validates that a document hash is a 64-character
// lowercase hex string (SHA-256). Rejects path traversal attempts.
var documentHashPattern = regexp.MustCompile(`^[a-f0-9]{64}$`)

func validateDocumentHash(documentHash string) error {
	if !documentHashPattern.MatchString(documentHash) {
		return fmt.Errorf("ingest: invalid document hash %q", documentHash)
	}
	return nil
}

func statePath(prefix, documentHash string) brain.Path {
	return brain.Path(fmt.Sprintf("%s/%s.json", prefix, documentHash))
}

// FilePipelineStateStoreConfig configures the file-based state store.
type FilePipelineStateStoreConfig struct {
	// Store is the brain Store backend used for persistence.
	Store brain.Store
	// Prefix is the path prefix inside the store where state files are written.
	// Defaults to "raw/.pipeline-state" when empty.
	Prefix string
}

// FilePipelineStateStore implements PipelineStateStore by persisting each
// document's state as a JSON file at {prefix}/{hash}.json using the
// brain.Store interface. The default prefix is "raw/.pipeline-state".
type FilePipelineStateStore struct {
	store  brain.Store
	prefix string
}

// NewFilePipelineStateStore creates a file-based state store backed by the
// given brain.Store using the default prefix.
func NewFilePipelineStateStore(store brain.Store) *FilePipelineStateStore {
	return &FilePipelineStateStore{store: store, prefix: defaultStatePrefix}
}

// NewFilePipelineStateStoreWithConfig creates a file-based state store with
// explicit configuration, including a custom path prefix.
func NewFilePipelineStateStoreWithConfig(cfg FilePipelineStateStoreConfig) *FilePipelineStateStore {
	prefix := cfg.Prefix
	if prefix == "" {
		prefix = defaultStatePrefix
	}
	return &FilePipelineStateStore{store: cfg.Store, prefix: prefix}
}

func (s *FilePipelineStateStore) Get(ctx context.Context, documentHash string) (*PipelineStateEntry, error) {
	if err := validateDocumentHash(documentHash); err != nil {
		return nil, err
	}
	data, err := s.store.Read(ctx, statePath(s.prefix, documentHash))
	if err != nil {
		if errors.Is(err, brain.ErrNotFound) {
			return nil, nil
		}
		return nil, fmt.Errorf("ingest: state get %s: %w", documentHash, err)
	}
	var entry PipelineStateEntry
	if err := json.Unmarshal(data, &entry); err != nil {
		return nil, fmt.Errorf("ingest: state decode %s: %w", documentHash, err)
	}
	return &entry, nil
}

func (s *FilePipelineStateStore) Set(ctx context.Context, entry PipelineStateEntry) error {
	if err := validateDocumentHash(entry.DocumentHash); err != nil {
		return err
	}
	data, err := json.MarshalIndent(entry, "", "  ")
	if err != nil {
		return fmt.Errorf("ingest: state encode %s: %w", entry.DocumentHash, err)
	}
	if err := s.store.Write(ctx, statePath(s.prefix, entry.DocumentHash), data); err != nil {
		return fmt.Errorf("ingest: state set %s: %w", entry.DocumentHash, err)
	}
	return nil
}

func (s *FilePipelineStateStore) ListIncomplete(ctx context.Context, brainID string) ([]PipelineStateEntry, error) {
	dir := brain.Path(s.prefix)
	exists, err := s.store.Exists(ctx, dir)
	if err != nil {
		return nil, fmt.Errorf("ingest: state list check dir: %w", err)
	}
	if !exists {
		return nil, nil
	}

	files, err := s.store.List(ctx, dir, brain.ListOpts{Recursive: false, Glob: "*.json"})
	if err != nil {
		return nil, fmt.Errorf("ingest: state list: %w", err)
	}

	entries := make([]PipelineStateEntry, 0, len(files))
	for _, file := range files {
		if file.IsDir {
			continue
		}
		data, readErr := s.store.Read(ctx, file.Path)
		if readErr != nil {
			if errors.Is(readErr, brain.ErrNotFound) {
				continue
			}
			return nil, fmt.Errorf("ingest: state list read %s: %w", file.Path, readErr)
		}
		var entry PipelineStateEntry
		if decodeErr := json.Unmarshal(data, &entry); decodeErr != nil {
			return nil, fmt.Errorf("ingest: state list decode %s: %w", file.Path, decodeErr)
		}
		if entry.BrainID == brainID && !entry.Stage.IsTerminal() {
			entries = append(entries, entry)
		}
	}

	return entries, nil
}

func (s *FilePipelineStateStore) Delete(ctx context.Context, documentHash string) error {
	if err := validateDocumentHash(documentHash); err != nil {
		return err
	}
	err := s.store.Delete(ctx, statePath(s.prefix, documentHash))
	if err != nil {
		if errors.Is(err, brain.ErrNotFound) {
			return nil
		}
		return fmt.Errorf("ingest: state delete %s: %w", documentHash, err)
	}
	return nil
}

var _ PipelineStateStore = (*FilePipelineStateStore)(nil)
