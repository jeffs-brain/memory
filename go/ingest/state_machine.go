// SPDX-License-Identifier: Apache-2.0
package ingest

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"
)

// PipelineStage represents the processing stage a document has reached
// within the ingestion pipeline.
type PipelineStage string

const (
	StageReceived  PipelineStage = "received"
	StageStored    PipelineStage = "stored"
	StageChunked   PipelineStage = "chunked"
	StageEmbedded  PipelineStage = "embedded"
	StageIndexed   PipelineStage = "indexed"
	StageCompleted PipelineStage = "completed"
	StageDeadLetter    PipelineStage = "dead_letter"
)

// stageOrder defines the ordinal position of each stage for comparison.
// Failed is excluded from the forward-progress chain.
var stageOrder = map[PipelineStage]int{
	StageReceived:  0,
	StageStored:    1,
	StageChunked:   2,
	StageEmbedded:  3,
	StageIndexed:   4,
	StageCompleted: 5,
}

// validForwardTransitions maps each stage to its single valid next stage.
var validForwardTransitions = map[PipelineStage]PipelineStage{
	StageReceived: StageStored,
	StageStored:   StageChunked,
	StageChunked:  StageEmbedded,
	StageEmbedded: StageIndexed,
	StageIndexed:  StageCompleted,
}

// TransitionEvent describes what triggered a state machine transition.
type TransitionEvent string

const (
	EventStageCompleted TransitionEvent = "stage_completed"
	EventStageFailed    TransitionEvent = "stage_failed"
	EventRetryExhausted TransitionEvent = "retry_exhausted"
)

// maxDefaultRetries is the default number of retries before a document
// moves to the dead_letter stage.
const maxDefaultRetries = 3

// ErrInvalidTransition signals that a requested stage transition is
// not permitted by the state machine rules.
var ErrInvalidTransition = errors.New("ingest: invalid state transition")

// PipelineStateEntry tracks the processing state of a single document
// through the ingestion pipeline.
type PipelineStateEntry struct {
	DocumentHash string
	Stage        PipelineStage
	RetryCount   int
	LastError    string
	CreatedAt    time.Time
	UpdatedAt    time.Time
}

// PipelineStateStore is the persistence layer for pipeline state entries.
// Implementations may back this with a database, file system, or in-memory
// map. P1-1 provides the production implementation.
type PipelineStateStore interface {
	// Load retrieves the state entry for a document hash. Returns nil if
	// no entry exists.
	Load(ctx context.Context, documentHash string) (*PipelineStateEntry, error)

	// Save persists the state entry, creating or overwriting as needed.
	Save(ctx context.Context, entry *PipelineStateEntry) error

	// ListIncomplete returns all entries not in a terminal stage
	// (completed or failed).
	ListIncomplete(ctx context.Context) ([]*PipelineStateEntry, error)
}

// TransitionCallback is invoked after every successful state transition.
// Implementations use this for observability (logging, metrics, events).
type TransitionCallback func(documentHash string, from, to PipelineStage, event TransitionEvent)

// PipelineStateMachine manages stage transitions for documents flowing
// through the ingestion pipeline. It validates transitions, tracks retry
// counts, and persists state via a PipelineStateStore.
type PipelineStateMachine struct {
	store      PipelineStateStore
	maxRetries int
	callback   TransitionCallback
	mu         sync.Mutex
}

// StateMachineConfig holds construction parameters for a PipelineStateMachine.
type StateMachineConfig struct {
	Store      PipelineStateStore
	MaxRetries int
	OnTransition TransitionCallback
}

// NewPipelineStateMachine creates a state machine with the given config.
// MaxRetries defaults to 3 if zero or negative.
func NewPipelineStateMachine(cfg StateMachineConfig) *PipelineStateMachine {
	retries := cfg.MaxRetries
	if retries <= 0 {
		retries = maxDefaultRetries
	}
	return &PipelineStateMachine{
		store:      cfg.Store,
		maxRetries: retries,
		callback:   cfg.OnTransition,
	}
}

// Transition validates that moving from current to the next stage implied
// by the event is permitted and returns the resulting stage. Returns
// ErrInvalidTransition for disallowed moves.
func Transition(current PipelineStage, event TransitionEvent) (PipelineStage, error) {
	switch event {
	case EventStageCompleted:
		next, ok := validForwardTransitions[current]
		if !ok {
			return current, fmt.Errorf("%w: no forward transition from %s", ErrInvalidTransition, current)
		}
		return next, nil
	case EventStageFailed:
		return current, nil
	case EventRetryExhausted:
		return StageDeadLetter, nil
	}
	return current, fmt.Errorf("%w: unknown event %s", ErrInvalidTransition, event)
}

// IsValidTransition reports whether moving from current to target is a
// valid single-step forward transition.
func IsValidTransition(current, target PipelineStage) bool {
	next, ok := validForwardTransitions[current]
	if !ok {
		return false
	}
	return next == target
}

// Advance loads the current state for documentHash, applies the event,
// and persists the resulting state. Returns the updated entry.
func (sm *PipelineStateMachine) Advance(ctx context.Context, documentHash string, event TransitionEvent) error {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	entry, err := sm.store.Load(ctx, documentHash)
	if err != nil {
		return fmt.Errorf("ingest: loading state for %s: %w", documentHash, err)
	}
	if entry == nil {
		return fmt.Errorf("ingest: no state entry for document %s", documentHash)
	}

	from := entry.Stage
	next, transErr := Transition(entry.Stage, event)
	if transErr != nil {
		return transErr
	}

	entry.Stage = next
	entry.UpdatedAt = time.Now().UTC()

	if err := sm.store.Save(ctx, entry); err != nil {
		return fmt.Errorf("ingest: saving state for %s: %w", documentHash, err)
	}

	if sm.callback != nil && from != next {
		sm.callback(documentHash, from, next, event)
	}
	return nil
}

// ShouldRetry returns true if the entry has not yet exhausted its retry
// budget (retryCount < maxRetries).
func (sm *PipelineStateMachine) ShouldRetry(entry *PipelineStateEntry) bool {
	return entry.RetryCount < sm.maxRetries
}

// RecordFailure increments the retry counter. If retries are exhausted,
// the document transitions to the failed stage; otherwise it remains in
// the current stage for retry.
func (sm *PipelineStateMachine) RecordFailure(ctx context.Context, documentHash string, failErr error) error {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	entry, err := sm.store.Load(ctx, documentHash)
	if err != nil {
		return fmt.Errorf("ingest: loading state for %s: %w", documentHash, err)
	}
	if entry == nil {
		return fmt.Errorf("ingest: no state entry for document %s", documentHash)
	}

	from := entry.Stage
	entry.RetryCount++
	entry.LastError = failErr.Error()
	entry.UpdatedAt = time.Now().UTC()

	if entry.RetryCount >= sm.maxRetries {
		entry.Stage = StageDeadLetter
		if err := sm.store.Save(ctx, entry); err != nil {
			return fmt.Errorf("ingest: saving state for %s: %w", documentHash, err)
		}
		if sm.callback != nil {
			sm.callback(documentHash, from, StageDeadLetter, EventRetryExhausted)
		}
		return nil
	}

	if err := sm.store.Save(ctx, entry); err != nil {
		return fmt.Errorf("ingest: saving state for %s: %w", documentHash, err)
	}
	if sm.callback != nil {
		sm.callback(documentHash, from, entry.Stage, EventStageFailed)
	}
	return nil
}

// MarkDeadLetter unconditionally moves the document to the dead_letter
// stage, recording the error regardless of retry count.
func (sm *PipelineStateMachine) MarkDeadLetter(ctx context.Context, documentHash string, failErr error) error {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	entry, err := sm.store.Load(ctx, documentHash)
	if err != nil {
		return fmt.Errorf("ingest: loading state for %s: %w", documentHash, err)
	}
	if entry == nil {
		return fmt.Errorf("ingest: no state entry for document %s", documentHash)
	}

	from := entry.Stage
	entry.Stage = StageDeadLetter
	entry.LastError = failErr.Error()
	entry.UpdatedAt = time.Now().UTC()

	if err := sm.store.Save(ctx, entry); err != nil {
		return fmt.Errorf("ingest: saving state for %s: %w", documentHash, err)
	}
	if sm.callback != nil && from != StageDeadLetter {
		sm.callback(documentHash, from, StageDeadLetter, EventRetryExhausted)
	}
	return nil
}

// ListIncomplete returns all pipeline state entries that are not in a
// terminal stage (indexed/completed or dead_letter). Delegates to the
// underlying PipelineStateStore.
func (sm *PipelineStateMachine) ListIncomplete(ctx context.Context) ([]*PipelineStateEntry, error) {
	sm.mu.Lock()
	defer sm.mu.Unlock()
	return sm.store.ListIncomplete(ctx)
}

// V1PipelineStateEntry represents the pipeline state format from P0-1.
// It lacks retry tracking and uses a different field layout.
type V1PipelineStateEntry struct {
	DocumentID string
	Hash       string
	Stage      string // "stored", "chunked", "embedded", "indexed"
	UpdatedAt  time.Time
	ChunkCount int
}

// v1StageMap maps V1 stage strings to the V2 PipelineStage type.
var v1StageMap = map[string]PipelineStage{
	"stored":   StageStored,
	"chunked":  StageChunked,
	"embedded": StageEmbedded,
	"indexed":  StageIndexed,
}

// MigrateFromV1 converts a V1 pipeline state entry to the V2 format
// used by the state machine. V1 entries from P0-1 lack retry tracking.
func MigrateFromV1(v1 V1PipelineStateEntry) PipelineStateEntry {
	stage, ok := v1StageMap[v1.Stage]
	if !ok {
		stage = StageReceived
	}
	return PipelineStateEntry{
		DocumentHash: v1.Hash,
		Stage:        stage,
		RetryCount:   0,
		LastError:    "",
		CreatedAt:    v1.UpdatedAt,
		UpdatedAt:    v1.UpdatedAt,
	}
}
