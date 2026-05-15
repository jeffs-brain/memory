// SPDX-License-Identifier: Apache-2.0
package ingest

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"
)

// memStateStore is a test-only in-memory implementation of PipelineStateStore.
type memStateStore struct {
	mu      sync.Mutex
	entries map[string]*PipelineStateEntry
}

func newMemStateStore() *memStateStore {
	return &memStateStore{entries: make(map[string]*PipelineStateEntry)}
}

func (m *memStateStore) Load(_ context.Context, documentHash string) (*PipelineStateEntry, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	entry, ok := m.entries[documentHash]
	if !ok {
		return nil, nil
	}
	cp := *entry
	return &cp, nil
}

func (m *memStateStore) Save(_ context.Context, entry *PipelineStateEntry) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	cp := *entry
	m.entries[entry.DocumentHash] = &cp
	return nil
}

func (m *memStateStore) ListIncomplete(_ context.Context) ([]*PipelineStateEntry, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	var result []*PipelineStateEntry
	for _, entry := range m.entries {
		if entry.Stage != StageCompleted && entry.Stage != StageDeadLetter {
			cp := *entry
			result = append(result, &cp)
		}
	}
	return result, nil
}

func seedEntry(store *memStateStore, hash string, stage PipelineStage) {
	store.entries[hash] = &PipelineStateEntry{
		DocumentHash: hash,
		Stage:        stage,
		RetryCount:   0,
		CreatedAt:    time.Now().UTC(),
		UpdatedAt:    time.Now().UTC(),
	}
}

func TestTransition_ValidForwardTransitions(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name     string
		current  PipelineStage
		expected PipelineStage
	}{
		{"received to stored", StageReceived, StageStored},
		{"stored to chunked", StageStored, StageChunked},
		{"chunked to embedded", StageChunked, StageEmbedded},
		{"embedded to indexed", StageEmbedded, StageIndexed},
		{"indexed to completed", StageIndexed, StageCompleted},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			got, err := Transition(tc.current, EventStageCompleted)
			if err != nil {
				t.Fatalf("Transition(%s, StageCompleted) unexpected error: %v", tc.current, err)
			}
			if got != tc.expected {
				t.Fatalf("Transition(%s, StageCompleted) = %s, want %s", tc.current, got, tc.expected)
			}
		})
	}
}

func TestTransition_InvalidSkipTransitions(t *testing.T) {
	t.Parallel()
	// completed has no forward transition
	_, err := Transition(StageCompleted, EventStageCompleted)
	if err == nil {
		t.Fatal("expected error for transition from completed")
	}
	if !errors.Is(err, ErrInvalidTransition) {
		t.Fatalf("expected ErrInvalidTransition, got: %v", err)
	}
	// failed has no forward transition
	_, err = Transition(StageDeadLetter, EventStageCompleted)
	if err == nil {
		t.Fatal("expected error for transition from failed")
	}
	if !errors.Is(err, ErrInvalidTransition) {
		t.Fatalf("expected ErrInvalidTransition, got: %v", err)
	}
}

func TestIsValidTransition(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name    string
		current PipelineStage
		target  PipelineStage
		valid   bool
	}{
		{"received to stored valid", StageReceived, StageStored, true},
		{"stored to chunked valid", StageStored, StageChunked, true},
		{"received to chunked skip invalid", StageReceived, StageChunked, false},
		{"stored to received backward invalid", StageStored, StageReceived, false},
		{"received to embedded skip invalid", StageReceived, StageEmbedded, false},
		{"indexed to completed valid", StageIndexed, StageCompleted, true},
		{"completed to received invalid", StageCompleted, StageReceived, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			got := IsValidTransition(tc.current, tc.target)
			if got != tc.valid {
				t.Fatalf("IsValidTransition(%s, %s) = %v, want %v", tc.current, tc.target, got, tc.valid)
			}
		})
	}
}

func TestAdvance_PersistsState(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	store := newMemStateStore()
	seedEntry(store, "abc123", StageReceived)

	sm := NewPipelineStateMachine(StateMachineConfig{Store: store})

	if err := sm.Advance(ctx, "abc123", EventStageCompleted); err != nil {
		t.Fatalf("Advance: %v", err)
	}

	entry, err := store.Load(ctx, "abc123")
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if entry.Stage != StageStored {
		t.Fatalf("expected stage stored, got %s", entry.Stage)
	}
}

func TestAdvance_FullPipeline(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	store := newMemStateStore()
	seedEntry(store, "doc1", StageReceived)

	var transitions []PipelineStage
	sm := NewPipelineStateMachine(StateMachineConfig{
		Store: store,
		OnTransition: func(_ string, _, to PipelineStage, _ TransitionEvent) {
			transitions = append(transitions, to)
		},
	})

	stages := []PipelineStage{StageStored, StageChunked, StageEmbedded, StageIndexed, StageCompleted}
	for _, expected := range stages {
		if err := sm.Advance(ctx, "doc1", EventStageCompleted); err != nil {
			t.Fatalf("Advance to %s: %v", expected, err)
		}
		entry, _ := store.Load(ctx, "doc1")
		if entry.Stage != expected {
			t.Fatalf("after advance expected %s, got %s", expected, entry.Stage)
		}
	}

	if len(transitions) != len(stages) {
		t.Fatalf("expected %d transition callbacks, got %d", len(stages), len(transitions))
	}
	for i, s := range stages {
		if transitions[i] != s {
			t.Fatalf("callback %d: expected %s, got %s", i, s, transitions[i])
		}
	}
}

func TestAdvance_InvalidTransitionRejected(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	store := newMemStateStore()
	seedEntry(store, "doc2", StageCompleted)

	sm := NewPipelineStateMachine(StateMachineConfig{Store: store})

	err := sm.Advance(ctx, "doc2", EventStageCompleted)
	if err == nil {
		t.Fatal("expected error advancing from completed")
	}
	if !errors.Is(err, ErrInvalidTransition) {
		t.Fatalf("expected ErrInvalidTransition, got: %v", err)
	}
}

func TestShouldRetry_UnderLimit(t *testing.T) {
	t.Parallel()
	sm := NewPipelineStateMachine(StateMachineConfig{Store: newMemStateStore()})

	entry := &PipelineStateEntry{RetryCount: 0}
	if !sm.ShouldRetry(entry) {
		t.Fatal("expected ShouldRetry=true for retryCount=0")
	}
	entry.RetryCount = 1
	if !sm.ShouldRetry(entry) {
		t.Fatal("expected ShouldRetry=true for retryCount=1")
	}
	entry.RetryCount = 2
	if !sm.ShouldRetry(entry) {
		t.Fatal("expected ShouldRetry=true for retryCount=2")
	}
}

func TestShouldRetry_AtLimit(t *testing.T) {
	t.Parallel()
	sm := NewPipelineStateMachine(StateMachineConfig{Store: newMemStateStore()})

	entry := &PipelineStateEntry{RetryCount: 3}
	if sm.ShouldRetry(entry) {
		t.Fatal("expected ShouldRetry=false for retryCount=3 (at max)")
	}
	entry.RetryCount = 4
	if sm.ShouldRetry(entry) {
		t.Fatal("expected ShouldRetry=false for retryCount=4 (over max)")
	}
}

func TestRecordFailure_IncrementsRetry(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	store := newMemStateStore()
	seedEntry(store, "retry-doc", StageChunked)

	sm := NewPipelineStateMachine(StateMachineConfig{Store: store})

	if err := sm.RecordFailure(ctx, "retry-doc", errors.New("timeout")); err != nil {
		t.Fatalf("RecordFailure: %v", err)
	}

	entry, _ := store.Load(ctx, "retry-doc")
	if entry.RetryCount != 1 {
		t.Fatalf("expected retryCount=1, got %d", entry.RetryCount)
	}
	if entry.Stage != StageChunked {
		t.Fatalf("expected stage chunked (still retryable), got %s", entry.Stage)
	}
	if entry.LastError != "timeout" {
		t.Fatalf("expected lastError=timeout, got %s", entry.LastError)
	}
}

func TestRecordFailure_ExhaustsRetries(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	store := newMemStateStore()
	store.entries["exhaust-doc"] = &PipelineStateEntry{
		DocumentHash: "exhaust-doc",
		Stage:        StageEmbedded,
		RetryCount:   2,
		CreatedAt:    time.Now().UTC(),
		UpdatedAt:    time.Now().UTC(),
	}

	var callbackStage PipelineStage
	sm := NewPipelineStateMachine(StateMachineConfig{
		Store: store,
		OnTransition: func(_ string, _, to PipelineStage, _ TransitionEvent) {
			callbackStage = to
		},
	})

	if err := sm.RecordFailure(ctx, "exhaust-doc", errors.New("permanent failure")); err != nil {
		t.Fatalf("RecordFailure: %v", err)
	}

	entry, _ := store.Load(ctx, "exhaust-doc")
	if entry.Stage != StageDeadLetter {
		t.Fatalf("expected stage failed after exhaustion, got %s", entry.Stage)
	}
	if entry.RetryCount != 3 {
		t.Fatalf("expected retryCount=3, got %d", entry.RetryCount)
	}
	if callbackStage != StageDeadLetter {
		t.Fatalf("expected callback with failed stage, got %s", callbackStage)
	}
}

func TestMarkDeadLetter_MovesToDeadLetter(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	store := newMemStateStore()
	seedEntry(store, "mark-doc", StageStored)

	sm := NewPipelineStateMachine(StateMachineConfig{Store: store})

	if err := sm.MarkDeadLetter(ctx, "mark-doc", errors.New("unrecoverable")); err != nil {
		t.Fatalf("MarkFailed: %v", err)
	}

	entry, _ := store.Load(ctx, "mark-doc")
	if entry.Stage != StageDeadLetter {
		t.Fatalf("expected stage failed, got %s", entry.Stage)
	}
	if entry.LastError != "unrecoverable" {
		t.Fatalf("expected lastError=unrecoverable, got %s", entry.LastError)
	}
}

func TestListIncomplete_ReturnsFailed(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	store := newMemStateStore()
	seedEntry(store, "incomplete1", StageChunked)
	seedEntry(store, "incomplete2", StageReceived)
	seedEntry(store, "done", StageCompleted)
	store.entries["dead"] = &PipelineStateEntry{
		DocumentHash: "dead",
		Stage:        StageDeadLetter,
		CreatedAt:    time.Now().UTC(),
		UpdatedAt:    time.Now().UTC(),
	}

	incomplete, err := store.ListIncomplete(ctx)
	if err != nil {
		t.Fatalf("ListIncomplete: %v", err)
	}
	if len(incomplete) != 2 {
		t.Fatalf("expected 2 incomplete entries, got %d", len(incomplete))
	}
	hashes := make(map[string]bool, len(incomplete))
	for _, entry := range incomplete {
		hashes[entry.DocumentHash] = true
	}
	if !hashes["incomplete1"] || !hashes["incomplete2"] {
		t.Fatalf("expected incomplete1 and incomplete2 in results, got %v", hashes)
	}
}

func TestTransition_RetryExhaustedMovesToFailed(t *testing.T) {
	t.Parallel()
	got, err := Transition(StageChunked, EventRetryExhausted)
	if err != nil {
		t.Fatalf("Transition with RetryExhausted: %v", err)
	}
	if got != StageDeadLetter {
		t.Fatalf("expected failed, got %s", got)
	}
}

func TestTransition_StageDeadLetter_KeepsCurrent(t *testing.T) {
	t.Parallel()
	got, err := Transition(StageEmbedded, EventStageFailed)
	if err != nil {
		t.Fatalf("Transition with StageDeadLetter: %v", err)
	}
	if got != StageEmbedded {
		t.Fatalf("expected stage to remain embedded, got %s", got)
	}
}

func TestPipelineStateMachine_ListIncomplete(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	store := newMemStateStore()
	seedEntry(store, "active-1", StageChunked)
	seedEntry(store, "active-2", StageReceived)
	seedEntry(store, "done", StageCompleted)
	store.entries["dead"] = &PipelineStateEntry{
		DocumentHash: "dead",
		Stage:        StageDeadLetter,
		CreatedAt:    time.Now().UTC(),
		UpdatedAt:    time.Now().UTC(),
	}

	sm := NewPipelineStateMachine(StateMachineConfig{Store: store})
	incomplete, err := sm.ListIncomplete(ctx)
	if err != nil {
		t.Fatalf("ListIncomplete: %v", err)
	}
	if len(incomplete) != 2 {
		t.Fatalf("expected 2 incomplete entries, got %d", len(incomplete))
	}
	hashes := make(map[string]bool, len(incomplete))
	for _, e := range incomplete {
		hashes[e.DocumentHash] = true
	}
	if !hashes["active-1"] || !hashes["active-2"] {
		t.Fatalf("expected active-1 and active-2, got %v", hashes)
	}
}

func TestMigrateFromV1(t *testing.T) {
	t.Parallel()
	v1 := V1PipelineStateEntry{
		DocumentID: "brain-1:abc123",
		Hash:       "abc123",
		Stage:      "embedded",
		UpdatedAt:  time.Date(2026, 5, 9, 12, 0, 0, 0, time.UTC),
		ChunkCount: 5,
	}
	v2 := MigrateFromV1(v1)
	if v2.DocumentHash != "abc123" {
		t.Fatalf("expected DocumentHash=abc123, got %s", v2.DocumentHash)
	}
	if v2.Stage != StageEmbedded {
		t.Fatalf("expected Stage=embedded, got %s", v2.Stage)
	}
	if v2.RetryCount != 0 {
		t.Fatalf("expected RetryCount=0, got %d", v2.RetryCount)
	}
	if v2.LastError != "" {
		t.Fatalf("expected empty LastError, got %s", v2.LastError)
	}
}

func TestMigrateFromV1_AllStages(t *testing.T) {
	t.Parallel()
	stages := map[string]PipelineStage{
		"stored":   StageStored,
		"chunked":  StageChunked,
		"embedded": StageEmbedded,
		"indexed":  StageIndexed,
	}
	for v1Stage, expected := range stages {
		v1 := V1PipelineStateEntry{
			DocumentID: "brain-1:test",
			Hash:       "test-hash",
			Stage:      v1Stage,
			UpdatedAt:  time.Now().UTC(),
		}
		v2 := MigrateFromV1(v1)
		if v2.Stage != expected {
			t.Errorf("stage %s: expected %s, got %s", v1Stage, expected, v2.Stage)
		}
	}
}

func TestMigrateFromV1_UnknownStage(t *testing.T) {
	t.Parallel()
	v1 := V1PipelineStateEntry{
		DocumentID: "brain-1:test",
		Hash:       "test-hash",
		Stage:      "unknown",
		UpdatedAt:  time.Now().UTC(),
	}
	v2 := MigrateFromV1(v1)
	if v2.Stage != StageReceived {
		t.Fatalf("expected received for unknown stage, got %s", v2.Stage)
	}
}
