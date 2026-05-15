// SPDX-License-Identifier: Apache-2.0
package hooks

import (
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/jeffs-brain/memory/go/brain"
)

func TestCreatedEventInRawDocumentsDispatches(t *testing.T) {
	var dispatched atomic.Int32
	var dispatchedPath string
	var mu sync.Mutex

	hook := NewMutationHook(MutationHookOptions{
		BrainID:          "brain-1",
		DebounceInterval: 10 * time.Millisecond,
		Dispatch: func(brainID, path string) error {
			mu.Lock()
			dispatchedPath = path
			mu.Unlock()
			dispatched.Add(1)
			return nil
		},
	})
	defer hook.Close()

	sink := hook.Sink()
	sink.OnBrainChange(brain.ChangeEvent{
		Kind: brain.ChangeCreated,
		Path: "raw/documents/readme.md",
		When: time.Now(),
	})

	time.Sleep(50 * time.Millisecond)

	if got := dispatched.Load(); got != 1 {
		t.Fatalf("expected 1 dispatch, got %d", got)
	}
	mu.Lock()
	if dispatchedPath != "raw/documents/readme.md" {
		t.Fatalf("expected raw/documents/readme.md, got %q", dispatchedPath)
	}
	mu.Unlock()
}

func TestUpdatedEventInRawDocumentsDispatches(t *testing.T) {
	var dispatched atomic.Int32

	hook := NewMutationHook(MutationHookOptions{
		BrainID:          "brain-1",
		DebounceInterval: 10 * time.Millisecond,
		Dispatch: func(_, _ string) error {
			dispatched.Add(1)
			return nil
		},
	})
	defer hook.Close()

	sink := hook.Sink()
	sink.OnBrainChange(brain.ChangeEvent{
		Kind: brain.ChangeUpdated,
		Path: "raw/documents/notes.md",
		When: time.Now(),
	})

	time.Sleep(50 * time.Millisecond)

	if got := dispatched.Load(); got != 1 {
		t.Fatalf("expected 1 dispatch, got %d", got)
	}
}

func TestDeletedEventIgnored(t *testing.T) {
	var dispatched atomic.Int32

	hook := NewMutationHook(MutationHookOptions{
		BrainID:          "brain-1",
		DebounceInterval: 10 * time.Millisecond,
		Dispatch: func(_, _ string) error {
			dispatched.Add(1)
			return nil
		},
	})
	defer hook.Close()

	sink := hook.Sink()
	sink.OnBrainChange(brain.ChangeEvent{
		Kind: brain.ChangeDeleted,
		Path: "raw/documents/readme.md",
		When: time.Now(),
	})

	time.Sleep(50 * time.Millisecond)

	if got := dispatched.Load(); got != 0 {
		t.Fatalf("expected 0 dispatches for delete, got %d", got)
	}
}

func TestPathOutsideRawDocumentsIgnored(t *testing.T) {
	var dispatched atomic.Int32

	hook := NewMutationHook(MutationHookOptions{
		BrainID:          "brain-1",
		DebounceInterval: 10 * time.Millisecond,
		Dispatch: func(_, _ string) error {
			dispatched.Add(1)
			return nil
		},
	})
	defer hook.Close()

	sink := hook.Sink()
	sink.OnBrainChange(brain.ChangeEvent{
		Kind: brain.ChangeCreated,
		Path: "memory/global/fact.md",
		When: time.Now(),
	})

	time.Sleep(50 * time.Millisecond)

	if got := dispatched.Load(); got != 0 {
		t.Fatalf("expected 0 dispatches for non-matching path, got %d", got)
	}
}

func TestCustomPathMatcher(t *testing.T) {
	var dispatched atomic.Int32

	hook := NewMutationHook(MutationHookOptions{
		BrainID:          "brain-1",
		DebounceInterval: 10 * time.Millisecond,
		PathMatchers:     []PathMatcher{PrefixPathMatcher("custom/")},
		Dispatch: func(_, _ string) error {
			dispatched.Add(1)
			return nil
		},
	})
	defer hook.Close()

	sink := hook.Sink()
	sink.OnBrainChange(brain.ChangeEvent{
		Kind: brain.ChangeCreated,
		Path: "custom/data.json",
		When: time.Now(),
	})

	time.Sleep(50 * time.Millisecond)

	if got := dispatched.Load(); got != 1 {
		t.Fatalf("expected 1 dispatch for custom path, got %d", got)
	}
}

func TestDebounceCoalescesRapidWrites(t *testing.T) {
	var dispatched atomic.Int32

	hook := NewMutationHook(MutationHookOptions{
		BrainID:          "brain-1",
		DebounceInterval: 100 * time.Millisecond,
		Dispatch: func(_, _ string) error {
			dispatched.Add(1)
			return nil
		},
	})
	defer hook.Close()

	sink := hook.Sink()
	// Rapid writes to the same path within debounce window.
	for range 5 {
		sink.OnBrainChange(brain.ChangeEvent{
			Kind: brain.ChangeUpdated,
			Path: "raw/documents/rapid.md",
			When: time.Now(),
		})
		time.Sleep(10 * time.Millisecond)
	}

	time.Sleep(200 * time.Millisecond)

	if got := dispatched.Load(); got != 1 {
		t.Fatalf("expected 1 coalesced dispatch, got %d", got)
	}
}

func TestOptOutByBatchReason(t *testing.T) {
	var dispatched atomic.Int32

	hook := NewMutationHook(MutationHookOptions{
		BrainID:          "brain-1",
		DebounceInterval: 10 * time.Millisecond,
		OptOutReasons:    map[string]bool{"pipeline": true, "ingest": true},
		Dispatch: func(_, _ string) error {
			dispatched.Add(1)
			return nil
		},
	})
	defer hook.Close()

	sink := hook.Sink()

	// Event with opt-out reason should be ignored.
	sink.OnBrainChange(brain.ChangeEvent{
		Kind:   brain.ChangeCreated,
		Path:   "raw/documents/pipeline-output.md",
		Reason: "pipeline",
		When:   time.Now(),
	})

	time.Sleep(50 * time.Millisecond)

	if got := dispatched.Load(); got != 0 {
		t.Fatalf("expected 0 dispatches for opt-out reason, got %d", got)
	}

	// Event without opt-out reason should dispatch.
	sink.OnBrainChange(brain.ChangeEvent{
		Kind:   brain.ChangeCreated,
		Path:   "raw/documents/user-upload.md",
		Reason: "user-write",
		When:   time.Now(),
	})

	time.Sleep(50 * time.Millisecond)

	if got := dispatched.Load(); got != 1 {
		t.Fatalf("expected 1 dispatch for non-opt-out reason, got %d", got)
	}
}

func TestDifferentPathsDispatchIndependently(t *testing.T) {
	var dispatched atomic.Int32

	hook := NewMutationHook(MutationHookOptions{
		BrainID:          "brain-1",
		DebounceInterval: 10 * time.Millisecond,
		Dispatch: func(_, _ string) error {
			dispatched.Add(1)
			return nil
		},
	})
	defer hook.Close()

	sink := hook.Sink()
	sink.OnBrainChange(brain.ChangeEvent{
		Kind: brain.ChangeCreated,
		Path: "raw/documents/a.md",
		When: time.Now(),
	})
	sink.OnBrainChange(brain.ChangeEvent{
		Kind: brain.ChangeCreated,
		Path: "raw/documents/b.md",
		When: time.Now(),
	})

	time.Sleep(50 * time.Millisecond)

	if got := dispatched.Load(); got != 2 {
		t.Fatalf("expected 2 dispatches for different paths, got %d", got)
	}
}

func TestGlobPathMatcher(t *testing.T) {
	matcher := GlobPathMatcher("raw/documents/**/*.md")

	tests := []struct {
		path  string
		match bool
	}{
		{"raw/documents/readme.md", true},
		{"raw/documents/sub/notes.md", true},
		{"raw/documents/sub/deep/nested.md", true},
		{"raw/documents/readme.txt", false},
		{"memory/global/fact.md", false},
	}

	for _, tt := range tests {
		if got := matcher(tt.path); got != tt.match {
			t.Errorf("glob match %q: expected %v, got %v", tt.path, tt.match, got)
		}
	}
}

func TestCloseStopsPendingTimers(t *testing.T) {
	var dispatched atomic.Int32

	hook := NewMutationHook(MutationHookOptions{
		BrainID:          "brain-1",
		DebounceInterval: 200 * time.Millisecond,
		Dispatch: func(_, _ string) error {
			dispatched.Add(1)
			return nil
		},
	})

	sink := hook.Sink()
	sink.OnBrainChange(brain.ChangeEvent{
		Kind: brain.ChangeCreated,
		Path: "raw/documents/closing.md",
		When: time.Now(),
	})

	// Close before debounce fires.
	hook.Close()
	time.Sleep(300 * time.Millisecond)

	if got := dispatched.Load(); got != 0 {
		t.Fatalf("expected 0 dispatches after close, got %d", got)
	}
}
