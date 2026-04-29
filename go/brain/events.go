// SPDX-License-Identifier: Apache-2.0

package brain

import "time"

// ChangeKind classifies a mutation emitted by a [Store] to its subscribers.
type ChangeKind int

const (
	// ChangeCreated signals a new path appeared.
	ChangeCreated ChangeKind = iota + 1
	// ChangeUpdated signals an existing path was rewritten or appended to.
	ChangeUpdated
	// ChangeDeleted signals a path was removed.
	ChangeDeleted
	// ChangeRenamed signals a path was moved. Path holds the destination;
	// OldPath holds the source.
	ChangeRenamed
)

// String returns a human-readable name for the change kind.
func (k ChangeKind) String() string {
	switch k {
	case ChangeCreated:
		return "created"
	case ChangeUpdated:
		return "updated"
	case ChangeDeleted:
		return "deleted"
	case ChangeRenamed:
		return "renamed"
	default:
		return "unknown"
	}
}

// ChangeEvent describes a single successful mutation. For renames, Path holds
// the destination and OldPath holds the source. Reason is propagated from
// the enclosing [BatchOptions] when the mutation is part of a batch, and is
// empty for standalone writes.
type ChangeEvent struct {
	Kind    ChangeKind
	Path    Path
	OldPath Path
	Reason  string
	When    time.Time
}

// EventSink consumes [ChangeEvent]s emitted by a [Store]. Sinks are
// called synchronously from the goroutine that performs the mutation:
// immediately after a standalone write, or immediately after the
// enclosing [Store.Batch] commits. Within a batch, events are buffered
// until commit and discarded on rollback, so subscribers never see
// speculative changes. Sinks must not block; dispatch expensive work on
// a background worker.
type EventSink interface {
	OnBrainChange(evt ChangeEvent)
}

// EventSinkFunc adapts a function to the [EventSink] interface.
type EventSinkFunc func(evt ChangeEvent)

// OnBrainChange implements [EventSink].
func (f EventSinkFunc) OnBrainChange(evt ChangeEvent) { f(evt) }
