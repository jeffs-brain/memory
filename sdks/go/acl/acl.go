// SPDX-License-Identifier: Apache-2.0

// Package acl defines the authorisation surface for jeffs-brain/memory.
//
// The package ships the [Provider] contract used by stores and HTTP
// handlers to authorise reads and writes, plus an in-process RBAC
// implementation in rbac.go and a [brain.Store] wrapper in wrap.go.
//
// Production deployments typically pair the [Provider] interface with
// the OpenFGA HTTP adapter shipped in the sibling aclopenfga package.
// The authorisation model that adapter speaks is documented in
// spec/openfga/schema.fga.
package acl

import (
	"context"
	"errors"
	"fmt"

	"github.com/jeffs-brain/memory/go/brain"
)

// ResourceType enumerates the kinds of resource a [Provider] can
// authorise against. The hierarchy is workspace > brain > collection >
// document and is honoured by both the in-process RBAC adapter and the
// canonical OpenFGA model.
type ResourceType string

const (
	ResourceWorkspace  ResourceType = "workspace"
	ResourceBrain      ResourceType = "brain"
	ResourceCollection ResourceType = "collection"
	ResourceDocument   ResourceType = "document"
)

// Resource identifies a single resource within the hierarchy. The
// concrete shape of [Resource.ID] is opaque to the provider; callers
// supply UUIDs, slugs or path-derived strings as appropriate.
type Resource struct {
	Type ResourceType
	ID   string
}

// SubjectKind enumerates the principal types the system recognises.
// Service principals are reserved for internal callers (cron jobs,
// background workers) that act on behalf of the system rather than a
// human or programmatic API user.
type SubjectKind string

const (
	SubjectUser    SubjectKind = "user"
	SubjectAPIKey  SubjectKind = "api_key"
	SubjectService SubjectKind = "service"
)

// Subject identifies the principal whose access is being checked.
type Subject struct {
	Kind SubjectKind
	ID   string
}

// Action names the high-level operation being authorised. Adapters map
// these to backend-specific relations: read and write are universal,
// the remainder are carried for richer policy expression.
type Action string

const (
	ActionRead   Action = "read"
	ActionWrite  Action = "write"
	ActionDelete Action = "delete"
	ActionAdmin  Action = "admin"
	ActionExport Action = "export"
)

// CheckResult is the outcome of a [Provider.Check] call. Reason is a
// free-form diagnostic string surfaced in error messages and logs;
// callers must not branch on it.
type CheckResult struct {
	Allowed bool
	Reason  string
}

// Tuple is a single relationship lifted straight from the
// Zanzibar/OpenFGA vocabulary so adapters can round-trip without
// translation.
type Tuple struct {
	Subject  Subject
	Relation string
	Resource Resource
}

// WriteTuplesRequest groups a set of additions and removals. Adapters
// SHOULD apply both sets atomically when the backend supports it.
type WriteTuplesRequest struct {
	Writes  []Tuple
	Deletes []Tuple
}

// ReadTuplesQuery filters tuples returned by [Provider.Read]. A nil
// Subject or Resource means "any"; an empty Relation means "any
// relation".
type ReadTuplesQuery struct {
	Subject  *Subject
	Relation string
	Resource *Resource
}

// Provider is the pluggable authorisation surface. Adapters MUST
// implement Check; Write and Read MAY return [ErrUnsupported] when the
// backend has no tuple persistence (for example, a stateless policy
// evaluator).
type Provider interface {
	// Name identifies the adapter for diagnostics and metrics.
	Name() string
	// Check evaluates whether subject is permitted to perform action
	// on resource.
	Check(ctx context.Context, subject Subject, action Action, resource Resource) (CheckResult, error)
	// Write applies the given tuple mutations.
	Write(ctx context.Context, req WriteTuplesRequest) error
	// Read returns tuples matching the query.
	Read(ctx context.Context, query ReadTuplesQuery) ([]Tuple, error)
	// Close releases any resources held by the provider (HTTP
	// transports, file handles, etc.). Adapters that own no releasable
	// state return nil.
	Close() error
}

// Allow returns a CheckResult that permits the action. Reason is
// optional; supply an empty string when no diagnostic is needed.
func Allow(reason string) CheckResult { return CheckResult{Allowed: true, Reason: reason} }

// Deny returns a CheckResult that denies the action.
func Deny(reason string) CheckResult { return CheckResult{Allowed: false, Reason: reason} }

// SubjectKey returns the canonical "kind:id" encoding used by tuple
// stores and the OpenFGA wire protocol.
func SubjectKey(s Subject) string { return string(s.Kind) + ":" + s.ID }

// ResourceKey returns the canonical "type:id" encoding.
func ResourceKey(r Resource) string { return string(r.Type) + ":" + r.ID }

// TupleKey returns a stable string key for a tuple, suitable for use
// as a map index by in-memory tuple stores.
func TupleKey(t Tuple) string {
	return SubjectKey(t.Subject) + "|" + t.Relation + "|" + ResourceKey(t.Resource)
}

// ErrUnsupported is returned by adapters that do not implement a given
// operation (for example, a stateless policy evaluator that has no
// tuple persistence). Callers should branch on errors.Is.
var ErrUnsupported = errors.New("acl: operation not supported")

// ErrUnsupportedRelation is returned by tuple-persisting adapters when
// the request names a relation the adapter does not understand. The
// in-process RBAC provider uses this to reject write attempts on
// relations outside its fixed vocabulary.
var ErrUnsupportedRelation = errors.New("acl: unsupported relation")

// ForbiddenError is returned by the [Wrap]-applied store and by
// callers that translate a denied [CheckResult] into an error. It
// satisfies errors.Is(err, brain.ErrForbidden) so existing handlers
// that branch on the brain sentinel keep working unchanged.
type ForbiddenError struct {
	Subject  Subject
	Action   Action
	Resource Resource
	Reason   string
}

// Error implements the error interface. The message shape mirrors the
// TypeScript ForbiddenError so logs are uniform across SDKs.
func (e *ForbiddenError) Error() string {
	if e.Reason == "" {
		return fmt.Sprintf("acl: forbidden: %s:%s %s %s:%s",
			e.Subject.Kind, e.Subject.ID, e.Action, e.Resource.Type, e.Resource.ID)
	}
	return fmt.Sprintf("acl: forbidden: %s:%s %s %s:%s (%s)",
		e.Subject.Kind, e.Subject.ID, e.Action, e.Resource.Type, e.Resource.ID, e.Reason)
}

// Is reports whether target is the brain forbidden sentinel. This
// keeps existing errors.Is(err, brain.ErrForbidden) checks honest
// without forcing callers to branch on a second sentinel.
func (e *ForbiddenError) Is(target error) bool {
	return target == brain.ErrForbidden
}
