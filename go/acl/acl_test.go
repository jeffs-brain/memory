// SPDX-License-Identifier: Apache-2.0

package acl

import (
	"errors"
	"strings"
	"testing"

	"github.com/jeffs-brain/memory/go/brain"
)

func TestSubjectKey(t *testing.T) {
	got := SubjectKey(Subject{Kind: SubjectUser, ID: "alice"})
	if got != "user:alice" {
		t.Fatalf("SubjectKey = %q, want user:alice", got)
	}
}

func TestResourceKey(t *testing.T) {
	got := ResourceKey(Resource{Type: ResourceBrain, ID: "notes"})
	if got != "brain:notes" {
		t.Fatalf("ResourceKey = %q, want brain:notes", got)
	}
}

func TestTupleKey(t *testing.T) {
	got := TupleKey(Tuple{
		Subject:  Subject{Kind: SubjectUser, ID: "alice"},
		Relation: "writer",
		Resource: Resource{Type: ResourceBrain, ID: "notes"},
	})
	want := "user:alice|writer|brain:notes"
	if got != want {
		t.Fatalf("TupleKey = %q, want %q", got, want)
	}
}

func TestAllowDeny(t *testing.T) {
	if a := Allow("ok"); !a.Allowed || a.Reason != "ok" {
		t.Errorf("Allow returned %+v", a)
	}
	if d := Deny("nope"); d.Allowed || d.Reason != "nope" {
		t.Errorf("Deny returned %+v", d)
	}
}

func TestForbiddenErrorMessage(t *testing.T) {
	err := &ForbiddenError{
		Subject:  Subject{Kind: SubjectUser, ID: "alice"},
		Action:   ActionWrite,
		Resource: Resource{Type: ResourceBrain, ID: "notes"},
		Reason:   "no grant",
	}
	want := "acl: forbidden: user:alice write brain:notes (no grant)"
	if err.Error() != want {
		t.Fatalf("Error = %q, want %q", err.Error(), want)
	}

	// Empty reason omits the parenthesised suffix.
	err.Reason = ""
	want = "acl: forbidden: user:alice write brain:notes"
	if err.Error() != want {
		t.Fatalf("Error (no reason) = %q, want %q", err.Error(), want)
	}
}

func TestForbiddenErrorIsBrainSentinel(t *testing.T) {
	// errors.Is(err, brain.ErrForbidden) must return true so existing
	// handlers branching on the brain sentinel continue to match.
	err := &ForbiddenError{Subject: Subject{Kind: SubjectUser, ID: "alice"}, Action: ActionRead, Resource: Resource{Type: ResourceBrain, ID: "x"}}
	if !errors.Is(err, brain.ErrForbidden) {
		t.Fatalf("errors.Is(err, brain.ErrForbidden) = false, want true")
	}
	// Other sentinels must not match.
	if errors.Is(err, brain.ErrNotFound) {
		t.Fatalf("errors.Is matched ErrNotFound by mistake")
	}
}

func TestForbiddenErrorAs(t *testing.T) {
	var sentinel error = &ForbiddenError{Subject: Subject{Kind: SubjectUser, ID: "x"}, Action: ActionRead, Resource: Resource{Type: ResourceBrain, ID: "y"}}
	var fe *ForbiddenError
	if !errors.As(sentinel, &fe) {
		t.Fatal("errors.As failed to extract ForbiddenError")
	}
	if !strings.Contains(fe.Error(), "user:x") {
		t.Errorf("extracted error missing subject id: %v", fe)
	}
}
