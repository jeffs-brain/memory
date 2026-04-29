// SPDX-License-Identifier: Apache-2.0

package acl

import (
	"context"
	"errors"
	"strings"
	"testing"
)

func userSubject(id string) Subject    { return Subject{Kind: SubjectUser, ID: id} }
func workspaceRes(id string) Resource  { return Resource{Type: ResourceWorkspace, ID: id} }
func brainRes(id string) Resource      { return Resource{Type: ResourceBrain, ID: id} }
func collectionRes(id string) Resource { return Resource{Type: ResourceCollection, ID: id} }
func documentRes(id string) Resource   { return Resource{Type: ResourceDocument, ID: id} }

// seed builds a fresh RBAC provider with the given tuples written.
func seed(t *testing.T, tuples ...Tuple) Provider {
	t.Helper()
	p := NewRbacProvider(RbacOptions{})
	if err := p.Write(context.Background(), WriteTuplesRequest{Writes: tuples}); err != nil {
		t.Fatalf("seed write: %v", err)
	}
	return p
}

func TestRbacAdminOnWorkspaceCascades(t *testing.T) {
	alice := userSubject("alice")
	ws := workspaceRes("acme")
	br := brainRes("notes")

	provider := seed(t,
		ParentTuple(br, ws),
		GrantTuple(alice, RoleAdmin, ws),
	)
	got, err := provider.Check(context.Background(), alice, ActionAdmin, br)
	if err != nil {
		t.Fatalf("Check: %v", err)
	}
	if !got.Allowed {
		t.Fatalf("admin on workspace did not cascade: %+v", got)
	}

	// Inheritance must keep cascading through the rest of the chain.
	col := collectionRes("meetings")
	doc := documentRes("doc-1")
	provider = seed(t,
		ParentTuple(br, ws),
		ParentTuple(col, br),
		ParentTuple(doc, col),
		GrantTuple(alice, RoleAdmin, ws),
	)
	for _, action := range []Action{ActionWrite, ActionDelete} {
		res, err := provider.Check(context.Background(), alice, action, doc)
		if err != nil {
			t.Fatalf("Check %s: %v", action, err)
		}
		if !res.Allowed {
			t.Errorf("%s on doc-1 not allowed: %+v", action, res)
		}
	}
}

func TestRbacDenyAtBrainOverridesWorkspaceGrant(t *testing.T) {
	alice := userSubject("alice")
	ws := workspaceRes("acme")
	br := brainRes("notes")

	provider := seed(t,
		ParentTuple(br, ws),
		GrantTuple(alice, RoleAdmin, ws),
		DenyTuple(alice, RoleAdmin, br),
	)

	adminRes, err := provider.Check(context.Background(), alice, ActionAdmin, br)
	if err != nil {
		t.Fatalf("Check admin: %v", err)
	}
	if adminRes.Allowed {
		t.Errorf("deny should have blocked admin: %+v", adminRes)
	}
	if !strings.Contains(adminRes.Reason, "deny:admin") {
		t.Errorf("reason %q missing deny:admin", adminRes.Reason)
	}

	// A deny on admin (rank 3) covers any action requiring rank <= 3,
	// so write (which only needs writer, rank 2) is also blocked. This
	// matches the documented TS semantics; a tighter policy would deny
	// the lower role explicitly.
	writeRes, err := provider.Check(context.Background(), alice, ActionWrite, br)
	if err != nil {
		t.Fatalf("Check write: %v", err)
	}
	if writeRes.Allowed {
		t.Errorf("deny:admin should also block write: %+v", writeRes)
	}
}

func TestRbacReaderCannotWrite(t *testing.T) {
	bob := userSubject("bob")
	doc := documentRes("doc-1")
	provider := seed(t, GrantTuple(bob, RoleReader, doc))

	read, err := provider.Check(context.Background(), bob, ActionRead, doc)
	if err != nil {
		t.Fatalf("Check read: %v", err)
	}
	if !read.Allowed {
		t.Errorf("reader cannot read its own grant: %+v", read)
	}

	write, err := provider.Check(context.Background(), bob, ActionWrite, doc)
	if err != nil {
		t.Fatalf("Check write: %v", err)
	}
	if write.Allowed {
		t.Errorf("reader should not be able to write: %+v", write)
	}
	if !strings.Contains(write.Reason, "insufficient") {
		t.Errorf("reason %q missing 'insufficient'", write.Reason)
	}
}

func TestRbacUnknownSubjectDenied(t *testing.T) {
	eve := userSubject("eve")
	doc := documentRes("doc-1")
	provider := seed(t, GrantTuple(userSubject("alice"), RoleAdmin, doc))

	res, err := provider.Check(context.Background(), eve, ActionRead, doc)
	if err != nil {
		t.Fatalf("Check: %v", err)
	}
	if res.Allowed {
		t.Errorf("unknown subject was allowed: %+v", res)
	}
	if !strings.Contains(res.Reason, "no grant") {
		t.Errorf("reason %q missing 'no grant'", res.Reason)
	}
}

func TestRbacRejectsUnsupportedRelation(t *testing.T) {
	provider := NewRbacProvider(RbacOptions{})
	err := provider.Write(context.Background(), WriteTuplesRequest{
		Writes: []Tuple{{
			Subject:  userSubject("a"),
			Relation: "bogus",
			Resource: brainRes("x"),
		}},
	})
	if err == nil {
		t.Fatal("expected error for bogus relation")
	}
	if !errors.Is(err, ErrUnsupportedRelation) {
		t.Fatalf("error does not wrap ErrUnsupportedRelation: %v", err)
	}
}

func TestRbacReadReturnsStoredTuples(t *testing.T) {
	alice := userSubject("alice")
	br := brainRes("notes")
	provider := seed(t, GrantTuple(alice, RoleWriter, br))

	got, err := provider.Read(context.Background(), ReadTuplesQuery{Subject: &alice})
	if err != nil {
		t.Fatalf("Read: %v", err)
	}
	if len(got) != 1 {
		t.Fatalf("Read returned %d tuples, want 1", len(got))
	}
	if got[0].Relation != string(RoleWriter) {
		t.Errorf("relation = %q, want writer", got[0].Relation)
	}
}

func TestParentSubjectRoundTrip(t *testing.T) {
	parent := workspaceRes("acme")
	subj := ParentSubject(parent)
	if subj.Kind != SubjectService {
		t.Errorf("kind = %q, want service", subj.Kind)
	}
	got, ok := DecodeParentSubject(subj)
	if !ok {
		t.Fatal("DecodeParentSubject returned ok=false")
	}
	if got != parent {
		t.Fatalf("decoded = %+v, want %+v", got, parent)
	}
}

func TestDecodeParentSubjectRejectsBadInput(t *testing.T) {
	cases := []Subject{
		{Kind: SubjectUser, ID: "workspace:x"},     // wrong kind
		{Kind: SubjectService, ID: "no-colon"},     // missing separator
		{Kind: SubjectService, ID: "unknown:type"}, // unknown resource type
	}
	for _, s := range cases {
		if _, ok := DecodeParentSubject(s); ok {
			t.Errorf("DecodeParentSubject(%+v) = ok=true, want false", s)
		}
	}
}

func TestRbacWriteAcceptsParentAndDeny(t *testing.T) {
	provider := NewRbacProvider(RbacOptions{})
	err := provider.Write(context.Background(), WriteTuplesRequest{
		Writes: []Tuple{
			ParentTuple(brainRes("a"), workspaceRes("b")),
			DenyTuple(userSubject("c"), RoleReader, brainRes("a")),
			GrantTuple(userSubject("d"), RoleWriter, brainRes("a")),
		},
	})
	if err != nil {
		t.Fatalf("Write rejected supported relations: %v", err)
	}
}

func TestRbacCloseReturnsNil(t *testing.T) {
	provider := NewRbacProvider(RbacOptions{})
	if err := provider.Close(); err != nil {
		t.Fatalf("Close returned %v, want nil", err)
	}
	// Double-close must also be safe: the adapter holds no closeable
	// state, so a second call must remain a no-op.
	if err := provider.Close(); err != nil {
		t.Fatalf("second Close returned %v, want nil", err)
	}
}

func TestMemoryTupleStoreFiltering(t *testing.T) {
	store := NewMemoryTupleStore()
	store.Add(GrantTuple(userSubject("a"), RoleReader, brainRes("x")))
	store.Add(GrantTuple(userSubject("b"), RoleWriter, brainRes("x")))
	store.Add(GrantTuple(userSubject("a"), RoleReader, brainRes("y")))

	// No filter: everything.
	if got := store.List(ReadTuplesQuery{}); len(got) != 3 {
		t.Errorf("unfiltered list returned %d, want 3", len(got))
	}
	// Filter by subject only.
	a := userSubject("a")
	if got := store.List(ReadTuplesQuery{Subject: &a}); len(got) != 2 {
		t.Errorf("subject filter returned %d, want 2", len(got))
	}
	// Combined subject + resource.
	x := brainRes("x")
	if got := store.List(ReadTuplesQuery{Subject: &a, Resource: &x}); len(got) != 1 {
		t.Errorf("subject+resource filter returned %d, want 1", len(got))
	}
	// Remove and re-list.
	store.Remove(GrantTuple(userSubject("a"), RoleReader, brainRes("x")))
	if got := store.List(ReadTuplesQuery{Subject: &a, Resource: &x}); len(got) != 0 {
		t.Errorf("after remove, list returned %d, want 0", len(got))
	}
}
