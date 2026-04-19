// SPDX-License-Identifier: Apache-2.0

package acl

import (
	"context"
	"fmt"
	"strings"
	"sync"
)

// Role enumerates the three collapsed roles the in-process RBAC
// adapter recognises. The adapter does not model the full OpenFGA
// vocabulary; consumers that need finer-grained relations should use
// the OpenFGA adapter instead.
type Role string

const (
	RoleAdmin  Role = "admin"
	RoleWriter Role = "writer"
	RoleReader Role = "reader"
)

// rbac semantics:
//
//  - Hierarchy is workspace > brain > collection > document.
//  - A grant of role R at any ancestor satisfies any action requiring
//    role R or lower, with admin > writer > reader.
//  - A deny tuple at any ancestor that names a role >= the required
//    role blocks the action, regardless of grants below it.
//  - Parent edges are themselves stored as tuples so the persistence
//    layer stays uniform.

const (
	// denyPrefix marks a deny tuple. The remainder of the relation is a
	// [Role] value: deny:admin, deny:writer, deny:reader.
	denyPrefix = "deny:"
	// parentRelation wires a child resource to its parent. The subject
	// of a parent tuple is the parent resource encoded via [ParentSubject].
	parentRelation = "parent"
)

// rolesByRank lists [Role] values from highest to lowest precedence.
// Encoded as a slice rather than a map so ordering is deterministic
// when iterating during evaluation.
var rolesByRank = []Role{RoleAdmin, RoleWriter, RoleReader}

// roleRank assigns a comparable priority to each role; admin > writer
// > reader. The default zero value of int means "unknown role" and
// must never satisfy a check.
var roleRank = map[Role]int{
	RoleAdmin:  3,
	RoleWriter: 2,
	RoleReader: 1,
}

// actionToMinRole maps each [Action] to the minimum [Role] required
// to satisfy it. Export is intentionally treated as a read-class
// action, matching the canonical OpenFGA model.
var actionToMinRole = map[Action]Role{
	ActionRead:   RoleReader,
	ActionWrite:  RoleWriter,
	ActionDelete: RoleAdmin,
	ActionAdmin:  RoleAdmin,
	ActionExport: RoleReader,
}

// TupleStore is the minimum persistence surface the RBAC provider
// needs. Implementations MAY back the store with Postgres, Redis or
// any other persistence layer; the in-memory implementation is
// supplied by [NewMemoryTupleStore].
type TupleStore interface {
	Add(t Tuple)
	Remove(t Tuple)
	List(query ReadTuplesQuery) []Tuple
}

// RbacOptions configures a new RBAC provider. Store defaults to a
// fresh in-memory store when nil.
type RbacOptions struct {
	Store TupleStore
}

// NewRbacProvider constructs an in-process RBAC provider. The provider
// is safe for concurrent use to the extent that the supplied
// [TupleStore] is.
func NewRbacProvider(opts RbacOptions) Provider {
	store := opts.Store
	if store == nil {
		store = NewMemoryTupleStore()
	}
	return &rbacProvider{store: store}
}

type rbacProvider struct {
	store TupleStore
}

func (p *rbacProvider) Name() string { return "rbac" }

func (p *rbacProvider) Check(ctx context.Context, subject Subject, action Action, resource Resource) (CheckResult, error) {
	required, ok := actionToMinRole[action]
	if !ok {
		return Deny(fmt.Sprintf("unknown action %q", action)), nil
	}
	chain := p.walkAncestors(resource)

	// A deny tuple anywhere from the target resource upwards that
	// covers the required role blocks the request outright. We check
	// denies before grants so a deny in a higher ancestor wins.
	for _, node := range chain {
		denied := p.deniedRolesAt(subject, node)
		for _, role := range rolesByRank {
			if !denied[role] {
				continue
			}
			if roleRank[role] >= roleRank[required] {
				return Deny(fmt.Sprintf("deny:%s on %s", role, ResourceKey(node))), nil
			}
		}
	}

	granted, ok := p.highestGrantedRole(subject, chain)
	if !ok {
		return Deny(fmt.Sprintf("no grant for %s on %s", SubjectKey(subject), ResourceKey(resource))), nil
	}
	if roleRank[granted] >= roleRank[required] {
		return Allow(fmt.Sprintf("%s grant satisfies %s", granted, required)), nil
	}
	return Deny(fmt.Sprintf("%s insufficient for %s", granted, required)), nil
}

func (p *rbacProvider) Write(ctx context.Context, req WriteTuplesRequest) error {
	for _, t := range req.Writes {
		if err := assertSupportedRelation(t.Relation); err != nil {
			return err
		}
	}
	for _, t := range req.Deletes {
		if err := assertSupportedRelation(t.Relation); err != nil {
			return err
		}
	}
	for _, t := range req.Writes {
		p.store.Add(t)
	}
	for _, t := range req.Deletes {
		p.store.Remove(t)
	}
	return nil
}

func (p *rbacProvider) Read(ctx context.Context, query ReadTuplesQuery) ([]Tuple, error) {
	return p.store.List(query), nil
}

// Close is a no-op for the in-process RBAC provider: the in-memory
// tuple store holds no releasable resources. The method exists to
// satisfy the [Provider] contract and keep lifecycle handling uniform
// across adapters.
func (p *rbacProvider) Close() error { return nil }

// walkAncestors returns the chain of resources from resource up to
// the topmost ancestor reachable via parent tuples. The first element
// is always resource itself. Cycles are guarded by the visited set.
func (p *rbacProvider) walkAncestors(resource Resource) []Resource {
	chain := []Resource{resource}
	seen := map[string]struct{}{ResourceKey(resource): {}}
	cursor := resource
	for {
		copyResource := cursor
		parents := p.store.List(ReadTuplesQuery{Resource: &copyResource, Relation: parentRelation})
		if len(parents) == 0 {
			break
		}
		parent, ok := DecodeParentSubject(parents[0].Subject)
		if !ok {
			break
		}
		key := ResourceKey(parent)
		if _, dup := seen[key]; dup {
			break
		}
		seen[key] = struct{}{}
		chain = append(chain, parent)
		cursor = parent
	}
	return chain
}

func (p *rbacProvider) highestGrantedRole(subject Subject, chain []Resource) (Role, bool) {
	var best Role
	bestRank := 0
	subj := subject
	for _, resource := range chain {
		res := resource
		for _, role := range rolesByRank {
			rows := p.store.List(ReadTuplesQuery{Subject: &subj, Relation: string(role), Resource: &res})
			if len(rows) > 0 && roleRank[role] > bestRank {
				best = role
				bestRank = roleRank[role]
			}
		}
	}
	return best, bestRank > 0
}

func (p *rbacProvider) deniedRolesAt(subject Subject, resource Resource) map[Role]bool {
	denied := make(map[Role]bool, len(rolesByRank))
	subj := subject
	res := resource
	for _, role := range rolesByRank {
		rows := p.store.List(ReadTuplesQuery{Subject: &subj, Relation: denyPrefix + string(role), Resource: &res})
		if len(rows) > 0 {
			denied[role] = true
		}
	}
	return denied
}

// ParentSubject encodes a parent resource as a service-typed [Subject]
// so the relationship can be stored as a regular tuple. The mapping
// is symmetric with [DecodeParentSubject].
func ParentSubject(parent Resource) Subject {
	return Subject{Kind: SubjectService, ID: string(parent.Type) + ":" + parent.ID}
}

// ParentTuple builds the canonical parent tuple linking child to parent.
func ParentTuple(child, parent Resource) Tuple {
	return Tuple{Subject: ParentSubject(parent), Relation: parentRelation, Resource: child}
}

// GrantTuple builds a grant tuple for subject at the given role on
// resource.
func GrantTuple(subject Subject, role Role, resource Resource) Tuple {
	return Tuple{Subject: subject, Relation: string(role), Resource: resource}
}

// DenyTuple builds a deny tuple. Denies override any grant reachable
// at the same level or below.
func DenyTuple(subject Subject, role Role, resource Resource) Tuple {
	return Tuple{Subject: subject, Relation: denyPrefix + string(role), Resource: resource}
}

// DecodeParentSubject reverses [ParentSubject]. Returns false when the
// subject is not a recognisable parent encoding (wrong kind, missing
// separator, unknown resource type).
func DecodeParentSubject(s Subject) (Resource, bool) {
	if s.Kind != SubjectService {
		return Resource{}, false
	}
	idx := strings.IndexByte(s.ID, ':')
	if idx == -1 {
		return Resource{}, false
	}
	rt := ResourceType(s.ID[:idx])
	switch rt {
	case ResourceWorkspace, ResourceBrain, ResourceCollection, ResourceDocument:
		return Resource{Type: rt, ID: s.ID[idx+1:]}, true
	}
	return Resource{}, false
}

// assertSupportedRelation gates [rbacProvider.Write] to the small,
// fixed vocabulary the adapter understands. Anything else is rejected
// with [ErrUnsupportedRelation] so callers can errors.Is on it.
func assertSupportedRelation(relation string) error {
	if relation == parentRelation {
		return nil
	}
	if rest, ok := strings.CutPrefix(relation, denyPrefix); ok {
		if isRole(rest) {
			return nil
		}
	}
	if isRole(relation) {
		return nil
	}
	return fmt.Errorf("rbac: %q: %w", relation, ErrUnsupportedRelation)
}

func isRole(s string) bool {
	switch Role(s) {
	case RoleAdmin, RoleWriter, RoleReader:
		return true
	}
	return false
}

// NewMemoryTupleStore returns an in-memory [TupleStore] backed by a
// map keyed by [TupleKey]. The store is safe for concurrent use.
func NewMemoryTupleStore() TupleStore {
	return &memoryTupleStore{rows: make(map[string]Tuple)}
}

type memoryTupleStore struct {
	mu   sync.RWMutex
	rows map[string]Tuple
}

func (s *memoryTupleStore) Add(t Tuple) {
	s.mu.Lock()
	s.rows[TupleKey(t)] = t
	s.mu.Unlock()
}

func (s *memoryTupleStore) Remove(t Tuple) {
	s.mu.Lock()
	delete(s.rows, TupleKey(t))
	s.mu.Unlock()
}

func (s *memoryTupleStore) List(query ReadTuplesQuery) []Tuple {
	s.mu.RLock()
	defer s.mu.RUnlock()
	out := make([]Tuple, 0)
	for _, t := range s.rows {
		if query.Subject != nil && SubjectKey(*query.Subject) != SubjectKey(t.Subject) {
			continue
		}
		if query.Relation != "" && query.Relation != t.Relation {
			continue
		}
		if query.Resource != nil && ResourceKey(*query.Resource) != ResourceKey(t.Resource) {
			continue
		}
		out = append(out, t)
	}
	return out
}
