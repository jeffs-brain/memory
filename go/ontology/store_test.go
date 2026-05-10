// SPDX-License-Identifier: Apache-2.0
package ontology_test

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/jeffs-brain/memory/go/brain"
	"github.com/jeffs-brain/memory/go/ontology"
	"github.com/jeffs-brain/memory/go/store/mem"
)

func newTestStore(t *testing.T) (*ontology.FileOntologyStore, *mem.Store) {
	t.Helper()
	ms := mem.New()
	t.Cleanup(func() { _ = ms.Close() })
	cfg := ontology.FileOntologyStoreConfig{
		BrainID:   "brain-1",
		ProjectID: "proj-1",
		OrgID:     "org-1",
	}
	store, err := ontology.NewFileOntologyStore(ms, cfg)
	if err != nil {
		t.Fatalf("NewFileOntologyStore: %v", err)
	}
	return store, ms
}

func TestGetType_BuiltInNodeType(t *testing.T) {
	t.Parallel()
	s, _ := newTestStore(t)
	ctx := context.Background()

	def, err := s.GetType(ctx, ontology.ScopeBuiltIn, "entity.customer")
	if err != nil {
		t.Fatalf("GetType(built-in, entity.customer) = %v", err)
	}
	if def.Type != "entity.customer" {
		t.Fatalf("expected type entity.customer, got %q", def.Type)
	}
	if def.Status != ontology.TypeStatusActive {
		t.Fatalf("expected status active, got %q", def.Status)
	}
	if def.Label == "" {
		t.Fatal("expected non-empty label")
	}
	if def.Description == "" {
		t.Fatal("expected non-empty description")
	}
}

func TestGetType_BuiltInEdgeType(t *testing.T) {
	t.Parallel()
	s, _ := newTestStore(t)
	ctx := context.Background()

	def, err := s.GetType(ctx, ontology.ScopeBuiltIn, "triggers")
	if err != nil {
		t.Fatalf("GetType(built-in, triggers) = %v", err)
	}
	if def.Type != "triggers" {
		t.Fatalf("expected type triggers, got %q", def.Type)
	}
}

func TestGetType_BuiltInNotFound(t *testing.T) {
	t.Parallel()
	s, _ := newTestStore(t)
	ctx := context.Background()

	_, err := s.GetType(ctx, ontology.ScopeBuiltIn, "nonexistent.type")
	if err == nil {
		t.Fatal("expected error for nonexistent built-in type")
	}
}

func TestUpsertType_BrainScope(t *testing.T) {
	t.Parallel()
	s, _ := newTestStore(t)
	ctx := context.Background()

	def := ontology.TypeDefinition{
		Type:        "entity.invoice",
		Label:       "Invoice",
		Description: "Financial invoice documents",
		CreatedAt:   "2026-05-10T00:00:00.000Z",
		Status:      ontology.TypeStatusActive,
	}
	if err := s.UpsertType(ctx, ontology.ScopeBrain, def); err != nil {
		t.Fatalf("UpsertType = %v", err)
	}

	got, err := s.GetType(ctx, ontology.ScopeBrain, "entity.invoice")
	if err != nil {
		t.Fatalf("GetType after upsert = %v", err)
	}
	if got.Type != "entity.invoice" {
		t.Fatalf("expected entity.invoice, got %q", got.Type)
	}
	if got.Label != "Invoice" {
		t.Fatalf("expected label Invoice, got %q", got.Label)
	}
}

func TestUpsertType_OverwritesExisting(t *testing.T) {
	t.Parallel()
	s, _ := newTestStore(t)
	ctx := context.Background()

	def := ontology.TypeDefinition{
		Type:        "entity.invoice",
		Label:       "Invoice V1",
		Description: "First version",
		CreatedAt:   "2026-05-10T00:00:00.000Z",
		Status:      ontology.TypeStatusActive,
	}
	if err := s.UpsertType(ctx, ontology.ScopeBrain, def); err != nil {
		t.Fatalf("first UpsertType = %v", err)
	}

	def.Label = "Invoice V2"
	def.Description = "Second version"
	if err := s.UpsertType(ctx, ontology.ScopeBrain, def); err != nil {
		t.Fatalf("second UpsertType = %v", err)
	}

	got, err := s.GetType(ctx, ontology.ScopeBrain, "entity.invoice")
	if err != nil {
		t.Fatalf("GetType after overwrite = %v", err)
	}
	if got.Label != "Invoice V2" {
		t.Fatalf("expected Invoice V2, got %q", got.Label)
	}
}

func TestUpsertType_RejectsBuiltInScope(t *testing.T) {
	t.Parallel()
	s, _ := newTestStore(t)
	ctx := context.Background()

	def := ontology.TypeDefinition{
		Type:        "entity.test",
		Label:       "Test",
		Description: "A test type",
		CreatedAt:   "2026-05-10T00:00:00.000Z",
		Status:      ontology.TypeStatusActive,
	}
	err := s.UpsertType(ctx, ontology.ScopeBuiltIn, def)
	if err == nil {
		t.Fatal("expected error when upserting to built-in scope")
	}
}

func TestUpsertType_RejectsInvalidDefinition(t *testing.T) {
	t.Parallel()
	s, _ := newTestStore(t)
	ctx := context.Background()

	def := ontology.TypeDefinition{
		Type:   "entity.test",
		Label:  "",
		Status: ontology.TypeStatusActive,
	}
	err := s.UpsertType(ctx, ontology.ScopeBrain, def)
	if err == nil {
		t.Fatal("expected error for invalid definition")
	}
}

func TestDeleteType_RemovesFromScope(t *testing.T) {
	t.Parallel()
	s, _ := newTestStore(t)
	ctx := context.Background()

	def := ontology.TypeDefinition{
		Type:        "entity.invoice",
		Label:       "Invoice",
		Description: "Financial invoices",
		CreatedAt:   "2026-05-10T00:00:00.000Z",
		Status:      ontology.TypeStatusActive,
	}
	if err := s.UpsertType(ctx, ontology.ScopeBrain, def); err != nil {
		t.Fatalf("UpsertType = %v", err)
	}

	if err := s.DeleteType(ctx, ontology.ScopeBrain, "entity.invoice"); err != nil {
		t.Fatalf("DeleteType = %v", err)
	}

	_, err := s.GetType(ctx, ontology.ScopeBrain, "entity.invoice")
	if err == nil {
		t.Fatal("expected error after delete")
	}
}

func TestDeleteType_NotFoundReturnsError(t *testing.T) {
	t.Parallel()
	s, _ := newTestStore(t)
	ctx := context.Background()

	err := s.DeleteType(ctx, ontology.ScopeBrain, "entity.nonexistent")
	if err == nil {
		t.Fatal("expected error when deleting nonexistent type")
	}
}

func TestDeleteType_RejectsBuiltInScope(t *testing.T) {
	t.Parallel()
	s, _ := newTestStore(t)
	ctx := context.Background()

	err := s.DeleteType(ctx, ontology.ScopeBuiltIn, "entity.customer")
	if err == nil {
		t.Fatal("expected error when deleting from built-in scope")
	}
}

func TestListTypes_BuiltIn(t *testing.T) {
	t.Parallel()
	s, _ := newTestStore(t)
	ctx := context.Background()

	types, err := s.ListTypes(ctx, ontology.ScopeBuiltIn, ontology.ListTypesOpts{})
	if err != nil {
		t.Fatalf("ListTypes(built-in) = %v", err)
	}
	if len(types) != 50 {
		t.Fatalf("expected 50 built-in types (31 nodes + 19 edges), got %d", len(types))
	}
}

func TestListTypes_FilterByPrefix(t *testing.T) {
	t.Parallel()
	s, _ := newTestStore(t)
	ctx := context.Background()

	types, err := s.ListTypes(ctx, ontology.ScopeBuiltIn, ontology.ListTypesOpts{Prefix: "entity."})
	if err != nil {
		t.Fatalf("ListTypes(prefix=entity.) = %v", err)
	}
	if len(types) != 6 {
		t.Fatalf("expected 6 entity types, got %d", len(types))
	}
	for _, td := range types {
		if !hasEntityPrefix(td.Type) {
			t.Fatalf("unexpected type %q in entity prefix filter", td.Type)
		}
	}
}

func TestListTypes_FilterByStatus(t *testing.T) {
	t.Parallel()
	s, _ := newTestStore(t)
	ctx := context.Background()

	active := ontology.TypeDefinition{
		Type:        "entity.active_one",
		Label:       "Active One",
		Description: "An active type",
		CreatedAt:   "2026-05-10T00:00:00.000Z",
		Status:      ontology.TypeStatusActive,
	}
	proposed := ontology.TypeDefinition{
		Type:        "entity.proposed_one",
		Label:       "Proposed One",
		Description: "A proposed type",
		CreatedAt:   "2026-05-10T00:00:00.000Z",
		Status:      ontology.TypeStatusProposed,
	}
	if err := s.UpsertType(ctx, ontology.ScopeBrain, active); err != nil {
		t.Fatalf("UpsertType active = %v", err)
	}
	if err := s.UpsertType(ctx, ontology.ScopeBrain, proposed); err != nil {
		t.Fatalf("UpsertType proposed = %v", err)
	}

	types, err := s.ListTypes(ctx, ontology.ScopeBrain, ontology.ListTypesOpts{Status: "proposed"})
	if err != nil {
		t.Fatalf("ListTypes(status=proposed) = %v", err)
	}
	if len(types) != 1 {
		t.Fatalf("expected 1 proposed type, got %d", len(types))
	}
	if types[0].Type != "entity.proposed_one" {
		t.Fatalf("expected entity.proposed_one, got %q", types[0].Type)
	}
}

func TestGetResolvedOntology_BuiltInOnly(t *testing.T) {
	t.Parallel()
	s, _ := newTestStore(t)
	ctx := context.Background()

	resolved, err := s.GetResolvedOntology(ctx, "brain-1", "proj-1", "org-1")
	if err != nil {
		t.Fatalf("GetResolvedOntology = %v", err)
	}
	if len(resolved.NodeTypes) != 31 {
		t.Fatalf("expected 31 node types, got %d", len(resolved.NodeTypes))
	}
	if len(resolved.EdgeTypes) != 19 {
		t.Fatalf("expected 19 edge types, got %d", len(resolved.EdgeTypes))
	}
	if len(resolved.BusinessCategories) != 8 {
		t.Fatalf("expected 8 categories, got %d", len(resolved.BusinessCategories))
	}
}

func TestGetResolvedOntology_BrainOverridesBuiltIn(t *testing.T) {
	t.Parallel()
	s, _ := newTestStore(t)
	ctx := context.Background()

	override := ontology.TypeDefinition{
		Type:        "entity.customer",
		Label:       "Custom Customer",
		Description: "Brain-level override of built-in customer type",
		CreatedAt:   "2026-05-10T00:00:00.000Z",
		Status:      ontology.TypeStatusActive,
	}
	if err := s.UpsertType(ctx, ontology.ScopeBrain, override); err != nil {
		t.Fatalf("UpsertType = %v", err)
	}

	resolved, err := s.GetResolvedOntology(ctx, "brain-1", "proj-1", "org-1")
	if err != nil {
		t.Fatalf("GetResolvedOntology = %v", err)
	}

	for _, rt := range resolved.NodeTypes {
		if rt.Type == "entity.customer" {
			if rt.Label != "Custom Customer" {
				t.Fatalf("expected brain override label, got %q", rt.Label)
			}
			if rt.Scope != ontology.ScopeBrain {
				t.Fatalf("expected scope brain, got %q", rt.Scope)
			}
			return
		}
	}
	t.Fatal("entity.customer not found in resolved ontology")
}

func TestGetResolvedOntology_ScopeHierarchy(t *testing.T) {
	t.Parallel()
	s, _ := newTestStore(t)
	ctx := context.Background()

	orgDef := ontology.TypeDefinition{
		Type:        "entity.tenant",
		Label:       "Org Tenant",
		Description: "Organisation-level tenant",
		CreatedAt:   "2026-05-10T00:00:00.000Z",
		Status:      ontology.TypeStatusActive,
	}
	if err := s.UpsertType(ctx, ontology.ScopeOrganisation, orgDef); err != nil {
		t.Fatalf("UpsertType org = %v", err)
	}

	projectDef := ontology.TypeDefinition{
		Type:        "entity.tenant",
		Label:       "Project Tenant",
		Description: "Project-level tenant override",
		CreatedAt:   "2026-05-10T00:00:00.000Z",
		Status:      ontology.TypeStatusActive,
	}
	if err := s.UpsertType(ctx, ontology.ScopeProject, projectDef); err != nil {
		t.Fatalf("UpsertType project = %v", err)
	}

	brainDef := ontology.TypeDefinition{
		Type:        "entity.tenant",
		Label:       "Brain Tenant",
		Description: "Brain-level tenant override",
		CreatedAt:   "2026-05-10T00:00:00.000Z",
		Status:      ontology.TypeStatusActive,
	}
	if err := s.UpsertType(ctx, ontology.ScopeBrain, brainDef); err != nil {
		t.Fatalf("UpsertType brain = %v", err)
	}

	resolved, err := s.GetResolvedOntology(ctx, "brain-1", "proj-1", "org-1")
	if err != nil {
		t.Fatalf("GetResolvedOntology = %v", err)
	}

	for _, rt := range resolved.NodeTypes {
		if rt.Type == "entity.tenant" {
			if rt.Label != "Brain Tenant" {
				t.Fatalf("expected brain to shadow project and org, got label %q", rt.Label)
			}
			if rt.Scope != ontology.ScopeBrain {
				t.Fatalf("expected scope brain, got %q", rt.Scope)
			}
			return
		}
	}
	t.Fatal("entity.tenant not found in resolved ontology")
}

func TestGetResolvedOntology_ProjectOverridesOrg(t *testing.T) {
	t.Parallel()
	s, _ := newTestStore(t)
	ctx := context.Background()

	orgDef := ontology.TypeDefinition{
		Type:        "entity.account",
		Label:       "Org Account",
		Description: "Organisation-level account",
		CreatedAt:   "2026-05-10T00:00:00.000Z",
		Status:      ontology.TypeStatusActive,
	}
	if err := s.UpsertType(ctx, ontology.ScopeOrganisation, orgDef); err != nil {
		t.Fatalf("UpsertType org = %v", err)
	}

	projectDef := ontology.TypeDefinition{
		Type:        "entity.account",
		Label:       "Project Account",
		Description: "Project-level account override",
		CreatedAt:   "2026-05-10T00:00:00.000Z",
		Status:      ontology.TypeStatusActive,
	}
	if err := s.UpsertType(ctx, ontology.ScopeProject, projectDef); err != nil {
		t.Fatalf("UpsertType project = %v", err)
	}

	resolved, err := s.GetResolvedOntology(ctx, "brain-1", "proj-1", "org-1")
	if err != nil {
		t.Fatalf("GetResolvedOntology = %v", err)
	}

	for _, rt := range resolved.NodeTypes {
		if rt.Type == "entity.account" {
			if rt.Label != "Project Account" {
				t.Fatalf("expected project to shadow org, got label %q", rt.Label)
			}
			if rt.Scope != ontology.ScopeProject {
				t.Fatalf("expected scope project, got %q", rt.Scope)
			}
			return
		}
	}
	t.Fatal("entity.account not found in resolved ontology")
}

func TestGetResolvedOntology_DeprecatedExcluded(t *testing.T) {
	t.Parallel()
	s, _ := newTestStore(t)
	ctx := context.Background()

	deprecated := ontology.TypeDefinition{
		Type:        "entity.legacy",
		Label:       "Legacy Entity",
		Description: "A deprecated type",
		CreatedAt:   "2026-05-10T00:00:00.000Z",
		Status:      ontology.TypeStatusDeprecated,
	}
	if err := s.UpsertType(ctx, ontology.ScopeBrain, deprecated); err != nil {
		t.Fatalf("UpsertType = %v", err)
	}

	resolved, err := s.GetResolvedOntology(ctx, "brain-1", "proj-1", "org-1")
	if err != nil {
		t.Fatalf("GetResolvedOntology = %v", err)
	}

	for _, rt := range resolved.NodeTypes {
		if rt.Type == "entity.legacy" {
			t.Fatal("deprecated type should not appear in resolved ontology")
		}
	}
}

func TestGetResolvedOntology_CustomCategories(t *testing.T) {
	t.Parallel()
	_, ms := newTestStore(t)
	cfg := ontology.FileOntologyStoreConfig{
		BrainID:   "brain-1",
		ProjectID: "proj-1",
		OrgID:     "org-1",
	}
	s, err := ontology.NewFileOntologyStore(ms, cfg)
	if err != nil {
		t.Fatalf("NewFileOntologyStore: %v", err)
	}
	ctx := context.Background()

	stored := ontology.StoredOntology{
		CustomBusinessCategories: []string{"logistics", "healthcare"},
	}
	data, err := json.Marshal(stored)
	if err != nil {
		t.Fatalf("marshal = %v", err)
	}
	if err := ms.Write(ctx, brain.Path("ontology/org/org-1/types.json"), data); err != nil {
		t.Fatalf("Write = %v", err)
	}

	resolved, err := s.GetResolvedOntology(ctx, "brain-1", "proj-1", "org-1")
	if err != nil {
		t.Fatalf("GetResolvedOntology = %v", err)
	}

	categorySet := make(map[string]struct{}, len(resolved.BusinessCategories))
	for _, c := range resolved.BusinessCategories {
		categorySet[c] = struct{}{}
	}
	if _, ok := categorySet["logistics"]; !ok {
		t.Fatal("expected logistics category in resolved ontology")
	}
	if _, ok := categorySet["healthcare"]; !ok {
		t.Fatal("expected healthcare category in resolved ontology")
	}
	if _, ok := categorySet["customer"]; !ok {
		t.Fatal("expected built-in customer category in resolved ontology")
	}
}

func TestGetResolvedOntology_Cached(t *testing.T) {
	t.Parallel()
	s, _ := newTestStore(t)
	ctx := context.Background()

	resolved1, err := s.GetResolvedOntology(ctx, "brain-1", "proj-1", "org-1")
	if err != nil {
		t.Fatalf("first GetResolvedOntology = %v", err)
	}
	resolved2, err := s.GetResolvedOntology(ctx, "brain-1", "proj-1", "org-1")
	if err != nil {
		t.Fatalf("second GetResolvedOntology = %v", err)
	}
	if resolved1 != resolved2 {
		t.Fatal("expected cached pointer equality")
	}
}

func TestGetResolvedOntology_CacheInvalidatedOnUpsert(t *testing.T) {
	t.Parallel()
	s, _ := newTestStore(t)
	ctx := context.Background()

	resolved1, err := s.GetResolvedOntology(ctx, "brain-1", "proj-1", "org-1")
	if err != nil {
		t.Fatalf("first GetResolvedOntology = %v", err)
	}

	def := ontology.TypeDefinition{
		Type:        "entity.fresh",
		Label:       "Fresh",
		Description: "A fresh type",
		CreatedAt:   "2026-05-10T00:00:00.000Z",
		Status:      ontology.TypeStatusActive,
	}
	if err := s.UpsertType(ctx, ontology.ScopeBrain, def); err != nil {
		t.Fatalf("UpsertType = %v", err)
	}

	resolved2, err := s.GetResolvedOntology(ctx, "brain-1", "proj-1", "org-1")
	if err != nil {
		t.Fatalf("second GetResolvedOntology = %v", err)
	}
	if resolved1 == resolved2 {
		t.Fatal("expected cache to be invalidated after upsert")
	}
}

func TestUpsertType_EdgeType(t *testing.T) {
	t.Parallel()
	s, _ := newTestStore(t)
	ctx := context.Background()

	def := ontology.TypeDefinition{
		Type:        "manages",
		Label:       "Manages",
		Description: "Source manages target",
		CreatedAt:   "2026-05-10T00:00:00.000Z",
		Status:      ontology.TypeStatusActive,
	}
	if err := s.UpsertType(ctx, ontology.ScopeBrain, def); err != nil {
		t.Fatalf("UpsertType edge = %v", err)
	}

	got, err := s.GetType(ctx, ontology.ScopeBrain, "manages")
	if err != nil {
		t.Fatalf("GetType edge = %v", err)
	}
	if got.Type != "manages" {
		t.Fatalf("expected manages, got %q", got.Type)
	}
}

func hasEntityPrefix(s string) bool {
	return len(s) > 7 && s[:7] == "entity."
}
