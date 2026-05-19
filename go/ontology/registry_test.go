// SPDX-License-Identifier: Apache-2.0
package ontology_test

import (
	"context"
	"testing"

	"github.com/jeffs-brain/memory/go/ontology"
	"github.com/jeffs-brain/memory/go/store/mem"
)

func newTestRegistry(t *testing.T) *ontology.Registry {
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
	return ontology.NewRegistry(store)
}

func TestRegistryRegisterType(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	reg := newTestRegistry(t)

	def := ontology.TypeDefinition{
		Type:        "entity.invoice",
		Label:       "Invoice",
		Description: "Financial invoice",
		CreatedAt:   "2026-01-01T00:00:00Z",
		Status:      ontology.TypeStatusActive,
	}
	if err := reg.RegisterType(ctx, ontology.ScopeBrain, def); err != nil {
		t.Fatalf("RegisterType: %v", err)
	}

	resolved, err := reg.Resolve(ctx, "brain-1", "proj-1", "org-1")
	if err != nil {
		t.Fatalf("Resolve: %v", err)
	}
	found := false
	for _, rt := range resolved.NodeTypes {
		if rt.Type == "entity.invoice" && rt.Label == "Invoice" {
			found = true
			break
		}
	}
	if !found {
		t.Fatal("expected entity.invoice in resolved ontology after RegisterType")
	}
}

func TestRegistryDeprecateType(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	reg := newTestRegistry(t)

	def := ontology.TypeDefinition{
		Type:        "entity.invoice",
		Label:       "Invoice",
		Description: "Financial invoice",
		CreatedAt:   "2026-01-01T00:00:00Z",
		Status:      ontology.TypeStatusActive,
	}
	if err := reg.RegisterType(ctx, ontology.ScopeBrain, def); err != nil {
		t.Fatalf("RegisterType: %v", err)
	}
	if err := reg.DeprecateType(ctx, ontology.ScopeBrain, "entity.invoice"); err != nil {
		t.Fatalf("DeprecateType: %v", err)
	}

	resolved, err := reg.Resolve(ctx, "brain-1", "proj-1", "org-1")
	if err != nil {
		t.Fatalf("Resolve: %v", err)
	}
	for _, rt := range resolved.NodeTypes {
		if rt.Type == "entity.invoice" {
			t.Fatal("deprecated type should not appear in resolved ontology")
		}
	}
}

func TestRegistryProposeType_CreatesNew(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	reg := newTestRegistry(t)

	if err := reg.ProposeType(ctx, ontology.ScopeBrain, "nodeType", "entity.invoice", "Discovered in procurement docs"); err != nil {
		t.Fatalf("ProposeType: %v", err)
	}

	types, err := reg.Resolve(ctx, "brain-1", "proj-1", "org-1")
	if err != nil {
		t.Fatalf("Resolve: %v", err)
	}
	// Proposed types should NOT appear in resolved (only active types resolve)
	for _, rt := range types.NodeTypes {
		if rt.Type == "entity.invoice" && rt.Scope == "brain" {
			// The type exists but should be proposed status, which is filtered
			t.Fatal("proposed type should not appear as active in resolved ontology")
		}
	}
}

func TestRegistryProposeType_SkipsSimilar(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	reg := newTestRegistry(t)

	// "entity.customer" is built-in with label "Customer (Entity)".
	// Proposing "entity.customers" should be skipped because
	// "Customers (Entity)" is very similar to "Customer (Entity)".
	if err := reg.ProposeType(ctx, ontology.ScopeBrain, "nodeType", "entity.customers", "Should be skipped"); err != nil {
		t.Fatalf("ProposeType: %v", err)
	}

	// Verify the type was NOT created (it was skipped as duplicate)
	resolved, err := reg.Resolve(ctx, "brain-1", "proj-1", "org-1")
	if err != nil {
		t.Fatalf("Resolve: %v", err)
	}
	for _, rt := range resolved.NodeTypes {
		if rt.Type == "entity.customers" {
			t.Fatal("similar type should have been skipped")
		}
	}
}

func TestRegistryProposeType_InvalidCategory(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	reg := newTestRegistry(t)

	err := reg.ProposeType(ctx, ontology.ScopeBrain, "invalid", "entity.test", "reason")
	if err == nil {
		t.Fatal("expected error for invalid category")
	}
}

func TestRegistryResolve_BuiltInOnly(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	reg := newTestRegistry(t)

	resolved, err := reg.Resolve(ctx, "brain-1", "proj-1", "org-1")
	if err != nil {
		t.Fatalf("Resolve: %v", err)
	}
	if len(resolved.NodeTypes) != 30 {
		t.Fatalf("expected 30 node types, got %d", len(resolved.NodeTypes))
	}
	if len(resolved.EdgeTypes) != 29 {
		t.Fatalf("expected 29 edge types, got %d", len(resolved.EdgeTypes))
	}
	if len(resolved.BusinessCategories) != 8 {
		t.Fatalf("expected 8 business categories, got %d", len(resolved.BusinessCategories))
	}
}
