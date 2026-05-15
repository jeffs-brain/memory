// SPDX-License-Identifier: Apache-2.0
package ontology_test

import (
	"context"
	"testing"
	"time"

	"github.com/jeffs-brain/memory/go/ontology"
	"github.com/jeffs-brain/memory/go/store/mem"
)

// fixedClock returns a clock function that always returns the same time.
func fixedClock(t time.Time) func() time.Time {
	return func() time.Time { return t }
}

func newTestWorkflow(t *testing.T) (*ontology.ProposalWorkflow, *ontology.Registry) {
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
	reg := ontology.NewRegistry(store)
	wf := ontology.NewProposalWorkflow(ontology.ProposalWorkflowConfig{
		Registry:  reg,
		Store:     ms,
		BrainID:   "brain-1",
		ProjectID: "proj-1",
		OrgID:     "org-1",
		Clock:     fixedClock(time.Date(2026, 5, 15, 12, 0, 0, 0, time.UTC)),
	})
	return wf, reg
}

func sampleExtraction() ontology.ExtractionResult {
	return ontology.ExtractionResult{
		NodeTypes: []ontology.TypeEntry{
			{Type: "entity.invoice", Label: "Invoice", Description: "A financial invoice document"},
			{Type: "entity.warehouse", Label: "Warehouse", Description: "Physical storage location"},
		},
		EdgeTypes: []ontology.TypeEntry{
			{Type: "ships_to", Label: "Ships To", Description: "Logistics shipping relationship"},
		},
		BusinessCategories: []string{"logistics"},
		Domain:             "logistics",
		Confidence:         0.85,
	}
}

func TestProposeFromExtraction_CreatesProposals(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	wf, _ := newTestWorkflow(t)

	batch, err := wf.ProposeFromExtraction(ctx, sampleExtraction(), "test-doc.pdf")
	if err != nil {
		t.Fatalf("ProposeFromExtraction: %v", err)
	}

	if batch.ID == "" {
		t.Fatal("expected non-empty batch ID")
	}
	if batch.Domain != "logistics" {
		t.Fatalf("expected domain %q, got %q", "logistics", batch.Domain)
	}
	if batch.SourceDocument != "test-doc.pdf" {
		t.Fatalf("expected source %q, got %q", "test-doc.pdf", batch.SourceDocument)
	}

	// All 3 types should become proposals (2 node + 1 edge, none are built-in)
	if len(batch.Proposals) != 3 {
		t.Fatalf("expected 3 proposals, got %d", len(batch.Proposals))
	}

	for _, p := range batch.Proposals {
		if p.Status != ontology.ProposalStatusPending {
			t.Errorf("proposal %q should be pending, got %q", p.Type, p.Status)
		}
		if p.ID == "" {
			t.Errorf("proposal %q has empty ID", p.Type)
		}
	}
}

func TestProposeFromExtraction_DedupExcluded(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	wf, _ := newTestWorkflow(t)

	// Extract with types that match built-in types
	result := ontology.ExtractionResult{
		NodeTypes: []ontology.TypeEntry{
			{Type: "entity.customer", Label: "Customer (Entity)", Description: "A customer entity"},
			{Type: "entity.invoice", Label: "Invoice", Description: "A financial invoice"},
		},
		EdgeTypes: []ontology.TypeEntry{
			{Type: "triggers", Label: "Triggers", Description: "Triggers relationship"},
		},
		Domain: "mixed",
	}

	batch, err := wf.ProposeFromExtraction(ctx, result, "test.pdf")
	if err != nil {
		t.Fatalf("ProposeFromExtraction: %v", err)
	}

	// entity.customer and triggers are built-in, only entity.invoice should be proposed
	if len(batch.Proposals) != 1 {
		t.Fatalf("expected 1 proposal (entity.invoice), got %d", len(batch.Proposals))
	}
	if batch.Proposals[0].Type != "entity.invoice" {
		t.Fatalf("expected entity.invoice proposal, got %q", batch.Proposals[0].Type)
	}
}

func TestProposeFromExtraction_EmptyExtraction(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	wf, _ := newTestWorkflow(t)

	batch, err := wf.ProposeFromExtraction(ctx, ontology.ExtractionResult{}, "empty.pdf")
	if err != nil {
		t.Fatalf("ProposeFromExtraction: %v", err)
	}

	if len(batch.Proposals) != 0 {
		t.Fatalf("expected 0 proposals, got %d", len(batch.Proposals))
	}
}

func TestAccept_ActivatesType(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	wf, reg := newTestWorkflow(t)

	batch, err := wf.ProposeFromExtraction(ctx, sampleExtraction(), "test.pdf")
	if err != nil {
		t.Fatalf("ProposeFromExtraction: %v", err)
	}

	// Find the entity.invoice proposal
	var invoiceProposal *ontology.Proposal
	for i := range batch.Proposals {
		if batch.Proposals[i].Type == "entity.invoice" {
			invoiceProposal = &batch.Proposals[i]
			break
		}
	}
	if invoiceProposal == nil {
		t.Fatal("entity.invoice proposal not found")
	}

	if err := wf.Accept(ctx, batch.ID, invoiceProposal.ID, "reviewer@test.com"); err != nil {
		t.Fatalf("Accept: %v", err)
	}

	// Verify the type is now active in the registry
	resolved, err := reg.Resolve(ctx, "brain-1", "proj-1", "org-1")
	if err != nil {
		t.Fatalf("Resolve: %v", err)
	}
	found := false
	for _, rt := range resolved.NodeTypes {
		if rt.Type == "entity.invoice" {
			found = true
			break
		}
	}
	if !found {
		t.Fatal("accepted type entity.invoice not found in resolved ontology")
	}
}

func TestAccept_AlreadyAccepted(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	wf, _ := newTestWorkflow(t)

	batch, err := wf.ProposeFromExtraction(ctx, sampleExtraction(), "test.pdf")
	if err != nil {
		t.Fatalf("ProposeFromExtraction: %v", err)
	}

	pid := batch.Proposals[0].ID
	if err := wf.Accept(ctx, batch.ID, pid, "reviewer@test.com"); err != nil {
		t.Fatalf("first Accept: %v", err)
	}
	// Second accept should be idempotent
	if err := wf.Accept(ctx, batch.ID, pid, "reviewer@test.com"); err != nil {
		t.Fatalf("second Accept (idempotent): %v", err)
	}
}

func TestMerge_SetsTarget(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	wf, _ := newTestWorkflow(t)

	batch, err := wf.ProposeFromExtraction(ctx, sampleExtraction(), "test.pdf")
	if err != nil {
		t.Fatalf("ProposeFromExtraction: %v", err)
	}

	pid := batch.Proposals[0].ID
	if err := wf.Merge(ctx, batch.ID, pid, "entity.customer", "reviewer@test.com"); err != nil {
		t.Fatalf("Merge: %v", err)
	}

	// Verify the proposal is marked as merged
	updated, err := wf.GetBatch(ctx, batch.ID)
	if err != nil {
		t.Fatalf("GetBatch: %v", err)
	}
	for _, p := range updated.Proposals {
		if p.ID == pid {
			if p.Status != ontology.ProposalStatusMerged {
				t.Fatalf("expected merged status, got %q", p.Status)
			}
			if p.MergedInto != "entity.customer" {
				t.Fatalf("expected mergedInto %q, got %q", "entity.customer", p.MergedInto)
			}
			if p.ReviewedBy != "reviewer@test.com" {
				t.Fatalf("expected reviewedBy %q, got %q", "reviewer@test.com", p.ReviewedBy)
			}
			return
		}
	}
	t.Fatal("proposal not found after merge")
}

func TestReject_SetsStatus(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	wf, reg := newTestWorkflow(t)

	batch, err := wf.ProposeFromExtraction(ctx, sampleExtraction(), "test.pdf")
	if err != nil {
		t.Fatalf("ProposeFromExtraction: %v", err)
	}

	pid := batch.Proposals[0].ID
	typeName := batch.Proposals[0].Type
	if err := wf.Reject(ctx, batch.ID, pid, "reviewer@test.com"); err != nil {
		t.Fatalf("Reject: %v", err)
	}

	// Verify proposal is rejected
	updated, err := wf.GetBatch(ctx, batch.ID)
	if err != nil {
		t.Fatalf("GetBatch: %v", err)
	}
	for _, p := range updated.Proposals {
		if p.ID == pid {
			if p.Status != ontology.ProposalStatusRejected {
				t.Fatalf("expected rejected status, got %q", p.Status)
			}
			break
		}
	}

	// Verify rejected type is NOT in the registry
	resolved, err := reg.Resolve(ctx, "brain-1", "proj-1", "org-1")
	if err != nil {
		t.Fatalf("Resolve: %v", err)
	}
	for _, rt := range resolved.NodeTypes {
		if rt.Type == typeName && rt.Scope == "brain" {
			t.Fatal("rejected type should not appear in resolved ontology")
		}
	}
}

func TestAcceptAll_BulkActivates(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	wf, reg := newTestWorkflow(t)

	batch, err := wf.ProposeFromExtraction(ctx, sampleExtraction(), "test.pdf")
	if err != nil {
		t.Fatalf("ProposeFromExtraction: %v", err)
	}

	if err := wf.AcceptAll(ctx, batch.ID, "reviewer@test.com"); err != nil {
		t.Fatalf("AcceptAll: %v", err)
	}

	// Verify all proposals are accepted
	updated, err := wf.GetBatch(ctx, batch.ID)
	if err != nil {
		t.Fatalf("GetBatch: %v", err)
	}
	for _, p := range updated.Proposals {
		if p.Status != ontology.ProposalStatusAccepted {
			t.Fatalf("expected accepted status for %q, got %q", p.Type, p.Status)
		}
		if p.ReviewedBy != "reviewer@test.com" {
			t.Fatalf("expected reviewedBy %q, got %q", "reviewer@test.com", p.ReviewedBy)
		}
	}

	// Verify types are in resolved ontology
	resolved, err := reg.Resolve(ctx, "brain-1", "proj-1", "org-1")
	if err != nil {
		t.Fatalf("Resolve: %v", err)
	}
	expectedTypes := map[string]bool{
		"entity.invoice":   false,
		"entity.warehouse": false,
		"ships_to":         false,
	}
	for _, rt := range resolved.NodeTypes {
		if _, ok := expectedTypes[rt.Type]; ok {
			expectedTypes[rt.Type] = true
		}
	}
	for _, rt := range resolved.EdgeTypes {
		if _, ok := expectedTypes[rt.Type]; ok {
			expectedTypes[rt.Type] = true
		}
	}
	for typeName, found := range expectedTypes {
		if !found {
			t.Errorf("expected type %q in resolved ontology after AcceptAll", typeName)
		}
	}
}

func TestList_FilterByStatus(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	wf, _ := newTestWorkflow(t)

	batch, err := wf.ProposeFromExtraction(ctx, sampleExtraction(), "test.pdf")
	if err != nil {
		t.Fatalf("ProposeFromExtraction: %v", err)
	}

	// Accept one, reject one
	if err := wf.Accept(ctx, batch.ID, batch.Proposals[0].ID, "reviewer"); err != nil {
		t.Fatalf("Accept: %v", err)
	}
	if err := wf.Reject(ctx, batch.ID, batch.Proposals[1].ID, "reviewer"); err != nil {
		t.Fatalf("Reject: %v", err)
	}

	// List pending only
	pending, err := wf.List(ctx, ontology.ProposalFilter{Status: ontology.ProposalStatusPending})
	if err != nil {
		t.Fatalf("List pending: %v", err)
	}
	totalPending := 0
	for _, b := range pending {
		totalPending += len(b.Proposals)
	}
	if totalPending != 1 {
		t.Fatalf("expected 1 pending proposal, got %d", totalPending)
	}

	// List accepted
	accepted, err := wf.List(ctx, ontology.ProposalFilter{Status: ontology.ProposalStatusAccepted})
	if err != nil {
		t.Fatalf("List accepted: %v", err)
	}
	totalAccepted := 0
	for _, b := range accepted {
		totalAccepted += len(b.Proposals)
	}
	if totalAccepted != 1 {
		t.Fatalf("expected 1 accepted proposal, got %d", totalAccepted)
	}
}

func TestList_CategoryFilter(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	wf, _ := newTestWorkflow(t)

	_, err := wf.ProposeFromExtraction(ctx, sampleExtraction(), "test.pdf")
	if err != nil {
		t.Fatalf("ProposeFromExtraction: %v", err)
	}

	// Filter by nodeType category -- sampleExtraction has 2 node + 1 edge
	nodeOnly, err := wf.List(ctx, ontology.ProposalFilter{Category: "nodeType"})
	if err != nil {
		t.Fatalf("List nodeType: %v", err)
	}
	totalNodeType := 0
	for _, b := range nodeOnly {
		totalNodeType += len(b.Proposals)
	}
	if totalNodeType != 2 {
		t.Fatalf("expected 2 nodeType proposals, got %d", totalNodeType)
	}

	// Filter by edgeType category
	edgeOnly, err := wf.List(ctx, ontology.ProposalFilter{Category: "edgeType"})
	if err != nil {
		t.Fatalf("List edgeType: %v", err)
	}
	totalEdgeType := 0
	for _, b := range edgeOnly {
		totalEdgeType += len(b.Proposals)
	}
	if totalEdgeType != 1 {
		t.Fatalf("expected 1 edgeType proposal, got %d", totalEdgeType)
	}
}

func TestList_NoFilter(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	wf, _ := newTestWorkflow(t)

	_, err := wf.ProposeFromExtraction(ctx, sampleExtraction(), "test.pdf")
	if err != nil {
		t.Fatalf("ProposeFromExtraction: %v", err)
	}

	all, err := wf.List(ctx, ontology.ProposalFilter{})
	if err != nil {
		t.Fatalf("List: %v", err)
	}
	if len(all) != 1 {
		t.Fatalf("expected 1 batch, got %d", len(all))
	}
	if len(all[0].Proposals) != 3 {
		t.Fatalf("expected 3 proposals, got %d", len(all[0].Proposals))
	}
}

func TestStoragePersistence(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
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
	reg := ontology.NewRegistry(store)

	clock := fixedClock(time.Date(2026, 5, 15, 12, 0, 0, 0, time.UTC))

	wf1 := ontology.NewProposalWorkflow(ontology.ProposalWorkflowConfig{
		Registry:  reg,
		Store:     ms,
		BrainID:   "brain-1",
		ProjectID: "proj-1",
		OrgID:     "org-1",
		Clock:     clock,
	})

	batch, err := wf1.ProposeFromExtraction(ctx, sampleExtraction(), "test.pdf")
	if err != nil {
		t.Fatalf("ProposeFromExtraction: %v", err)
	}

	// Create a new workflow instance (simulating restart)
	wf2 := ontology.NewProposalWorkflow(ontology.ProposalWorkflowConfig{
		Registry:  reg,
		Store:     ms,
		BrainID:   "brain-1",
		ProjectID: "proj-1",
		OrgID:     "org-1",
		Clock:     clock,
	})

	// Should be able to read the batch from the second workflow
	readBack, err := wf2.GetBatch(ctx, batch.ID)
	if err != nil {
		t.Fatalf("GetBatch on new workflow: %v", err)
	}
	if len(readBack.Proposals) != len(batch.Proposals) {
		t.Fatalf("expected %d proposals, got %d", len(batch.Proposals), len(readBack.Proposals))
	}
}

func TestProposalCategories(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	wf, _ := newTestWorkflow(t)

	batch, err := wf.ProposeFromExtraction(ctx, sampleExtraction(), "test.pdf")
	if err != nil {
		t.Fatalf("ProposeFromExtraction: %v", err)
	}

	nodeTypeCount := 0
	edgeTypeCount := 0
	for _, p := range batch.Proposals {
		switch p.Category {
		case "nodeType":
			nodeTypeCount++
		case "edgeType":
			edgeTypeCount++
		default:
			t.Errorf("unexpected category %q for proposal %q", p.Category, p.Type)
		}
	}
	if nodeTypeCount != 2 {
		t.Errorf("expected 2 nodeType proposals, got %d", nodeTypeCount)
	}
	if edgeTypeCount != 1 {
		t.Errorf("expected 1 edgeType proposal, got %d", edgeTypeCount)
	}
}

func TestReject_CannotRejectAccepted(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	wf, _ := newTestWorkflow(t)

	batch, err := wf.ProposeFromExtraction(ctx, sampleExtraction(), "test.pdf")
	if err != nil {
		t.Fatalf("ProposeFromExtraction: %v", err)
	}

	pid := batch.Proposals[0].ID
	if err := wf.Accept(ctx, batch.ID, pid, "reviewer"); err != nil {
		t.Fatalf("Accept: %v", err)
	}

	err = wf.Reject(ctx, batch.ID, pid, "reviewer")
	if err == nil {
		t.Fatal("expected error when rejecting already-accepted proposal")
	}
}

func TestProposalRoundTrip(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	wf, reg := newTestWorkflow(t)

	// Extract -> propose -> accept -> verify in resolved ontology
	result := ontology.ExtractionResult{
		NodeTypes: []ontology.TypeEntry{
			{Type: "entity.invoice", Label: "Invoice", Description: "A financial invoice"},
		},
		Domain: "finance",
	}

	batch, err := wf.ProposeFromExtraction(ctx, result, "finance.pdf")
	if err != nil {
		t.Fatalf("ProposeFromExtraction: %v", err)
	}
	if len(batch.Proposals) != 1 {
		t.Fatalf("expected 1 proposal, got %d", len(batch.Proposals))
	}

	// Before accept: type should not be in resolved
	resolved, err := reg.Resolve(ctx, "brain-1", "proj-1", "org-1")
	if err != nil {
		t.Fatalf("Resolve: %v", err)
	}
	for _, rt := range resolved.NodeTypes {
		if rt.Type == "entity.invoice" && rt.Scope == "brain" {
			t.Fatal("proposed type should not appear in resolved ontology before accept")
		}
	}

	// Accept
	if err := wf.Accept(ctx, batch.ID, batch.Proposals[0].ID, "reviewer@test.com"); err != nil {
		t.Fatalf("Accept: %v", err)
	}

	// After accept: type should be in resolved
	resolved, err = reg.Resolve(ctx, "brain-1", "proj-1", "org-1")
	if err != nil {
		t.Fatalf("Resolve: %v", err)
	}
	found := false
	for _, rt := range resolved.NodeTypes {
		if rt.Type == "entity.invoice" {
			found = true
			if rt.Scope != "brain" {
				t.Fatalf("expected scope brain, got %q", rt.Scope)
			}
			break
		}
	}
	if !found {
		t.Fatal("accepted type entity.invoice not found in resolved ontology")
	}
}
