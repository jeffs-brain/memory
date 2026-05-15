// SPDX-License-Identifier: Apache-2.0
package ontology

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"sort"
	"strings"
	"time"

	"github.com/jeffs-brain/memory/go/brain"
	"github.com/jeffs-brain/memory/go/ingest"
)

// ProposalStatus is the lifecycle state of a type proposal.
type ProposalStatus string

const (
	ProposalStatusPending  ProposalStatus = "proposed"
	ProposalStatusAccepted ProposalStatus = "accepted"
	ProposalStatusMerged   ProposalStatus = "merged"
	ProposalStatusRejected ProposalStatus = "rejected"
)

// Proposal is a candidate type awaiting review.
type Proposal struct {
	ID             string         `json:"id"`
	Type           string         `json:"type"`
	Label          string         `json:"label"`
	Description    string         `json:"description"`
	Category       string         `json:"category"` // "nodeType" | "edgeType"
	DiscoveredFrom string         `json:"discoveredFrom"`
	CreatedAt      string         `json:"createdAt"`
	Status         ProposalStatus `json:"status"`
	ReviewedBy     string         `json:"reviewedBy,omitempty"`
	ReviewedAt     string         `json:"reviewedAt,omitempty"`
	MergedInto     string         `json:"mergedInto,omitempty"`
}

// ProposalBatch groups proposals from a single extraction.
type ProposalBatch struct {
	ID             string     `json:"id"`
	Proposals      []Proposal `json:"proposals"`
	Domain         string     `json:"domain"`
	SourceDocument string     `json:"sourceDocument"`
	CreatedAt      string     `json:"createdAt"`
}

// ProposalFilter controls which proposals are returned by List.
type ProposalFilter struct {
	Status   ProposalStatus
	Category string
}

// ProposalWorkflow manages the type proposal lifecycle. It creates
// proposals from extraction results, runs deduplication to filter
// already-known types, and provides accept/merge/reject operations
// that transition types through the workflow.
type ProposalWorkflow struct {
	registry  *Registry
	store     brain.Store
	brainID   string
	projectID string
	orgID     string
	clock     func() time.Time
}

// ProposalWorkflowConfig configures the proposal workflow.
type ProposalWorkflowConfig struct {
	// Registry provides access to the resolved ontology and type registration.
	Registry *Registry
	// Store provides raw file access for reading/writing proposal batch JSON.
	Store brain.Store
	// BrainID is the brain to resolve the ontology for during proposal deduplication.
	BrainID string
	// ProjectID is the project to resolve the ontology for during proposal deduplication.
	ProjectID string
	// OrgID is the organisation to resolve the ontology for during proposal deduplication.
	OrgID string
	// Clock overrides time.Now for deterministic testing. Defaults to time.Now.
	Clock func() time.Time
}

// NewProposalWorkflow creates a workflow backed by the given registry
// and brain store.
func NewProposalWorkflow(cfg ProposalWorkflowConfig) *ProposalWorkflow {
	clock := cfg.Clock
	if clock == nil {
		clock = time.Now
	}
	return &ProposalWorkflow{
		registry:  cfg.Registry,
		store:     cfg.Store,
		brainID:   cfg.BrainID,
		projectID: cfg.ProjectID,
		orgID:     cfg.OrgID,
		clock:     clock,
	}
}

// proposalBatchPath returns the brain.Path for a proposal batch file.
func proposalBatchPath(batchID string) brain.Path {
	return brain.Path("ontology/proposals/" + batchID + ".json")
}

// proposalDirPath is the directory under which all proposal batches are stored.
const proposalDirPath = brain.Path("ontology/proposals/")

// proposalIDFromInputs computes a deterministic proposal ID from the
// type name and batch ID using BLAKE3 (first 16 hex characters).
func proposalIDFromInputs(typeName, batchID string) string {
	return ingest.HashString(typeName + ":" + batchID)[:16]
}

// computeBatchID computes a deterministic batch ID from the domain,
// source document, and timestamp using BLAKE3 (first 16 hex characters).
func computeBatchID(domain, sourceDocument, timestamp string) string {
	return ingest.HashString(domain + ":" + sourceDocument + ":" + timestamp)[:16]
}

// ProposeFromExtraction creates proposals for unique types from an
// extraction result. Runs deduplication against the resolved ontology
// first, so only genuinely new types become proposals.
func (w *ProposalWorkflow) ProposeFromExtraction(ctx context.Context, result ExtractionResult, sourceDocument string) (*ProposalBatch, error) {
	resolved, err := w.registry.Resolve(ctx, w.brainID, w.projectID, w.orgID)
	if err != nil {
		return nil, fmt.Errorf("ontology: propose from extraction resolve: %w", err)
	}

	existingTypes := resolvedToDefinitions(resolved)
	dedup := NewDeduplicator(nil)
	extractedDefs := extractionToDefinitions(result, w.clock)

	dedupResult, err := dedup.Deduplicate(ctx, extractedDefs, existingTypes)
	if err != nil {
		return nil, fmt.Errorf("ontology: propose from extraction dedup: %w", err)
	}

	now := w.clock().UTC().Format(time.RFC3339)
	bid := computeBatchID(result.Domain, sourceDocument, now)

	batch := &ProposalBatch{
		ID:             bid,
		Proposals:      make([]Proposal, 0, len(dedupResult.Unique)),
		Domain:         result.Domain,
		SourceDocument: sourceDocument,
		CreatedAt:      now,
	}

	for _, unique := range dedupResult.Unique {
		category := "nodeType"
		if HasPrefix(unique.Type) == "" {
			category = "edgeType"
		}
		pid := proposalIDFromInputs(unique.Type, bid)
		batch.Proposals = append(batch.Proposals, Proposal{
			ID:             pid,
			Type:           unique.Type,
			Label:          unique.Label,
			Description:    unique.Description,
			Category:       category,
			DiscoveredFrom: sourceDocument,
			CreatedAt:      now,
			Status:         ProposalStatusPending,
		})
	}

	if err := w.writeBatch(ctx, batch); err != nil {
		return nil, fmt.Errorf("ontology: propose from extraction write: %w", err)
	}

	return batch, nil
}

// Accept activates a proposed type in the registry at brain scope.
// The proposal status transitions from "proposed" to "accepted" and
// the type is registered as active. Idempotent for already-accepted
// proposals.
func (w *ProposalWorkflow) Accept(ctx context.Context, batchID, proposalID, reviewedBy string) error {
	batch, err := w.readBatch(ctx, batchID)
	if err != nil {
		return fmt.Errorf("ontology: accept proposal read batch %q: %w", batchID, err)
	}

	proposal, idx := findProposal(batch, proposalID)
	if proposal == nil {
		return fmt.Errorf("ontology: proposal %q not found in batch %q", proposalID, batchID)
	}

	if proposal.Status == ProposalStatusAccepted {
		return nil
	}
	if proposal.Status != ProposalStatusPending {
		return fmt.Errorf("ontology: cannot accept proposal %q with status %q", proposalID, proposal.Status)
	}

	now := w.clock().UTC().Format(time.RFC3339)
	label := proposal.Label
	if label == "" {
		if proposal.Category == "nodeType" {
			label = FormatNodeTypeLabel(proposal.Type)
		} else {
			label = FormatEdgeTypeLabel(proposal.Type)
		}
	}

	def := TypeDefinition{
		Type:           proposal.Type,
		Label:          label,
		Description:    proposal.Description,
		DiscoveredFrom: proposal.DiscoveredFrom,
		CreatedAt:      now,
		Status:         TypeStatusActive,
	}

	if err := w.registry.RegisterType(ctx, ScopeBrain, def); err != nil {
		return fmt.Errorf("ontology: accept proposal register: %w", err)
	}

	batch.Proposals[idx].Status = ProposalStatusAccepted
	batch.Proposals[idx].ReviewedBy = reviewedBy
	batch.Proposals[idx].ReviewedAt = now

	return w.writeBatch(ctx, batch)
}

// Merge marks a proposal as merged into an existing type. No registry
// change occurs; the decision is recorded for audit purposes.
// Idempotent for already-merged proposals.
func (w *ProposalWorkflow) Merge(ctx context.Context, batchID, proposalID, targetType, reviewedBy string) error {
	batch, err := w.readBatch(ctx, batchID)
	if err != nil {
		return fmt.Errorf("ontology: merge proposal read batch %q: %w", batchID, err)
	}

	proposal, idx := findProposal(batch, proposalID)
	if proposal == nil {
		return fmt.Errorf("ontology: proposal %q not found in batch %q", proposalID, batchID)
	}

	if proposal.Status == ProposalStatusMerged {
		return nil
	}
	if proposal.Status != ProposalStatusPending {
		return fmt.Errorf("ontology: cannot merge proposal %q with status %q", proposalID, proposal.Status)
	}

	now := w.clock().UTC().Format(time.RFC3339)
	batch.Proposals[idx].Status = ProposalStatusMerged
	batch.Proposals[idx].MergedInto = targetType
	batch.Proposals[idx].ReviewedBy = reviewedBy
	batch.Proposals[idx].ReviewedAt = now

	return w.writeBatch(ctx, batch)
}

// Reject marks a proposal as rejected. No registry change occurs.
// Idempotent for already-rejected proposals.
func (w *ProposalWorkflow) Reject(ctx context.Context, batchID, proposalID, reviewedBy string) error {
	batch, err := w.readBatch(ctx, batchID)
	if err != nil {
		return fmt.Errorf("ontology: reject proposal read batch %q: %w", batchID, err)
	}

	proposal, idx := findProposal(batch, proposalID)
	if proposal == nil {
		return fmt.Errorf("ontology: proposal %q not found in batch %q", proposalID, batchID)
	}

	if proposal.Status == ProposalStatusRejected {
		return nil
	}
	if proposal.Status != ProposalStatusPending {
		return fmt.Errorf("ontology: cannot reject proposal %q with status %q", proposalID, proposal.Status)
	}

	now := w.clock().UTC().Format(time.RFC3339)
	batch.Proposals[idx].Status = ProposalStatusRejected
	batch.Proposals[idx].ReviewedBy = reviewedBy
	batch.Proposals[idx].ReviewedAt = now

	return w.writeBatch(ctx, batch)
}

// AcceptAll bulk-accepts all pending proposals in a batch. Individual
// registration failures are collected rather than short-circuiting, and
// the batch is written atomically at the end regardless of partial
// failures so that successfully accepted proposals are persisted.
func (w *ProposalWorkflow) AcceptAll(ctx context.Context, batchID, reviewedBy string) error {
	batch, err := w.readBatch(ctx, batchID)
	if err != nil {
		return fmt.Errorf("ontology: accept all read batch %q: %w", batchID, err)
	}

	now := w.clock().UTC().Format(time.RFC3339)
	var errs []error

	for i := range batch.Proposals {
		if batch.Proposals[i].Status != ProposalStatusPending {
			continue
		}

		label := batch.Proposals[i].Label
		if label == "" {
			if batch.Proposals[i].Category == "nodeType" {
				label = FormatNodeTypeLabel(batch.Proposals[i].Type)
			} else {
				label = FormatEdgeTypeLabel(batch.Proposals[i].Type)
			}
		}

		def := TypeDefinition{
			Type:           batch.Proposals[i].Type,
			Label:          label,
			Description:    batch.Proposals[i].Description,
			DiscoveredFrom: batch.Proposals[i].DiscoveredFrom,
			CreatedAt:      now,
			Status:         TypeStatusActive,
		}

		if regErr := w.registry.RegisterType(ctx, ScopeBrain, def); regErr != nil {
			errs = append(errs, fmt.Errorf("register %q: %w", batch.Proposals[i].Type, regErr))
			continue
		}

		batch.Proposals[i].Status = ProposalStatusAccepted
		batch.Proposals[i].ReviewedBy = reviewedBy
		batch.Proposals[i].ReviewedAt = now
	}

	// Always write the batch so successfully accepted proposals are persisted.
	if writeErr := w.writeBatch(ctx, batch); writeErr != nil {
		errs = append(errs, fmt.Errorf("write batch: %w", writeErr))
	}

	if len(errs) > 0 {
		return fmt.Errorf("ontology: accept all batch %q: %w", batchID, errors.Join(errs...))
	}
	return nil
}

// List returns proposal batches matching the filter. All batches are
// scanned from the store.
func (w *ProposalWorkflow) List(ctx context.Context, filter ProposalFilter) ([]ProposalBatch, error) {
	batches, err := w.listBatches(ctx)
	if err != nil {
		return nil, fmt.Errorf("ontology: list proposals: %w", err)
	}

	if filter.Status == "" && filter.Category == "" {
		return batches, nil
	}

	filtered := make([]ProposalBatch, 0, len(batches))
	for _, batch := range batches {
		matchedProposals := make([]Proposal, 0, len(batch.Proposals))
		for _, p := range batch.Proposals {
			if filter.Status != "" && p.Status != filter.Status {
				continue
			}
			if filter.Category != "" && p.Category != filter.Category {
				continue
			}
			matchedProposals = append(matchedProposals, p)
		}
		if len(matchedProposals) > 0 {
			filteredBatch := batch
			filteredBatch.Proposals = matchedProposals
			filtered = append(filtered, filteredBatch)
		}
	}

	return filtered, nil
}

// GetBatch reads a single proposal batch by ID.
func (w *ProposalWorkflow) GetBatch(ctx context.Context, batchID string) (*ProposalBatch, error) {
	return w.readBatch(ctx, batchID)
}

// readBatch reads a proposal batch from the store. The batch ID is
// validated to prevent path traversal.
func (w *ProposalWorkflow) readBatch(ctx context.Context, id string) (*ProposalBatch, error) {
	if err := validateID(id); err != nil {
		return nil, fmt.Errorf("read batch: %w", err)
	}
	data, err := w.store.Read(ctx, proposalBatchPath(id))
	if err != nil {
		return nil, fmt.Errorf("read batch %q: %w", id, err)
	}
	var batch ProposalBatch
	if err := json.Unmarshal(data, &batch); err != nil {
		return nil, fmt.Errorf("unmarshal batch %q: %w", id, err)
	}
	return &batch, nil
}

// writeBatch writes a proposal batch to the store. The batch ID is
// validated to prevent path traversal.
func (w *ProposalWorkflow) writeBatch(ctx context.Context, batch *ProposalBatch) error {
	if err := validateID(batch.ID); err != nil {
		return fmt.Errorf("write batch: %w", err)
	}
	data, err := json.Marshal(batch)
	if err != nil {
		return fmt.Errorf("marshal batch %q: %w", batch.ID, err)
	}
	return w.store.Write(ctx, proposalBatchPath(batch.ID), data)
}

// listBatches reads all proposal batches from the store by listing
// files under ontology/proposals/.
func (w *ProposalWorkflow) listBatches(ctx context.Context) ([]ProposalBatch, error) {
	entries, err := w.store.List(ctx, proposalDirPath, brain.ListOpts{})
	if err != nil {
		if errors.Is(err, brain.ErrNotFound) {
			return nil, nil
		}
		return nil, fmt.Errorf("list proposal batches: %w", err)
	}

	batches := make([]ProposalBatch, 0, len(entries))
	for _, entry := range entries {
		if entry.IsDir || !strings.HasSuffix(string(entry.Path), ".json") {
			continue
		}
		data, readErr := w.store.Read(ctx, entry.Path)
		if readErr != nil {
			slog.Warn("ontology: skipping unreadable proposal batch", "path", string(entry.Path), "error", readErr)
			continue
		}
		var batch ProposalBatch
		if unmarshalErr := json.Unmarshal(data, &batch); unmarshalErr != nil {
			slog.Warn("ontology: skipping corrupt proposal batch", "path", string(entry.Path), "error", unmarshalErr)
			continue
		}
		batches = append(batches, batch)
	}

	sort.Slice(batches, func(i, j int) bool {
		return batches[i].CreatedAt < batches[j].CreatedAt
	})

	return batches, nil
}

// findProposal locates a proposal within a batch by ID.
func findProposal(batch *ProposalBatch, id string) (*Proposal, int) {
	for i := range batch.Proposals {
		if batch.Proposals[i].ID == id {
			return &batch.Proposals[i], i
		}
	}
	return nil, -1
}

// resolvedToDefinitions converts resolved types to TypeDefinitions for
// deduplication comparison.
func resolvedToDefinitions(resolved *ResolvedOntology) []TypeDefinition {
	defs := make([]TypeDefinition, 0, len(resolved.NodeTypes)+len(resolved.EdgeTypes))
	for _, rt := range resolved.NodeTypes {
		defs = append(defs, rt.TypeDefinition)
	}
	for _, rt := range resolved.EdgeTypes {
		defs = append(defs, rt.TypeDefinition)
	}
	return defs
}

// extractionToDefinitions converts extraction result entries to
// TypeDefinitions for deduplication comparison. Uses the injected
// clock instead of time.Now for deterministic testing.
func extractionToDefinitions(result ExtractionResult, clock func() time.Time) []TypeDefinition {
	now := clock().UTC().Format(time.RFC3339)
	defs := make([]TypeDefinition, 0, len(result.NodeTypes)+len(result.EdgeTypes))
	for _, nt := range result.NodeTypes {
		defs = append(defs, TypeDefinition{
			Type:        nt.Type,
			Label:       nt.Label,
			Description: nt.Description,
			CreatedAt:   now,
			Status:      TypeStatusProposed,
		})
	}
	for _, et := range result.EdgeTypes {
		defs = append(defs, TypeDefinition{
			Type:        et.Type,
			Label:       et.Label,
			Description: et.Description,
			CreatedAt:   now,
			Status:      TypeStatusProposed,
		})
	}
	return defs
}
