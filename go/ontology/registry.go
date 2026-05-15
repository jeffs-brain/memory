// SPDX-License-Identifier: Apache-2.0
package ontology

import (
	"context"
	"fmt"
	"strings"
	"time"
)

// ProposeDedupThreshold is the Jaro-Winkler similarity threshold above
// which ProposeType skips registration (considers the type a duplicate).
const ProposeDedupThreshold = 0.85

// Registry provides high-level ontology operations on top of an
// OntologyStore. It adds ProposeType with fuzzy dedup guard,
// RegisterType, DeprecateType, and ApplyTemplate.
type Registry struct {
	store OntologyStore
}

// NewRegistry creates a Registry backed by the given OntologyStore.
func NewRegistry(store OntologyStore) *Registry {
	return &Registry{store: store}
}

// RegisterType adds or updates a type at the specified scope.
func (r *Registry) RegisterType(ctx context.Context, scope Scope, def TypeDefinition) error {
	return r.store.UpsertType(ctx, scope, def)
}

// DeprecateType marks a type as deprecated at the specified scope.
func (r *Registry) DeprecateType(ctx context.Context, scope Scope, typeName string) error {
	existing, err := r.store.GetType(ctx, scope, typeName)
	if err != nil {
		return fmt.Errorf("ontology: deprecate type %q at %s: %w", typeName, scope, err)
	}
	deprecated := *existing
	deprecated.Status = TypeStatusDeprecated
	return r.store.UpsertType(ctx, scope, deprecated)
}

// ProposeType creates a proposed type, skipping if a fuzzy-similar type
// already exists (Jaro-Winkler >= 0.85). Returns nil without error when
// the type is skipped as a duplicate.
func (r *Registry) ProposeType(ctx context.Context, scope Scope, category string, typeName string, reason string) error {
	resolved, err := r.store.GetResolvedOntology(ctx, "", "", "")
	if err != nil {
		return fmt.Errorf("ontology: propose type resolve: %w", err)
	}

	var existingLabels []string
	switch category {
	case "nodeType":
		for _, rt := range resolved.NodeTypes {
			existingLabels = append(existingLabels, rt.Label)
		}
	case "edgeType":
		for _, rt := range resolved.EdgeTypes {
			existingLabels = append(existingLabels, rt.Label)
		}
	default:
		return fmt.Errorf("ontology: category must be nodeType or edgeType, got %q", category)
	}

	proposedLabel := FormatNodeTypeLabel(typeName)
	if category == "edgeType" {
		proposedLabel = FormatEdgeTypeLabel(typeName)
	}

	for _, existing := range existingLabels {
		sim := jaroWinkler(strings.ToLower(proposedLabel), strings.ToLower(existing))
		if sim >= ProposeDedupThreshold {
			return nil
		}
	}

	now := time.Now().UTC().Format(time.RFC3339)
	def := TypeDefinition{
		Type:           typeName,
		Label:          proposedLabel,
		Description:    reason,
		DiscoveredFrom: "proposal",
		CreatedAt:      now,
		Status:         TypeStatusProposed,
	}
	return r.store.UpsertType(ctx, scope, def)
}

// Resolve returns the fully merged ontology from the store.
func (r *Registry) Resolve(ctx context.Context, brainID, projectID, orgID string) (*ResolvedOntology, error) {
	return r.store.GetResolvedOntology(ctx, brainID, projectID, orgID)
}

// jaroWinkler computes Jaro-Winkler similarity between two lowercase
// strings. This is a standalone implementation to avoid depending on
// the dedup package (P6-5).
func jaroWinkler(s1, s2 string) float64 {
	if s1 == s2 {
		return 1.0
	}
	if len(s1) == 0 || len(s2) == 0 {
		return 0.0
	}

	maxDist := len(s1)
	if len(s2) > maxDist {
		maxDist = len(s2)
	}
	matchWindow := maxDist/2 - 1
	if matchWindow < 0 {
		matchWindow = 0
	}

	s1Matches := make([]bool, len(s1))
	s2Matches := make([]bool, len(s2))

	matches := 0
	transpositions := 0

	for i := 0; i < len(s1); i++ {
		start := i - matchWindow
		if start < 0 {
			start = 0
		}
		end := i + matchWindow + 1
		if end > len(s2) {
			end = len(s2)
		}
		for j := start; j < end; j++ {
			if s2Matches[j] || s1[i] != s2[j] {
				continue
			}
			s1Matches[i] = true
			s2Matches[j] = true
			matches++
			break
		}
	}

	if matches == 0 {
		return 0.0
	}

	k := 0
	for i := 0; i < len(s1); i++ {
		if !s1Matches[i] {
			continue
		}
		for !s2Matches[k] {
			k++
		}
		if s1[i] != s2[k] {
			transpositions++
		}
		k++
	}

	jaro := (float64(matches)/float64(len(s1)) +
		float64(matches)/float64(len(s2)) +
		float64(matches-transpositions/2)/float64(matches)) / 3.0

	commonPrefix := 0
	limit := 4
	if len(s1) < limit {
		limit = len(s1)
	}
	if len(s2) < limit {
		limit = len(s2)
	}
	for i := 0; i < limit; i++ {
		if s1[i] != s2[i] {
			break
		}
		commonPrefix++
	}

	return jaro + float64(commonPrefix)*0.1*(1.0-jaro)
}
