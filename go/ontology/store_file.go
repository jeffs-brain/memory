// SPDX-License-Identifier: Apache-2.0
package ontology

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"regexp"
	"strings"
	"sync"

	"github.com/jeffs-brain/memory/go/brain"
)

// validIDPattern matches IDs that start with an alphanumeric character followed
// by zero or more alphanumeric, underscore, or hyphen characters.
var validIDPattern = regexp.MustCompile(`^[a-zA-Z0-9][a-zA-Z0-9_-]*$`)

// validateID checks that an ID is safe for path interpolation. IDs must start
// with an alphanumeric character and contain only alphanumeric, underscore, or
// hyphen characters. This prevents path traversal attacks via ".." or "/" in IDs.
func validateID(id string) error {
	if id == "" {
		return fmt.Errorf("ontology: ID must not be empty")
	}
	if !validIDPattern.MatchString(id) {
		return fmt.Errorf("ontology: invalid ID %q: must match ^[a-zA-Z0-9][a-zA-Z0-9_-]*$", id)
	}
	return nil
}

// FileOntologyStoreConfig holds the IDs used by FileOntologyStore to
// determine which scope files to read and write for CRUD operations.
type FileOntologyStoreConfig struct {
	BrainID   string
	ProjectID string
	OrgID     string
}

// FileOntologyStore implements OntologyStore using a brain.Store backend.
// Types are stored as JSON files at well-known paths:
//   - Organisation: ontology/org/{orgID}/types.json
//   - Project: ontology/project/{projectID}/types.json
//   - Brain: ontology/brain/{brainID}/types.json
//
// Built-in types are hardcoded and always available without any file I/O.
type FileOntologyStore struct {
	store  brain.Store
	config FileOntologyStoreConfig
	mu     sync.RWMutex
	cache  map[string]*ResolvedOntology
}

var _ OntologyStore = (*FileOntologyStore)(nil)

// NewFileOntologyStore creates a FileOntologyStore backed by the given store.
// Returns an error if any configured ID fails validation (path traversal prevention).
func NewFileOntologyStore(store brain.Store, cfg FileOntologyStoreConfig) (*FileOntologyStore, error) {
	if cfg.BrainID != "" {
		if err := validateID(cfg.BrainID); err != nil {
			return nil, fmt.Errorf("ontology: invalid BrainID: %w", err)
		}
	}
	if cfg.ProjectID != "" {
		if err := validateID(cfg.ProjectID); err != nil {
			return nil, fmt.Errorf("ontology: invalid ProjectID: %w", err)
		}
	}
	if cfg.OrgID != "" {
		if err := validateID(cfg.OrgID); err != nil {
			return nil, fmt.Errorf("ontology: invalid OrgID: %w", err)
		}
	}
	return &FileOntologyStore{
		store:  store,
		config: cfg,
		cache:  make(map[string]*ResolvedOntology),
	}, nil
}

// GetType returns a single type definition at the given scope.
func (f *FileOntologyStore) GetType(ctx context.Context, scope Scope, typeID string) (*TypeDefinition, error) {
	if scope == ScopeBuiltIn {
		return getBuiltInType(typeID)
	}
	stored, err := f.readScopeByID(ctx, scope, f.idForScope(scope))
	if err != nil {
		return nil, fmt.Errorf("ontology: get type %q at %s: %w", typeID, scope, err)
	}
	for i := range stored.CustomNodeTypes {
		if stored.CustomNodeTypes[i].Type == typeID {
			return &stored.CustomNodeTypes[i], nil
		}
	}
	for i := range stored.CustomEdgeTypes {
		if stored.CustomEdgeTypes[i].Type == typeID {
			return &stored.CustomEdgeTypes[i], nil
		}
	}
	return nil, fmt.Errorf("ontology: type %q not found at scope %s: %w", typeID, scope, brain.ErrNotFound)
}

// ListTypes returns all types at the given scope, optionally filtered.
func (f *FileOntologyStore) ListTypes(ctx context.Context, scope Scope, opts ListTypesOpts) ([]TypeDefinition, error) {
	if scope == ScopeBuiltIn {
		return listBuiltInTypes(opts), nil
	}
	stored, err := f.readScopeByID(ctx, scope, f.idForScope(scope))
	if err != nil {
		return nil, fmt.Errorf("ontology: list types at %s: %w", scope, err)
	}
	combined := make([]TypeDefinition, 0, len(stored.CustomNodeTypes)+len(stored.CustomEdgeTypes))
	combined = append(combined, stored.CustomNodeTypes...)
	combined = append(combined, stored.CustomEdgeTypes...)
	return filterTypes(combined, opts), nil
}

// UpsertType creates or updates a type definition at the given scope.
func (f *FileOntologyStore) UpsertType(ctx context.Context, scope Scope, def TypeDefinition) error {
	if scope == ScopeBuiltIn {
		return fmt.Errorf("ontology: cannot upsert to built-in scope")
	}
	if err := ValidateTypeDefinition(def); err != nil {
		return err
	}
	id := f.idForScope(scope)
	stored, err := f.readScopeByID(ctx, scope, id)
	if err != nil {
		return fmt.Errorf("ontology: upsert type at %s: %w", scope, err)
	}
	if HasPrefix(def.Type) != "" {
		stored.CustomNodeTypes = upsertInSlice(stored.CustomNodeTypes, def)
	} else {
		stored.CustomEdgeTypes = upsertInSlice(stored.CustomEdgeTypes, def)
	}
	if err := f.writeScopeByID(ctx, scope, id, stored); err != nil {
		return fmt.Errorf("ontology: upsert type at %s: %w", scope, err)
	}
	f.invalidateCache()
	return nil
}

// DeleteType removes a type from the given scope.
func (f *FileOntologyStore) DeleteType(ctx context.Context, scope Scope, typeID string) error {
	if scope == ScopeBuiltIn {
		return fmt.Errorf("ontology: cannot delete from built-in scope")
	}
	id := f.idForScope(scope)
	stored, err := f.readScopeByID(ctx, scope, id)
	if err != nil {
		return fmt.Errorf("ontology: delete type at %s: %w", scope, err)
	}
	var found bool
	stored.CustomNodeTypes, found = removeFromSlice(stored.CustomNodeTypes, typeID)
	if !found {
		stored.CustomEdgeTypes, found = removeFromSlice(stored.CustomEdgeTypes, typeID)
	}
	if !found {
		return fmt.Errorf("ontology: type %q not found at scope %s: %w", typeID, scope, brain.ErrNotFound)
	}
	if err := f.writeScopeByID(ctx, scope, id, stored); err != nil {
		return fmt.Errorf("ontology: delete type at %s: %w", scope, err)
	}
	f.invalidateCache()
	return nil
}

// GetResolvedOntology returns the fully merged ontology across all 4 layers.
func (f *FileOntologyStore) GetResolvedOntology(ctx context.Context, brainID, projectID, orgID string) (*ResolvedOntology, error) {
	for _, pair := range []struct{ name, id string }{
		{"brainID", brainID},
		{"projectID", projectID},
		{"orgID", orgID},
	} {
		if pair.id != "" {
			if err := validateID(pair.id); err != nil {
				return nil, fmt.Errorf("ontology: GetResolvedOntology invalid %s: %w", pair.name, err)
			}
		}
	}

	cacheKey := buildCacheKey(brainID, projectID, orgID)
	f.mu.RLock()
	cached, ok := f.cache[cacheKey]
	f.mu.RUnlock()
	if ok {
		return cached, nil
	}

	nodeMap := make(map[string]ResolvedType, len(BuiltInNodeTypes))
	edgeMap := make(map[string]ResolvedType, len(BuiltInEdgeTypes))
	categorySet := make(map[string]struct{}, len(BusinessCategories))

	for _, c := range BusinessCategories {
		categorySet[c] = struct{}{}
	}
	for _, nt := range BuiltInNodeTypes {
		nodeMap[nt] = ResolvedType{
			TypeDefinition: TypeDefinition{
				Type:        nt,
				Label:       FormatNodeTypeLabel(nt),
				Description: GetBuiltInNodeTypeDescription(nt),
				CreatedAt:   "1970-01-01T00:00:00.000Z",
				Status:      TypeStatusActive,
			},
			Scope: ScopeBuiltIn,
		}
	}
	for _, et := range BuiltInEdgeTypes {
		edgeMap[et] = ResolvedType{
			TypeDefinition: TypeDefinition{
				Type:        et,
				Label:       FormatEdgeTypeLabel(et),
				Description: GetBuiltInEdgeTypeDescription(et),
				CreatedAt:   "1970-01-01T00:00:00.000Z",
				Status:      TypeStatusActive,
			},
			Scope: ScopeBuiltIn,
		}
	}

	layers := []struct {
		scope Scope
		id    string
	}{
		{ScopeOrganisation, orgID},
		{ScopeProject, projectID},
		{ScopeBrain, brainID},
	}
	for _, layer := range layers {
		if layer.id == "" {
			continue
		}
		stored, err := f.readScopeByID(ctx, layer.scope, layer.id)
		if err != nil {
			continue
		}
		for _, nt := range stored.CustomNodeTypes {
			nodeMap[nt.Type] = ResolvedType{TypeDefinition: nt, Scope: layer.scope}
		}
		for _, et := range stored.CustomEdgeTypes {
			edgeMap[et.Type] = ResolvedType{TypeDefinition: et, Scope: layer.scope}
		}
		for _, cat := range stored.CustomBusinessCategories {
			categorySet[cat] = struct{}{}
		}
	}

	resolved := &ResolvedOntology{
		NodeTypes:          filterActiveResolved(nodeMap),
		EdgeTypes:          filterActiveResolved(edgeMap),
		BusinessCategories: sortedKeys(categorySet),
	}

	f.mu.Lock()
	f.cache[cacheKey] = resolved
	f.mu.Unlock()

	return resolved, nil
}

// Close releases resources held by the store.
func (f *FileOntologyStore) Close() error {
	f.mu.Lock()
	f.cache = nil
	f.mu.Unlock()
	return nil
}

func (f *FileOntologyStore) idForScope(scope Scope) string {
	switch scope {
	case ScopeOrganisation:
		return f.config.OrgID
	case ScopeProject:
		return f.config.ProjectID
	case ScopeBrain:
		return f.config.BrainID
	default:
		return ""
	}
}

func (f *FileOntologyStore) readScopeByID(ctx context.Context, scope Scope, id string) (*StoredOntology, error) {
	p := scopePath(scope, id)
	data, err := f.store.Read(ctx, p)
	if err != nil {
		if errors.Is(err, brain.ErrNotFound) {
			return &StoredOntology{}, nil
		}
		return nil, err
	}
	var stored StoredOntology
	if err := json.Unmarshal(data, &stored); err != nil {
		return nil, fmt.Errorf("ontology: unmarshal %s: %w", p, err)
	}
	return &stored, nil
}

func (f *FileOntologyStore) writeScopeByID(ctx context.Context, scope Scope, id string, stored *StoredOntology) error {
	p := scopePath(scope, id)
	data, err := json.Marshal(stored)
	if err != nil {
		return fmt.Errorf("ontology: marshal %s: %w", p, err)
	}
	return f.store.Write(ctx, p, data)
}

func (f *FileOntologyStore) invalidateCache() {
	f.mu.Lock()
	f.cache = make(map[string]*ResolvedOntology)
	f.mu.Unlock()
}

func getBuiltInType(typeID string) (*TypeDefinition, error) {
	if IsBuiltInNodeType(typeID) {
		return &TypeDefinition{
			Type:        typeID,
			Label:       FormatNodeTypeLabel(typeID),
			Description: GetBuiltInNodeTypeDescription(typeID),
			CreatedAt:   "1970-01-01T00:00:00.000Z",
			Status:      TypeStatusActive,
		}, nil
	}
	if IsBuiltInEdgeType(typeID) {
		return &TypeDefinition{
			Type:        typeID,
			Label:       FormatEdgeTypeLabel(typeID),
			Description: GetBuiltInEdgeTypeDescription(typeID),
			CreatedAt:   "1970-01-01T00:00:00.000Z",
			Status:      TypeStatusActive,
		}, nil
	}
	return nil, fmt.Errorf("ontology: type %q not found at scope built-in: %w", typeID, brain.ErrNotFound)
}

func listBuiltInTypes(opts ListTypesOpts) []TypeDefinition {
	defs := make([]TypeDefinition, 0, len(BuiltInNodeTypes)+len(BuiltInEdgeTypes))
	for _, nt := range BuiltInNodeTypes {
		defs = append(defs, TypeDefinition{
			Type:        nt,
			Label:       FormatNodeTypeLabel(nt),
			Description: GetBuiltInNodeTypeDescription(nt),
			CreatedAt:   "1970-01-01T00:00:00.000Z",
			Status:      TypeStatusActive,
		})
	}
	for _, et := range BuiltInEdgeTypes {
		defs = append(defs, TypeDefinition{
			Type:        et,
			Label:       FormatEdgeTypeLabel(et),
			Description: GetBuiltInEdgeTypeDescription(et),
			CreatedAt:   "1970-01-01T00:00:00.000Z",
			Status:      TypeStatusActive,
		})
	}
	return filterTypes(defs, opts)
}

func buildCacheKey(brainID, projectID, orgID string) string {
	return "org:" + orgID + ":proj:" + projectID + ":brain:" + brainID
}

func scopePath(scope Scope, id string) brain.Path {
	switch scope {
	case ScopeOrganisation:
		return brain.Path("ontology/org/" + id + "/types.json")
	case ScopeProject:
		return brain.Path("ontology/project/" + id + "/types.json")
	case ScopeBrain:
		return brain.Path("ontology/brain/" + id + "/types.json")
	default:
		return brain.Path("ontology/types.json")
	}
}

func upsertInSlice(slice []TypeDefinition, def TypeDefinition) []TypeDefinition {
	for i := range slice {
		if slice[i].Type == def.Type {
			slice[i] = def
			return slice
		}
	}
	return append(slice, def)
}

func removeFromSlice(slice []TypeDefinition, typeID string) ([]TypeDefinition, bool) {
	for i := range slice {
		if slice[i].Type == typeID {
			return append(slice[:i], slice[i+1:]...), true
		}
	}
	return slice, false
}

func filterTypes(defs []TypeDefinition, opts ListTypesOpts) []TypeDefinition {
	if opts.Prefix == "" && opts.Status == "" {
		return defs
	}
	filtered := make([]TypeDefinition, 0, len(defs))
	for _, d := range defs {
		if opts.Prefix != "" && !strings.HasPrefix(d.Type, opts.Prefix) {
			continue
		}
		if opts.Status != "" && string(d.Status) != opts.Status {
			continue
		}
		filtered = append(filtered, d)
	}
	return filtered
}

func filterActiveResolved(m map[string]ResolvedType) []ResolvedType {
	result := make([]ResolvedType, 0, len(m))
	for _, rt := range m {
		if rt.Status == TypeStatusActive {
			result = append(result, rt)
		}
	}
	return result
}

func sortedKeys(m map[string]struct{}) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sortStrings(keys)
	return keys
}

func sortStrings(s []string) {
	for i := 1; i < len(s); i++ {
		for j := i; j > 0 && s[j] < s[j-1]; j-- {
			s[j], s[j-1] = s[j-1], s[j]
		}
	}
}
