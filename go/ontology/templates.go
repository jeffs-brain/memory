// SPDX-License-Identifier: Apache-2.0
package ontology

import (
	"fmt"
	"sort"
	"sync"
)

// TypeEntry is a single type within an industry template.
type TypeEntry struct {
	Type        string `json:"type"`
	Label       string `json:"label"`
	Description string `json:"description"`
}

// IndustryTemplate defines a domain-specific set of types that seed an
// ontology for a particular industry vertical.
type IndustryTemplate struct {
	Label              string      `json:"label"`
	Description        string      `json:"description"`
	NodeTypes          []TypeEntry `json:"nodeTypes"`
	EdgeTypes          []TypeEntry `json:"edgeTypes"`
	BusinessCategories []string    `json:"businessCategories,omitempty"`
}

// templates is the mutable registry of industry templates. Starts with
// the 6 built-in templates and can be extended via RegisterTemplate.
// Protected by templateMu.
var templates = map[string]IndustryTemplate{
	"server_hardware":  serverHardwareTemplate,
	"insurance":        insuranceTemplate,
	"logistics":        logisticsTemplate,
	"finance":          financeTemplate,
	"healthcare":       healthcareTemplate,
	"order_processing": orderProcessingTemplate,
}

// templateMu protects concurrent access to the templates map.
var templateMu sync.RWMutex

// Templates returns a snapshot of the current template registry.
// The returned map is a copy safe for iteration.
var Templates = templates

// ListTemplates returns sorted template keys.
func ListTemplates() []string {
	templateMu.RLock()
	defer templateMu.RUnlock()
	keys := make([]string, 0, len(templates))
	for k := range templates {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

// GetTemplate returns a template by key. The second return value is
// false when the key is not registered.
func GetTemplate(key string) (IndustryTemplate, bool) {
	templateMu.RLock()
	defer templateMu.RUnlock()
	t, ok := templates[key]
	return t, ok
}

// RegisterTemplate adds a custom industry template to the registry at
// runtime. Returns an error if the key is empty or already registered.
// The template's types are validated before registration.
func RegisterTemplate(key string, tmpl IndustryTemplate) error {
	if key == "" {
		return fmt.Errorf("ontology: template key must not be empty")
	}
	templateMu.Lock()
	defer templateMu.Unlock()
	if _, exists := templates[key]; exists {
		return fmt.Errorf("ontology: template %q is already registered", key)
	}
	for _, nt := range tmpl.NodeTypes {
		if !IsValidNodeType(nt.Type) {
			return fmt.Errorf("ontology: template %q contains invalid node type %q", key, nt.Type)
		}
		if nt.Label == "" || nt.Description == "" {
			return fmt.Errorf("ontology: template %q node type %q has empty label or description", key, nt.Type)
		}
	}
	for _, et := range tmpl.EdgeTypes {
		if !IsValidEdgeType(et.Type) {
			return fmt.Errorf("ontology: template %q contains invalid edge type %q", key, et.Type)
		}
		if et.Label == "" || et.Description == "" {
			return fmt.Errorf("ontology: template %q edge type %q has empty label or description", key, et.Type)
		}
	}
	for _, cat := range tmpl.BusinessCategories {
		if !IsValidBusinessCategory(cat) {
			return fmt.Errorf("ontology: template %q contains invalid business category %q", key, cat)
		}
	}
	templates[key] = tmpl
	return nil
}

// resetTemplates restores the 6 built-in templates only (for testing).
func resetTemplates() {
	templateMu.Lock()
	defer templateMu.Unlock()
	templates = map[string]IndustryTemplate{
		"server_hardware":  serverHardwareTemplate,
		"insurance":        insuranceTemplate,
		"logistics":        logisticsTemplate,
		"finance":          financeTemplate,
		"healthcare":       healthcareTemplate,
		"order_processing": orderProcessingTemplate,
	}
}
