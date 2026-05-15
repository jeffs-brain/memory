// SPDX-License-Identifier: Apache-2.0
package ontology

import (
	"testing"
)

func TestListTemplates(t *testing.T) {
	keys := ListTemplates()
	expected := []string{
		"finance",
		"healthcare",
		"insurance",
		"logistics",
		"order_processing",
		"server_hardware",
	}
	if len(keys) != len(expected) {
		t.Fatalf("ListTemplates() returned %d keys, want %d", len(keys), len(expected))
	}
	for i, key := range keys {
		if key != expected[i] {
			t.Errorf("ListTemplates()[%d] = %q, want %q", i, key, expected[i])
		}
	}
}

func TestGetTemplate_Exists(t *testing.T) {
	cases := []struct {
		name string
		key  string
	}{
		{"server_hardware", "server_hardware"},
		{"insurance", "insurance"},
		{"logistics", "logistics"},
		{"finance", "finance"},
		{"healthcare", "healthcare"},
		{"order_processing", "order_processing"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			tmpl, err := GetTemplate(tc.key)
			if err != nil {
				t.Fatalf("GetTemplate(%q) returned error: %v", tc.key, err)
			}
			if tmpl.Label == "" {
				t.Errorf("GetTemplate(%q).Label is empty", tc.key)
			}
			if tmpl.Description == "" {
				t.Errorf("GetTemplate(%q).Description is empty", tc.key)
			}
		})
	}
}

func TestGetTemplate_NotFound(t *testing.T) {
	_, err := GetTemplate("nonexistent_template")
	if err == nil {
		t.Fatal("GetTemplate(\"nonexistent_template\") returned nil error, want error")
	}
}

func TestTemplateCounts(t *testing.T) {
	cases := []struct {
		name       string
		key        string
		nodeCount  int
		edgeCount  int
		catCount   int
	}{
		{"server_hardware", "server_hardware", 9, 6, 1},
		{"insurance", "insurance", 20, 10, 7},
		{"logistics", "logistics", 27, 11, 8},
		{"finance", "finance", 22, 12, 9},
		{"healthcare", "healthcare", 30, 15, 9},
		{"order_processing", "order_processing", 8, 3, 1},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			tmpl, err := GetTemplate(tc.key)
			if err != nil {
				t.Fatalf("GetTemplate(%q) error: %v", tc.key, err)
			}
			if len(tmpl.NodeTypes) != tc.nodeCount {
				t.Errorf("template %q has %d node types, want %d", tc.key, len(tmpl.NodeTypes), tc.nodeCount)
			}
			if len(tmpl.EdgeTypes) != tc.edgeCount {
				t.Errorf("template %q has %d edge types, want %d", tc.key, len(tmpl.EdgeTypes), tc.edgeCount)
			}
			if len(tmpl.BusinessCategories) != tc.catCount {
				t.Errorf("template %q has %d business categories, want %d", tc.key, len(tmpl.BusinessCategories), tc.catCount)
			}
		})
	}
}

func TestTemplateNodeTypesValid(t *testing.T) {
	for key, tmpl := range Templates {
		t.Run(key, func(t *testing.T) {
			for _, nt := range tmpl.NodeTypes {
				if !IsValidNodeType(nt.Type) {
					t.Errorf("template %q: invalid node type %q", key, nt.Type)
				}
				if nt.Label == "" {
					t.Errorf("template %q: node type %q has empty label", key, nt.Type)
				}
				if nt.Description == "" {
					t.Errorf("template %q: node type %q has empty description", key, nt.Type)
				}
			}
		})
	}
}

func TestTemplateEdgeTypesValid(t *testing.T) {
	for key, tmpl := range Templates {
		t.Run(key, func(t *testing.T) {
			for _, et := range tmpl.EdgeTypes {
				if !IsValidEdgeType(et.Type) {
					t.Errorf("template %q: invalid edge type %q", key, et.Type)
				}
				if et.Label == "" {
					t.Errorf("template %q: edge type %q has empty label", key, et.Type)
				}
				if et.Description == "" {
					t.Errorf("template %q: edge type %q has empty description", key, et.Type)
				}
			}
		})
	}
}

func TestTemplateBusinessCategoriesValid(t *testing.T) {
	for key, tmpl := range Templates {
		t.Run(key, func(t *testing.T) {
			for _, cat := range tmpl.BusinessCategories {
				if !IsValidBusinessCategory(cat) {
					t.Errorf("template %q: invalid business category %q", key, cat)
				}
			}
		})
	}
}

func TestTemplateNoDuplicateNodeTypes(t *testing.T) {
	for key, tmpl := range Templates {
		t.Run(key, func(t *testing.T) {
			seen := make(map[string]struct{}, len(tmpl.NodeTypes))
			for _, nt := range tmpl.NodeTypes {
				if _, exists := seen[nt.Type]; exists {
					t.Errorf("template %q: duplicate node type %q", key, nt.Type)
				}
				seen[nt.Type] = struct{}{}
			}
		})
	}
}

func TestTemplateNoDuplicateEdgeTypes(t *testing.T) {
	for key, tmpl := range Templates {
		t.Run(key, func(t *testing.T) {
			seen := make(map[string]struct{}, len(tmpl.EdgeTypes))
			for _, et := range tmpl.EdgeTypes {
				if _, exists := seen[et.Type]; exists {
					t.Errorf("template %q: duplicate edge type %q", key, et.Type)
				}
				seen[et.Type] = struct{}{}
			}
		})
	}
}

func TestTemplateNoDuplicateBusinessCategories(t *testing.T) {
	for key, tmpl := range Templates {
		t.Run(key, func(t *testing.T) {
			seen := make(map[string]struct{}, len(tmpl.BusinessCategories))
			for _, cat := range tmpl.BusinessCategories {
				if _, exists := seen[cat]; exists {
					t.Errorf("template %q: duplicate business category %q", key, cat)
				}
				seen[cat] = struct{}{}
			}
		})
	}
}

func TestRegisterTemplate(t *testing.T) {
	t.Cleanup(resetTemplates)

	custom := IndustryTemplate{
		Label:       "Education",
		Description: "Schools, courses, and student management",
		NodeTypes: []TypeEntry{
			{Type: "entity.student", Label: "Student", Description: "A student enrolled in courses"},
			{Type: "entity.course", Label: "Course", Description: "An academic course"},
		},
		EdgeTypes: []TypeEntry{
			{Type: "enrolled_in", Label: "Enrolled In", Description: "A student is enrolled in a course"},
		},
		BusinessCategories: []string{"education"},
	}
	if err := RegisterTemplate("education", custom); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	keys := ListTemplates()
	found := false
	for _, k := range keys {
		if k == "education" {
			found = true
			break
		}
	}
	if !found {
		t.Fatal("expected 'education' template to be registered")
	}
	if len(keys) != 7 {
		t.Fatalf("expected 7 templates, got %d", len(keys))
	}

	tmpl, err := GetTemplate("education")
	if err != nil {
		t.Fatalf("GetTemplate(education) error: %v", err)
	}
	if tmpl.Label != "Education" {
		t.Fatalf("expected label 'Education', got %q", tmpl.Label)
	}
}

func TestRegisterTemplate_Errors(t *testing.T) {
	t.Cleanup(resetTemplates)

	if err := RegisterTemplate("", IndustryTemplate{}); err == nil {
		t.Fatal("expected error for empty key")
	}
	if err := RegisterTemplate("server_hardware", IndustryTemplate{}); err == nil {
		t.Fatal("expected error for duplicate key")
	}
	invalid := IndustryTemplate{
		NodeTypes: []TypeEntry{{Type: "INVALID", Label: "Bad", Description: "Bad type"}},
	}
	if err := RegisterTemplate("bad_template", invalid); err == nil {
		t.Fatal("expected error for invalid node type")
	}
}
