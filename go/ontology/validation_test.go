// SPDX-License-Identifier: Apache-2.0
package ontology

import "testing"

func TestIsValidNodeType(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name    string
		value   string
		isValid bool
	}{
		{"built-in entity.customer", "entity.customer", true},
		{"built-in rule.combined", "rule.combined", true},
		{"built-in exception.special_case", "exception.special_case", true},
		{"built-in decision.table", "decision.table", true},
		{"built-in process.subworkflow", "process.subworkflow", true},
		{"custom entity.invoice", "entity.invoice", true},
		{"custom rule.custom_thing", "rule.custom_thing", true},
		{"custom exception.timeout_handling", "exception.timeout_handling", true},
		{"custom decision.routing_logic", "decision.routing_logic", true},
		{"custom process.batch_job", "process.batch_job", true},
		{"custom multi underscore", "entity.multi_word_name", true},
		{"invalid no prefix", "invalid", false},
		{"invalid wrong prefix", "wrong.prefix", false},
		{"invalid empty", "", false},
		{"invalid prefix only", "entity.", false},
		{"invalid uppercase name", "entity.Customer", false},
		{"invalid name starts with number", "entity.1thing", false},
		{"invalid name has hyphen", "entity.my-thing", false},
		{"invalid name has space", "entity.my thing", false},
		{"invalid double dot", "entity..customer", false},
		{"invalid trailing underscore", "entity.name_", false},
		{"invalid double underscore", "entity.name__thing", false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			got := IsValidNodeType(tc.value)
			if got != tc.isValid {
				t.Fatalf("IsValidNodeType(%q) = %v, want %v", tc.value, got, tc.isValid)
			}
		})
	}
}

func TestIsValidNodeType_AllBuiltIn(t *testing.T) {
	t.Parallel()
	for _, typ := range BuiltInNodeTypes {
		if !IsValidNodeType(typ) {
			t.Fatalf("built-in node type %q failed validation", typ)
		}
	}
}

func TestIsValidEdgeType(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name    string
		value   string
		isValid bool
	}{
		{"built-in triggers", "triggers", true},
		{"built-in requires_approval_from", "requires_approval_from", true},
		{"built-in validates", "validates", true},
		{"custom ships_to", "ships_to", true},
		{"custom custom_relation", "custom_relation", true},
		{"custom single word", "connects", true},
		{"invalid uppercase", "Invalid", false},
		{"invalid has spaces", "has spaces", false},
		{"invalid starts with number", "123start", false},
		{"invalid empty", "", false},
		{"invalid has dot", "has.dot", false},
		{"invalid has hyphen", "has-hyphen", false},
		{"invalid trailing underscore", "name_", false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			got := IsValidEdgeType(tc.value)
			if got != tc.isValid {
				t.Fatalf("IsValidEdgeType(%q) = %v, want %v", tc.value, got, tc.isValid)
			}
		})
	}
}

func TestIsValidEdgeType_AllBuiltIn(t *testing.T) {
	t.Parallel()
	for _, typ := range BuiltInEdgeTypes {
		if !IsValidEdgeType(typ) {
			t.Fatalf("built-in edge type %q failed validation", typ)
		}
	}
}

func TestIsValidBusinessCategory(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name    string
		value   string
		isValid bool
	}{
		{"built-in customer", "customer", true},
		{"built-in authorization", "authorization", true},
		{"built-in general", "general", true},
		{"custom category", "custom_category", true},
		{"custom multi word", "server_hardware", true},
		{"invalid Has Spaces", "Has Spaces", false},
		{"invalid starts with number", "123", false},
		{"invalid empty", "", false},
		{"invalid uppercase", "Customer", false},
		{"invalid has dot", "my.category", false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			got := IsValidBusinessCategory(tc.value)
			if got != tc.isValid {
				t.Fatalf("IsValidBusinessCategory(%q) = %v, want %v", tc.value, got, tc.isValid)
			}
		})
	}
}

func TestIsValidBusinessCategory_AllBuiltIn(t *testing.T) {
	t.Parallel()
	for _, cat := range BusinessCategories {
		if !IsValidBusinessCategory(cat) {
			t.Fatalf("built-in business category %q failed validation", cat)
		}
	}
}

func TestHasPrefix(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name     string
		nodeType string
		want     string
	}{
		{"entity prefix", "entity.customer", "entity."},
		{"rule prefix", "rule.validation", "rule."},
		{"exception prefix", "exception.override", "exception."},
		{"decision prefix", "decision.branch", "decision."},
		{"process prefix", "process.workflow", "process."},
		{"no prefix", "triggers", ""},
		{"invalid prefix", "wrong.thing", ""},
		{"empty string", "", ""},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			got := HasPrefix(tc.nodeType)
			if got != tc.want {
				t.Fatalf("HasPrefix(%q) = %q, want %q", tc.nodeType, got, tc.want)
			}
		})
	}
}

func TestValidateNodeType_ReturnsError(t *testing.T) {
	t.Parallel()
	if err := ValidateNodeType("entity.customer"); err != nil {
		t.Fatalf("unexpected error for valid type: %v", err)
	}
	if err := ValidateNodeType("invalid"); err == nil {
		t.Fatal("expected error for invalid type, got nil")
	}
}

func TestValidateEdgeType_ReturnsError(t *testing.T) {
	t.Parallel()
	if err := ValidateEdgeType("triggers"); err != nil {
		t.Fatalf("unexpected error for valid type: %v", err)
	}
	if err := ValidateEdgeType("INVALID"); err == nil {
		t.Fatal("expected error for invalid type, got nil")
	}
}

func TestValidateBusinessCategory_ReturnsError(t *testing.T) {
	t.Parallel()
	if err := ValidateBusinessCategory("customer"); err != nil {
		t.Fatalf("unexpected error for valid category: %v", err)
	}
	if err := ValidateBusinessCategory(""); err == nil {
		t.Fatal("expected error for empty category, got nil")
	}
}

func TestValidateTypeDefinition(t *testing.T) {
	t.Parallel()
	validDef := TypeDefinition{
		Type:        "entity.customer",
		Label:       "Customer",
		Description: "A customer entity",
		CreatedAt:   "2026-01-01T00:00:00Z",
		Status:      TypeStatusActive,
	}
	cases := []struct {
		name    string
		def     TypeDefinition
		wantErr bool
	}{
		{"valid definition", validDef, false},
		{"valid proposed", TypeDefinition{Type: "entity.x", Label: "X", Description: "X", CreatedAt: "2026-01-01T00:00:00Z", Status: TypeStatusProposed}, false},
		{"valid deprecated", TypeDefinition{Type: "entity.x", Label: "X", Description: "X", CreatedAt: "2026-01-01T00:00:00Z", Status: TypeStatusDeprecated}, false},
		{"empty type", TypeDefinition{Type: "", Label: "L", Description: "D", CreatedAt: "2026-01-01T00:00:00Z", Status: TypeStatusActive}, true},
		{"empty label", TypeDefinition{Type: "entity.x", Label: "", Description: "D", CreatedAt: "2026-01-01T00:00:00Z", Status: TypeStatusActive}, true},
		{"empty description", TypeDefinition{Type: "entity.x", Label: "L", Description: "", CreatedAt: "2026-01-01T00:00:00Z", Status: TypeStatusActive}, true},
		{"empty createdAt", TypeDefinition{Type: "entity.x", Label: "L", Description: "D", CreatedAt: "", Status: TypeStatusActive}, true},
		{"invalid status", TypeDefinition{Type: "entity.x", Label: "L", Description: "D", CreatedAt: "2026-01-01T00:00:00Z", Status: "invalid"}, true},
		{"empty status", TypeDefinition{Type: "entity.x", Label: "L", Description: "D", CreatedAt: "2026-01-01T00:00:00Z", Status: ""}, true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			err := ValidateTypeDefinition(tc.def)
			if (err != nil) != tc.wantErr {
				t.Fatalf("ValidateTypeDefinition() err=%v, wantErr=%v", err, tc.wantErr)
			}
		})
	}
}

func TestFormatNodeTypeLabel(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name  string
		input string
		want  string
	}{
		{"entity customer", "entity.customer", "Customer (Entity)"},
		{"process approval chain", "process.approval_chain", "Approval Chain (Process)"},
		{"rule classification", "rule.classification", "Classification (Rule)"},
		{"exception special case", "exception.special_case", "Special Case (Exception)"},
		{"decision escalation", "decision.escalation", "Escalation (Decision)"},
		{"no dot fallback", "invalid", "invalid"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			got := FormatNodeTypeLabel(tc.input)
			if got != tc.want {
				t.Fatalf("FormatNodeTypeLabel(%q) = %q, want %q", tc.input, got, tc.want)
			}
		})
	}
}

func TestFormatEdgeTypeLabel(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name  string
		input string
		want  string
	}{
		{"requires approval from", "requires_approval_from", "Requires Approval From"},
		{"triggers", "triggers", "Triggers"},
		{"feeds into", "feeds_into", "Feeds Into"},
		{"related to", "related_to", "Related To"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			got := FormatEdgeTypeLabel(tc.input)
			if got != tc.want {
				t.Fatalf("FormatEdgeTypeLabel(%q) = %q, want %q", tc.input, got, tc.want)
			}
		})
	}
}
