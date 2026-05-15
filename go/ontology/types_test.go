// SPDX-License-Identifier: Apache-2.0
package ontology

import "testing"

func TestBuiltInNodeTypeCount(t *testing.T) {
	t.Parallel()
	if got := len(BuiltInNodeTypes); got != 31 {
		t.Fatalf("expected 31 built-in node types, got %d", got)
	}
}

func TestBuiltInEdgeTypeCount(t *testing.T) {
	t.Parallel()
	if got := len(BuiltInEdgeTypes); got != 19 {
		t.Fatalf("expected 19 built-in edge types, got %d", got)
	}
}

func TestBusinessCategoryCount(t *testing.T) {
	t.Parallel()
	if got := len(BusinessCategories); got != 8 {
		t.Fatalf("expected 8 business categories, got %d", got)
	}
}

func TestNodeTypePrefixCount(t *testing.T) {
	t.Parallel()
	if got := len(NodeTypePrefixesList()); got != 5 {
		t.Fatalf("expected 5 node type prefixes, got %d", got)
	}
}

func TestBuiltInNodeTypesNoDuplicates(t *testing.T) {
	t.Parallel()
	seen := make(map[string]struct{}, len(BuiltInNodeTypes))
	for _, typ := range BuiltInNodeTypes {
		if _, exists := seen[typ]; exists {
			t.Fatalf("duplicate node type: %s", typ)
		}
		seen[typ] = struct{}{}
	}
}

func TestBuiltInEdgeTypesNoDuplicates(t *testing.T) {
	t.Parallel()
	seen := make(map[string]struct{}, len(BuiltInEdgeTypes))
	for _, typ := range BuiltInEdgeTypes {
		if _, exists := seen[typ]; exists {
			t.Fatalf("duplicate edge type: %s", typ)
		}
		seen[typ] = struct{}{}
	}
}

func TestBusinessCategoriesNoDuplicates(t *testing.T) {
	t.Parallel()
	seen := make(map[string]struct{}, len(BusinessCategories))
	for _, cat := range BusinessCategories {
		if _, exists := seen[cat]; exists {
			t.Fatalf("duplicate business category: %s", cat)
		}
		seen[cat] = struct{}{}
	}
}

func TestAllNodeTypesHaveDescriptions(t *testing.T) {
	t.Parallel()
	for _, typ := range BuiltInNodeTypes {
		desc := GetBuiltInNodeTypeDescription(typ)
		if desc == "A business intelligence node type" {
			t.Fatalf("node type %s missing description", typ)
		}
		if desc == "" {
			t.Fatalf("node type %s has empty description", typ)
		}
	}
}

func TestAllEdgeTypesHaveDescriptions(t *testing.T) {
	t.Parallel()
	for _, typ := range BuiltInEdgeTypes {
		desc := GetBuiltInEdgeTypeDescription(typ)
		if desc == "A relationship between intelligence nodes" {
			t.Fatalf("edge type %s missing description", typ)
		}
		if desc == "" {
			t.Fatalf("edge type %s has empty description", typ)
		}
	}
}

func TestAllNodeTypesHaveValidPrefix(t *testing.T) {
	t.Parallel()
	for _, typ := range BuiltInNodeTypes {
		prefix := HasPrefix(typ)
		if prefix == "" {
			t.Fatalf("node type %s has no valid prefix", typ)
		}
	}
}

func TestNodeTypePrefixesEndWithDot(t *testing.T) {
	t.Parallel()
	for _, prefix := range NodeTypePrefixesList() {
		if prefix[len(prefix)-1] != '.' {
			t.Fatalf("prefix %q does not end with dot", prefix)
		}
	}
}

func TestRegisterPrefix(t *testing.T) {
	t.Cleanup(resetPrefixes)

	if err := RegisterPrefix("metric."); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	prefixes := NodeTypePrefixesList()
	found := false
	for _, p := range prefixes {
		if p == "metric." {
			found = true
			break
		}
	}
	if !found {
		t.Fatal("expected 'metric.' prefix to be registered")
	}
	if len(prefixes) != 6 {
		t.Fatalf("expected 6 prefixes after registration, got %d", len(prefixes))
	}

	if !IsValidNodeType("metric.cpu_usage") {
		t.Fatal("expected metric.cpu_usage to be valid after registering metric. prefix")
	}
}

func TestRegisterPrefix_Errors(t *testing.T) {
	t.Cleanup(resetPrefixes)

	if err := RegisterPrefix(""); err == nil {
		t.Fatal("expected error for empty prefix")
	}
	if err := RegisterPrefix("noDot"); err == nil {
		t.Fatal("expected error for prefix without trailing dot")
	}
	if err := RegisterPrefix("entity."); err == nil {
		t.Fatal("expected error for already-registered prefix")
	}
}

func TestIsBuiltInNodeType(t *testing.T) {
	t.Parallel()
	for _, typ := range BuiltInNodeTypes {
		if !IsBuiltInNodeType(typ) {
			t.Fatalf("IsBuiltInNodeType(%q) returned false", typ)
		}
	}
	if IsBuiltInNodeType("entity.invoice") {
		t.Fatal("custom type should not be built-in")
	}
	if IsBuiltInNodeType("") {
		t.Fatal("empty string should not be built-in")
	}
}

func TestIsBuiltInEdgeType(t *testing.T) {
	t.Parallel()
	for _, typ := range BuiltInEdgeTypes {
		if !IsBuiltInEdgeType(typ) {
			t.Fatalf("IsBuiltInEdgeType(%q) returned false", typ)
		}
	}
	if IsBuiltInEdgeType("ships_to") {
		t.Fatal("custom type should not be built-in")
	}
}

func TestIsBuiltInBusinessCategory(t *testing.T) {
	t.Parallel()
	for _, cat := range BusinessCategories {
		if !IsBuiltInBusinessCategory(cat) {
			t.Fatalf("IsBuiltInBusinessCategory(%q) returned false", cat)
		}
	}
	if IsBuiltInBusinessCategory("custom_area") {
		t.Fatal("custom category should not be built-in")
	}
}
