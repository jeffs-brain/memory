// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strings"
	"testing"

	"github.com/modelcontextprotocol/go-sdk/mcp"
)

// specField is one top-level input field parsed from spec/MCP-TOOLS.md.
type specField struct {
	name       string
	required   bool
	typ        string
	enumValues []string
}

var (
	specSectionRe = regexp.MustCompile("(?m)^## `(memory_[a-z_]+)`")
	specInputRe   = regexp.MustCompile("\\*\\*Input schema\\*\\*(?::\\s*`\\{\\}`|\\s*```[a-z]*\\n((?s:.*?))```)")
	specFieldRe   = regexp.MustCompile(`(?m)^ {2}([A-Za-z_]+)(\?)?:\s*(.+)$`)
	specEnumRe    = regexp.MustCompile(`'([^']+)'`)
)

func specTypeOf(t *testing.T, raw string) (string, []string) {
	t.Helper()
	decl, _, _ := strings.Cut(raw, "#")
	decl = strings.TrimSpace(decl)
	if strings.HasPrefix(decl, "'") {
		var values []string
		for _, m := range specEnumRe.FindAllStringSubmatch(decl, -1) {
			values = append(values, m[1])
		}
		return "string", values
	}
	for _, typ := range []string{"string", "integer", "number", "boolean"} {
		if strings.HasPrefix(decl, typ) {
			return typ, nil
		}
	}
	if strings.HasPrefix(decl, "Array<") {
		return "array", nil
	}
	t.Fatalf("unrecognised spec type: %s", raw)
	return "", nil
}

// parseToolSpec returns the top-level input fields of every tool in the
// spec, keyed by tool name. The TypeScript and Python servers run the
// same parse.
func parseToolSpec(t *testing.T) map[string][]specField {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("..", "..", "..", "spec", "MCP-TOOLS.md"))
	if err != nil {
		t.Fatalf("read spec: %v", err)
	}
	doc := string(raw)
	out := map[string][]specField{}
	locs := specSectionRe.FindAllStringSubmatchIndex(doc, -1)
	for i, loc := range locs {
		end := len(doc)
		if i+1 < len(locs) {
			end = locs[i+1][0]
		}
		name := doc[loc[2]:loc[3]]
		section := doc[loc[1]:end]
		if next := strings.Index(section, "\n## "); next >= 0 {
			section = section[:next]
		}
		block := specInputRe.FindStringSubmatch(section)
		if block == nil {
			t.Fatalf("%s: no input schema in the spec", name)
		}
		fields := []specField{}
		for _, m := range specFieldRe.FindAllStringSubmatch(block[1], -1) {
			typ, enumValues := specTypeOf(t, m[3])
			fields = append(fields, specField{name: m[1], required: m[2] == "", typ: typ, enumValues: enumValues})
		}
		out[name] = fields
	}
	return out
}

type advertisedProperty struct {
	Type any      `json:"type"`
	Enum []string `json:"enum"`
}

type advertisedSchema struct {
	Type       string                        `json:"type"`
	Properties map[string]advertisedProperty `json:"properties"`
	Required   []string                      `json:"required"`
}

// propertyTypes flattens a JSON Schema "type", which may be a string or
// an array such as ["null", "array"] for a nil-able Go slice.
func propertyTypes(v any) []string {
	switch typ := v.(type) {
	case string:
		return []string{typ}
	case []any:
		out := []string{}
		for _, item := range typ {
			if s, ok := item.(string); ok && s != "null" {
				out = append(out, s)
			}
		}
		return out
	}
	return nil
}

func TestToolSchemasMatchSpec(t *testing.T) {
	spec := parseToolSpec(t)
	session := newInMemoryServer(t, t.TempDir())
	list, err := session.ListTools(context.Background(), &mcp.ListToolsParams{})
	if err != nil {
		t.Fatalf("list tools: %v", err)
	}

	advertised := []string{}
	for _, tool := range list.Tools {
		advertised = append(advertised, tool.Name)
	}
	want := []string{}
	for name := range spec {
		want = append(want, name)
	}
	slices.Sort(advertised)
	slices.Sort(want)
	if !slices.Equal(advertised, want) {
		t.Fatalf("tools = %v, want the spec's %v", advertised, want)
	}

	for _, tool := range list.Tools {
		t.Run(tool.Name, func(t *testing.T) {
			raw, err := json.Marshal(tool.InputSchema)
			if err != nil {
				t.Fatalf("marshal schema: %v", err)
			}
			var schema advertisedSchema
			if err := json.Unmarshal(raw, &schema); err != nil {
				t.Fatalf("decode schema: %v", err)
			}
			if schema.Type != "object" {
				t.Fatalf("type = %q, want object", schema.Type)
			}
			fields := spec[tool.Name]
			gotNames, wantNames, wantRequired := []string{}, []string{}, []string{}
			for name := range schema.Properties {
				gotNames = append(gotNames, name)
			}
			for _, f := range fields {
				wantNames = append(wantNames, f.name)
				if f.required {
					wantRequired = append(wantRequired, f.name)
				}
			}
			gotRequired := slices.Clone(schema.Required)
			for _, s := range [][]string{gotNames, wantNames, gotRequired, wantRequired} {
				slices.Sort(s)
			}
			if !slices.Equal(gotNames, wantNames) {
				t.Errorf("properties = %v, want %v", gotNames, wantNames)
			}
			if !slices.Equal(gotRequired, wantRequired) {
				t.Errorf("required = %v, want %v", gotRequired, wantRequired)
			}
			for _, f := range fields {
				prop, ok := schema.Properties[f.name]
				if !ok {
					continue
				}
				accepted := []string{f.typ}
				if f.typ == "number" {
					accepted = append(accepted, "integer")
				}
				types := propertyTypes(prop.Type)
				if len(types) != 1 || !slices.Contains(accepted, types[0]) {
					t.Errorf("%s: type = %v, want one of %v", f.name, prop.Type, accepted)
				}
				if f.enumValues != nil {
					got := slices.Clone(prop.Enum)
					wantEnum := slices.Clone(f.enumValues)
					slices.Sort(got)
					slices.Sort(wantEnum)
					if !slices.Equal(got, wantEnum) {
						t.Errorf("%s: enum = %v, want %v", f.name, got, wantEnum)
					}
				}
			}
		})
	}
}
