// SPDX-License-Identifier: Apache-2.0

package search

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

// TestQueryParserGolden loads the TS-maintained fixture in
// spec/fixtures/query-parser/cases.json and asserts this port's
// parser and compiler produce outputs structurally equivalent to
// the goldens. The fixture is the cross-SDK contract defined in
// spec/QUERY-DSL.md: parser divergence between SDKs surfaces here
// first.
func TestQueryParserGolden(t *testing.T) {
	fixturePath := locateGoldenFixture(t)

	raw, err := os.ReadFile(fixturePath)
	if err != nil {
		t.Fatalf("reading golden fixture: %v", err)
	}

	var fixture goldenFixture
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatalf("decoding golden fixture: %v", err)
	}

	if len(fixture.Cases) == 0 {
		t.Fatal("fixture has zero parse cases, expected 15")
	}
	if len(fixture.CompileToFTS) == 0 {
		t.Fatal("fixture has zero compile cases, expected 4")
	}

	for _, tc := range fixture.Cases {
		t.Run("parse/"+tc.Name, func(t *testing.T) {
			ast := Parse(tc.Input)
			got := astToGolden(ast)
			if !goldenASTEqual(got, tc.ExpectedAST) {
				t.Errorf("parse mismatch for %q\n got  = %s\n want = %s",
					tc.Input, jsonString(got), jsonString(tc.ExpectedAST))
			}
		})
	}

	for _, tc := range fixture.CompileToFTS {
		t.Run("compile/"+tc.Name, func(t *testing.T) {
			ast := Parse(tc.Input)
			got := BuildFTS5Expr(ast.Tokens)
			if got != tc.ExpectedFTS {
				t.Errorf("compile mismatch for %q\n got  = %q\n want = %q",
					tc.Input, got, tc.ExpectedFTS)
			}
		})
	}
}

// goldenFixture mirrors the JSON shape of
// spec/fixtures/query-parser/cases.json. Names match the fixture
// keys one-for-one.
type goldenFixture struct {
	Description  string          `json:"description"`
	Cases        []goldenParseCase   `json:"cases"`
	CompileToFTS []goldenCompileCase `json:"compileToFTS"`
}

type goldenParseCase struct {
	Name        string   `json:"name"`
	Input       string   `json:"input"`
	ExpectedAST goldenAST `json:"expectedAst"`
}

type goldenCompileCase struct {
	Name        string `json:"name"`
	Input       string `json:"input"`
	ExpectedFTS string `json:"expectedFTS"`
}

type goldenAST struct {
	Raw          string        `json:"raw"`
	Tokens       []goldenToken `json:"tokens"`
	HasOperators bool          `json:"hasOperators"`
}

type goldenToken struct {
	Kind     string `json:"kind"`
	Text     string `json:"text"`
	Operator string `json:"operator,omitempty"`
}

func astToGolden(ast AST) goldenAST {
	out := goldenAST{
		Raw:          ast.Raw,
		HasOperators: ast.HasOperators,
		Tokens:       make([]goldenToken, 0, len(ast.Tokens)),
	}
	for _, tok := range ast.Tokens {
		out.Tokens = append(out.Tokens, goldenToken{
			Kind:     tok.Kind.String(),
			Text:     tok.Text,
			Operator: tok.Operator,
		})
	}
	return out
}

func goldenASTEqual(a, b goldenAST) bool {
	if a.Raw != b.Raw || a.HasOperators != b.HasOperators {
		return false
	}
	if len(a.Tokens) != len(b.Tokens) {
		return false
	}
	for i := range a.Tokens {
		if a.Tokens[i] != b.Tokens[i] {
			return false
		}
	}
	return true
}

func jsonString(v any) string {
	b, err := json.Marshal(v)
	if err != nil {
		return err.Error()
	}
	return string(b)
}

// locateGoldenFixture walks up from the package directory to find
// the spec/fixtures/query-parser/cases.json file. The Go SDK lives
// below spec/ in the wider monorepo (jeffs-brain/memory/sdks/go/
// against jeffs-brain/memory/spec/fixtures/), so an upward search
// is the robust way to locate the fixture without hard-coding the
// module layout.
func locateGoldenFixture(t *testing.T) string {
	t.Helper()
	wd, err := os.Getwd()
	if err != nil {
		t.Fatalf("getwd: %v", err)
	}
	dir := wd
	for {
		candidate := filepath.Join(dir, "spec", "fixtures", "query-parser", "cases.json")
		if _, err := os.Stat(candidate); err == nil {
			return candidate
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			break
		}
		dir = parent
	}
	t.Fatalf("unable to locate spec/fixtures/query-parser/cases.json from %s", wd)
	return ""
}
