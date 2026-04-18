// SPDX-License-Identifier: Apache-2.0

package brain

import (
	"errors"
	"testing"
)

func TestPathHelpers(t *testing.T) {
	cases := []struct {
		name string
		got  Path
		want string
	}{
		{"global index", MemoryGlobalIndex(), "memory/global/MEMORY.md"},
		{"global topic", MemoryGlobalTopic("user-role"), "memory/global/user-role.md"},
		{"global prefix", MemoryGlobalPrefix(), "memory/global"},
		{"project index", MemoryProjectIndex("home-alex-code-jeff"), "memory/project/home-alex-code-jeff/MEMORY.md"},
		{"project topic", MemoryProjectTopic("home-alex-code-jeff", "arch"), "memory/project/home-alex-code-jeff/arch.md"},
		{"project prefix", MemoryProjectPrefix("home-alex-code-jeff"), "memory/project/home-alex-code-jeff"},
		{"projects prefix", MemoryProjectsPrefix(), "memory/project"},
		{"tenant topic", MemoryTenantTopic("lleverage", "user-role"), "memory/tenant/lleverage/user-role.md"},
		{"tenant codec", MemoryTenantCodec("lleverage"), "tenants/lleverage/codec.md"},
		{"thread topic", MemoryThreadTopic("abc-123", "context"), "memory/thread/abc-123/context.md"},
		{"thread root", MemoryThreadMemoryRoot("abc-123"), "memory/thread/abc-123"},
		{"buffer prefix", MemoryBufferPrefix(), "memory/buffer"},
		{"buffer global", MemoryBufferGlobal(), "memory/buffer/global.md"},
		{"buffer project", MemoryBufferProject("home-alex-code-jeff"), "memory/buffer/project/home-alex-code-jeff.md"},
		{"wiki master index", WikiMasterIndex(), "wiki/_index.md"},
		{"wiki concepts", WikiConceptsIndex(), "wiki/_concepts.md"},
		{"wiki health", WikiHealth(), "wiki/_health.md"},
		{"wiki log", WikiLog(), "wiki/_log.md"},
		{"wiki topic index", WikiTopicIndex("go"), "wiki/go/_index.md"},
		{"wiki article", WikiArticle("go", "interfaces"), "wiki/go/interfaces.md"},
		{"wiki article from rel", WikiArticleFromRel("go/interfaces.md"), "wiki/go/interfaces.md"},
		{"wiki prefix", WikiPrefix(), "wiki"},
		{"raw ingest", RawIngest("web", "example-com.md"), "raw/web/example-com.md"},
		{"raw document", RawDocument("hedgehogs"), "raw/documents/hedgehogs.md"},
		{"raw source", RawSource("web/example-com.md"), "raw/.sources/web/example-com.md"},
		{"raw prefix", RawPrefix(), "raw"},
		{"raw documents prefix", RawDocumentsPrefix(), "raw/documents"},
		{"sources prefix", SourcesPrefix(), "raw/.sources"},
		{"schema", Schema(), "kb-schema.md"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := string(tc.got); got != tc.want {
				t.Fatalf("got %q, want %q", got, tc.want)
			}
			if err := ValidatePath(tc.got); err != nil {
				t.Fatalf("ValidatePath(%q) returned error: %v", tc.got, err)
			}
		})
	}
}

func TestValidatePath_Rejects(t *testing.T) {
	cases := []struct {
		name string
		p    Path
	}{
		{"empty", ""},
		{"leading slash", "/memory/global"},
		{"dotdot", "memory/../etc/passwd"},
		{"backslash", "memory\\global\\foo.md"},
		{"trailing dot segment", "memory/global/./foo.md"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := ValidatePath(tc.p)
			if err == nil {
				t.Fatalf("ValidatePath(%q) returned nil, want error", tc.p)
			}
			if !errors.Is(err, ErrInvalidPath) {
				t.Fatalf("error does not wrap ErrInvalidPath: %v", err)
			}
		})
	}
}

func TestIsGenerated(t *testing.T) {
	cases := map[Path]bool{
		"wiki/_index.md":          true,
		"wiki/_log.md":            true,
		"wiki/go/_index.md":       true,
		"wiki/go/interfaces.md":   false,
		"memory/global/MEMORY.md": false, // all-caps index, not generated in the underscore sense
		"memory/global/foo.md":    false,
	}
	for p, want := range cases {
		if got := IsGenerated(p); got != want {
			t.Errorf("IsGenerated(%q) = %v, want %v", p, got, want)
		}
	}
}

func TestIsSourceRaw(t *testing.T) {
	cases := map[Path]bool{
		"raw/.sources/web/foo.md": true,
		"raw/.sources":            true,
		"raw/web/foo.md":          false,
		"raw":                     false,
		"wiki/foo.md":             false,
	}
	for p, want := range cases {
		if got := IsSourceRaw(p); got != want {
			t.Errorf("IsSourceRaw(%q) = %v, want %v", p, got, want)
		}
	}
}

func TestSanitiseScopeID(t *testing.T) {
	cases := []struct {
		input string
		want  string
	}{
		{"lleverage", "lleverage"},
		{"Lleverage-AI", "lleverage-ai"},
		{"my workspace!", "my-workspace"},
		{"  spaces  ", "spaces"},
		{"UPPER_case_123", "upper_case_123"},
		{"---dashes---", "dashes"},
		{"", "unknown"},
		{"   ", "unknown"},
		{"abc@def.com", "abc-def-com"},
	}
	for _, tc := range cases {
		t.Run(tc.input, func(t *testing.T) {
			got := sanitiseScopeID(tc.input)
			if got != tc.want {
				t.Fatalf("sanitiseScopeID(%q) = %q, want %q", tc.input, got, tc.want)
			}
		})
	}
}
