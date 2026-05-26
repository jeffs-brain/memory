// SPDX-License-Identifier: Apache-2.0

package graph

import "testing"

func TestResolveEntityAndScope(t *testing.T) {
	tests := []struct {
		path   string
		entity EntityType
		scope  string
	}{
		{path: "memory/project/doc.md", entity: EntityMemory, scope: "project"},
		{path: "memory/global/doc.md", entity: EntityMemory, scope: "global"},
		{path: "memory/agent/doc.md", entity: EntityMemory, scope: "agent"},
		{path: "memory/_procedural/runbook.md", entity: EntityProcedural, scope: "agent"},
		{path: "memory/team/heuristic-alpha.md", entity: EntityHeuristic, scope: "other"},
		{path: "memory/team/anti-pattern-beta.md", entity: EntityHeuristic, scope: "other"},
		{path: "ontology/concept.md", entity: EntityOntology, scope: "other"},
		{path: "wiki/topic.md", entity: EntityArticle, scope: "other"},
		{path: "drafts/draft.md", entity: EntityArticle, scope: "other"},
		{path: "episodes/episode-1.md", entity: EntityEpisode, scope: "other"},
		{path: "reflections/episode-1.md", entity: EntityEpisode, scope: "other"},
		{path: "raw/import.md", entity: EntityIngested, scope: "other"},
		{path: "unknown/path.md", entity: EntityMemory, scope: "other"},
	}

	for _, tc := range tests {
		resolved := resolveEntityAndScope(tc.path)
		if resolved.entity != tc.entity || resolved.scope != tc.scope {
			t.Fatalf("path=%q expected (%s,%s), got (%s,%s)", tc.path, tc.entity, tc.scope, resolved.entity, resolved.scope)
		}
	}
}

func TestFileLabel(t *testing.T) {
	tests := []struct {
		path  string
		label string
	}{
		{path: "memory/project/note.md", label: "note.md"},
		{path: "single", label: "single"},
		{path: "path/with/trailing/", label: "path/with/trailing/"},
		{path: "", label: ""},
	}

	for _, tc := range tests {
		got := fileLabel(tc.path)
		if got != tc.label {
			t.Fatalf("path=%q expected label=%q, got=%q", tc.path, tc.label, got)
		}
	}
}
