// SPDX-License-Identifier: Apache-2.0

package graph

import (
	"regexp"
	"strings"
)

var heuristicPathRE = regexp.MustCompile(`^memory/.*/(heuristic-|anti-pattern-)`)

type entityScope struct {
	entity EntityType
	scope  string
}

type entityRule struct {
	predicate func(string) bool
	result    entityScope
}

var entityRules = []entityRule{
	{predicate: func(path string) bool { return heuristicPathRE.MatchString(path) }, result: entityScope{entity: EntityHeuristic, scope: "other"}},
	{predicate: func(path string) bool { return strings.HasPrefix(path, "memory/_procedural/") }, result: entityScope{entity: EntityProcedural, scope: "agent"}},
	{predicate: func(path string) bool { return strings.HasPrefix(path, "memory/project/") }, result: entityScope{entity: EntityMemory, scope: "project"}},
	{predicate: func(path string) bool { return strings.HasPrefix(path, "memory/global/") }, result: entityScope{entity: EntityMemory, scope: "global"}},
	{predicate: func(path string) bool { return strings.HasPrefix(path, "memory/agent/") }, result: entityScope{entity: EntityMemory, scope: "agent"}},
	{predicate: func(path string) bool { return strings.HasPrefix(path, "ontology/") }, result: entityScope{entity: EntityOntology, scope: "other"}},
	{predicate: func(path string) bool { return strings.HasPrefix(path, "wiki/") }, result: entityScope{entity: EntityArticle, scope: "other"}},
	{predicate: func(path string) bool { return strings.HasPrefix(path, "drafts/") }, result: entityScope{entity: EntityArticle, scope: "other"}},
	{predicate: func(path string) bool { return strings.HasPrefix(path, "episodes/") }, result: entityScope{entity: EntityEpisode, scope: "other"}},
	{predicate: func(path string) bool { return strings.HasPrefix(path, "raw/") }, result: entityScope{entity: EntityIngested, scope: "other"}},
	{predicate: func(path string) bool { return strings.HasPrefix(path, "reflections/") }, result: entityScope{entity: EntityEpisode, scope: "other"}},
}

var fallbackScope = entityScope{entity: EntityMemory, scope: "other"}

func resolveEntityAndScope(path string) entityScope {
	for _, rule := range entityRules {
		if rule.predicate(path) {
			return rule.result
		}
	}
	return fallbackScope
}

func fileLabel(path string) string {
	parts := strings.Split(path, "/")
	if len(parts) == 0 {
		return path
	}
	last := parts[len(parts)-1]
	if last == "" {
		return path
	}
	return last
}
