// SPDX-License-Identifier: Apache-2.0

package memory

import (
	"context"
	"regexp"
	"strings"

	"github.com/jeffs-brain/memory/go/brain"
)

// WikilinkPattern matches [[topic]], [[global:topic]], and
// [[topic|Display Text]] syntax.
var WikilinkPattern = regexp.MustCompile(`\[\[([^\]]+)\]\]`)

// ResolveWikilink resolves a wikilink reference to a logical brain path.
//
// For [[topic]], searches project memory first, falls back to global.
// For [[global:topic]], searches only global memory. The |Display Text
// part (if present) is stripped before resolution.
func (m *Memory) ResolveWikilink(ctx context.Context, link, projectPath string) brain.Path {
	if idx := strings.Index(link, "|"); idx >= 0 {
		link = link[:idx]
	}
	link = strings.TrimSpace(link)
	if link == "" {
		return ""
	}

	slug := ProjectSlug(projectPath)

	if strings.HasPrefix(link, "global:") {
		topic := strings.TrimSpace(strings.TrimPrefix(link, "global:"))
		return m.resolveTopicIn(ctx, topic, brain.MemoryGlobalPrefix())
	}

	if resolved := m.resolveTopicIn(ctx, link, brain.MemoryProjectPrefix(slug)); resolved != "" {
		return resolved
	}
	return m.resolveTopicIn(ctx, link, brain.MemoryGlobalPrefix())
}

// resolveTopicIn normalises a topic and checks whether the corresponding
// .md file exists under the given logical prefix.
func (m *Memory) resolveTopicIn(ctx context.Context, topic string, prefix brain.Path) brain.Path {
	normalised := normaliseTopic(topic)
	if normalised == "" {
		return ""
	}
	candidate := brain.Path(string(prefix) + "/" + normalised + ".md")
	exists, err := m.store.Exists(ctx, candidate)
	if err != nil || !exists {
		return ""
	}
	return candidate
}

// ResolveAllWikilinks resolves all wikilinks in content and returns the
// set of resolved paths (deduplicated).
func (m *Memory) ResolveAllWikilinks(ctx context.Context, content, projectPath string) []brain.Path {
	links := ExtractWikilinks(content)
	if len(links) == 0 {
		return nil
	}
	seen := make(map[brain.Path]struct{})
	var paths []brain.Path
	for _, link := range links {
		resolved := m.ResolveWikilink(ctx, link, projectPath)
		if resolved == "" {
			continue
		}
		if _, ok := seen[resolved]; ok {
			continue
		}
		seen[resolved] = struct{}{}
		paths = append(paths, resolved)
	}
	return paths
}

// ExtractWikilinks returns all wikilink references found in content.
func ExtractWikilinks(content string) []string {
	matches := WikilinkPattern.FindAllStringSubmatch(content, -1)
	if len(matches) == 0 {
		return nil
	}
	links := make([]string, 0, len(matches))
	for _, m := range matches {
		links = append(links, m[1])
	}
	return links
}

// normaliseTopic converts a human-readable topic name to a kebab-case
// filename stem.
func normaliseTopic(topic string) string {
	topic = strings.TrimSpace(topic)
	topic = strings.ToLower(topic)
	topic = strings.ReplaceAll(topic, " ", "-")
	return topic
}
