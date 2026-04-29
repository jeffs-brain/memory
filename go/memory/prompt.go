// SPDX-License-Identifier: Apache-2.0

package memory

import (
	"context"
	"fmt"
	"strings"

	"github.com/jeffs-brain/memory/go/brain"
)

// MemoryInstructions is appended to the base system prompt so the model
// knows how to use the persistent memory system.
const MemoryInstructions = `# Persistent Memory

You have a two-tier persistent memory system. Memories are stored as markdown files and persist across sessions.

## Scopes

- **Global memory** — cross-project knowledge about the user, stored at the global memory directory. Shared across all projects.
- **Project memory** — codebase-specific knowledge, stored at the project memory directory. Scoped to the current project.

Each scope has its own **MEMORY.md** index file (loaded into your context) and individual **topic files** (.md with YAML frontmatter).

## Memory Types

**Global scope** (user-level, cross-project):
- **user**: Role, preferences, expertise, working style.
- **feedback**: Corrections, validated approaches, lessons the user has taught you.

**Project scope** (codebase-specific):
- **project**: Architecture, decisions, constraints, patterns specific to this codebase.
- **reference**: Pointers to external systems, documentation, APIs relevant to this project.

Choose the scope based on the type of knowledge. If it would be useful in other projects, it belongs in global. If it only makes sense in the context of the current codebase, it belongs in project.

## Storage Format

- **MEMORY.md** index entries should be one line each, under ~150 characters.
- **Topic files** sit alongside MEMORY.md and should have YAML frontmatter with name, description, and type fields.
- You can read and write memory files using your existing read, write, and edit tools.

## What NOT to Save

- Code patterns, file paths, or project structure — derive these from the code.
- Git history or recent changes — use git log / git blame.
- Debugging solutions — the fix is in the code, the context is in the commit.
- Ephemeral task details or current conversation context.

## Using Memories

- Before acting on a memory, verify it is still accurate — files may have been renamed, functions removed.
- If a memory names a file path, check the file exists before recommending it.
- Update or remove memories that are wrong or outdated.
- Do not write duplicate memories. Check for existing ones first.

Example topic file:

` + "```" + `markdown
---
name: Architecture Overview
description: High-level system architecture and key decisions
type: project
---

Content goes here...
` + "```" + ``

// BuildMemoryPromptFor reads both global and project MEMORY.md indexes
// and formats them for injection into the system prompt. Returns an
// empty string if neither index file exists.
func (m *Memory) BuildMemoryPromptFor(projectPath string) string {
	ctx := context.Background()
	globalIndex := m.LoadGlobalIndex(ctx)
	projectIndex := m.LoadProjectIndex(ctx, projectPath)

	if globalIndex == "" && projectIndex == "" {
		return ""
	}

	var b strings.Builder
	if globalIndex != "" {
		label := "memory/global"
		if phys, ok := m.store.LocalPath(brain.MemoryGlobalPrefix()); ok {
			label = fmt.Sprintf("memory/global (%s)", phys)
		}
		b.WriteString("# Global Memory\n\n")
		b.WriteString(globalIndex)
		b.WriteString(fmt.Sprintf("\n\n**Global memory directory:** %s\n\n", label))
	}
	if projectIndex != "" {
		slug := ProjectSlug(projectPath)
		label := "memory/project/" + slug
		if phys, ok := m.store.LocalPath(brain.MemoryProjectPrefix(slug)); ok {
			label = fmt.Sprintf("memory/project/%s (%s)", slug, phys)
		}
		b.WriteString("# Project Memory\n\n")
		b.WriteString(projectIndex)
		b.WriteString(fmt.Sprintf("\n\n**Project memory directory:** %s", label))
	}
	return strings.TrimSpace(b.String())
}
