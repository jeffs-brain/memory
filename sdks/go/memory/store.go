// SPDX-License-Identifier: Apache-2.0

package memory

import (
	"context"
	"errors"
	"strings"

	"github.com/jeffs-brain/memory/go/brain"
)

// maxIndexLines caps the number of lines read from MEMORY.md to keep
// the system prompt manageable.
const maxIndexLines = 200

// TopicFile describes a memory topic file discovered in the memory directory.
// Path is a logical [brain.Path]; production code never stores OS paths here.
type TopicFile struct {
	Name        string
	Description string
	Type        string
	Path        brain.Path
	Created     string
	Modified    string
	Tags        []string
	Confidence  string
	Source      string
	Scope       string
}

// Frontmatter holds parsed YAML frontmatter fields from a memory file.
type Frontmatter struct {
	Name         string
	Description  string
	Type         string
	Created      string
	Modified     string
	Tags         []string
	Confidence   string
	Source       string
	Supersedes   string
	SupersededBy string
}

// LoadIndexAt reads a MEMORY.md at the given logical path, capped at
// maxIndexLines.
func (m *Memory) LoadIndexAt(ctx context.Context, p brain.Path) string {
	data, err := m.store.Read(ctx, p)
	if err != nil {
		return ""
	}
	content := string(data)
	lines := strings.SplitN(content, "\n", maxIndexLines+1)
	if len(lines) > maxIndexLines {
		lines = lines[:maxIndexLines]
		lines = append(lines, "[...truncated]")
	}
	return strings.TrimSpace(strings.Join(lines, "\n"))
}

// LoadProjectIndex reads the project's MEMORY.md and returns its contents.
func (m *Memory) LoadProjectIndex(ctx context.Context, projectPath string) string {
	slug := ProjectSlug(projectPath)
	return m.LoadIndexAt(ctx, brain.MemoryProjectIndex(slug))
}

// LoadGlobalIndex reads the global MEMORY.md.
func (m *Memory) LoadGlobalIndex(ctx context.Context) string {
	return m.LoadIndexAt(ctx, brain.MemoryGlobalIndex())
}

// listTopicsUnder enumerates topic files under a logical prefix and sets
// the given scope on each.
func (m *Memory) listTopicsUnder(ctx context.Context, prefix brain.Path, scope string) ([]TopicFile, error) {
	entries, err := m.store.List(ctx, prefix, brain.ListOpts{IncludeGenerated: true})
	if err != nil {
		if errors.Is(err, brain.ErrNotFound) {
			return nil, nil
		}
		return nil, err
	}

	var topics []TopicFile
	for _, entry := range entries {
		if entry.IsDir {
			continue
		}
		name := baseName(string(entry.Path))
		if !strings.HasSuffix(name, ".md") {
			continue
		}
		if strings.EqualFold(name, "MEMORY.md") {
			continue
		}

		content, err := m.store.Read(ctx, entry.Path)
		if err != nil {
			continue
		}

		fm, _ := ParseFrontmatter(string(content))
		topicName := fm.Name
		if topicName == "" {
			topicName = strings.TrimSuffix(name, ".md")
		}

		topics = append(topics, TopicFile{
			Name:        topicName,
			Description: fm.Description,
			Type:        fm.Type,
			Path:        entry.Path,
			Created:     fm.Created,
			Modified:    fm.Modified,
			Tags:        fm.Tags,
			Confidence:  fm.Confidence,
			Source:      fm.Source,
			Scope:       scope,
		})
	}

	return topics, nil
}

// ListProjectTopics returns all project-scoped topic files for the given
// project path.
func (m *Memory) ListProjectTopics(ctx context.Context, projectPath string) ([]TopicFile, error) {
	slug := ProjectSlug(projectPath)
	return m.listTopicsUnder(ctx, brain.MemoryProjectPrefix(slug), "project")
}

// ListGlobalTopics returns all global-scoped topic files.
func (m *Memory) ListGlobalTopics(ctx context.Context) ([]TopicFile, error) {
	return m.listTopicsUnder(ctx, brain.MemoryGlobalPrefix(), "global")
}

// ReadTopic reads a topic file at the given logical [brain.Path].
func (m *Memory) ReadTopic(ctx context.Context, p brain.Path) (string, error) {
	data, err := m.store.Read(ctx, p)
	if err != nil {
		return "", err
	}
	return string(data), nil
}

// baseName returns the final path segment of a forward-slash path.
func baseName(p string) string {
	if idx := strings.LastIndex(p, "/"); idx >= 0 {
		return p[idx+1:]
	}
	return p
}

// ParseFrontmatter extracts YAML frontmatter delimited by --- lines at the
// start of a markdown file. It returns the parsed frontmatter struct and the
// remaining body after the frontmatter block.
func ParseFrontmatter(content string) (Frontmatter, string) {
	lines := strings.Split(content, "\n")
	if len(lines) < 2 || strings.TrimSpace(lines[0]) != "---" {
		return Frontmatter{}, content
	}

	closeIdx := -1
	for i := 1; i < len(lines); i++ {
		if strings.TrimSpace(lines[i]) == "---" {
			closeIdx = i
			break
		}
	}
	if closeIdx < 0 {
		return Frontmatter{}, content
	}

	var fm Frontmatter
	var currentListKey string

	for _, line := range lines[1:closeIdx] {
		trimmed := strings.TrimSpace(line)

		if currentListKey != "" && strings.HasPrefix(trimmed, "- ") {
			val := strings.TrimSpace(strings.TrimPrefix(trimmed, "-"))
			if currentListKey == "tags" {
				fm.Tags = append(fm.Tags, val)
			}
			continue
		}

		key, val, ok := parseKV(line)
		if !ok {
			currentListKey = ""
			continue
		}

		if val == "" {
			currentListKey = key
			continue
		}
		currentListKey = ""

		switch key {
		case "name":
			fm.Name = val
		case "description":
			fm.Description = val
		case "type":
			fm.Type = val
		case "created":
			fm.Created = val
		case "modified":
			fm.Modified = val
		case "confidence":
			fm.Confidence = val
		case "source":
			fm.Source = val
		case "supersedes":
			fm.Supersedes = val
		case "superseded_by":
			fm.SupersededBy = val
		case "tags":
			for _, tag := range strings.Split(val, ",") {
				tag = strings.TrimSpace(tag)
				if tag != "" {
					fm.Tags = append(fm.Tags, tag)
				}
			}
		}
	}

	remaining := lines[closeIdx+1:]
	body := strings.TrimSpace(strings.Join(remaining, "\n"))

	return fm, body
}

// parseKV splits a "key: value" line.
func parseKV(line string) (string, string, bool) {
	idx := strings.Index(line, ":")
	if idx < 0 {
		return "", "", false
	}

	key := strings.TrimSpace(line[:idx])
	val := strings.TrimSpace(line[idx+1:])

	if len(val) >= 2 {
		if (val[0] == '"' && val[len(val)-1] == '"') || (val[0] == '\'' && val[len(val)-1] == '\'') {
			val = val[1 : len(val)-1]
		}
	}

	return key, val, true
}
