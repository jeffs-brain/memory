// SPDX-License-Identifier: Apache-2.0

package search

import "strings"

// memoryFrontmatter mirrors the subset of [memory.Frontmatter] fields the
// search package consumes during indexing. Ported from
// jeff/internal/memory/store.go so the search package stays free of a
// cross-package dependency on the memory SDK.
type memoryFrontmatter struct {
	Name         string
	Description  string
	Tags         []string
	Modified     string
	SupersededBy string
}

// wikiFrontmatter mirrors the subset of [knowledge.WikiFrontmatter]
// fields the search package consumes during indexing. Ported from
// jeff/internal/knowledge/frontmatter.go.
type wikiFrontmatter struct {
	Title    string
	Summary  string
	Tags     []string
	Modified string
}

// parseMemoryFrontmatter extracts YAML frontmatter delimited by `---`
// lines at the start of a markdown file. Returns the parsed struct and
// the remaining body after the frontmatter block.
func parseMemoryFrontmatter(content string) (memoryFrontmatter, string) {
	lines := strings.Split(content, "\n")
	if len(lines) < 2 || strings.TrimSpace(lines[0]) != "---" {
		return memoryFrontmatter{}, content
	}

	closeIdx := -1
	for i := 1; i < len(lines); i++ {
		if strings.TrimSpace(lines[i]) == "---" {
			closeIdx = i
			break
		}
	}
	if closeIdx < 0 {
		return memoryFrontmatter{}, content
	}

	var fm memoryFrontmatter
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

		key, val, ok := parseFrontmatterKV(line)
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
		case "modified":
			fm.Modified = val
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

// parseWikiFrontmatter extracts YAML frontmatter from a wiki article.
// Returns the parsed struct and the body that follows.
func parseWikiFrontmatter(content string) (wikiFrontmatter, string) {
	lines := strings.Split(content, "\n")
	if len(lines) < 2 || strings.TrimSpace(lines[0]) != "---" {
		return wikiFrontmatter{}, content
	}

	closeIdx := -1
	for i := 1; i < len(lines); i++ {
		if strings.TrimSpace(lines[i]) == "---" {
			closeIdx = i
			break
		}
	}
	if closeIdx < 0 {
		return wikiFrontmatter{}, content
	}

	var fm wikiFrontmatter
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

		key, val, ok := parseFrontmatterKV(line)
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
		case "title":
			fm.Title = val
		case "summary":
			fm.Summary = val
		case "modified":
			fm.Modified = val
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

// parseFrontmatterKV splits a "key: value" line. Returns the key,
// value, and whether the split was successful. Values are trimmed of
// surrounding whitespace and optional single or double quotes.
func parseFrontmatterKV(line string) (string, string, bool) {
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
