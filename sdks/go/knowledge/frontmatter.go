// SPDX-License-Identifier: Apache-2.0

package knowledge

import (
	"strings"

	"gopkg.in/yaml.v3"
)

// Frontmatter carries the fields parsed from a markdown YAML header.
//
// Ported verbatim from jeff/apps/jeff/internal/knowledge/frontmatter.go;
// the shape is tuned for memory retrieval so the parser stays
// lenient about quoting and list shapes the hand-written markdown
// habitually uses.
type Frontmatter struct {
	Title    string   `yaml:"title"`
	Summary  string   `yaml:"summary"`
	Tags     []string `yaml:"tags"`
	Sources  []string `yaml:"sources"`
	Source   string   `yaml:"source"`
	Created  string   `yaml:"created"`
	Modified string   `yaml:"modified"`
	// Name and Description mirror the memory-scope frontmatter used
	// by the memory SDK. Present so the same parser covers both
	// article and memory frontmatter without a separate function.
	Name        string `yaml:"name"`
	Description string `yaml:"description"`
	// SourceType tags the origin of the ingest (e.g. "web", "file",
	// "pdf"). Used to route downstream compilation.
	SourceType string `yaml:"source_type"`
	// Ingested is the RFC3339 timestamp written at ingest time.
	Ingested string `yaml:"ingested"`
}

// ParseFrontmatter extracts YAML frontmatter from a markdown document.
//
// The parser accepts the familiar `---\n<yaml>\n---\n<body>` layout and
// returns the parsed [Frontmatter] plus the body that follows. When the
// content has no frontmatter block the returned Frontmatter is zero and
// body is the entire input. A malformed YAML header is handled
// defensively: the structured parse falls back to a line-by-line scan
// so the caller always receives a usable Frontmatter even when a user
// hand-authored the block with inconsistent indentation.
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

	header := strings.Join(lines[1:closeIdx], "\n")
	body := strings.TrimSpace(strings.Join(lines[closeIdx+1:], "\n"))

	var fm Frontmatter
	if err := yaml.Unmarshal([]byte(header), &fm); err == nil {
		// YAML path produced clean results. Fall through to the
		// line-scan only when the struct is entirely blank, which
		// happens when the YAML parses but does not match any of our
		// fields.
		if fm.hasAnything() {
			return fm, body
		}
	}

	// Line-scan fallback mirrors jeff's hand-rolled parser.
	fm = parseFrontmatterLineScan(lines[1:closeIdx])
	return fm, body
}

// hasAnything reports whether any field carries content. Used to pick
// between the YAML result and the line-scan fallback.
func (f Frontmatter) hasAnything() bool {
	return f.Title != "" || f.Summary != "" || f.Source != "" ||
		f.Created != "" || f.Modified != "" || f.Name != "" ||
		f.Description != "" || f.SourceType != "" || f.Ingested != "" ||
		len(f.Tags) > 0 || len(f.Sources) > 0
}

// parseFrontmatterLineScan implements the jeff-style fallback parser
// that walks the YAML block line by line. Handles bullet lists
// continuing from an empty `key:` header.
func parseFrontmatterLineScan(lines []string) Frontmatter {
	var fm Frontmatter
	var currentListKey string

	for _, line := range lines {
		trimmed := strings.TrimSpace(line)

		if currentListKey != "" && strings.HasPrefix(trimmed, "- ") {
			val := strings.TrimSpace(strings.TrimPrefix(trimmed, "-"))
			val = stripQuotes(val)
			switch currentListKey {
			case "tags":
				fm.Tags = append(fm.Tags, val)
			case "sources":
				fm.Sources = append(fm.Sources, val)
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
		case "title":
			fm.Title = val
		case "summary":
			fm.Summary = val
		case "source":
			fm.Source = val
		case "source_type":
			fm.SourceType = val
		case "created":
			fm.Created = val
		case "modified":
			fm.Modified = val
		case "ingested":
			fm.Ingested = val
		case "name":
			fm.Name = val
		case "description":
			fm.Description = val
		case "tags":
			for _, tag := range strings.Split(val, ",") {
				tag = stripQuotes(strings.TrimSpace(tag))
				if tag != "" {
					fm.Tags = append(fm.Tags, tag)
				}
			}
		case "sources":
			for _, src := range strings.Split(val, ",") {
				src = stripQuotes(strings.TrimSpace(src))
				if src != "" {
					fm.Sources = append(fm.Sources, src)
				}
			}
		}
	}

	return fm
}

// parseKV splits a "key: value" line. Mirrors the original jeff helper.
func parseKV(line string) (string, string, bool) {
	idx := strings.Index(line, ":")
	if idx < 0 {
		return "", "", false
	}
	key := strings.TrimSpace(line[:idx])
	val := stripQuotes(strings.TrimSpace(line[idx+1:]))
	return key, val, true
}

// stripQuotes removes a single matched pair of single or double quotes
// from val. Other quote shapes (smart quotes, mismatched pairs) are
// left untouched.
func stripQuotes(val string) string {
	if len(val) < 2 {
		return val
	}
	first, last := val[0], val[len(val)-1]
	if (first == '"' && last == '"') || (first == '\'' && last == '\'') {
		return val[1 : len(val)-1]
	}
	return val
}
