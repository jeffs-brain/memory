// SPDX-License-Identifier: Apache-2.0

package memory

import (
	"fmt"
	"regexp"
	"strings"
)

const structuredArtefactMemoryLimit = 12

var (
	artefactFenceStartRe     = regexp.MustCompile("^\\s*```")
	artefactHeadingRe        = regexp.MustCompile("^\\s{0,3}#{1,6}\\s+\\S")
	artefactTableLineRe      = regexp.MustCompile("^\\s*\\|.*\\|\\s*$")
	artefactNumberedLineRe   = regexp.MustCompile("^\\s*(\\d{1,3})[.)]\\s+\\S")
	artefactBulletLineRe     = regexp.MustCompile("^\\s{0,6}[-*+]\\s+\\S")
	artefactRecipeHeadingRe  = regexp.MustCompile("(?i)^\\s*(ingredients|instructions|method|steps|directions)\\s*:?\\s*$")
	artefactSongSectionRe    = regexp.MustCompile("(?i)^\\s*(verse\\s*\\d*|chorus|bridge|pre-chorus|outro|refrain)\\s*:?\\s*$")
	artefactSectionLabelRe   = regexp.MustCompile("(?i)^\\s*([A-Z][A-Za-z0-9 /&'-]{1,48})\\s*:\\s*$")
	artefactTitleTokenRe     = regexp.MustCompile(`[A-Za-z0-9]+`)
	artefactOrdinalCaptureRe = regexp.MustCompile(`(?m)^\s*(\d{1,3})[.)]\s+`)
)

type structuredArtefact struct {
	kind           string
	title          string
	content        string
	section        string
	itemOrdinals   []string
	messageOrdinal int
}

func deriveStructuredArtefactMemories(messages []Message, existing []ExtractedMemory, sessionID, sessionDate string) []ExtractedMemory {
	seen := buildExistingMemoryTextSet(existing)
	iso := deriveHeuristicISODate(messages, sessionDate)
	out := make([]ExtractedMemory, 0, 4)
	assistantOrdinal := 0

	for _, message := range messages {
		if message.Role != RoleAssistant {
			continue
		}
		assistantOrdinal++
		for _, artefact := range detectStructuredArtefacts(message.Content, assistantOrdinal) {
			canonical := normaliseMemoryText(artefact.kind + "\n" + artefact.content)
			if canonical == "" || seen[canonical] {
				continue
			}
			slug := artefactSlug(artefact)
			if slug == "" {
				continue
			}
			out = append(out, buildStructuredArtefactMemory(artefact, iso, sessionID, sessionDate, slug))
			seen[canonical] = true
			if len(out) >= structuredArtefactMemoryLimit {
				return out
			}
		}
	}

	return out
}

func detectStructuredArtefacts(content string, messageOrdinal int) []structuredArtefact {
	lines := strings.Split(strings.ReplaceAll(content, "\r\n", "\n"), "\n")
	out := make([]structuredArtefact, 0, 4)
	out = append(out, detectFencedCodeArtefacts(lines, messageOrdinal)...)
	out = append(out, detectMarkdownTableArtefacts(lines, messageOrdinal)...)
	if recipe := detectRecipeArtefact(content, messageOrdinal); recipe.content != "" {
		out = append(out, recipe)
	}
	if song := detectSongArtefact(content, messageOrdinal); song.content != "" {
		out = append(out, song)
	}
	out = append(out, detectNumberedArtefacts(lines, messageOrdinal)...)
	out = append(out, detectOutlineArtefacts(lines, messageOrdinal)...)
	out = append(out, detectSectionArtefacts(lines, messageOrdinal)...)
	return dedupeStructuredArtefacts(out)
}

func detectFencedCodeArtefacts(lines []string, messageOrdinal int) []structuredArtefact {
	out := make([]structuredArtefact, 0)
	for idx := 0; idx < len(lines); idx++ {
		if !artefactFenceStartRe.MatchString(lines[idx]) {
			continue
		}
		start := idx
		idx++
		for idx < len(lines) && !artefactFenceStartRe.MatchString(lines[idx]) {
			idx++
		}
		if idx >= len(lines) {
			break
		}
		block := strings.TrimSpace(strings.Join(lines[start:idx+1], "\n"))
		if block == "" {
			continue
		}
		out = append(out, structuredArtefact{
			kind:           "code_block",
			title:          "Code block",
			content:        block,
			messageOrdinal: messageOrdinal,
		})
	}
	return out
}

func detectMarkdownTableArtefacts(lines []string, messageOrdinal int) []structuredArtefact {
	out := make([]structuredArtefact, 0)
	for start := 0; start < len(lines); {
		if !artefactTableLineRe.MatchString(lines[start]) {
			start++
			continue
		}
		end := start
		for end < len(lines) && artefactTableLineRe.MatchString(lines[end]) {
			end++
		}
		if end-start >= 3 && isHeuristicMarkdownSeparatorRow(parseHeuristicMarkdownTableCells(lines[start+1])) {
			out = append(out, structuredArtefact{
				kind:           "markdown_table",
				title:          tableArtefactTitle(lines[start]),
				content:        strings.TrimSpace(strings.Join(lines[start:end], "\n")),
				section:        nearestPreviousHeading(lines, start),
				messageOrdinal: messageOrdinal,
			})
		}
		start = end
	}
	return out
}

func detectRecipeArtefact(content string, messageOrdinal int) structuredArtefact {
	lines := strings.Split(strings.ReplaceAll(content, "\r\n", "\n"), "\n")
	hasIngredients := false
	hasMethod := false
	for _, line := range lines {
		heading := strings.ToLower(cleanArtefactHeading(line))
		switch heading {
		case "ingredients":
			hasIngredients = true
		case "instructions", "method", "steps", "directions":
			hasMethod = true
		}
	}
	if !hasIngredients || !hasMethod {
		return structuredArtefact{}
	}
	return structuredArtefact{
		kind:           "recipe",
		title:          "Recipe",
		content:        strings.TrimSpace(content),
		messageOrdinal: messageOrdinal,
	}
}

func detectSongArtefact(content string, messageOrdinal int) structuredArtefact {
	lines := strings.Split(strings.ReplaceAll(content, "\r\n", "\n"), "\n")
	sections := make([]string, 0, 3)
	for _, line := range lines {
		heading := cleanArtefactHeading(line)
		if artefactSongSectionRe.MatchString(heading) {
			sections = append(sections, heading)
		}
	}
	if len(sections) < 2 {
		return structuredArtefact{}
	}
	return structuredArtefact{
		kind:           "song",
		title:          "Song",
		content:        strings.TrimSpace(content),
		section:        strings.Join(sections, ", "),
		messageOrdinal: messageOrdinal,
	}
}

func detectNumberedArtefacts(lines []string, messageOrdinal int) []structuredArtefact {
	out := make([]structuredArtefact, 0)
	for start := 0; start < len(lines); {
		if !artefactNumberedLineRe.MatchString(lines[start]) {
			start++
			continue
		}
		end := start
		ordinals := make([]string, 0)
		for end < len(lines) {
			if matched := artefactNumberedLineRe.FindStringSubmatch(lines[end]); len(matched) == 2 {
				ordinals = append(ordinals, matched[1])
				end++
				continue
			}
			if strings.TrimSpace(lines[end]) == "" && end+1 < len(lines) && artefactNumberedLineRe.MatchString(lines[end+1]) {
				end++
				continue
			}
			break
		}
		if len(ordinals) >= 2 {
			out = append(out, structuredArtefact{
				kind:           "numbered_options",
				title:          "Numbered options",
				content:        strings.TrimSpace(strings.Join(lines[start:end], "\n")),
				section:        nearestPreviousHeading(lines, start),
				itemOrdinals:   ordinals,
				messageOrdinal: messageOrdinal,
			})
		}
		start = end
	}
	return out
}

func detectOutlineArtefacts(lines []string, messageOrdinal int) []structuredArtefact {
	out := make([]structuredArtefact, 0)
	for start := 0; start < len(lines); {
		if !artefactHeadingRe.MatchString(lines[start]) {
			start++
			continue
		}
		end := start + 1
		hasOutlineLine := false
		for end < len(lines) {
			if end > start && artefactHeadingRe.MatchString(lines[end]) {
				break
			}
			if artefactBulletLineRe.MatchString(lines[end]) || artefactNumberedLineRe.MatchString(lines[end]) {
				hasOutlineLine = true
			}
			end++
		}
		if hasOutlineLine {
			out = append(out, structuredArtefact{
				kind:           "outline",
				title:          headingTitle(lines[start]),
				content:        strings.TrimSpace(strings.Join(lines[start:end], "\n")),
				section:        headingTitle(lines[start]),
				itemOrdinals:   numberedOrdinals(strings.Join(lines[start:end], "\n")),
				messageOrdinal: messageOrdinal,
			})
		}
		start = end
	}
	return out
}

func detectSectionArtefacts(lines []string, messageOrdinal int) []structuredArtefact {
	out := make([]structuredArtefact, 0)
	for start := 0; start < len(lines); start++ {
		if !artefactHeadingRe.MatchString(lines[start]) && !artefactRecipeHeadingRe.MatchString(lines[start]) && !artefactSongSectionRe.MatchString(lines[start]) && !artefactSectionLabelRe.MatchString(lines[start]) {
			continue
		}
		end := start + 1
		for end < len(lines) {
			if artefactRecipeHeadingRe.MatchString(lines[end]) || artefactSongSectionRe.MatchString(lines[end]) || artefactSectionLabelRe.MatchString(lines[end]) || artefactHeadingRe.MatchString(lines[end]) {
				break
			}
			end++
		}
		block := strings.TrimSpace(strings.Join(lines[start:end], "\n"))
		if len(strings.Fields(block)) < 6 {
			continue
		}
		if blockHasArtefactListLine(block) {
			continue
		}
		title := cleanArtefactHeading(lines[start])
		out = append(out, structuredArtefact{
			kind:           "section",
			title:          title,
			content:        block,
			section:        title,
			itemOrdinals:   numberedOrdinals(block),
			messageOrdinal: messageOrdinal,
		})
	}
	return out
}

func buildStructuredArtefactMemory(artefact structuredArtefact, iso, sessionID, sessionDate, slug string) ExtractedMemory {
	summary := fmt.Sprintf("Assistant %s from message %d", strings.ReplaceAll(artefact.kind, "_", " "), artefact.messageOrdinal)
	if artefact.title != "" {
		summary += ": " + artefact.title
	}
	content := buildStructuredArtefactContent(artefact)
	return ExtractedMemory{
		Action:             "create",
		Filename:           buildHeuristicSessionFilename("assistant-artefact", iso, sessionID, artefact.kind+"-"+slug),
		Name:               toTitleCase(strings.ReplaceAll(artefact.kind, "_", " ")) + ": " + artefact.title,
		Description:        truncateOneLine(summary, 140),
		Type:               "project",
		Scope:              "project",
		Content:            content,
		IndexEntry:         truncateOneLine(summary, 140),
		SessionID:          strings.TrimSpace(sessionID),
		SessionDate:        strings.TrimSpace(sessionDate),
		SourceRole:         "assistant",
		ArtefactType:       artefact.kind,
		ArtefactOrdinal:    artefact.messageOrdinal,
		ArtefactSection:    artefact.section,
		ArtefactDescriptor: artefact.title,
		Tags:               []string{"artefact", "assistant", artefact.kind},
	}
}

func buildStructuredArtefactContent(artefact structuredArtefact) string {
	var b strings.Builder
	b.WriteString("Kind: ")
	b.WriteString(artefact.kind)
	b.WriteString("\nMessage ordinal: ")
	b.WriteString(fmt.Sprintf("%d", artefact.messageOrdinal))
	if artefact.section != "" {
		b.WriteString("\nSection: ")
		b.WriteString(artefact.section)
	}
	if len(artefact.itemOrdinals) > 0 {
		b.WriteString("\nItem ordinals: ")
		b.WriteString(strings.Join(artefact.itemOrdinals, ", "))
	}
	b.WriteString("\n\nContent:\n")
	b.WriteString(strings.TrimSpace(artefact.content))
	return b.String()
}

func dedupeStructuredArtefacts(input []structuredArtefact) []structuredArtefact {
	seen := make(map[string]bool, len(input))
	out := make([]structuredArtefact, 0, len(input))
	for _, artefact := range input {
		canonical := normaliseMemoryText(artefact.kind + "\n" + artefact.content)
		if canonical == "" || seen[canonical] {
			continue
		}
		seen[canonical] = true
		out = append(out, artefact)
	}
	return out
}

func artefactSlug(artefact structuredArtefact) string {
	source := artefact.title
	if source == "" {
		source = artefact.content
	}
	tokens := artefactTitleTokenRe.FindAllString(strings.ToLower(source), -1)
	if len(tokens) == 0 {
		return ""
	}
	if len(tokens) > 5 {
		tokens = tokens[:5]
	}
	return strings.Join(tokens, "-")
}

func headingTitle(line string) string {
	return cleanArtefactHeading(line)
}

func cleanArtefactHeading(line string) string {
	trimmed := strings.TrimSpace(line)
	trimmed = strings.TrimLeft(trimmed, "#")
	trimmed = strings.TrimSpace(trimmed)
	trimmed = strings.TrimPrefix(trimmed, "**")
	trimmed = strings.TrimSuffix(trimmed, "**")
	trimmed = strings.TrimSuffix(trimmed, ":")
	return strings.TrimSpace(trimmed)
}

func nearestPreviousHeading(lines []string, start int) string {
	for idx := start - 1; idx >= 0; idx-- {
		if strings.TrimSpace(lines[idx]) == "" {
			continue
		}
		if artefactHeadingRe.MatchString(lines[idx]) {
			return headingTitle(lines[idx])
		}
		return ""
	}
	return ""
}

func tableArtefactTitle(header string) string {
	cells := parseHeuristicMarkdownTableCells(header)
	if len(cells) == 0 {
		return "Markdown table"
	}
	return "Table: " + strings.Join(cells, " / ")
}

func numberedOrdinals(content string) []string {
	matches := artefactOrdinalCaptureRe.FindAllStringSubmatch(content, -1)
	out := make([]string, 0, len(matches))
	for _, matched := range matches {
		if len(matched) == 2 {
			out = append(out, matched[1])
		}
	}
	return out
}

func blockHasArtefactListLine(content string) bool {
	for _, line := range strings.Split(content, "\n") {
		if artefactBulletLineRe.MatchString(line) || artefactNumberedLineRe.MatchString(line) {
			return true
		}
	}
	return false
}
