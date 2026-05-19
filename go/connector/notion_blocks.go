// SPDX-License-Identifier: Apache-2.0

package connector

import (
	"encoding/json"
	"fmt"
	"strings"
)

// notionSearchResponse models the Notion search API response.
type notionSearchResponse struct {
	Results    []json.RawMessage `json:"results"`
	NextCursor string            `json:"next_cursor"`
	HasMore    bool              `json:"has_more"`
}

// notionDatabaseQueryResponse models the database query response.
type notionDatabaseQueryResponse struct {
	Results    []json.RawMessage `json:"results"`
	NextCursor string            `json:"next_cursor"`
	HasMore    bool              `json:"has_more"`
}

// notionBlockChildrenResponse models the block children response.
type notionBlockChildrenResponse struct {
	Results    []notionBlock `json:"results"`
	NextCursor string       `json:"next_cursor"`
	HasMore    bool         `json:"has_more"`
}

// notionBlock represents a single Notion block.
type notionBlock struct {
	ID          string          `json:"id"`
	Type        string          `json:"type"`
	HasChildren bool            `json:"has_children"`
	RawData     json.RawMessage `json:"-"`
}

// UnmarshalJSON implements custom unmarshalling for notionBlock to
// capture the full raw data alongside parsed fields.
func (b *notionBlock) UnmarshalJSON(data []byte) error {
	type alias notionBlock
	var a alias
	if err := json.Unmarshal(data, &a); err != nil {
		return err
	}
	a.RawData = data
	*b = notionBlock(a)
	return nil
}

// notionRichText represents a Notion rich text object.
type notionRichText struct {
	Type        string             `json:"type"`
	PlainText   string             `json:"plain_text"`
	Href        string             `json:"href"`
	Annotations notionAnnotations  `json:"annotations"`
}

// notionAnnotations contains formatting flags for rich text.
type notionAnnotations struct {
	Bold          bool   `json:"bold"`
	Italic        bool   `json:"italic"`
	Strikethrough bool   `json:"strikethrough"`
	Underline     bool   `json:"underline"`
	Code          bool   `json:"code"`
	Colour        string `json:"color"`
}

// notionListBlockTypes is the set of block types considered list items
// for indentation purposes. Module-level constant avoids per-call
// allocation.
var notionListBlockTypes = map[string]struct{}{
	"bulleted_list_item": {},
	"numbered_list_item": {},
	"to_do":              {},
	"toggle":             {},
}

// isListBlock returns true if the block type is a list item.
func isListBlock(blockType string) bool {
	_, ok := notionListBlockTypes[blockType]
	return ok
}

// blockToMarkdown converts a single Notion block to its markdown
// representation.
func (c *NotionConnector) blockToMarkdown(block notionBlock) string {
	// Skip blocks excluded by the block type filter.
	if len(c.config.BlockTypeFilter) > 0 {
		if _, allowed := c.config.BlockTypeFilter[block.Type]; !allowed {
			return ""
		}
	}

	var raw map[string]json.RawMessage
	if err := json.Unmarshal(block.RawData, &raw); err != nil {
		return ""
	}

	blockData, ok := raw[block.Type]
	if !ok {
		return ""
	}

	var typed struct {
		RichText   []notionRichText `json:"rich_text"`
		Caption    []notionRichText `json:"caption"`
		Language   string           `json:"language"`
		URL        string           `json:"url"`
		Checked    bool             `json:"checked"`
		Expression string           `json:"expression"`
	}
	if err := json.Unmarshal(blockData, &typed); err != nil {
		return ""
	}

	text := renderRichText(typed.RichText)

	blockRenderers := map[string]func() string{
		"paragraph":          func() string { return text + "\n" },
		"heading_1":          func() string { return "# " + text + "\n" },
		"heading_2":          func() string { return "## " + text + "\n" },
		"heading_3":          func() string { return "### " + text + "\n" },
		"bulleted_list_item": func() string { return "- " + text },
		"numbered_list_item": func() string { return "1. " + text },
		"to_do": func() string {
			marker := "[ ]"
			if typed.Checked {
				marker = "[x]"
			}
			return "- " + marker + " " + text
		},
		"toggle":    func() string { return "- " + text },
		"quote":     func() string { return "> " + text + "\n" },
		"callout":   func() string { return "> " + text + "\n" },
		"divider":   func() string { return "---\n" },
		"code": func() string {
			lang := typed.Language
			return "```" + lang + "\n" + text + "\n```\n"
		},
		"equation": func() string {
			return "$$\n" + typed.Expression + "\n$$\n"
		},
		"image": func() string {
			caption := renderRichText(typed.Caption)
			url := extractFileURL(blockData)
			if caption != "" {
				return fmt.Sprintf("![%s](%s)\n", caption, url)
			}
			return fmt.Sprintf("![](%s)\n", url)
		},
		"file": func() string {
			url := extractFileURL(blockData)
			caption := renderRichText(typed.Caption)
			if caption == "" {
				caption = "file"
			}
			return fmt.Sprintf("[%s](%s)\n", caption, url)
		},
		"bookmark": func() string {
			caption := renderRichText(typed.Caption)
			if typed.URL != "" {
				if caption != "" {
					return fmt.Sprintf("[%s](%s)\n", caption, typed.URL)
				}
				return typed.URL + "\n"
			}
			return ""
		},
		"embed": func() string {
			if typed.URL != "" {
				return typed.URL + "\n"
			}
			return ""
		},
		"table_of_contents": func() string { return "" },
		"breadcrumb":        func() string { return "" },
		"column_list":       func() string { return "" },
		"column":            func() string { return "" },
		"child_page":        func() string { return "" },
		"child_database":    func() string { return "" },
		"synced_block":      func() string { return "" },
		"link_preview": func() string {
			if typed.URL != "" {
				return typed.URL + "\n"
			}
			return ""
		},
		"table":     func() string { return "" },
		"table_row": func() string { return renderTableRow(blockData) },
	}

	renderer, found := blockRenderers[block.Type]
	if !found {
		return text
	}
	return renderer()
}

// renderRichText converts a slice of Notion rich text objects to a
// markdown string with formatting annotations.
func renderRichText(texts []notionRichText) string {
	var builder strings.Builder
	for _, t := range texts {
		text := t.PlainText

		if t.Annotations.Code {
			text = "`" + text + "`"
		}
		if t.Annotations.Bold {
			text = "**" + text + "**"
		}
		if t.Annotations.Italic {
			text = "*" + text + "*"
		}
		if t.Annotations.Strikethrough {
			text = "~~" + text + "~~"
		}

		if t.Href != "" {
			text = "[" + text + "](" + t.Href + ")"
		}

		builder.WriteString(text)
	}
	return builder.String()
}

// renderPlainRichText extracts plain text from a rich text array
// without formatting.
func renderPlainRichText(texts []notionRichText) string {
	var parts []string
	for _, t := range texts {
		parts = append(parts, t.PlainText)
	}
	return strings.Join(parts, "")
}

// extractFileURL pulls the URL from a file-type block's JSON data.
func extractFileURL(data json.RawMessage) string {
	var fileBlock struct {
		External struct {
			URL string `json:"url"`
		} `json:"external"`
		File struct {
			URL string `json:"url"`
		} `json:"file"`
	}
	if err := json.Unmarshal(data, &fileBlock); err != nil {
		return ""
	}
	if fileBlock.File.URL != "" {
		return fileBlock.File.URL
	}
	return fileBlock.External.URL
}

// renderTableRow converts a table_row block to a markdown table
// row.
func renderTableRow(data json.RawMessage) string {
	var row struct {
		Cells [][]notionRichText `json:"cells"`
	}
	if err := json.Unmarshal(data, &row); err != nil {
		return ""
	}

	cells := make([]string, 0, len(row.Cells))
	for _, cell := range row.Cells {
		cells = append(cells, renderRichText(cell))
	}

	return "| " + strings.Join(cells, " | ") + " |"
}
