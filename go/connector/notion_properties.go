// SPDX-License-Identifier: Apache-2.0

package connector

import (
	"encoding/json"
	"fmt"
	"sort"
	"strings"
	"time"
)

// extractTitleFromProperties finds the title property in a page's
// properties map and returns its plain text value.
func extractTitleFromProperties(properties map[string]json.RawMessage) string {
	for _, propRaw := range properties {
		var prop struct {
			Type  string `json:"type"`
			Title []struct {
				PlainText string `json:"plain_text"`
			} `json:"title"`
		}
		if err := json.Unmarshal(propRaw, &prop); err != nil {
			continue
		}
		if prop.Type == "title" && len(prop.Title) > 0 {
			var parts []string
			for _, t := range prop.Title {
				parts = append(parts, t.PlainText)
			}
			return strings.Join(parts, "")
		}
	}
	return ""
}

// extractPropertyValue converts a Notion property value to its
// string representation.
func extractPropertyValue(raw json.RawMessage) string {
	var prop struct {
		Type           string           `json:"type"`
		Title          []notionRichText `json:"title"`
		RichText       []notionRichText `json:"rich_text"`
		Number         *float64         `json:"number"`
		Select         *struct {
			Name string `json:"name"`
		} `json:"select"`
		MultiSelect []struct {
			Name string `json:"name"`
		} `json:"multi_select"`
		Date *struct {
			Start string `json:"start"`
			End   string `json:"end"`
		} `json:"date"`
		Checkbox bool   `json:"checkbox"`
		URL      string `json:"url"`
		Email    string `json:"email"`
		Phone    string `json:"phone_number"`
		Status   *struct {
			Name string `json:"name"`
		} `json:"status"`
		People []struct {
			Name string `json:"name"`
		} `json:"people"`
		Relation []struct {
			ID string `json:"id"`
		} `json:"relation"`
	}
	if err := json.Unmarshal(raw, &prop); err != nil {
		return ""
	}

	renderers := map[string]func() string{
		"title":    func() string { return renderPlainRichText(prop.Title) },
		"rich_text": func() string { return renderPlainRichText(prop.RichText) },
		"number": func() string {
			if prop.Number == nil {
				return ""
			}
			return fmt.Sprintf("%g", *prop.Number)
		},
		"select": func() string {
			if prop.Select == nil {
				return ""
			}
			return prop.Select.Name
		},
		"multi_select": func() string {
			names := make([]string, 0, len(prop.MultiSelect))
			for _, ms := range prop.MultiSelect {
				names = append(names, ms.Name)
			}
			return strings.Join(names, ", ")
		},
		"date": func() string {
			if prop.Date == nil {
				return ""
			}
			if prop.Date.End != "" {
				return prop.Date.Start + " to " + prop.Date.End
			}
			return prop.Date.Start
		},
		"checkbox": func() string {
			if prop.Checkbox {
				return "true"
			}
			return "false"
		},
		"url":          func() string { return prop.URL },
		"email":        func() string { return prop.Email },
		"phone_number": func() string { return prop.Phone },
		"status": func() string {
			if prop.Status == nil {
				return ""
			}
			return prop.Status.Name
		},
		"people": func() string {
			names := make([]string, 0, len(prop.People))
			for _, p := range prop.People {
				names = append(names, p.Name)
			}
			return strings.Join(names, ", ")
		},
		"relation": func() string {
			ids := make([]string, 0, len(prop.Relation))
			for _, r := range prop.Relation {
				ids = append(ids, r.ID)
			}
			return strings.Join(ids, ", ")
		},
	}

	renderer, found := renderers[prop.Type]
	if !found {
		return ""
	}
	return renderer()
}

// databaseEntryParsed holds parsed fields from a database entry.
type databaseEntryParsed struct {
	pageID             string
	title              string
	url                string
	lastEditedTime     time.Time
	propertiesMarkdown string
}

// parseDatabaseEntry extracts structured data from a raw database
// entry JSON.
func parseDatabaseEntry(raw json.RawMessage) (databaseEntryParsed, error) {
	var entry struct {
		ID             string                     `json:"id"`
		URL            string                     `json:"url"`
		LastEditedTime string                     `json:"last_edited_time"`
		Properties     map[string]json.RawMessage `json:"properties"`
	}
	if err := json.Unmarshal(raw, &entry); err != nil {
		return databaseEntryParsed{}, err
	}

	editedTime, _ := time.Parse(time.RFC3339, entry.LastEditedTime)

	title := ""
	var propLines []string

	// Sort property keys for deterministic output.
	keys := make([]string, 0, len(entry.Properties))
	for k := range entry.Properties {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	for _, key := range keys {
		propRaw := entry.Properties[key]
		val := extractPropertyValue(propRaw)

		// Detect title property.
		var propType struct {
			Type string `json:"type"`
		}
		if err := json.Unmarshal(propRaw, &propType); err == nil && propType.Type == "title" {
			title = val
			continue
		}

		if val != "" {
			propLines = append(propLines, fmt.Sprintf("- **%s**: %s", key, val))
		}
	}

	var builder strings.Builder
	if title != "" {
		builder.WriteString("## " + title + "\n\n")
	}
	if len(propLines) > 0 {
		builder.WriteString(strings.Join(propLines, "\n"))
		builder.WriteString("\n")
	}

	return databaseEntryParsed{
		pageID:             entry.ID,
		title:              title,
		url:                entry.URL,
		lastEditedTime:     editedTime,
		propertiesMarkdown: builder.String(),
	}, nil
}

// parsePageResponse extracts page metadata from a Notion page API
// response.
func parsePageResponse(data []byte) (NotionPage, error) {
	var resp struct {
		ID             string                     `json:"id"`
		URL            string                     `json:"url"`
		LastEditedTime string                     `json:"last_edited_time"`
		Properties     map[string]json.RawMessage `json:"properties"`
		Parent         struct {
			Type   string `json:"type"`
			PageID string `json:"page_id"`
		} `json:"parent"`
	}
	if err := json.Unmarshal(data, &resp); err != nil {
		return NotionPage{}, fmt.Errorf("parsing page: %w", err)
	}

	editedTime, _ := time.Parse(time.RFC3339, resp.LastEditedTime)

	// Extract title from properties.
	title := extractTitleFromProperties(resp.Properties)

	return NotionPage{
		ID:             resp.ID,
		Title:          title,
		URL:            resp.URL,
		LastEditedTime: editedTime,
		ParentID:       resp.Parent.PageID,
	}, nil
}
