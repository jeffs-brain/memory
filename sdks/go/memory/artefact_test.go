// SPDX-License-Identifier: Apache-2.0

package memory

import (
	"context"
	"strings"
	"testing"
)

func TestDeriveStructuredArtefactMemories_PreservesAssistantArtefacts(t *testing.T) {
	messages := []Message{
		{Role: RoleSystem, Content: "This conversation took place on 2024/03/13 (Wed) 09:30."},
		{Role: RoleUser, Content: "Give me useful options and a recipe."},
		{Role: RoleAssistant, Content: "## Dinner options\n1. Lentil soup with lemon\n2. Mushroom risotto\n3. Tomato pasta\n\nIngredients:\n- 200g lentils\n- 1 lemon\n\nMethod:\n1. Simmer the lentils.\n2. Finish with lemon."},
	}

	out := deriveStructuredArtefactMemories(messages, nil, "session-a", "2024-03-13")

	numbered := findArtefactMemory(out, "numbered_options")
	if numbered == nil {
		t.Fatalf("expected numbered options artefact in %#v", out)
	}
	if numbered.SourceRole != "assistant" {
		t.Fatalf("source role = %q, want assistant", numbered.SourceRole)
	}
	if numbered.SessionID != "session-a" || numbered.SessionDate != "2024-03-13" {
		t.Fatalf("session metadata = %q/%q, want session-a/2024-03-13", numbered.SessionID, numbered.SessionDate)
	}
	if !strings.Contains(numbered.Content, "Message ordinal: 1") {
		t.Fatalf("numbered content missing message ordinal:\n%s", numbered.Content)
	}
	if !strings.Contains(numbered.Content, "Section: Dinner options") {
		t.Fatalf("numbered content missing section:\n%s", numbered.Content)
	}
	if !strings.Contains(numbered.Content, "Item ordinals: 1, 2, 3") {
		t.Fatalf("numbered content missing ordinals:\n%s", numbered.Content)
	}
	if !strings.Contains(numbered.Content, "1. Lentil soup with lemon\n2. Mushroom risotto\n3. Tomato pasta") {
		t.Fatalf("numbered content did not preserve exact list:\n%s", numbered.Content)
	}

	recipe := findArtefactMemory(out, "recipe")
	if recipe == nil {
		t.Fatalf("expected recipe artefact in %#v", out)
	}
	if !strings.Contains(recipe.Content, "Ingredients:\n- 200g lentils\n- 1 lemon\n\nMethod:\n1. Simmer the lentils.\n2. Finish with lemon.") {
		t.Fatalf("recipe content did not preserve exact recipe:\n%s", recipe.Content)
	}
}

func TestExtractFromMessages_AddsAssistantStructuredArtefacts(t *testing.T) {
	mem, _ := newTestMemory(t)
	provider := &extractStubProvider{reply: `{"memories":[]}`}

	out, err := ExtractFromMessagesWithSession(context.Background(), provider, "test-model", mem, "/project", []Message{
		{Role: RoleSystem, Content: "This conversation took place on 2024/03/13 (Wed) 09:30."},
		{Role: RoleUser, Content: "Write the song and show me the table."},
		{Role: RoleAssistant, Content: "Verse 1:\nThe kettle hums beside the rain\n\nChorus:\nKeep the porch light burning\n\n| Name | Shift |\n| --- | --- |\n| Alex | 8 am - 4 pm |"},
	}, "session-b", "2024-03-13")
	if err != nil {
		t.Fatalf("ExtractFromMessagesWithSession: %v", err)
	}

	song := findArtefactMemory(out, "song")
	if song == nil {
		t.Fatalf("expected song artefact in %#v", out)
	}
	if !strings.Contains(song.Content, "Section: Verse 1, Chorus") {
		t.Fatalf("song content missing section metadata:\n%s", song.Content)
	}
	if !strings.Contains(song.Content, "Verse 1:\nThe kettle hums beside the rain\n\nChorus:\nKeep the porch light burning") {
		t.Fatalf("song content did not preserve exact content:\n%s", song.Content)
	}

	table := findArtefactMemory(out, "markdown_table")
	if table == nil {
		t.Fatalf("expected markdown table artefact in %#v", out)
	}
	if !strings.Contains(table.Content, "| Name | Shift |\n| --- | --- |\n| Alex | 8 am - 4 pm |") {
		t.Fatalf("table content did not preserve exact table:\n%s", table.Content)
	}
	if table.SourceRole != "assistant" {
		t.Fatalf("table source role = %q, want assistant", table.SourceRole)
	}
}

func TestDeriveStructuredArtefactMemories_DetectsCodeOutlineAndSections(t *testing.T) {
	messages := []Message{
		{Role: RoleUser, Content: "Draft the plan and include code."},
		{Role: RoleAssistant, Content: "```go\nfmt.Println(\"hello\")\n```\n\n## Launch outline\n- Prepare release notes\n- Publish package\n\n## Risks\nThe main risk is stale documentation after the package is published."},
	}

	out := deriveStructuredArtefactMemories(messages, nil, "", "")

	code := findArtefactMemory(out, "code_block")
	if code == nil {
		t.Fatalf("expected code block artefact in %#v", out)
	}
	if !strings.Contains(code.Content, "```go\nfmt.Println(\"hello\")\n```") {
		t.Fatalf("code content did not preserve exact fence:\n%s", code.Content)
	}

	outline := findArtefactMemory(out, "outline")
	if outline == nil {
		t.Fatalf("expected outline artefact in %#v", out)
	}
	if !strings.Contains(outline.Content, "Section: Launch outline") {
		t.Fatalf("outline content missing section metadata:\n%s", outline.Content)
	}
	if !strings.Contains(outline.Content, "- Prepare release notes\n- Publish package") {
		t.Fatalf("outline content did not preserve bullets:\n%s", outline.Content)
	}

	section := findArtefactMemory(out, "section")
	if section == nil {
		t.Fatalf("expected section artefact in %#v", out)
	}
	if !strings.Contains(section.Content, "Section: Risks") {
		t.Fatalf("section content missing heading metadata:\n%s", section.Content)
	}
	if !strings.Contains(section.Content, "The main risk is stale documentation after the package is published.") {
		t.Fatalf("section content did not preserve prose:\n%s", section.Content)
	}
}

func findArtefactMemory(memories []ExtractedMemory, kind string) *ExtractedMemory {
	for i := range memories {
		if strings.Contains(memories[i].Content, "Kind: "+kind) {
			return &memories[i]
		}
	}
	return nil
}
