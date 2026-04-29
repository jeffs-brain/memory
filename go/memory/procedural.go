// SPDX-License-Identifier: Apache-2.0

package memory

import (
	"encoding/json"
	"fmt"
	"strings"
	"time"
)

// ProceduralRecord is a structured record of a skill or tool sequence
// invocation, used to build procedural memory over repeated use.
type ProceduralRecord struct {
	Tier        string
	Name        string
	TaskContext string
	Outcome     string
	ObservedAt  time.Time
	ToolCalls   []string
	Tags        []string
}

// DetectProcedurals scans messages for skill invocations (Tier A) and
// agent invocations (Tier B).
func DetectProcedurals(messages []Message) []ProceduralRecord {
	var records []ProceduralRecord
	now := time.Now()

	skills := detectSkillInvocations(messages)
	for _, s := range skills {
		records = append(records, ProceduralRecord{
			Tier:        "skill",
			Name:        s.name,
			TaskContext: s.context,
			Outcome:     s.outcome,
			ObservedAt:  now,
			ToolCalls:   []string{"skill"},
			Tags:        []string{"procedural", "skill", s.name},
		})
	}

	agents := detectAgentInvocations(messages)
	for _, a := range agents {
		records = append(records, ProceduralRecord{
			Tier:        "agent",
			Name:        a.name,
			TaskContext: a.context,
			Outcome:     a.outcome,
			ObservedAt:  now,
			ToolCalls:   []string{"agent"},
			Tags:        []string{"procedural", "agent", a.name},
		})
	}

	return records
}

type invocation struct {
	name    string
	context string
	outcome string
}

func detectSkillInvocations(messages []Message) []invocation {
	var results []invocation

	for i, m := range messages {
		if m.Role != RoleAssistant {
			continue
		}
		for _, tc := range m.ToolCalls {
			if tc.Name != "skill" {
				continue
			}
			var args struct {
				Skill string `json:"skill"`
				Args  string `json:"args"`
			}
			if err := json.Unmarshal(tc.Arguments, &args); err != nil || args.Skill == "" {
				continue
			}

			outcome := inferToolCallOutcome(messages, i, tc.ID)
			ctx := inferProceduralContext(messages, i)

			results = append(results, invocation{
				name:    args.Skill,
				context: ctx,
				outcome: outcome,
			})
		}
	}

	return results
}

func detectAgentInvocations(messages []Message) []invocation {
	var results []invocation

	for i, m := range messages {
		if m.Role != RoleAssistant {
			continue
		}
		for _, tc := range m.ToolCalls {
			if tc.Name != "agent" {
				continue
			}
			var args struct {
				Type   string `json:"type"`
				Prompt string `json:"prompt"`
			}
			if err := json.Unmarshal(tc.Arguments, &args); err != nil || args.Type == "" {
				continue
			}

			outcome := inferToolCallOutcome(messages, i, tc.ID)
			ctx := args.Prompt
			if len(ctx) > 160 {
				ctx = ctx[:160]
			}
			if ctx == "" {
				ctx = inferProceduralContext(messages, i)
			}

			results = append(results, invocation{
				name:    args.Type,
				context: ctx,
				outcome: outcome,
			})
		}
	}

	return results
}

// inferToolCallOutcome looks at the tool result for a given tool call
// to determine outcome.
func inferToolCallOutcome(messages []Message, afterIndex int, toolCallID string) string {
	for _, m := range messages[afterIndex:] {
		if m.Role != RoleTool {
			continue
		}
		if m.ToolCallID == toolCallID || (toolCallID == "" && m.Name == "skill") {
			content := strings.ToLower(m.Content)
			if strings.Contains(content, "error") || strings.Contains(content, "failed") {
				return "error"
			}
			return "ok"
		}
	}
	return "partial"
}

// inferProceduralContext extracts a brief task context from the
// nearest preceding user message.
func inferProceduralContext(messages []Message, beforeIndex int) string {
	for i := beforeIndex - 1; i >= 0; i-- {
		if messages[i].Role == RoleUser {
			ctx := messages[i].Content
			if len(ctx) > 160 {
				ctx = ctx[:160]
			}
			return ctx
		}
	}
	return ""
}

// FormatProceduralRecord formats a procedural record as markdown
// content.
func FormatProceduralRecord(r ProceduralRecord) string {
	var b strings.Builder
	fmt.Fprintf(&b, "---\nname: %s\n", r.Name)
	fmt.Fprintf(&b, "type: procedural\n")
	fmt.Fprintf(&b, "tier: %s\n", r.Tier)
	fmt.Fprintf(&b, "outcome: %s\n", r.Outcome)
	if len(r.Tags) > 0 {
		fmt.Fprintf(&b, "tags: [%s]\n", strings.Join(r.Tags, ", "))
	}
	fmt.Fprintf(&b, "observed: %s\n", r.ObservedAt.Format(time.RFC3339))
	b.WriteString("---\n\n")
	if r.TaskContext != "" {
		fmt.Fprintf(&b, "## Context\n\n%s\n\n", r.TaskContext)
	}
	if len(r.ToolCalls) > 0 {
		fmt.Fprintf(&b, "## Tool sequence\n\n%s\n", strings.Join(r.ToolCalls, " -> "))
	}
	return b.String()
}
