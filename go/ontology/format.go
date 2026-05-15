// SPDX-License-Identifier: Apache-2.0
package ontology

import "strings"

// FormatNodeTypeLabel converts a dotted node type identifier into a
// human-readable label. For example, "entity.customer" becomes "Customer (Entity)"
// and "process.approval_chain" becomes "Approval Chain (Process)".
func FormatNodeTypeLabel(typ string) string {
	parts := strings.SplitN(typ, ".", 2)
	if len(parts) != 2 {
		return typ
	}
	prefix := parts[0]
	name := parts[1]
	return titleCaseSnake(name) + " (" + titleCaseWord(prefix) + ")"
}

// FormatEdgeTypeLabel converts a snake_case edge type identifier into a
// human-readable label. For example, "requires_approval_from" becomes
// "Requires Approval From".
func FormatEdgeTypeLabel(typ string) string {
	return titleCaseSnake(typ)
}

// titleCaseSnake splits a snake_case string on underscores and title-cases
// each word, joining with spaces.
func titleCaseSnake(s string) string {
	words := strings.Split(s, "_")
	var b strings.Builder
	for i, word := range words {
		if i > 0 {
			b.WriteByte(' ')
		}
		b.WriteString(titleCaseWord(word))
	}
	return b.String()
}

// titleCaseWord capitalises the first character of a single word.
func titleCaseWord(word string) string {
	if len(word) == 0 {
		return word
	}
	return strings.ToUpper(word[:1]) + word[1:]
}
