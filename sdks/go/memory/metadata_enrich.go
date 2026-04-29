// SPDX-License-Identifier: Apache-2.0

package memory

import (
	"regexp"
	"strings"
)

var (
	stateSignalRe       = regexp.MustCompile(`(?i)\b(?:current(?:ly)?|now|still|no longer|changed|switched|settled on|replaced|own(?:s)?|have|has|use|attend|go to|take|learn(?:ing)?|play(?:ing)?|prefer|favo(?:u)?rite)\b`)
	stateOwnedRe        = regexp.MustCompile(`(?i)\b(?:own(?:s)?|have|has)\b`)
	stateLocationRe     = regexp.MustCompile(`(?i)\b(?:attend|go to|take|venue|studio|centre|center|class)\b`)
	stateActivityRe     = regexp.MustCompile(`(?i)\b(?:learn(?:ing)?|play(?:ing)?|practis(?:e|ing)|practic(?:e|ing))\b`)
	statePreferenceRe   = regexp.MustCompile(`(?i)\b(?:prefer|favo(?:u)?rite|settled on)\b`)
	metadataTokenRe     = regexp.MustCompile(`[A-Za-z0-9][A-Za-z0-9'-]*`)
	metadataSentenceEnd = regexp.MustCompile(`[\n.!?]+`)
)

var metadataStopTokens = map[string]bool{
	"about": true, "after": true, "also": true, "and": true, "assistant": true,
	"been": true, "before": true, "content": true, "currently": true, "date": true,
	"from": true, "have": true, "kind": true, "message": true, "mixed": true,
	"now": true, "observed": true, "ordinal": true, "section": true, "session": true,
	"source": true, "still": true, "that": true, "the": true, "this": true,
	"user": true, "with": true,
}

func enrichDerivedMemoryMetadata(memories []ExtractedMemory) []ExtractedMemory {
	if len(memories) == 0 {
		return memories
	}
	out := make([]ExtractedMemory, len(memories))
	copy(out, memories)
	for i := range out {
		enrichOneMemoryMetadata(&out[i])
	}
	return out
}

func enrichOneMemoryMetadata(memory *ExtractedMemory) {
	if memory == nil {
		return
	}
	if memory.ValidFrom == "" {
		memory.ValidFrom = firstNonEmptyMemoryValue(memory.EventDate, memory.ObservedOn, memory.SessionDate, memory.ModifiedOverride)
	}
	if memory.ClaimKey == "" {
		memory.ClaimKey = deriveMemoryClaimKey(*memory)
	}
	if memory.ClaimStatus == "" && memory.ClaimKey != "" {
		memory.ClaimStatus = "asserted"
	}
	if memory.StateKey == "" {
		deriveStateMetadata(memory)
	}
	if memory.StateKey != "" && memory.ClaimKey == "" {
		memory.ClaimKey = "claim." + string(NormaliseClaimKey(memory.StateKey))
	}
}

func deriveMemoryClaimKey(memory ExtractedMemory) string {
	prefix := "claim"
	if memory.ArtefactType != "" {
		prefix = "artefact." + string(NormaliseClaimKey(memory.ArtefactType))
	}
	if memory.Type != "" {
		prefix = prefix + "." + string(NormaliseClaimKey(memory.Type))
	}
	tokens := metadataKeyTokens(memory.Content, 6)
	if len(tokens) == 0 {
		tokens = metadataKeyTokens(memory.Name+" "+memory.Description, 6)
	}
	if len(tokens) == 0 {
		return ""
	}
	return string(NormaliseClaimKey(prefix + "." + strings.Join(tokens, ".")))
}

func deriveStateMetadata(memory *ExtractedMemory) {
	if memory == nil || !stateSignalRe.MatchString(memory.Content) {
		return
	}
	role := strings.ToLower(strings.TrimSpace(memory.SourceRole))
	if role == "assistant" && !strings.Contains(strings.ToLower(memory.Content), "user") {
		return
	}

	kind := "current_state"
	switch {
	case stateOwnedRe.MatchString(memory.Content):
		kind = "owned_item_set"
	case stateLocationRe.MatchString(memory.Content):
		kind = "location"
	case stateActivityRe.MatchString(memory.Content):
		kind = "current_activity"
	case statePreferenceRe.MatchString(memory.Content):
		kind = "preference"
	}

	subjectTokens := metadataKeyTokens(firstMetadataSentence(memory.Content), 5)
	if len(subjectTokens) == 0 {
		return
	}
	memory.StateKind = firstNonEmptyMemoryValue(memory.StateKind, kind)
	memory.StateSubject = firstNonEmptyMemoryValue(memory.StateSubject, strings.Join(subjectTokens, " "))
	memory.StateValue = firstNonEmptyMemoryValue(memory.StateValue, truncateOneLine(memory.Content, 220))
	memory.StateKey = string(NormaliseStateKey("state." + kind + "." + strings.Join(subjectTokens, ".")))
}

func firstMetadataSentence(content string) string {
	content = strings.TrimSpace(content)
	if content == "" {
		return ""
	}
	parts := metadataSentenceEnd.Split(content, 2)
	if len(parts) == 0 {
		return content
	}
	return strings.TrimSpace(parts[0])
}

func metadataKeyTokens(content string, limit int) []string {
	out := make([]string, 0, limit)
	for _, token := range metadataTokenRe.FindAllString(strings.ToLower(content), -1) {
		token = strings.Trim(token, "'-")
		if len(token) < 3 || metadataStopTokens[token] {
			continue
		}
		out = append(out, token)
		if limit > 0 && len(out) >= limit {
			break
		}
	}
	return out
}

func firstNonEmptyMemoryValue(values ...string) string {
	for _, value := range values {
		if trimmed := strings.TrimSpace(value); trimmed != "" {
			return trimmed
		}
	}
	return ""
}
