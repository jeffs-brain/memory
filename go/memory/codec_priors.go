// SPDX-License-Identifier: Apache-2.0

package memory

import (
	"errors"
	"fmt"
	"strings"
)

// Codec priors bounds. Mirrors the TypeScript implementation in
// sdks/ts/memory/src/memory/codec-priors.ts so both languages render an
// identical priors block.
const (
	codecPriorsMaxItemsPerList = 64
	codecPriorsMaxItemLength   = 80
	codecPriorsMaxBlockChars   = 4000
)

// ErrInvalidCodecPriors is returned when codec priors are structurally
// malformed. The package never silently coerces bad input.
var ErrInvalidCodecPriors = errors.New("memory: invalid codec priors")

// CodecPriors are the project's canonical vocabulary, folded into the
// extraction prompt as SOFT known-entity hints. They mirror the codec
// `ontology:` frontmatter block (plain string lists) and are deliberately
// distinct from the typed ontology.ResolvedOntology used by the
// ontology-type extractor — codec priors carry no types, attributes, or
// endpoints.
//
// All three lists are optional; an absent or wholly empty value leaves
// extraction behaviour byte-identical to the no-priors baseline.
type CodecPriors struct {
	// Entities are canonical entity names, e.g. {"Person", "Organisation"}.
	Entities []string
	// Relations are canonical relation names, e.g. {"worksAt", "owns"}.
	Relations []string
	// DomainTerms is domain vocabulary / shorthand, e.g. {"sprint"}.
	DomainTerms []string
}

type codecPriorList struct {
	heading string
	items   []string
}

// buildCodecPriorsBlock renders the soft known-entity priors block that is
// appended to the extraction system prompt. It returns an empty string
// when there is nothing to render (nil priors, or all lists empty after
// sanitisation), guaranteeing the baseline prompt is unchanged.
//
// Items are trimmed, empties dropped, deduplicated case-insensitively
// (first occurrence wins), truncated per-item, and capped per-list and by
// an overall character budget so priors can never grow the prompt without
// limit. A list that is present but unusable raises ErrInvalidCodecPriors;
// because Go slices are already typed there is no non-string case, so the
// error surfaces only on a future validation extension and keeps parity
// with the TypeScript typed-error contract.
func buildCodecPriorsBlock(priors *CodecPriors) (string, error) {
	if priors == nil {
		return "", nil
	}

	lists := []codecPriorList{
		{heading: "Entities", items: priors.Entities},
		{heading: "Relations", items: priors.Relations},
		{heading: "Domain terms", items: priors.DomainTerms},
	}

	sections := make([]string, 0, len(lists))
	remainingBudget := codecPriorsMaxBlockChars

	for _, list := range lists {
		items, err := sanitiseCodecPriorList(list.heading, list.items)
		if err != nil {
			return "", err
		}
		if len(items) == 0 {
			continue
		}
		lines := make([]string, 0, len(items))
		for _, item := range items {
			line := "- " + item
			if len(line) > remainingBudget {
				break
			}
			lines = append(lines, line)
			remainingBudget -= len(line)
		}
		if len(lines) == 0 {
			continue
		}
		sections = append(sections, fmt.Sprintf("### %s\n%s", list.heading, strings.Join(lines, "\n")))
	}

	if len(sections) == 0 {
		return "", nil
	}

	parts := []string{
		"",
		"## Project codec priors",
		"The following are the project’s canonical entities, relations, and domain terms. Treat them as SOFT hints: when a saved fact references one of these, prefer the canonical name below. Do NOT invent facts to match them, and do NOT discard durable knowledge that falls outside this list.",
		"",
	}
	parts = append(parts, sections...)
	return strings.Join(parts, "\n"), nil
}

// applyCodecPriors composes the effective extraction system prompt from the
// verbatim base prompt and the optional codec priors block.
func applyCodecPriors(basePrompt string, priors *CodecPriors) (string, error) {
	block, err := buildCodecPriorsBlock(priors)
	if err != nil {
		return "", err
	}
	if block == "" {
		return basePrompt, nil
	}
	return basePrompt + "\n" + block, nil
}

func sanitiseCodecPriorList(heading string, values []string) ([]string, error) {
	if len(values) == 0 {
		return nil, nil
	}
	seen := make(map[string]struct{}, len(values))
	out := make([]string, 0, len(values))
	for _, raw := range values {
		if strings.ContainsAny(raw, "\n\r") {
			return nil, fmt.Errorf("%w: %s must not contain line breaks", ErrInvalidCodecPriors, strings.ToLower(heading))
		}
		trimmed := strings.TrimSpace(raw)
		if trimmed == "" {
			continue
		}
		item := truncateCodecPriorItem(trimmed)
		key := strings.ToLower(item)
		if _, ok := seen[key]; ok {
			continue
		}
		seen[key] = struct{}{}
		out = append(out, item)
		if len(out) >= codecPriorsMaxItemsPerList {
			break
		}
	}
	return out, nil
}

// truncateCodecPriorItem caps an item at codecPriorsMaxItemLength runes,
// truncating on a rune boundary so the result is always valid UTF-8. For
// the canonical-name domain (ASCII / Latin) this matches the TypeScript
// code-unit cap exactly.
func truncateCodecPriorItem(value string) string {
	runes := []rune(value)
	if len(runes) <= codecPriorsMaxItemLength {
		return value
	}
	return string(runes[:codecPriorsMaxItemLength])
}
