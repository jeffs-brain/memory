// SPDX-License-Identifier: Apache-2.0

package memory

import (
	"errors"
	"strings"
	"testing"
)

func priorLines(block string) []string {
	var out []string
	for _, line := range strings.Split(block, "\n") {
		if strings.HasPrefix(line, "- ") {
			out = append(out, line)
		}
	}
	return out
}

func TestBuildCodecPriorsBlock_Positive(t *testing.T) {
	block, err := buildCodecPriorsBlock(&CodecPriors{
		Entities:    []string{"Person", "Organisation", "Product"},
		Relations:   []string{"worksAt", "owns", "dependsOn"},
		DomainTerms: []string{"sprint", "deployment", "incident"},
	})
	if err != nil {
		t.Fatalf("buildCodecPriorsBlock: %v", err)
	}
	for _, want := range []string{
		"## Project codec priors",
		"### Entities",
		"- Person",
		"### Relations",
		"- worksAt",
		"### Domain terms",
		"- sprint",
	} {
		if !strings.Contains(block, want) {
			t.Fatalf("block missing %q\n%s", want, block)
		}
	}
}

func TestApplyCodecPriors_AppendsBlock(t *testing.T) {
	composed, err := applyCodecPriors(extractionPrompt, &CodecPriors{Entities: []string{"Person"}})
	if err != nil {
		t.Fatalf("applyCodecPriors: %v", err)
	}
	if !strings.HasPrefix(composed, extractionPrompt) {
		t.Fatalf("composed prompt does not start with the base prompt")
	}
	if !strings.Contains(composed, "- Person") {
		t.Fatalf("composed prompt missing prior")
	}
	if len(composed) <= len(extractionPrompt) {
		t.Fatalf("composed prompt did not grow")
	}
}

func TestBuildCodecPriorsBlock_BackwardCompatible(t *testing.T) {
	cases := map[string]*CodecPriors{
		"nil":         nil,
		"empty":       {},
		"empty lists": {Entities: []string{}, Relations: []string{}, DomainTerms: []string{}},
		"blank items": {Entities: []string{"   ", ""}},
	}
	for name, priors := range cases {
		t.Run(name, func(t *testing.T) {
			block, err := buildCodecPriorsBlock(priors)
			if err != nil {
				t.Fatalf("buildCodecPriorsBlock: %v", err)
			}
			if block != "" {
				t.Fatalf("expected empty block, got %q", block)
			}
			composed, err := applyCodecPriors(extractionPrompt, priors)
			if err != nil {
				t.Fatalf("applyCodecPriors: %v", err)
			}
			if composed != extractionPrompt {
				t.Fatalf("expected byte-identical base prompt for %s", name)
			}
		})
	}
}

func TestBuildCodecPriorsBlock_MalformedLineBreakIsTypedError(t *testing.T) {
	_, err := buildCodecPriorsBlock(&CodecPriors{Entities: []string{"Person\nInjected heading"}})
	if err == nil {
		t.Fatalf("expected error for line break in prior")
	}
	if !errors.Is(err, ErrInvalidCodecPriors) {
		t.Fatalf("error = %v, want ErrInvalidCodecPriors", err)
	}
}

func TestBuildCodecPriorsBlock_DeduplicatesCaseInsensitively(t *testing.T) {
	block, err := buildCodecPriorsBlock(&CodecPriors{Entities: []string{"Person", "person", "PERSON", "Org"}})
	if err != nil {
		t.Fatalf("buildCodecPriorsBlock: %v", err)
	}
	got := priorLines(block)
	want := []string{"- Person", "- Org"}
	if strings.Join(got, "|") != strings.Join(want, "|") {
		t.Fatalf("dedup lines = %v, want %v", got, want)
	}
}

func TestBuildCodecPriorsBlock_CapsItemsPerList(t *testing.T) {
	items := make([]string, 0, codecPriorsMaxItemsPerList+50)
	for i := 0; i < codecPriorsMaxItemsPerList+50; i++ {
		items = append(items, "Entity"+strings.Repeat("x", 0)+itoa(i))
	}
	block, err := buildCodecPriorsBlock(&CodecPriors{Entities: items})
	if err != nil {
		t.Fatalf("buildCodecPriorsBlock: %v", err)
	}
	if got := len(priorLines(block)); got != codecPriorsMaxItemsPerList {
		t.Fatalf("rendered %d items, want %d", got, codecPriorsMaxItemsPerList)
	}
}

func TestBuildCodecPriorsBlock_TruncatesItem(t *testing.T) {
	long := strings.Repeat("A", codecPriorsMaxItemLength+40)
	block, err := buildCodecPriorsBlock(&CodecPriors{Entities: []string{long}})
	if err != nil {
		t.Fatalf("buildCodecPriorsBlock: %v", err)
	}
	want := "- " + strings.Repeat("A", codecPriorsMaxItemLength)
	got := priorLines(block)
	if len(got) != 1 || got[0] != want {
		t.Fatalf("truncated line = %v, want %q", got, want)
	}
}

func TestBuildCodecPriorsBlock_BoundsTotalGrowth(t *testing.T) {
	huge := make([]string, 0, 500)
	for i := 0; i < 500; i++ {
		huge = append(huge, strings.Repeat("X", 80)+itoa(i))
	}
	block, err := buildCodecPriorsBlock(&CodecPriors{Entities: huge, Relations: huge, DomainTerms: huge})
	if err != nil {
		t.Fatalf("buildCodecPriorsBlock: %v", err)
	}
	if len(block) >= 6000 {
		t.Fatalf("block length %d not bounded", len(block))
	}
}

// GO ↔ TS PARITY: this golden string is asserted byte-for-byte in the TS
// test "buildCodecPriorsBlock — Go/TS parity"
// (sdks/ts/memory/src/memory/codec-priors.test.ts). Any change here MUST be
// mirrored there, and vice-versa.
func TestBuildCodecPriorsBlock_GoldenParity(t *testing.T) {
	priors := &CodecPriors{
		Entities:    []string{"Person", "person", "Organisation"},
		Relations:   []string{"worksAt"},
		DomainTerms: []string{"sprint", "deployment"},
	}
	golden := strings.Join([]string{
		"",
		"## Project codec priors",
		"The following are the project’s canonical entities, relations, and domain terms. Treat them as SOFT hints: when a saved fact references one of these, prefer the canonical name below. Do NOT invent facts to match them, and do NOT discard durable knowledge that falls outside this list.",
		"",
		"### Entities",
		"- Person",
		"- Organisation",
		"### Relations",
		"- worksAt",
		"### Domain terms",
		"- sprint",
		"- deployment",
	}, "\n")

	block, err := buildCodecPriorsBlock(priors)
	if err != nil {
		t.Fatalf("buildCodecPriorsBlock: %v", err)
	}
	if block != golden {
		t.Fatalf("golden parity mismatch:\n got: %q\nwant: %q", block, golden)
	}
}

// itoa avoids importing strconv solely for the test loop counters.
func itoa(i int) string {
	if i == 0 {
		return "0"
	}
	var digits []byte
	for i > 0 {
		digits = append([]byte{byte('0' + i%10)}, digits...)
		i /= 10
	}
	return string(digits)
}
