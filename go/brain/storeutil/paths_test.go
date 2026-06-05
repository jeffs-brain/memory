// SPDX-License-Identifier: Apache-2.0

package storeutil

import (
	"testing"

	"github.com/jeffs-brain/memory/go/brain"
)

func TestRelative_Conversations(t *testing.T) {
	cases := []struct {
		in   brain.Path
		want string
	}{
		{"conversations", "conversations"},
		{"conversations/slack/2026/05/2026-05-30-id-slug.md", "conversations/slack/2026/05/2026-05-30-id-slug.md"},
		{"conversations/_index.md", "conversations/_index.md"},
	}
	for _, tc := range cases {
		got, err := Relative(tc.in)
		if err != nil {
			t.Fatalf("Relative(%q) error: %v", tc.in, err)
		}
		if got != tc.want {
			t.Errorf("Relative(%q) = %q, want %q", tc.in, got, tc.want)
		}
	}
}

func TestLogicalFromRel_Conversations_RoundTrips(t *testing.T) {
	for _, p := range []brain.Path{
		"conversations",
		"conversations/telegram/2026/05/2026-05-30-id-slug.md",
		"conversations/slack/_index.md",
	} {
		rel, err := Relative(p)
		if err != nil {
			t.Fatalf("Relative(%q) error: %v", p, err)
		}
		if back := LogicalFromRel(rel); back != p {
			t.Errorf("round trip %q -> %q -> %q", p, rel, back)
		}
	}
}

func TestResolve_Conversations(t *testing.T) {
	abs, err := Resolve("/brain", "conversations/slack/2026/05/x.md")
	if err != nil {
		t.Fatalf("Resolve conversations path: %v", err)
	}
	if abs != "/brain/conversations/slack/2026/05/x.md" {
		t.Errorf("Resolve = %q, want /brain/conversations/slack/2026/05/x.md", abs)
	}
}
