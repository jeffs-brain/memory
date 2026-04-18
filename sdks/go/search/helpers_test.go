// SPDX-License-Identifier: Apache-2.0

package search

import "testing"

// TestTrailingSegment guards the small path helper used by the
// discovery layer when carving the trailing slug out of a project
// memory entry.
func TestTrailingSegment(t *testing.T) {
	cases := []struct {
		in, want string
	}{
		{"", ""},
		{"single", "single"},
		{"a/b", "b"},
		{"a/b/c.md", "c.md"},
		{"/leading/slash", "slash"},
	}
	for _, c := range cases {
		got := trailingSegment(c.in)
		if got != c.want {
			t.Errorf("trailingSegment(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}
