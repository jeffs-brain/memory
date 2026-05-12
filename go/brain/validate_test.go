// SPDX-License-Identifier: Apache-2.0

package brain

import (
	"strings"
	"testing"
)

func TestValidateBrainID(t *testing.T) {
	cases := []struct {
		name    string
		id      string
		wantErr bool
	}{
		// Positive cases.
		{name: "simple lowercase", id: "my-brain", wantErr: false},
		{name: "mixed case with dot", id: "Brain.v2", wantErr: false},
		{name: "minimum length", id: "a", wantErr: false},
		{name: "max length 128", id: strings.Repeat("a", 128), wantErr: false},
		{name: "multiple dots no traversal", id: "a.b.c.d", wantErr: false},
		{name: "all allowed special chars", id: "brain_v2-final.1", wantErr: false},
		{name: "numeric start", id: "1brain", wantErr: false},

		// Negative cases.
		{name: "empty", id: "", wantErr: true},
		{name: "path traversal passwd", id: "../../etc/passwd", wantErr: true},
		{name: "path traversal prefix", id: "../attack", wantErr: true},
		{name: "path separator slash", id: "brain/sub", wantErr: true},
		{name: "starts with dot", id: ".hidden", wantErr: true},
		{name: "starts with dash", id: "-dash", wantErr: true},
		{name: "exceeds max length", id: strings.Repeat("a", 129), wantErr: true},
		{name: "null byte", id: "brain\x00null", wantErr: true},
		{name: "contains space", id: "brain name", wantErr: true},
		{name: "backslash", id: "brain\\sub", wantErr: true},
		{name: "unicode", id: "brainé", wantErr: true},
		{name: "double dot traversal embedded", id: "foo..bar", wantErr: true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := ValidateBrainID(tc.id)
			if tc.wantErr && err == nil {
				t.Fatalf("ValidateBrainID(%q) = nil, want error", tc.id)
			}
			if !tc.wantErr && err != nil {
				t.Fatalf("ValidateBrainID(%q) = %v, want nil", tc.id, err)
			}
		})
	}
}
