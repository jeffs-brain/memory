// SPDX-License-Identifier: Apache-2.0

package connector

import (
	"testing"
)

func TestConvertMrkdwn(t *testing.T) {
	tests := []struct {
		name  string
		input string
		want  string
	}{
		{
			name:  "empty string",
			input: "",
			want:  "",
		},
		{
			name:  "plain text unchanged",
			input: "Hello world",
			want:  "Hello world",
		},
		{
			name:  "bold conversion",
			input: "this is *bold text* here",
			want:  "this is **bold text** here",
		},
		{
			name:  "italic conversion",
			input: "this is _italic text_ here",
			want:  "this is *italic text* here",
		},
		{
			name:  "strikethrough conversion",
			input: "this is ~struck text~ here",
			want:  "this is ~~struck text~~ here",
		},
		{
			name:  "labelled link",
			input: "<https://example.com|Example Site>",
			want:  "[Example Site](https://example.com)",
		},
		{
			name:  "bare link",
			input: "<https://example.com>",
			want:  "https://example.com",
		},
		{
			name:  "channel mention",
			input: "Check <#C123ABC|general> for updates",
			want:  "Check #general for updates",
		},
		{
			name:  "user mention",
			input: "Hey <@U123ABC> check this",
			want:  "Hey @U123ABC check this",
		},
		{
			name:  "inline code preserved",
			input: "Run `go test` to verify",
			want:  "Run `go test` to verify",
		},
		{
			name:  "code block preserved",
			input: "Here is code:\n```func main() {}```",
			want:  "Here is code:\n```\nfunc main() {}\n```",
		},
		{
			name:  "emoji shortcodes preserved",
			input: ":wave: Hello :smile:",
			want:  ":wave: Hello :smile:",
		},
		{
			name:  "multiple transformations",
			input: "Hey <@U001>, check <https://example.com|this link> and _remember_ to *review*",
			want:  "Hey @U001, check [this link](https://example.com) and *remember* to **review**",
		},
		{
			name:  "blockquote preserved",
			input: "> This is a quote",
			want:  "> This is a quote",
		},
		{
			name:  "bold at start of line",
			input: "*bold* text",
			want:  "**bold** text",
		},
		{
			name:  "italic at start of line",
			input: "_italic_ text",
			want:  "*italic* text",
		},
		{
			name:  "bold at end of line",
			input: "text *bold*",
			want:  "text **bold**",
		},
		{
			name:  "bold does not match inside words",
			input: "not*bold*here",
			want:  "not*bold*here",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := ConvertMrkdwn(tt.input)
			if got != tt.want {
				t.Errorf("ConvertMrkdwn(%q)\n  got:  %q\n  want: %q", tt.input, got, tt.want)
			}
		})
	}
}
