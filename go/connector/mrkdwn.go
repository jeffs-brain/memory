// SPDX-License-Identifier: Apache-2.0

package connector

import (
	"regexp"
	"strings"
)

// mrkdwn conversion patterns. Each pattern transforms one Slack mrkdwn
// construct into the equivalent standard Markdown syntax.
//
// Ordering matters: link patterns must be applied before bold/italic to
// avoid mangling URLs that contain underscores or asterisks.

var (
	// <https://example.com|label> -> [label](https://example.com)
	mrkdwnLabelledLink = regexp.MustCompile(`<(https?://[^|>]+)\|([^>]+)>`)

	// <https://example.com> -> https://example.com
	mrkdwnBareLink = regexp.MustCompile(`<(https?://[^>]+)>`)

	// <#C123ABC|channel-name> -> #channel-name
	mrkdwnChannel = regexp.MustCompile(`<#[A-Z0-9]+\|([^>]+)>`)

	// <@U123ABC> -> @U123ABC (user mention -- resolved separately)
	mrkdwnUserMention = regexp.MustCompile(`<@([A-Z0-9]+)>`)

	// *bold* -> **bold** (Slack uses single asterisks for bold)
	// Must not match inside code spans or URLs.
	mrkdwnBold = regexp.MustCompile(`(?:^|(?P<pre>\s))\*(?P<text>[^\s*][^*]*[^\s*]|[^\s*])\*(?:$|(?P<post>\s))`)

	// _italic_ -> *italic* (Slack uses underscores for italic)
	mrkdwnItalic = regexp.MustCompile(`(?:^|(?P<pre>\s))_(?P<text>[^\s_][^_]*[^\s_]|[^\s_])_(?:$|(?P<post>\s))`)

	// ~strike~ -> ~~strike~~
	mrkdwnStrike = regexp.MustCompile(`(?:^|(?P<pre>\s))~(?P<text>[^\s~][^~]*[^\s~]|[^\s~])~(?:$|(?P<post>\s))`)
)

// ConvertMrkdwn converts Slack mrkdwn formatted text to standard
// Markdown. It handles links, channel mentions, user mentions, bold,
// italic, strikethrough, and code blocks. Emoji shortcodes (:name:)
// are left as-is.
func ConvertMrkdwn(text string) string {
	if text == "" {
		return ""
	}

	// Protect code blocks from formatting conversions by extracting
	// them first and reinserting after all other transformations.
	result, codeBlocks := extractCodeBlocks(text)
	result, inlineCode := extractInlineCode(result)

	// Links (labelled and bare).
	result = mrkdwnLabelledLink.ReplaceAllString(result, "[$2]($1)")
	result = mrkdwnBareLink.ReplaceAllString(result, "$1")

	// Channel mentions.
	result = mrkdwnChannel.ReplaceAllString(result, "#$1")

	// User mentions -- leave as @USER_ID; the caller resolves names.
	result = mrkdwnUserMention.ReplaceAllString(result, "@$1")

	// Bold: *text* -> **text**
	result = mrkdwnBold.ReplaceAllStringFunc(result, func(m string) string {
		sub := mrkdwnBold.FindStringSubmatch(m)
		pre := sub[mrkdwnBold.SubexpIndex("pre")]
		text := sub[mrkdwnBold.SubexpIndex("text")]
		post := sub[mrkdwnBold.SubexpIndex("post")]
		return pre + "**" + text + "**" + post
	})

	// Italic: _text_ -> *text*
	result = mrkdwnItalic.ReplaceAllStringFunc(result, func(m string) string {
		sub := mrkdwnItalic.FindStringSubmatch(m)
		pre := sub[mrkdwnItalic.SubexpIndex("pre")]
		text := sub[mrkdwnItalic.SubexpIndex("text")]
		post := sub[mrkdwnItalic.SubexpIndex("post")]
		return pre + "*" + text + "*" + post
	})

	// Strikethrough: ~text~ -> ~~text~~
	result = mrkdwnStrike.ReplaceAllStringFunc(result, func(m string) string {
		sub := mrkdwnStrike.FindStringSubmatch(m)
		pre := sub[mrkdwnStrike.SubexpIndex("pre")]
		text := sub[mrkdwnStrike.SubexpIndex("text")]
		post := sub[mrkdwnStrike.SubexpIndex("post")]
		return pre + "~~" + text + "~~" + post
	})

	// Restore inline code and code blocks.
	result = restoreInlineCode(result, inlineCode)
	result = restoreCodeBlocks(result, codeBlocks)

	return result
}

// ---------------------------------------------------------------------------
// Code block extraction / restoration
// ---------------------------------------------------------------------------

const codeBlockPlaceholder = "\x00CB"
const inlineCodePlaceholder = "\x00IC"

func extractCodeBlocks(text string) (string, []string) {
	var blocks []string
	result := text
	for {
		start := strings.Index(result, "```")
		if start == -1 {
			break
		}
		end := strings.Index(result[start+3:], "```")
		if end == -1 {
			break
		}
		end += start + 3 + 3
		block := result[start:end]
		blocks = append(blocks, block)
		result = result[:start] + codeBlockPlaceholder + result[end:]
	}
	return result, blocks
}

func restoreCodeBlocks(text string, blocks []string) string {
	result := text
	for _, block := range blocks {
		// Convert Slack code blocks to fenced blocks with newlines.
		inner := block[3 : len(block)-3]
		replacement := "```\n" + strings.TrimSpace(inner) + "\n```"
		result = strings.Replace(result, codeBlockPlaceholder, replacement, 1)
	}
	return result
}

func extractInlineCode(text string) (string, []string) {
	var codes []string
	result := text
	for {
		start := strings.Index(result, "`")
		if start == -1 {
			break
		}
		end := strings.Index(result[start+1:], "`")
		if end == -1 {
			break
		}
		end += start + 1 + 1
		code := result[start:end]
		codes = append(codes, code)
		result = result[:start] + inlineCodePlaceholder + result[end:]
	}
	return result, codes
}

func restoreInlineCode(text string, codes []string) string {
	result := text
	for _, code := range codes {
		result = strings.Replace(result, inlineCodePlaceholder, code, 1)
	}
	return result
}
