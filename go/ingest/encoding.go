// SPDX-License-Identifier: Apache-2.0

package ingest

import (
	"fmt"
	"strings"
	"unicode/utf8"
)

// detectEncoding inspects the leading bytes of raw for a BOM marker.
// Returns the decoded string, the encoding name, and any error.
func detectEncoding(raw []byte) (string, string, error) {
	// UTF-8 BOM (EF BB BF).
	if len(raw) >= 3 && raw[0] == 0xEF && raw[1] == 0xBB && raw[2] == 0xBF {
		stripped := raw[3:]
		if !utf8.Valid(stripped) {
			return "", "", fmt.Errorf("structured: invalid UTF-8 after BOM")
		}
		return string(stripped), "utf-8-bom", nil
	}

	// UTF-16 LE BOM (FF FE).
	if len(raw) >= 2 && raw[0] == 0xFF && raw[1] == 0xFE {
		decoded, err := decodeUTF16LE(raw[2:])
		if err != nil {
			return "", "", fmt.Errorf("structured: UTF-16 LE decode: %w", err)
		}
		return decoded, "utf-16-le", nil
	}

	// UTF-16 BE BOM (FE FF).
	if len(raw) >= 2 && raw[0] == 0xFE && raw[1] == 0xFF {
		decoded, err := decodeUTF16BE(raw[2:])
		if err != nil {
			return "", "", fmt.Errorf("structured: UTF-16 BE decode: %w", err)
		}
		return decoded, "utf-16-be", nil
	}

	// Try UTF-8.
	if utf8.Valid(raw) {
		return string(raw), "utf-8", nil
	}

	// Fallback: Latin-1 (ISO-8859-1) — every byte is valid.
	var b strings.Builder
	b.Grow(len(raw))
	for _, by := range raw {
		b.WriteRune(rune(by))
	}
	return b.String(), "latin-1", nil
}

// decodeUTF16LE decodes UTF-16 little-endian bytes to a Go string.
func decodeUTF16LE(data []byte) (string, error) {
	if len(data)%2 != 0 {
		return "", fmt.Errorf("odd byte count for UTF-16")
	}
	var b strings.Builder
	b.Grow(len(data) / 2)
	for i := 0; i+1 < len(data); i += 2 {
		cp := rune(data[i]) | rune(data[i+1])<<8
		b.WriteRune(cp)
	}
	return b.String(), nil
}

// decodeUTF16BE decodes UTF-16 big-endian bytes to a Go string.
func decodeUTF16BE(data []byte) (string, error) {
	if len(data)%2 != 0 {
		return "", fmt.Errorf("odd byte count for UTF-16")
	}
	var b strings.Builder
	b.Grow(len(data) / 2)
	for i := 0; i+1 < len(data); i += 2 {
		cp := rune(data[i])<<8 | rune(data[i+1])
		b.WriteRune(cp)
	}
	return b.String(), nil
}
