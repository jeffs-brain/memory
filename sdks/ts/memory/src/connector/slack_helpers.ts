// SPDX-License-Identifier: Apache-2.0

/**
 * Helper functions for the Slack connector: mrkdwn conversion, SSRF
 * validation, response size limiting, and timestamp parsing.
 */

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

// IP ranges that are blocked for SSRF protection. Covers private
// networks (RFC 1918), loopback, link-local, and cloud metadata.
const BLOCKED_IP_PATTERNS: readonly RegExp[] = [
  /^127\./, // IPv4 loopback
  /^10\./, // Class A private
  /^172\.(1[6-9]|2\d|3[01])\./, // Class B private
  /^192\.168\./, // Class C private
  /^169\.254\./, // Link-local
  /^0\./, // Unspecified
  /^::1$/, // IPv6 loopback
  /^fe80:/i, // IPv6 link-local
  /^fc00:/i, // IPv6 ULA
  /^fd/i, // IPv6 ULA
]

const CODE_BLOCK_PLACEHOLDER = '\x00CB'
const INLINE_CODE_PLACEHOLDER = '\x00IC'

// ---------------------------------------------------------------------------
// SSRF protection
// ---------------------------------------------------------------------------

/**
 * Validates that a download URL is safe to fetch by checking the
 * hostname is not a private/internal IP address. This prevents SSRF
 * attacks via crafted download URLs.
 */
export function validateDownloadURL(rawURL: string): void {
  const parsed = new URL(rawURL)

  if (parsed.protocol !== 'https:' && parsed.protocol !== 'http:') {
    throw new Error(`slack: blocked URL scheme "${parsed.protocol}" (only http/https allowed)`)
  }

  const host = parsed.hostname
  if (!host) {
    throw new Error('slack: download URL has no hostname')
  }

  // Check if the hostname itself is a blocked IP address.
  for (const pattern of BLOCKED_IP_PATTERNS) {
    if (pattern.test(host)) {
      throw new Error(`slack: request to private/internal network blocked for ${host}`)
    }
  }
}

// ---------------------------------------------------------------------------
// Response size-limited reader
// ---------------------------------------------------------------------------

/**
 * Reads a response body in chunks up to `maxBytes`, stopping early if
 * the limit is reached. This prevents OOM from unbounded responses.
 * Returns a Buffer of the read data.
 */
export async function readResponseWithLimit(response: Response, maxBytes: number): Promise<Buffer> {
  const reader = response.body?.getReader()
  if (!reader) {
    throw new Error('slack: response body is not readable')
  }

  const chunks: Uint8Array[] = []
  let totalBytes = 0

  while (true) {
    const { done, value } = await reader.read()
    if (done) break

    totalBytes += value.byteLength
    if (totalBytes > maxBytes) {
      reader.cancel()
      throw new Error(`slack: response body exceeds ${maxBytes} byte limit`)
    }
    chunks.push(value)
  }

  return Buffer.concat(chunks)
}

// ---------------------------------------------------------------------------
// Mrkdwn conversion
// ---------------------------------------------------------------------------

/**
 * Converts Slack mrkdwn formatted text to standard Markdown. Handles
 * links, channel mentions, user mentions, bold, italic, strikethrough,
 * and code blocks. Emoji shortcodes (:name:) are left as-is.
 */
export function convertMrkdwn(text: string): string {
  if (!text) return ''

  // Protect code blocks and inline code from formatting conversions.
  const { text: withoutBlocks, blocks } = extractCodeBlocks(text)
  const { text: withoutInline, codes } = extractInlineCode(withoutBlocks)

  let result = withoutInline

  // Links (labelled and bare).
  result = result.replace(/<(https?:\/\/[^|>]+)\|([^>]+)>/g, '[$2]($1)')
  result = result.replace(/<(https?:\/\/[^>]+)>/g, '$1')

  // Channel mentions.
  result = result.replace(/<#[A-Z0-9]+\|([^>]+)>/g, '#$1')

  // User mentions.
  result = result.replace(/<@([A-Z0-9]+)>/g, '@$1')

  // Bold: *text* -> **text**
  result = result.replace(/(^|\s)\*([^\s*][^*]*[^\s*]|[^\s*])\*($|\s)/g, '$1**$2**$3')

  // Italic: _text_ -> *text*
  result = result.replace(/(^|\s)_([^\s_][^_]*[^\s_]|[^\s_])_($|\s)/g, '$1*$2*$3')

  // Strikethrough: ~text~ -> ~~text~~
  result = result.replace(/(^|\s)~([^\s~][^~]*[^\s~]|[^\s~])~($|\s)/g, '$1~~$2~~$3')

  // Restore inline code and code blocks.
  result = restoreInlineCode(result, codes)
  result = restoreCodeBlocks(result, blocks)

  return result
}

// ---------------------------------------------------------------------------
// Code extraction helpers
// ---------------------------------------------------------------------------

function extractCodeBlocks(text: string): { text: string; blocks: string[] } {
  const blocks: string[] = []
  let result = text
  let start = result.indexOf('```')

  while (start !== -1) {
    const end = result.indexOf('```', start + 3)
    if (end === -1) break
    const block = result.slice(start, end + 3)
    blocks.push(block)
    result = result.slice(0, start) + CODE_BLOCK_PLACEHOLDER + result.slice(end + 3)
    start = result.indexOf('```')
  }

  return { text: result, blocks }
}

function restoreCodeBlocks(text: string, blocks: string[]): string {
  let result = text
  for (const block of blocks) {
    const inner = block.slice(3, -3).trim()
    const replacement = `\`\`\`\n${inner}\n\`\`\``
    result = result.replace(CODE_BLOCK_PLACEHOLDER, replacement)
  }
  return result
}

function extractInlineCode(text: string): { text: string; codes: string[] } {
  const codes: string[] = []
  let result = text
  let start = result.indexOf('`')

  while (start !== -1) {
    const end = result.indexOf('`', start + 1)
    if (end === -1) break
    const code = result.slice(start, end + 1)
    codes.push(code)
    result = result.slice(0, start) + INLINE_CODE_PLACEHOLDER + result.slice(end + 1)
    start = result.indexOf('`')
  }

  return { text: result, codes }
}

function restoreInlineCode(text: string, codes: string[]): string {
  let result = text
  for (const code of codes) {
    result = result.replace(INLINE_CODE_PLACEHOLDER, code)
  }
  return result
}

// ---------------------------------------------------------------------------
// Timestamp helpers
// ---------------------------------------------------------------------------

/**
 * Parse a Slack epoch timestamp string (e.g. "1234567890.123456") to a
 * Date object.
 */
export function parseSlackTimestamp(ts: string): Date {
  const parts = ts.split('.')
  const secs = Number.parseInt(parts[0] ?? '0', 10)
  const micros = parts.length > 1 ? Number.parseInt(parts[1] ?? '0', 10) : 0
  return new Date(secs * 1000 + micros / 1000)
}

export function formatDate(date: Date): string {
  const year = date.getUTCFullYear()
  const month = String(date.getUTCMonth() + 1).padStart(2, '0')
  const day = String(date.getUTCDate()).padStart(2, '0')
  const hours = String(date.getUTCHours()).padStart(2, '0')
  const minutes = String(date.getUTCMinutes()).padStart(2, '0')
  return `${year}-${month}-${day} ${hours}:${minutes}`
}
