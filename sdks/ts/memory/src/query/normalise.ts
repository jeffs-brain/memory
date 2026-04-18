// SPDX-License-Identifier: Apache-2.0

/**
 * Normalisation helpers for raw query strings before parsing. Applies
 * Unicode NFC, collapses whitespace, and strips zero-width / BOM
 * characters so the parser works on a single canonical form. Boolean
 * operators (AND/OR/NOT, uppercase only) and double quotes are
 * preserved verbatim so the parser can detect them.
 */

const INVISIBLES = /[\u200B-\u200D\uFEFF]/g
const NBSP = /\u00A0/g
const WHITESPACE_RUN = /\s+/g

/**
 * Normalise raw user input into the canonical form the parser expects.
 * The transformation is:
 *   1. NFC Unicode normalisation
 *   2. Strip zero-width joiners and BOM characters
 *   3. Map non-breaking spaces to regular spaces
 *   4. Collapse whitespace runs and trim
 *
 * Quotes and boolean operator casing are left untouched.
 */
export function normalise(raw: string): string {
  if (raw === '') return ''
  let out = raw.normalize('NFC')
  out = out.replace(INVISIBLES, '')
  out = out.replace(NBSP, ' ')
  out = out.replace(WHITESPACE_RUN, ' ')
  return out.trim()
}

/**
 * Lowercase a token without disturbing any FTS5 control characters.
 * Kept as a separate helper so the parser can normalise case just once
 * per bare token and reuse the lowered form for both stop-word checks
 * and alias lookups.
 */
export function lowerToken(token: string): string {
  return token.toLocaleLowerCase('en')
}
