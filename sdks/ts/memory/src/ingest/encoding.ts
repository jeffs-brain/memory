// SPDX-License-Identifier: Apache-2.0

/**
 * Character encoding detection for structured data extractors.
 * Detects UTF-8 (with/without BOM), UTF-16 LE/BE, and Latin-1
 * (ISO-8859-1) fallback.
 */

export const detectEncoding = (raw: Buffer): { text: string; encoding: string } => {
  // UTF-8 BOM (EF BB BF).
  if (raw.length >= 3 && raw[0] === 0xef && raw[1] === 0xbb && raw[2] === 0xbf) {
    return { text: raw.subarray(3).toString('utf8'), encoding: 'utf-8-bom' }
  }

  // UTF-16 LE BOM (FF FE).
  if (raw.length >= 2 && raw[0] === 0xff && raw[1] === 0xfe) {
    return { text: raw.subarray(2).toString('utf16le'), encoding: 'utf-16-le' }
  }

  // UTF-16 BE BOM (FE FF).
  if (raw.length >= 2 && raw[0] === 0xfe && raw[1] === 0xff) {
    return { text: decodeUTF16BE(raw.subarray(2)), encoding: 'utf-16-be' }
  }

  // Try UTF-8. If all bytes are valid, use it.
  const utf8Text = raw.toString('utf8')
  const roundTrip = Buffer.from(utf8Text, 'utf8')
  if (roundTrip.length === raw.length && raw.equals(roundTrip)) {
    return { text: utf8Text, encoding: 'utf-8' }
  }

  // Fallback: Latin-1 (ISO-8859-1).
  return { text: raw.toString('latin1'), encoding: 'latin-1' }
}

const decodeUTF16BE = (data: Buffer): string => {
  const swapped = Buffer.alloc(data.length)
  for (let i = 0; i + 1 < data.length; i += 2) {
    swapped[i] = data[i + 1] ?? 0
    swapped[i + 1] = data[i] ?? 0
  }
  return swapped.toString('utf16le')
}
