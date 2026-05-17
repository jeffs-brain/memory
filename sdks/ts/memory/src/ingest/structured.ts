// SPDX-License-Identifier: Apache-2.0

/**
 * Structured data extractors for CSV/TSV and JSON documents.
 * Schema-aware chunking that preserves record boundaries and column
 * context instead of splitting arbitrarily by character count.
 */

// -------------------------------------------------------------------
// Shared types
// -------------------------------------------------------------------

export type ExtractionResult = {
  readonly content: string
  readonly mime: string
  readonly metadata: Readonly<Record<string, string>>
}

// -------------------------------------------------------------------
// Encoding detection
// -------------------------------------------------------------------

const detectEncoding = (raw: Buffer): { text: string; encoding: string } => {
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

// -------------------------------------------------------------------
// CSV Extractor
// -------------------------------------------------------------------

export type CsvExtractorConfig = {
  readonly rowsPerChunk?: number
  readonly maxRows?: number
  readonly forceDelimiter?: string
}

const defaultCsvConfig = {
  rowsPerChunk: 50,
  maxRows: 100000,
} as const

const resolveDelimiter = (text: string, forced: string | undefined): string => {
  if (forced !== undefined && forced.length > 0) return forced
  return detectDelimiter(text)
}

/**
 * Parse CSV text using the detected delimiter. Handles quoted fields
 * with embedded delimiters and newlines.
 */
const parseCSV = (text: string, delimiter: string): readonly string[][] => {
  const rows: string[][] = []
  let current: string[] = []
  let field = ''
  let inQuotes = false
  let i = 0

  while (i < text.length) {
    const ch = text[i] ?? ''

    if (inQuotes) {
      if (ch === '"') {
        const next = text[i + 1]
        if (next === '"') {
          field += '"'
          i += 2
          continue
        }
        inQuotes = false
        i++
        continue
      }
      field += ch
      i++
      continue
    }

    if (ch === '"') {
      inQuotes = true
      i++
      continue
    }

    if (ch === delimiter) {
      current.push(field)
      field = ''
      i++
      continue
    }

    if (ch === '\r') {
      const next = text[i + 1]
      if (next === '\n') {
        current.push(field)
        field = ''
        if (current.length > 0) rows.push(current)
        current = []
        i += 2
        continue
      }
      current.push(field)
      field = ''
      if (current.length > 0) rows.push(current)
      current = []
      i++
      continue
    }

    if (ch === '\n') {
      current.push(field)
      field = ''
      if (current.length > 0) rows.push(current)
      current = []
      i++
      continue
    }

    field += ch
    i++
  }

  // Flush remaining field.
  if (field.length > 0 || current.length > 0) {
    current.push(field)
    rows.push(current)
  }

  return rows
}

/**
 * Extract structured content from CSV/TSV bytes. Each chunk preserves
 * column headers for self-contained search context.
 */
export const extractCSV = (raw: Buffer, config: CsvExtractorConfig = {}): ExtractionResult => {
  if (raw.length === 0) {
    throw new Error('structured: empty csv file')
  }

  const { text, encoding } = detectEncoding(raw)
  const delimiter = resolveDelimiter(text, config.forceDelimiter)

  const allRows = parseCSV(text, delimiter)
  if (allRows.length === 0) {
    throw new Error('structured: empty csv file')
  }

  const { headers, dataRows: allDataRows } = splitHeaders(allRows)
  const maxRows = config.maxRows ?? defaultCsvConfig.maxRows
  const dataRows = allDataRows.length > maxRows ? allDataRows.slice(0, maxRows) : allDataRows
  const rpc = config.rowsPerChunk ?? defaultCsvConfig.rowsPerChunk

  const parts: string[] = []
  let chunkCount = 0

  for (let i = 0; i < dataRows.length; i += rpc) {
    const end = Math.min(i + rpc, dataRows.length)
    const batch = dataRows.slice(i, end)
    const lines: string[] = []

    for (let j = 0; j < batch.length; j++) {
      const row = batch[j]
      if (row === undefined) continue
      const rowNum = i + j + 1
      lines.push(`Row ${rowNum}:`)
      for (let k = 0; k < headers.length; k++) {
        const header = headers[k] ?? `Column_${k + 1}`
        const val = k < row.length ? (row[k] ?? '') : ''
        lines.push(`- ${header}: ${val}`)
      }
      if (j < batch.length - 1) {
        lines.push('')
      }
    }

    parts.push(lines.join('\n'))
    chunkCount++
  }

  const mime = delimiter === '\t' ? 'text/tab-separated-values' : 'text/csv'

  return {
    content: parts.join('\n\n---\n\n'),
    mime,
    metadata: {
      encoding,
      delimiter,
      column_count: String(headers.length),
      row_count: String(dataRows.length),
      chunk_count: String(chunkCount),
    },
  }
}

/**
 * Detect the most likely delimiter by testing comma, semicolon, tab,
 * and pipe across the first 10 rows and selecting the one with the
 * highest column count and consistency.
 */
const detectDelimiter = (text: string): string => {
  const candidates = [',', ';', '\t', '|']
  const lines = firstNLines(text, 10)
  if (lines.length === 0) return ','

  let bestDelim = ','
  let bestScore = 0

  for (const d of candidates) {
    const counts = lines.map((line) => line.split(d).length)
    const firstCount = counts[0] ?? 1
    if (firstCount <= 1) continue

    const mean = counts.reduce((a, b) => a + b, 0) / counts.length
    const variance = counts.reduce((a, c) => a + (c - mean) ** 2, 0) / counts.length
    const stddev = Math.sqrt(variance)
    const consistency = mean > 0 ? 1 - stddev / mean : 1
    const score = mean * consistency

    if (score > bestScore) {
      bestScore = score
      bestDelim = d
    }
  }

  return bestDelim
}

const firstNLines = (text: string, n: number): readonly string[] => {
  const out: string[] = []
  for (const line of text.split('\n')) {
    if (line.trim() === '') continue
    out.push(line)
    if (out.length >= n) break
  }
  return out
}

/**
 * Separate headers from data rows. The first row is treated as
 * headers when: (a) all values are non-empty, (b) none is purely
 * numeric, and (c) all values are unique.
 */
const splitHeaders = (
  rows: readonly (readonly string[])[],
): { headers: readonly string[]; dataRows: readonly (readonly string[])[] } => {
  if (rows.length === 0) return { headers: [], dataRows: [] }
  const first = rows[0]
  if (first === undefined) return { headers: [], dataRows: [] }

  if (looksLikeHeaders(first)) {
    return { headers: first, dataRows: rows.slice(1) }
  }

  const headers = first.map((_, i) => `Column_${i + 1}`)
  return { headers, dataRows: rows }
}

const looksLikeHeaders = (row: readonly string[]): boolean => {
  if (row.length === 0) return false
  const seen = new Set<string>()
  for (const v of row) {
    const trimmed = v.trim()
    if (trimmed === '') return false
    if (isNumeric(trimmed)) return false
    const lower = trimmed.toLowerCase()
    if (seen.has(lower)) return false
    seen.add(lower)
  }
  return true
}

const isNumeric = (s: string): boolean => {
  const trimmed = s.trim()
  if (trimmed === '') return false
  return !Number.isNaN(Number(trimmed)) && trimmed !== ''
}

// -------------------------------------------------------------------
// JSON Extractor
// -------------------------------------------------------------------

export type JsonExtractorConfig = {
  readonly objectsPerChunk?: number
  readonly maxDepth?: number
  readonly schemaSampleSize?: number
  readonly tableThreshold?: number
}

const defaultJsonConfig = {
  objectsPerChunk: 50,
  maxDepth: 10,
  schemaSampleSize: 20,
  tableThreshold: 3,
} as const

/**
 * Extract structured content from JSON bytes. Arrays of objects are
 * chunked by object boundaries. Uniform arrays render as markdown
 * tables. Deeply nested structures flatten to dot-notation.
 */
export const extractJSON = (raw: Buffer, config: JsonExtractorConfig = {}): ExtractionResult => {
  const trimmed = raw.toString('utf8').trim()
  if (trimmed === '') {
    throw new Error('structured: empty json input')
  }

  const { text, encoding } = detectEncoding(raw)

  let parsed: unknown
  try {
    parsed = JSON.parse(text)
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err)
    throw new Error(`structured: invalid json: ${msg}`)
  }

  const cfg = {
    objectsPerChunk: config.objectsPerChunk ?? defaultJsonConfig.objectsPerChunk,
    maxDepth: config.maxDepth ?? defaultJsonConfig.maxDepth,
    schemaSampleSize: config.schemaSampleSize ?? defaultJsonConfig.schemaSampleSize,
    tableThreshold: config.tableThreshold ?? defaultJsonConfig.tableThreshold,
  }

  const { content, structureType, schema } = renderJSON(parsed, cfg)

  const metadata: Record<string, string> = {
    encoding,
    structure_type: structureType,
  }
  if (schema !== '') {
    metadata.detected_schema = schema
  }

  return { content, mime: 'application/json', metadata }
}

type ResolvedJsonConfig = {
  readonly objectsPerChunk: number
  readonly maxDepth: number
  readonly schemaSampleSize: number
  readonly tableThreshold: number
}

type RenderResult = {
  readonly content: string
  readonly structureType: string
  readonly schema: string
}

const renderJSON = (v: unknown, cfg: ResolvedJsonConfig): RenderResult => {
  if (Array.isArray(v)) {
    return renderJSONArray(v, cfg)
  }
  if (v !== null && typeof v === 'object') {
    return renderJSONObject(v as Record<string, unknown>, cfg)
  }
  return { content: formatPrimitive(v), structureType: 'primitive', schema: '' }
}

const renderJSONArray = (arr: readonly unknown[], cfg: ResolvedJsonConfig): RenderResult => {
  if (arr.length === 0) {
    return { content: '[]', structureType: 'empty_array', schema: '' }
  }

  // Check if all elements are primitives.
  const allPrimitive = arr.every(
    (elem) => elem === null || (typeof elem !== 'object' && !Array.isArray(elem)),
  )
  if (allPrimitive) {
    const lines = arr.map(formatPrimitive)
    return { content: lines.join('\n'), structureType: 'primitive_array', schema: '' }
  }

  // Separate objects from non-objects.
  const objects: Record<string, unknown>[] = []
  for (const elem of arr) {
    if (elem !== null && typeof elem === 'object' && !Array.isArray(elem)) {
      objects.push(elem as Record<string, unknown>)
    }
  }

  if (objects.length === 0) {
    const lines: string[] = []
    for (let i = 0; i < arr.length; i++) {
      if (i > 0) lines.push('')
      lines.push(`Item ${i + 1}:`)
      lines.push(flattenValue('', arr[i], 0, cfg.maxDepth, new Set()))
    }
    return { content: lines.join('\n'), structureType: 'mixed_array', schema: '' }
  }

  // Schema detection from sampled objects.
  const sampleSize = Math.min(cfg.schemaSampleSize, objects.length)
  const sample = objects.slice(0, sampleSize)
  const commonKeys = detectCommonKeys(sample)
  const schemaStr = commonKeys.join(', ')
  const isUniform = commonKeys.length > 0 && areObjectsUniform(sample, commonKeys)

  if (isUniform && objects.length >= cfg.tableThreshold) {
    return {
      content: renderAsMarkdownTable(objects, commonKeys, cfg.objectsPerChunk),
      structureType: 'uniform_object_array',
      schema: schemaStr,
    }
  }

  return {
    content: renderObjectsIndividually(objects, commonKeys, cfg),
    structureType: 'heterogeneous_object_array',
    schema: schemaStr,
  }
}

const renderJSONObject = (
  obj: Record<string, unknown>,
  cfg: ResolvedJsonConfig,
): RenderResult => {
  const keys = Object.keys(obj).sort()
  if (keys.length === 0) {
    return { content: '{}', structureType: 'empty_object', schema: '' }
  }

  const visited = new Set<unknown>()
  const lines: string[] = []
  lines.push(`Object with ${keys.length} keys: ${keys.join(', ')}`)
  lines.push('')

  for (const key of keys) {
    const val = obj[key]
    const depth = maxNestingDepth(val, 0)
    if (depth >= 4) {
      lines.push(flattenValue(key, val, 0, cfg.maxDepth, visited))
    } else {
      lines.push(`${key}: ${formatValue(val)}`)
    }
  }

  return { content: lines.join('\n'), structureType: 'object', schema: keys.join(', ') }
}

const detectCommonKeys = (objects: readonly Record<string, unknown>[]): readonly string[] => {
  if (objects.length === 0) return []
  const counts = new Map<string, number>()
  for (const obj of objects) {
    for (const key of Object.keys(obj)) {
      counts.set(key, (counts.get(key) ?? 0) + 1)
    }
  }
  const threshold = Math.max(1, Math.floor(objects.length / 2))
  const keys: string[] = []
  for (const [key, count] of counts) {
    if (count >= threshold) keys.push(key)
  }
  return keys.sort()
}

const areObjectsUniform = (
  objects: readonly Record<string, unknown>[],
  commonKeys: readonly string[],
): boolean => {
  const keySet = new Set(commonKeys)
  for (const obj of objects) {
    const objKeys = Object.keys(obj)
    if (objKeys.length !== commonKeys.length) return false
    for (const key of objKeys) {
      if (!keySet.has(key)) return false
    }
  }
  return true
}

const renderAsMarkdownTable = (
  objects: readonly Record<string, unknown>[],
  keys: readonly string[],
  objectsPerChunk: number,
): string => {
  const parts: string[] = []

  for (let i = 0; i < objects.length; i += objectsPerChunk) {
    const end = Math.min(i + objectsPerChunk, objects.length)
    const batch = objects.slice(i, end)

    const lines: string[] = []
    // Header row.
    lines.push(`| ${keys.join(' | ')} |`)
    // Separator row.
    lines.push(`|${keys.map(() => ' --- |').join('')}`)
    // Data rows.
    for (const obj of batch) {
      const cells = keys.map((k) => escapeMdTableCell(formatPrimitive(obj[k])))
      lines.push(`| ${cells.join(' | ')} |`)
    }

    parts.push(lines.join('\n'))
  }

  return parts.join('\n\n---\n\n')
}

const renderObjectsIndividually = (
  objects: readonly Record<string, unknown>[],
  _commonKeys: readonly string[],
  cfg: ResolvedJsonConfig,
): string => {
  const opc = cfg.objectsPerChunk
  const parts: string[] = []

  for (let i = 0; i < objects.length; i += opc) {
    const end = Math.min(i + opc, objects.length)
    const batch = objects.slice(i, end)
    const lines: string[] = []

    for (let j = 0; j < batch.length; j++) {
      const obj = batch[j]
      if (obj === undefined) continue
      if (j > 0) lines.push('')
      lines.push(`Item ${i + j + 1}:`)
      const keys = Object.keys(obj).sort()
      const visited = new Set<unknown>()
      for (const key of keys) {
        const val = obj[key]
        const depth = maxNestingDepth(val, 0)
        if (depth >= 4) {
          lines.push(flattenValue(key, val, 0, cfg.maxDepth, visited))
        } else {
          lines.push(`- ${key}: ${formatPrimitive(val)}`)
        }
      }
    }

    parts.push(lines.join('\n'))
  }

  return parts.join('\n\n---\n\n')
}

const flattenValue = (
  prefix: string,
  v: unknown,
  depth: number,
  maxDepth: number,
  visited: Set<unknown>,
): string => {
  if (depth >= maxDepth) {
    return `${prefix}: [max depth exceeded]`
  }

  if (v !== null && typeof v === 'object') {
    if (visited.has(v)) {
      return `${prefix}: [circular reference]`
    }
    visited.add(v)

    if (Array.isArray(v)) {
      if (v.length === 0) return `${prefix}: []`
      const lines: string[] = []
      for (let i = 0; i < v.length; i++) {
        lines.push(flattenValue(`${prefix}[${i}]`, v[i], depth + 1, maxDepth, visited))
      }
      return lines.join('\n')
    }

    const obj = v as Record<string, unknown>
    const keys = Object.keys(obj).sort()
    if (keys.length === 0) return `${prefix}: {}`
    const lines: string[] = []
    for (const key of keys) {
      const childPrefix = prefix !== '' ? `${prefix}.${key}` : key
      lines.push(flattenValue(childPrefix, obj[key], depth + 1, maxDepth, visited))
    }
    return lines.join('\n')
  }

  const label = prefix !== '' ? prefix : 'value'
  return `${label}: ${formatPrimitive(v)}`
}

const formatPrimitive = (v: unknown): string => {
  if (v === null || v === undefined) return 'null'
  if (typeof v === 'string') return v
  if (typeof v === 'boolean') return v ? 'true' : 'false'
  if (typeof v === 'number') return String(v)
  return String(v)
}

const formatValue = (v: unknown): string => {
  if (v !== null && typeof v === 'object') {
    return JSON.stringify(v)
  }
  return formatPrimitive(v)
}

const maxNestingDepth = (v: unknown, current: number): number => {
  if (v === null || typeof v !== 'object') return current
  if (Array.isArray(v)) {
    let max = current
    for (const child of v) {
      const d = maxNestingDepth(child, current + 1)
      if (d > max) max = d
    }
    return max
  }
  const obj = v as Record<string, unknown>
  let max = current
  for (const val of Object.values(obj)) {
    const d = maxNestingDepth(val, current + 1)
    if (d > max) max = d
  }
  return max
}

const escapeMdTableCell = (s: string): string =>
  s.replaceAll('|', '\\|').replaceAll('\n', ' ')

// -------------------------------------------------------------------
// JSONL support
// -------------------------------------------------------------------

/**
 * Extract structured content from newline-delimited JSON. Each line
 * is parsed as a JSON value, collected into an array, then delegated
 * to extractJSON.
 */
export const extractJSONL = (raw: Buffer, config: JsonExtractorConfig = {}): ExtractionResult => {
  const text = raw.toString('utf8').trim()
  if (text === '') {
    throw new Error('structured: empty jsonl input')
  }

  const objects: unknown[] = []
  for (const line of text.split('\n')) {
    const trimmed = line.trim()
    if (trimmed === '') continue
    try {
      objects.push(JSON.parse(trimmed))
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err)
      throw new Error(`structured: invalid jsonl: ${msg}`)
    }
  }

  if (objects.length === 0) {
    throw new Error('structured: empty jsonl input')
  }

  const arrayJson = JSON.stringify(objects)
  const result = extractJSON(Buffer.from(arrayJson, 'utf8'), config)
  return { ...result, mime: 'application/jsonl' }
}
