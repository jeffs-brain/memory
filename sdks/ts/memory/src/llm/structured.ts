/**
 * Structured-output helpers shared by every provider. The extract /
 * validate logic is provider-agnostic; providers wire their own
 * request-building and retry-correction loop on top.
 */

import { SchemaValidationError } from './errors.js'

const FENCED_JSON_RE = /```(?:json)?\s*\n([\s\S]*?)\n```/

/**
 * Pull JSON out of a model response, tolerating fenced code blocks,
 * leading prose, and surrounding whitespace. Returns the raw JSON
 * string. Throws when no parseable object or array is present.
 */
export function extractJSON(content: string): string {
  const trimmed = content.trim()
  if (trimmed === '') throw new Error('empty response')

  const fenced = FENCED_JSON_RE.exec(trimmed)
  if (fenced && fenced[1] !== undefined) {
    const inner = fenced[1].trim()
    if (inner !== '') return inner
  }

  // Walk for the first '{' or '[' that starts a valid JSON document.
  for (let i = 0; i < trimmed.length; i++) {
    const ch = trimmed[i]
    if (ch !== '{' && ch !== '[') continue
    const candidate = trimmed.slice(i)
    const parsed = tryParse(candidate)
    if (parsed !== undefined) return parsed
  }
  throw new Error('no JSON object or array found in response')
}

/**
 * Attempt to parse a prefix of `s` as JSON. Returns the exact text of
 * the parsed prefix when successful, or undefined when the candidate
 * is not a standalone JSON value.
 *
 * The strategy mirrors Go's json.Decoder: walk the string counting
 * braces/brackets, honouring strings and escapes, and return the
 * smallest prefix that balances.
 */
function tryParse(s: string): string | undefined {
  const opener = s[0]
  if (opener !== '{' && opener !== '[') return undefined
  const closer = opener === '{' ? '}' : ']'
  let depth = 0
  let inString = false
  let escape = false
  for (let i = 0; i < s.length; i++) {
    const ch = s[i]
    if (escape) {
      escape = false
      continue
    }
    if (inString) {
      if (ch === '\\') escape = true
      else if (ch === '"') inString = false
      continue
    }
    if (ch === '"') {
      inString = true
      continue
    }
    if (ch === opener) depth++
    else if (ch === closer) {
      depth--
      if (depth === 0) {
        const slice = s.slice(0, i + 1)
        try {
          JSON.parse(slice)
          return slice
        } catch {
          return undefined
        }
      }
    }
  }
  return undefined
}

/**
 * Validate a JSON payload against a JSON Schema. This is a pragmatic
 * subset that covers the fields we actually use in the port (type,
 * required, properties, items, enum). Anything else is skipped so a
 * richer schema still validates the common shape without failing the
 * caller's flow.
 *
 * Throws the validator's reason string on failure.
 */
export function validateAgainstSchema(payload: unknown, schema: unknown): void {
  const err = walk(payload, schema, '$')
  if (err !== null) throw new Error(err)
}

function walk(value: unknown, schema: unknown, path: string): string | null {
  if (schema === null || typeof schema !== 'object') return null
  const s = schema as Record<string, unknown>

  if (Array.isArray(s.enum)) {
    if (!s.enum.some((e) => deepEqual(e, value))) {
      return `${path}: value not in enum`
    }
  }

  if (typeof s.type === 'string') {
    if (!typeMatches(value, s.type)) {
      return `${path}: expected type ${s.type}, got ${classify(value)}`
    }
  }

  if (s.type === 'object' || (typeof s.type === 'undefined' && isPlainObject(value))) {
    if (!isPlainObject(value)) return null
    const props = isPlainObject(s.properties) ? (s.properties as Record<string, unknown>) : {}
    if (Array.isArray(s.required)) {
      for (const key of s.required) {
        if (typeof key !== 'string') continue
        if (!(key in value)) return `${path}.${key}: required property missing`
      }
    }
    for (const [key, childSchema] of Object.entries(props)) {
      if (!(key in value)) continue
      const err = walk((value as Record<string, unknown>)[key], childSchema, `${path}.${key}`)
      if (err !== null) return err
    }
  }

  if (s.type === 'array' || (typeof s.type === 'undefined' && Array.isArray(value))) {
    if (!Array.isArray(value)) return null
    if (s.items !== undefined) {
      for (let i = 0; i < value.length; i++) {
        const err = walk(value[i], s.items, `${path}[${i}]`)
        if (err !== null) return err
      }
    }
  }

  return null
}

function typeMatches(value: unknown, type: string): boolean {
  switch (type) {
    case 'string':
      return typeof value === 'string'
    case 'number':
      return typeof value === 'number' && !Number.isNaN(value)
    case 'integer':
      return typeof value === 'number' && Number.isInteger(value)
    case 'boolean':
      return typeof value === 'boolean'
    case 'null':
      return value === null
    case 'array':
      return Array.isArray(value)
    case 'object':
      return isPlainObject(value)
    default:
      return true
  }
}

function classify(value: unknown): string {
  if (value === null) return 'null'
  if (Array.isArray(value)) return 'array'
  return typeof value
}

function isPlainObject(v: unknown): v is Record<string, unknown> {
  return typeof v === 'object' && v !== null && !Array.isArray(v)
}

function deepEqual(a: unknown, b: unknown): boolean {
  if (a === b) return true
  if (typeof a !== typeof b) return false
  if (a === null || b === null) return false
  if (Array.isArray(a) && Array.isArray(b)) {
    if (a.length !== b.length) return false
    for (let i = 0; i < a.length; i++) if (!deepEqual(a[i], b[i])) return false
    return true
  }
  if (isPlainObject(a) && isPlainObject(b)) {
    const ak = Object.keys(a)
    const bk = Object.keys(b)
    if (ak.length !== bk.length) return false
    for (const k of ak) if (!deepEqual(a[k], b[k])) return false
    return true
  }
  return false
}

export type StructuredRunner = {
  call(messages: ReadonlyArray<{ role: string; content: string }>): Promise<string>
}

/**
 * Run the validate-and-retry loop for structured output. The caller
 * supplies a {@link StructuredRunner} that performs a single round-trip
 * against the provider with the given message history. We extract the
 * JSON, validate against the schema, and on failure append a
 * corrective user message and retry until the budget is exhausted.
 */
export async function runStructured(
  runner: StructuredRunner,
  baseMessages: ReadonlyArray<{ role: string; content: string }>,
  schema: string,
  maxRetries: number,
): Promise<string> {
  if (maxRetries <= 0) maxRetries = 5
  let schemaObj: unknown
  try {
    schemaObj = JSON.parse(schema)
  } catch (err) {
    throw new Error(
      `invalid JSON schema: ${err instanceof Error ? err.message : String(err)}`,
    )
  }

  let messages = baseMessages.slice()
  let lastPayload = ''
  let lastReason = ''

  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    const content = await runner.call(messages)
    try {
      const payload = extractJSON(content)
      let instance: unknown
      try {
        instance = JSON.parse(payload)
      } catch (err) {
        throw new Error(
          `payload is not valid JSON: ${err instanceof Error ? err.message : String(err)}`,
        )
      }
      validateAgainstSchema(instance, schemaObj)
      return payload
    } catch (err) {
      lastPayload = content
      lastReason = err instanceof Error ? err.message : String(err)
      messages = [
        ...messages,
        { role: 'assistant', content },
        {
          role: 'user',
          content: `Your previous response failed schema validation: ${lastReason}. Return only valid JSON matching the schema.`,
        },
      ]
    }
  }

  throw new SchemaValidationError(lastPayload, lastReason)
}
