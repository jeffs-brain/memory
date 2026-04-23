import { SchemaValidationError } from './errors.js'

const FENCED_JSON_RE = /```(?:json)?\s*\n([\s\S]*?)\n```/

export const extractJson = (content: string): string => {
  const trimmed = content.trim()
  if (trimmed === '') throw new Error('empty response')

  const fenced = FENCED_JSON_RE.exec(trimmed)
  if (fenced?.[1] !== undefined) {
    const inner = fenced[1].trim()
    if (inner !== '') return inner
  }

  for (let index = 0; index < trimmed.length; index += 1) {
    const character = trimmed[index]
    if (character !== '{' && character !== '[') continue
    const candidate = trimmed.slice(index)
    const parsed = tryParse(candidate)
    if (parsed !== undefined) return parsed
  }
  throw new Error('no JSON object or array found in response')
}

const tryParse = (value: string): string | undefined => {
  const opener = value[0]
  if (opener !== '{' && opener !== '[') return undefined
  const closer = opener === '{' ? '}' : ']'
  let depth = 0
  let inString = false
  let escaping = false
  for (let index = 0; index < value.length; index += 1) {
    const character = value[index]
    if (escaping) {
      escaping = false
      continue
    }
    if (inString) {
      if (character === '\\') escaping = true
      else if (character === '"') inString = false
      continue
    }
    if (character === '"') {
      inString = true
      continue
    }
    if (character === opener) depth += 1
    else if (character === closer) {
      depth -= 1
      if (depth === 0) {
        const slice = value.slice(0, index + 1)
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

export const validateAgainstSchema = (payload: unknown, schema: unknown): void => {
  const error = walk(payload, schema, '$')
  if (error !== null) throw new Error(error)
}

const walk = (value: unknown, schema: unknown, path: string): string | null => {
  if (schema === null || typeof schema !== 'object') return null
  const current = schema as Record<string, unknown>

  if (Array.isArray(current.enum)) {
    if (!current.enum.some((entry) => deepEqual(entry, value))) {
      return `${path}: value not in enum`
    }
  }

  if (typeof current.type === 'string') {
    if (!typeMatches(value, current.type)) {
      return `${path}: expected type ${current.type}`
    }
  }

  if (current.type === 'object' || (current.type === undefined && isPlainObject(value))) {
    if (!isPlainObject(value)) return null
    const properties = isPlainObject(current.properties)
      ? (current.properties as Record<string, unknown>)
      : {}
    if (Array.isArray(current.required)) {
      for (const key of current.required) {
        if (typeof key !== 'string') continue
        if (!(key in value)) return `${path}.${key}: required property missing`
      }
    }
    for (const [key, childSchema] of Object.entries(properties)) {
      if (!(key in value)) continue
      const error = walk((value as Record<string, unknown>)[key], childSchema, `${path}.${key}`)
      if (error !== null) return error
    }
  }

  if (current.type === 'array' || (current.type === undefined && Array.isArray(value))) {
    if (!Array.isArray(value)) return null
    if (current.items !== undefined) {
      for (let index = 0; index < value.length; index += 1) {
        const error = walk(value[index], current.items, `${path}[${index}]`)
        if (error !== null) return error
      }
    }
  }

  return null
}

const typeMatches = (value: unknown, type: string): boolean => {
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

const isPlainObject = (value: unknown): value is Record<string, unknown> => {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

const deepEqual = (left: unknown, right: unknown): boolean => {
  if (left === right) return true
  if (typeof left !== typeof right) return false
  if (left === null || right === null) return false
  if (Array.isArray(left) && Array.isArray(right)) {
    if (left.length !== right.length) return false
    for (let index = 0; index < left.length; index += 1) {
      if (!deepEqual(left[index], right[index])) return false
    }
    return true
  }
  if (isPlainObject(left) && isPlainObject(right)) {
    const leftKeys = Object.keys(left)
    const rightKeys = Object.keys(right)
    if (leftKeys.length !== rightKeys.length) return false
    for (const key of leftKeys) {
      if (!deepEqual(left[key], right[key])) return false
    }
    return true
  }
  return false
}

export type StructuredRunner = {
  call(
    messages: ReadonlyArray<{ readonly role: string; readonly content: string }>,
  ): Promise<string>
}

export const runStructured = async (
  runner: StructuredRunner,
  baseMessages: ReadonlyArray<{ readonly role: string; readonly content: string }>,
  schema: string,
  maxRetries: number,
): Promise<string> => {
  const retryBudget = maxRetries <= 0 ? 5 : maxRetries
  let parsedSchema: unknown
  try {
    parsedSchema = JSON.parse(schema)
  } catch (error) {
    throw new Error(
      `invalid JSON schema: ${error instanceof Error ? error.message : String(error)}`,
    )
  }

  let messages = baseMessages.slice()
  let lastPayload = ''
  let lastReason = ''

  for (let attempt = 1; attempt <= retryBudget; attempt += 1) {
    const content = await runner.call(messages)
    try {
      const payload = extractJson(content)
      const instance = JSON.parse(payload) as unknown
      validateAgainstSchema(instance, parsedSchema)
      return payload
    } catch (error) {
      lastPayload = content
      lastReason = error instanceof Error ? error.message : String(error)
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
