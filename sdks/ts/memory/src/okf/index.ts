// SPDX-License-Identifier: Apache-2.0

/**
 * Open Knowledge Format helpers.
 *
 * OKF v0.1 is a lightweight markdown plus YAML-frontmatter convention. These
 * helpers keep the package permissive: parsing never rejects unknown fields,
 * normalisation maps Jeffs Brain legacy fields onto OKF names, and validation
 * reports conformance issues without mutating content.
 */

export const OKF_VERSION = '0.1' as const

const FENCE = '---'
const BOOLEAN_RE = /^(true|false|yes|no)$/i
const URI_SCHEME_RE = /^[a-z][a-z0-9+.-]*:/i
const MARKDOWN_LINK_RE = /(^|[^!])\[([^\]\n]+)\]\(([^)\s]+)(?:\s+"[^"]*")?\)/g
const WIKILINK_RE = /\[\[([^\]]+)\]\]/g

export type OkfFrontmatterValue = string | boolean | readonly string[]
export type OkfFrontmatter = Readonly<Record<string, OkfFrontmatterValue>>

export type ParsedOkfDocument = {
  readonly frontmatter: OkfFrontmatter
  readonly body: string
  readonly present: boolean
  readonly malformed: boolean
}

export type OkfConceptMetadata = {
  readonly conceptId?: string
  readonly type?: string
  readonly title?: string
  readonly description?: string
  readonly resource?: string
  readonly tags: readonly string[]
  readonly timestamp?: string
  readonly frontmatter: OkfFrontmatter
  readonly body: string
}

export type OkfValidationIssueSeverity = 'error' | 'warning'

export type OkfValidationIssue = {
  readonly severity: OkfValidationIssueSeverity
  readonly code:
    | 'missing_frontmatter'
    | 'malformed_frontmatter'
    | 'missing_type'
    | 'reserved_frontmatter'
    | 'invalid_log_heading'
  readonly path?: string
  readonly message: string
}

export type OkfValidationReport = {
  readonly ok: boolean
  readonly issues: readonly OkfValidationIssue[]
}

export type OkfLinkKind = 'markdown' | 'wikilink'

export type OkfLink = {
  readonly kind: OkfLinkKind
  readonly label: string
  readonly target: string
  readonly resolvedTarget?: string
  readonly external: boolean
}

export type NormaliseOkfOptions = {
  readonly path?: string
  readonly defaultType?: string
}

const stripQuotes = (value: string): string => {
  if (value.length < 2) return value
  const first = value[0]
  const last = value[value.length - 1]
  if ((first === '"' && last === '"') || (first === "'" && last === "'")) {
    return value.slice(1, -1)
  }
  return value
}

const parseInlineList = (value: string): readonly string[] => {
  const trimmed = value.trim()
  if (!trimmed.startsWith('[') || !trimmed.endsWith(']')) return []
  const inner = trimmed.slice(1, -1).trim()
  if (inner === '') return []
  return inner
    .split(',')
    .map((item) => stripQuotes(item.trim()))
    .filter((item) => item !== '')
}

const coerceScalar = (raw: string): string | boolean => {
  const value = stripQuotes(raw)
  if (BOOLEAN_RE.test(value)) return /^(true|yes)$/i.test(value)
  return value
}

const splitKV = (line: string): readonly [string, string] | undefined => {
  const idx = line.indexOf(':')
  if (idx < 0) return undefined
  const key = line.slice(0, idx).trim()
  const value = line.slice(idx + 1).trim()
  if (key === '') return undefined
  return [key, value]
}

const stringValue = (value: OkfFrontmatterValue | undefined): string | undefined => {
  if (value === undefined) return undefined
  if (typeof value === 'string') return value.trim() === '' ? undefined : value.trim()
  if (typeof value === 'boolean') return String(value)
  return undefined
}

const firstStringField = (
  frontmatter: OkfFrontmatter,
  keys: readonly string[],
): string | undefined => {
  for (const key of keys) {
    const value = stringValue(frontmatter[key])
    if (value !== undefined) return value
  }
  return undefined
}

const tagsField = (frontmatter: OkfFrontmatter): readonly string[] => {
  const value = frontmatter.tags
  if (Array.isArray(value)) {
    return value.map((tag) => tag.trim()).filter((tag) => tag !== '')
  }
  if (typeof value === 'string') {
    const trimmed = value.trim()
    if (trimmed === '') return []
    if (trimmed.startsWith('[') && trimmed.endsWith(']')) return parseInlineList(trimmed)
    return trimmed
      .split(',')
      .map((tag) => tag.trim())
      .filter((tag) => tag !== '')
  }
  return []
}

const titleFromHeading = (body: string): string | undefined => {
  const heading = body.match(/^#\s+(.+)$/m)?.[1]?.trim()
  return heading === undefined || heading === '' ? undefined : heading
}

const titleFromPath = (path: string | undefined): string | undefined => {
  if (path === undefined || path.trim() === '') return undefined
  const last = path
    .split('/')
    .filter((part) => part !== '')
    .pop()
  if (last === undefined) return undefined
  const stem = last.replace(/\.(md|markdown)$/i, '')
  const title = stem.replace(/[-_]+/g, ' ').replace(/\s+/g, ' ').trim()
  return title === '' ? undefined : title
}

export const conceptIdFromPath = (path: string | undefined): string | undefined => {
  if (path === undefined || path.trim() === '') return undefined
  const withoutLeading = path.replace(/^\/+/, '')
  const withoutMarkdown = withoutLeading.replace(/\.(md|markdown)$/i, '')
  return withoutMarkdown === '' ? undefined : withoutMarkdown
}

export const deriveOkfTypeFromPath = (path: string | undefined): string | undefined => {
  if (path === undefined || path.trim() === '') return undefined
  if (path === 'codec.md') return 'Codec'
  const first = path.replace(/^\/+/, '').split('/')[0] ?? ''
  switch (first) {
    case 'wiki':
    case 'drafts':
      return 'Article'
    case 'memory':
      return 'Memory'
    case 'raw':
      return 'Raw Document'
    case 'conversations':
    case 'episodes':
    case 'reflections':
      return 'Session'
    case 'ontology':
      return 'Ontology'
    case 'projects':
      return 'Project Memory'
    default:
      return 'Document'
  }
}

export const parseOkfDocument = (content: string): ParsedOkfDocument => {
  const lines = content.split('\n')
  if (lines.length < 2 || lines[0]?.trim() !== FENCE) {
    return { frontmatter: {}, body: content, present: false, malformed: false }
  }

  let closeIdx = -1
  for (let i = 1; i < lines.length; i++) {
    if (lines[i]?.trim() === FENCE) {
      closeIdx = i
      break
    }
  }

  if (closeIdx < 0) {
    return { frontmatter: {}, body: content, present: false, malformed: true }
  }

  const frontmatter: Record<string, OkfFrontmatterValue> = {}
  const listBuffers = new Map<string, string[]>()
  let currentListKey = ''

  for (let i = 1; i < closeIdx; i++) {
    const line = lines[i] ?? ''
    const trimmed = line.trim()

    if (currentListKey !== '' && trimmed.startsWith('- ')) {
      const value = stripQuotes(trimmed.slice(2).trim())
      if (value !== '') {
        const buffer = listBuffers.get(currentListKey) ?? []
        buffer.push(value)
        listBuffers.set(currentListKey, buffer)
        frontmatter[currentListKey] = buffer
      }
      continue
    }

    const kv = splitKV(line)
    if (kv === undefined) {
      currentListKey = ''
      continue
    }

    const [key, rawValue] = kv
    if (rawValue === '') {
      currentListKey = key
      if (!listBuffers.has(key)) {
        const buffer: string[] = []
        listBuffers.set(key, buffer)
        frontmatter[key] = buffer
      }
      continue
    }

    currentListKey = ''
    const inline = parseInlineList(rawValue)
    frontmatter[key] =
      inline.length > 0 || /^\[\s*\]$/.test(rawValue.trim()) ? inline : coerceScalar(rawValue)
  }

  const body = lines
    .slice(closeIdx + 1)
    .join('\n')
    .replace(/^\n+/, '')
    .replace(/\n+$/, '')

  return { frontmatter, body, present: true, malformed: false }
}

export const normaliseOkfMetadata = (
  frontmatter: OkfFrontmatter,
  body = '',
  options: NormaliseOkfOptions = {},
): OkfConceptMetadata => {
  const title =
    firstStringField(frontmatter, ['title']) ??
    titleFromHeading(body) ??
    firstStringField(frontmatter, ['name']) ??
    titleFromPath(options.path)
  const description = firstStringField(frontmatter, ['description', 'summary'])
  const explicitType = firstStringField(frontmatter, ['type'])
  const type = explicitType ?? options.defaultType
  const resource = firstStringField(frontmatter, ['resource', 'url', 'source_url', 'sourceUrl'])
  const timestamp = firstStringField(frontmatter, [
    'timestamp',
    'modified',
    'updated_at',
    'updatedAt',
    'created_at',
    'createdAt',
    'created',
  ])

  const conceptId = conceptIdFromPath(options.path)

  return {
    ...(conceptId !== undefined ? { conceptId } : {}),
    ...(type !== undefined ? { type } : {}),
    ...(title !== undefined ? { title } : {}),
    ...(description !== undefined ? { description } : {}),
    ...(resource !== undefined ? { resource } : {}),
    tags: tagsField(frontmatter),
    ...(timestamp !== undefined ? { timestamp } : {}),
    frontmatter,
    body,
  }
}

export const normaliseOkfDocument = (
  content: string,
  options: NormaliseOkfOptions = {},
): OkfConceptMetadata => {
  const parsed = parseOkfDocument(content)
  return normaliseOkfMetadata(parsed.frontmatter, parsed.body, options)
}

const isReservedPath = (path: string | undefined): boolean => {
  if (path === undefined) return false
  const last = path
    .split('/')
    .filter((part) => part !== '')
    .pop()
  return last === 'index.md' || last === 'log.md'
}

const isRootIndexPath = (path: string | undefined): boolean =>
  path === undefined || path === '' || path === 'index.md' || path === '/index.md'

export const validateOkfDocument = (
  content: string,
  options: { readonly path?: string } = {},
): OkfValidationReport => {
  const parsed = parseOkfDocument(content)
  const issues: OkfValidationIssue[] = []
  const path = options.path

  if (isReservedPath(path)) {
    if (parsed.present) {
      const keys = Object.keys(parsed.frontmatter)
      const allowedRootIndex = isRootIndexPath(path) && keys.every((key) => key === 'okf_version')
      if (!allowedRootIndex) {
        issues.push({
          severity: 'error',
          code: 'reserved_frontmatter',
          ...(path !== undefined ? { path } : {}),
          message: 'reserved OKF index.md and log.md files must not be concept documents',
        })
      }
    }
    if (path?.endsWith('log.md') === true) {
      for (const line of content.split('\n')) {
        if (/^##\s+/.test(line) && !/^##\s+\d{4}-\d{2}-\d{2}\s*$/.test(line)) {
          issues.push({
            severity: 'warning',
            code: 'invalid_log_heading',
            path,
            message: 'OKF log.md date headings should use YYYY-MM-DD',
          })
          break
        }
      }
    }
    return { ok: !issues.some((issue) => issue.severity === 'error'), issues }
  }

  if (parsed.malformed) {
    issues.push({
      severity: 'error',
      code: 'malformed_frontmatter',
      ...(path !== undefined ? { path } : {}),
      message: 'frontmatter starts with --- but has no closing fence',
    })
  }
  if (!parsed.present) {
    issues.push({
      severity: 'error',
      code: 'missing_frontmatter',
      ...(path !== undefined ? { path } : {}),
      message: 'OKF concept documents require YAML frontmatter',
    })
  }
  if (parsed.present && firstStringField(parsed.frontmatter, ['type']) === undefined) {
    issues.push({
      severity: 'error',
      code: 'missing_type',
      ...(path !== undefined ? { path } : {}),
      message: 'OKF concept documents require a non-empty type field',
    })
  }

  return { ok: issues.length === 0, issues }
}

const stripLinkDecorators = (target: string): string => {
  const trimmed = target.trim().replace(/^<|>$/g, '')
  return trimmed.replace(/[?#].*$/, '')
}

export const normaliseOkfLinkTarget = (target: string, sourcePath?: string): string | undefined => {
  const stripped = stripLinkDecorators(target)
  if (stripped === '' || stripped.startsWith('#') || URI_SCHEME_RE.test(stripped)) return undefined
  if (stripped.startsWith('/')) return stripped.replace(/^\/+/, '')

  const sourceDir =
    sourcePath === undefined ? [] : sourcePath.replace(/^\/+/, '').split('/').slice(0, -1)
  const rawSegments = stripped.split('/')
  const segments =
    stripped.startsWith('./') || stripped.startsWith('../')
      ? [...sourceDir, ...rawSegments]
      : rawSegments
  const resolved: string[] = []
  for (const segment of segments) {
    if (segment === '' || segment === '.') continue
    if (segment === '..') {
      resolved.pop()
      continue
    }
    resolved.push(segment)
  }
  const joined = resolved.join('/')
  return joined === '' ? undefined : joined
}

export const extractOkfLinks = (content: string, sourcePath?: string): readonly OkfLink[] => {
  const links: OkfLink[] = []

  for (const match of content.matchAll(WIKILINK_RE)) {
    const raw = (match[1] ?? '').trim()
    const target = raw.split('|')[0]?.trim() ?? ''
    if (target === '') continue
    const resolvedTarget = normaliseOkfLinkTarget(target, sourcePath)
    links.push({
      kind: 'wikilink',
      label: target,
      target,
      ...(resolvedTarget !== undefined ? { resolvedTarget } : {}),
      external: resolvedTarget === undefined,
    })
  }

  for (const match of content.matchAll(MARKDOWN_LINK_RE)) {
    const label = (match[2] ?? '').trim()
    const target = stripLinkDecorators(match[3] ?? '')
    if (target === '') continue
    const resolvedTarget = normaliseOkfLinkTarget(target, sourcePath)
    links.push({
      kind: 'markdown',
      label,
      target,
      ...(resolvedTarget !== undefined ? { resolvedTarget } : {}),
      external: resolvedTarget === undefined,
    })
  }

  return links
}
