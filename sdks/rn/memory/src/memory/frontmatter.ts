export type Frontmatter = {
  name?: string
  description?: string
  type?: string
  created?: string
  modified?: string
  tags?: readonly string[]
  confidence?: string
  source?: string
  supersedes?: string
  superseded_by?: string
  scope?: string
  session_id?: string
  session_date?: string
  observed_on?: string
  extra: Record<string, string>
}

export const parseFrontmatter = (content: string): { frontmatter: Frontmatter; body: string } => {
  const lines = content.split('\n')
  if (lines.length < 2 || (lines[0] ?? '').trim() !== '---') {
    return { frontmatter: { extra: {} }, body: content }
  }

  let closeIdx = -1
  for (let index = 1; index < lines.length; index += 1) {
    if ((lines[index] ?? '').trim() === '---') {
      closeIdx = index
      break
    }
  }
  if (closeIdx < 0) {
    return { frontmatter: { extra: {} }, body: content }
  }

  const frontmatter: Frontmatter = { extra: {} }
  const tags: string[] = []
  let currentListKey = ''

  for (let index = 1; index < closeIdx; index += 1) {
    const line = lines[index] ?? ''
    const trimmed = line.trim()

    if (currentListKey !== '' && trimmed.startsWith('- ')) {
      const value = trimmed.slice(2).trim()
      if (currentListKey === 'tags' && value !== '') tags.push(value)
      continue
    }

    const colon = line.indexOf(':')
    if (colon < 0) {
      currentListKey = ''
      continue
    }

    const key = line.slice(0, colon).trim()
    let value = line.slice(colon + 1).trim()
    if (value.length >= 2) {
      const first = value[0]
      const last = value[value.length - 1]
      if ((first === '"' && last === '"') || (first === "'" && last === "'")) {
        value = value.slice(1, -1)
      }
    }

    if (value === '') {
      currentListKey = key
      continue
    }
    currentListKey = ''

    switch (key) {
      case 'name':
        frontmatter.name = value
        break
      case 'description':
        frontmatter.description = value
        break
      case 'type':
        frontmatter.type = value
        break
      case 'created':
        frontmatter.created = value
        break
      case 'modified':
        frontmatter.modified = value
        break
      case 'confidence':
        frontmatter.confidence = value
        break
      case 'source':
        frontmatter.source = value
        break
      case 'supersedes':
        frontmatter.supersedes = value
        break
      case 'superseded_by':
        frontmatter.superseded_by = value
        break
      case 'scope':
        frontmatter.scope = value
        break
      case 'session_id':
        frontmatter.session_id = value
        break
      case 'session_date':
        frontmatter.session_date = value
        break
      case 'observed_on':
        frontmatter.observed_on = value
        break
      case 'tags':
        if (value.startsWith('[') && value.endsWith(']')) {
          const inner = value.slice(1, -1)
          for (const raw of inner.split(',')) {
            const tag = raw.trim()
            if (tag !== '') tags.push(tag)
          }
        } else {
          for (const raw of value.split(',')) {
            const tag = raw.trim()
            if (tag !== '') tags.push(tag)
          }
        }
        break
      default:
        frontmatter.extra[key] = value
    }
  }

  if (tags.length > 0) frontmatter.tags = tags
  return {
    frontmatter,
    body: lines
      .slice(closeIdx + 1)
      .join('\n')
      .trim(),
  }
}

export const buildFrontmatter = (frontmatter: Frontmatter): string => {
  const out: string[] = ['---']
  if (frontmatter.name) out.push(`name: ${frontmatter.name}`)
  if (frontmatter.description) out.push(`description: ${frontmatter.description}`)
  if (frontmatter.type) out.push(`type: ${frontmatter.type}`)
  if (frontmatter.scope) out.push(`scope: ${frontmatter.scope}`)
  if (frontmatter.created) out.push(`created: ${frontmatter.created}`)
  if (frontmatter.modified) out.push(`modified: ${frontmatter.modified}`)
  if (frontmatter.confidence) out.push(`confidence: ${frontmatter.confidence}`)
  if (frontmatter.source) out.push(`source: ${frontmatter.source}`)
  if (frontmatter.supersedes) out.push(`supersedes: ${frontmatter.supersedes}`)
  if (frontmatter.superseded_by) out.push(`superseded_by: ${frontmatter.superseded_by}`)
  if (frontmatter.session_id) out.push(`session_id: ${frontmatter.session_id}`)
  if (frontmatter.session_date) out.push(`session_date: ${frontmatter.session_date}`)
  if (frontmatter.observed_on) out.push(`observed_on: ${frontmatter.observed_on}`)
  if (frontmatter.tags && frontmatter.tags.length > 0) {
    out.push(`tags: [${frontmatter.tags.join(', ')}]`)
  }
  for (const [key, value] of Object.entries(frontmatter.extra ?? {})) {
    out.push(`${key}: ${value}`)
  }
  out.push('---', '')
  return out.join('\n')
}
