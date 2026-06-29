// SPDX-License-Identifier: Apache-2.0

/**
 * Minimal YAML-frontmatter parser. Matches the subset produced by the Go
 * memory writer: `key: value` scalars plus a `tags:` list in either inline
 * (`[a, b]`) or block (`- a\n- b`) form. Unknown keys land in `extra` so
 * callers can preserve round-trip fidelity when rewriting.
 *
 * The parser also tolerates the [Open Knowledge Format](https://github.com/GoogleCloudPlatform/knowledge-catalog/tree/main/okf)
 * key aliases (`title`→`name`, `timestamp`→`modified`), so brains written by
 * {@link buildFrontmatter} under the `okf` profile read back identically and a
 * brain may freely mix both formats. See {@link FrontmatterProfile}.
 */

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

/**
 * Serialisation profile for {@link buildFrontmatter}.
 *
 * - `default` — the native field set (`name`/`created`/`modified`/…). This is
 *   what every existing brain on disk uses and stays the default, so output is
 *   byte-for-byte unchanged unless a profile is explicitly selected.
 * - `okf` — emits [Open Knowledge Format](https://github.com/GoogleCloudPlatform/knowledge-catalog/tree/main/okf)
 *   shaped frontmatter (`type`/`title`/`description`/`tags`/`timestamp`) for
 *   interoperability with OKF tooling. Native-only fields (`scope`, `source`,
 *   `session_id`, …) are preserved as OKF extension keys so a brain round-trips
 *   losslessly through the profile.
 *
 * Reads are profile-agnostic — {@link parseFrontmatter} understands both key
 * sets — so a brain may mix formats and switching profiles needs no migration.
 */
export type FrontmatterProfile = 'default' | 'okf'

export const parseFrontmatter = (content: string): { frontmatter: Frontmatter; body: string } => {
  const lines = content.split('\n')
  if (lines.length < 2 || (lines[0] ?? '').trim() !== '---') {
    return { frontmatter: { extra: {} }, body: content }
  }

  let closeIdx = -1
  for (let i = 1; i < lines.length; i++) {
    if ((lines[i] ?? '').trim() === '---') {
      closeIdx = i
      break
    }
  }
  if (closeIdx < 0) {
    return { frontmatter: { extra: {} }, body: content }
  }

  const fm: Frontmatter = { extra: {} }
  const tags: string[] = []
  let currentListKey = ''
  // OKF aliases captured separately so the native keys win when both are present.
  let titleAlias: string | undefined
  let timestampAlias: string | undefined

  for (let i = 1; i < closeIdx; i++) {
    const line = lines[i] ?? ''
    const trimmed = line.trim()

    if (currentListKey !== '' && trimmed.startsWith('- ')) {
      const val = trimmed.slice(2).trim()
      if (currentListKey === 'tags' && val) tags.push(val)
      continue
    }

    const colon = line.indexOf(':')
    if (colon < 0) {
      currentListKey = ''
      continue
    }

    const key = line.slice(0, colon).trim()
    let val = line.slice(colon + 1).trim()
    if (val.length >= 2) {
      const first = val[0]
      const last = val[val.length - 1]
      if ((first === '"' && last === '"') || (first === "'" && last === "'")) {
        val = val.slice(1, -1)
      }
    }

    if (val === '') {
      currentListKey = key
      continue
    }
    currentListKey = ''

    switch (key) {
      case 'name':
        fm.name = val
        break
      case 'description':
        fm.description = val
        break
      case 'type':
        fm.type = val
        break
      case 'created':
        fm.created = val
        break
      case 'modified':
        fm.modified = val
        break
      case 'confidence':
        fm.confidence = val
        break
      case 'source':
        fm.source = val
        break
      case 'supersedes':
        fm.supersedes = val
        break
      case 'superseded_by':
        fm.superseded_by = val
        break
      case 'scope':
        fm.scope = val
        break
      case 'session_id':
        fm.session_id = val
        break
      case 'session_date':
        fm.session_date = val
        break
      case 'observed_on':
        fm.observed_on = val
        break
      // OKF profile aliases — resolved after the loop so an explicit native key wins.
      case 'title':
        titleAlias = val
        break
      case 'timestamp':
        timestampAlias = val
        break
      case 'tags': {
        // Inline [a, b, c] form.
        if (val.startsWith('[') && val.endsWith(']')) {
          const inner = val.slice(1, -1)
          for (const raw of inner.split(',')) {
            const t = raw.trim()
            if (t) tags.push(t)
          }
        } else {
          for (const raw of val.split(',')) {
            const t = raw.trim()
            if (t) tags.push(t)
          }
        }
        break
      }
      default:
        fm.extra[key] = val
    }
  }

  if (tags.length > 0) fm.tags = tags
  // OKF: `title`→name, `timestamp`→modified. Native keys, if also present, win.
  if (fm.name === undefined && titleAlias !== undefined) fm.name = titleAlias
  if (fm.modified === undefined && timestampAlias !== undefined) fm.modified = timestampAlias

  const body = lines
    .slice(closeIdx + 1)
    .join('\n')
    .trim()
  return { frontmatter: fm, body }
}

/**
 * Serialise frontmatter. The default profile is unchanged from prior releases
 * (byte-for-byte); pass `{ profile: 'okf' }` to emit Open Knowledge Format
 * shaped frontmatter instead. See {@link FrontmatterProfile}.
 */
export const buildFrontmatter = (
  fm: Frontmatter,
  opts: { readonly profile?: FrontmatterProfile | undefined } = {},
): string => {
  if (opts.profile === 'okf') return buildOkfFrontmatter(fm)
  const out: string[] = ['---']
  if (fm.name) out.push(`name: ${fm.name}`)
  if (fm.description) out.push(`description: ${fm.description}`)
  if (fm.type) out.push(`type: ${fm.type}`)
  if (fm.scope) out.push(`scope: ${fm.scope}`)
  if (fm.created) out.push(`created: ${fm.created}`)
  if (fm.modified) out.push(`modified: ${fm.modified}`)
  if (fm.confidence) out.push(`confidence: ${fm.confidence}`)
  if (fm.source) out.push(`source: ${fm.source}`)
  if (fm.supersedes) out.push(`supersedes: ${fm.supersedes}`)
  if (fm.superseded_by) out.push(`superseded_by: ${fm.superseded_by}`)
  if (fm.session_id) out.push(`session_id: ${fm.session_id}`)
  if (fm.session_date) out.push(`session_date: ${fm.session_date}`)
  if (fm.observed_on) out.push(`observed_on: ${fm.observed_on}`)
  if (fm.tags && fm.tags.length > 0) out.push(`tags: [${fm.tags.join(', ')}]`)
  for (const [k, v] of Object.entries(fm.extra ?? {})) out.push(`${k}: ${v}`)
  out.push('---', '')
  return out.join('\n')
}

/**
 * Open Knowledge Format projection of {@link Frontmatter}. OKF core fields come
 * first (`type`, `title`, `description`, `tags`, `timestamp`); the native-only
 * fields follow as OKF extension keys (OKF permits additional keys) so the note
 * round-trips back to an equivalent Frontmatter via {@link parseFrontmatter}.
 *
 * `timestamp` is sourced from `modified` (the recency signal recall ranks on),
 * falling back to `created`; `created` is still emitted as an extension key so
 * the original creation time survives the round-trip.
 */
const buildOkfFrontmatter = (fm: Frontmatter): string => {
  const out: string[] = ['---']
  // OKF core.
  if (fm.type) out.push(`type: ${fm.type}`)
  if (fm.name) out.push(`title: ${fm.name}`)
  if (fm.description) out.push(`description: ${fm.description}`)
  if (fm.tags && fm.tags.length > 0) out.push(`tags: [${fm.tags.join(', ')}]`)
  const timestamp = fm.modified ?? fm.created
  if (timestamp) out.push(`timestamp: ${timestamp}`)
  // Native fields preserved as OKF extension keys for lossless round-trips.
  if (fm.created) out.push(`created: ${fm.created}`)
  if (fm.confidence) out.push(`confidence: ${fm.confidence}`)
  if (fm.source) out.push(`source: ${fm.source}`)
  if (fm.supersedes) out.push(`supersedes: ${fm.supersedes}`)
  if (fm.superseded_by) out.push(`superseded_by: ${fm.superseded_by}`)
  if (fm.scope) out.push(`scope: ${fm.scope}`)
  if (fm.session_id) out.push(`session_id: ${fm.session_id}`)
  if (fm.session_date) out.push(`session_date: ${fm.session_date}`)
  if (fm.observed_on) out.push(`observed_on: ${fm.observed_on}`)
  for (const [k, v] of Object.entries(fm.extra ?? {})) out.push(`${k}: ${v}`)
  out.push('---', '')
  return out.join('\n')
}
