/**
 * Markdown-aware + plain-text chunkers used by the ingest pipeline. The
 * token count is an approximation (chars / 4) — good enough to keep chunks
 * bounded; swap in a real tokeniser via `countTokens` later if needed.
 *
 * The markdown splitter tracks heading hierarchy so each chunk carries a
 * `headingPath` that mirrors the ATX / setext boundary tree. Sections
 * larger than `maxTokens` are broken into overlapping windows at paragraph
 * boundaries where possible; paragraphs that already fit are never split
 * mid-paragraph.
 */

export type Chunk = {
  readonly content: string
  readonly ordinal: number
  readonly headingPath: readonly string[]
  readonly startLine: number
  readonly endLine: number
  readonly tokens: number
}

export type ChunkOptions = {
  readonly maxTokens?: number
  readonly overlapTokens?: number
}

const DEFAULT_MAX_TOKENS = 512
const DEFAULT_OVERLAP_TOKENS = 64
const MIN_MAX_TOKENS = 16

/** Rough token approximation. Monotonic in text length. Swap for a real
 * tokeniser if budget enforcement needs to be tight. */
export const countTokens = (text: string): number =>
  text.length === 0 ? 0 : Math.ceil(text.length / 4)

type HeadingLine = {
  readonly line: number
  readonly level: number
  readonly title: string
}

type Section = {
  readonly headingPath: readonly string[]
  readonly startLine: number
  readonly endLine: number
  readonly content: string
}

const normaliseOpts = (opts: ChunkOptions | undefined): Required<ChunkOptions> => {
  const maxTokens = Math.max(MIN_MAX_TOKENS, opts?.maxTokens ?? DEFAULT_MAX_TOKENS)
  const rawOverlap = opts?.overlapTokens ?? DEFAULT_OVERLAP_TOKENS
  const overlapTokens = Math.max(0, Math.min(rawOverlap, Math.floor(maxTokens / 2)))
  return { maxTokens, overlapTokens }
}

/**
 * Detect ATX (`# Heading`) + setext (underline-style) headings. Returns
 * them in input order together with their resolved level.
 */
const findHeadings = (lines: readonly string[]): readonly HeadingLine[] => {
  const out: HeadingLine[] = []
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i] ?? ''
    const atx = /^(#{1,6})\s+(.+?)\s*#*\s*$/.exec(line)
    if (atx !== null) {
      const hashes = atx[1] ?? ''
      const title = (atx[2] ?? '').trim()
      out.push({ line: i, level: hashes.length, title })
      continue
    }
    // Setext: the heading title is on line i, underline on line i+1.
    const next = lines[i + 1]
    if (
      next !== undefined &&
      line.trim() !== '' &&
      /^[=-]{2,}\s*$/.test(next)
    ) {
      const level = next.trimEnd().startsWith('=') ? 1 : 2
      out.push({ line: i, level, title: line.trim() })
    }
  }
  return out
}

/**
 * Given the parsed heading list, split the input into sections keyed by
 * the cumulative heading path at each point. A section runs from the
 * heading line until the next heading of equal or shallower depth.
 */
const splitSections = (lines: readonly string[], headings: readonly HeadingLine[]): readonly Section[] => {
  if (headings.length === 0) {
    return [
      {
        headingPath: [],
        startLine: 0,
        endLine: Math.max(0, lines.length - 1),
        content: lines.join('\n'),
      },
    ]
  }

  const sections: Section[] = []
  // Preamble before the first heading, when present.
  const first = headings[0]
  if (first !== undefined && first.line > 0) {
    const preambleLines = lines.slice(0, first.line)
    if (preambleLines.some((l) => l.trim() !== '')) {
      sections.push({
        headingPath: [],
        startLine: 0,
        endLine: first.line - 1,
        content: preambleLines.join('\n'),
      })
    }
  }

  const stack: HeadingLine[] = []
  for (let i = 0; i < headings.length; i++) {
    const h = headings[i]
    if (h === undefined) continue
    while (stack.length > 0 && (stack[stack.length - 1]?.level ?? 0) >= h.level) {
      stack.pop()
    }
    stack.push(h)
    const next = headings[i + 1]
    // Setext headings span two lines; include both in the content so the
    // chunker renders the heading + underline intact.
    const endExclusive =
      next?.line ?? lines.length
    const contentLines = lines.slice(h.line, endExclusive)
    sections.push({
      headingPath: stack.map((s) => s.title),
      startLine: h.line,
      endLine: endExclusive - 1,
      content: contentLines.join('\n'),
    })
  }
  return sections
}

/**
 * Split a section's text into paragraphs — blocks separated by blank lines.
 * Returns paragraph strings preserved verbatim (without trailing blank).
 */
const splitParagraphs = (text: string): readonly string[] => {
  if (text === '') return []
  const paras: string[] = []
  let current: string[] = []
  for (const line of text.split('\n')) {
    if (line.trim() === '') {
      if (current.length > 0) {
        paras.push(current.join('\n'))
        current = []
      }
    } else {
      current.push(line)
    }
  }
  if (current.length > 0) paras.push(current.join('\n'))
  return paras
}

/**
 * Window a section that overflows `maxTokens` into overlapping paragraph
 * packs. A paragraph that is itself larger than `maxTokens` is hard-split
 * by character count — rare but unavoidable.
 */
const windowSection = (
  section: Section,
  max: number,
  overlap: number,
): readonly { content: string; startLine: number; endLine: number }[] => {
  const paras = splitParagraphs(section.content)
  if (paras.length === 0) {
    return [
      {
        content: section.content,
        startLine: section.startLine,
        endLine: section.endLine,
      },
    ]
  }

  // Assign each paragraph a rough line span relative to the section so we
  // can report startLine/endLine per window. Maintain a running index.
  const withLines: { content: string; startLine: number; endLine: number; tokens: number }[] = []
  let cursor = section.startLine
  const lines = section.content.split('\n')
  let lineIdx = 0
  for (const para of paras) {
    // Skip blank prefix lines.
    while (lineIdx < lines.length && (lines[lineIdx] ?? '').trim() === '') {
      lineIdx++
      cursor++
    }
    const paraLineCount = para.split('\n').length
    const start = cursor
    const end = cursor + paraLineCount - 1
    withLines.push({
      content: para,
      startLine: start,
      endLine: end,
      tokens: countTokens(para),
    })
    lineIdx += paraLineCount
    cursor += paraLineCount
  }

  const out: { content: string; startLine: number; endLine: number }[] = []
  let i = 0
  while (i < withLines.length) {
    const batch: typeof withLines = []
    let tokenCount = 0
    let j = i
    while (j < withLines.length) {
      const p = withLines[j]
      if (p === undefined) break
      if (p.tokens > max) {
        // Hard-split oversized paragraph by character count.
        if (batch.length === 0) {
          const step = max * 4
          const stride = Math.max(1, step - overlap * 4)
          let k = 0
          while (k < p.content.length) {
            const slice = p.content.slice(k, k + step)
            out.push({ content: slice, startLine: p.startLine, endLine: p.endLine })
            k += stride
            if (k + (step - stride) >= p.content.length) break
          }
          j++
          i = j
          tokenCount = 0
          break
        }
        break
      }
      if (tokenCount + p.tokens > max && batch.length > 0) break
      batch.push(p)
      tokenCount += p.tokens
      j++
    }
    if (batch.length === 0) continue
    const first = batch[0]
    const last = batch[batch.length - 1]
    if (first === undefined || last === undefined) {
      i = j
      continue
    }
    out.push({
      content: batch.map((p) => p.content).join('\n\n'),
      startLine: first.startLine,
      endLine: last.endLine,
    })
    if (j >= withLines.length) break
    // Advance with overlap: walk back from j while the trailing tokens fit
    // the overlap budget so the next window's prefix repeats paragraphs.
    if (overlap === 0 || batch.length === 1) {
      i = j
      continue
    }
    let overlapTokens = 0
    let back = batch.length - 1
    while (back > 0 && overlapTokens + (batch[back]?.tokens ?? 0) <= overlap) {
      overlapTokens += batch[back]?.tokens ?? 0
      back--
    }
    const overlapCount = batch.length - 1 - back
    i = j - overlapCount
    if (i <= 0) i = j
  }
  return out
}

const renderHeadingPrefix = (path: readonly string[]): string => {
  if (path.length === 0) return ''
  return path.map((title, idx) => `${'#'.repeat(idx + 1)} ${title}`).join('\n')
}

export const chunkMarkdown = (text: string, opts?: ChunkOptions): readonly Chunk[] => {
  const { maxTokens, overlapTokens } = normaliseOpts(opts)
  if (text.trim() === '') return []
  const lines = text.split('\n')
  const headings = findHeadings(lines)
  const sections = splitSections(lines, headings)

  const chunks: Chunk[] = []
  let ordinal = 0
  for (const section of sections) {
    const trimmed = section.content.trim()
    if (trimmed === '') continue
    const tokens = countTokens(section.content)
    if (tokens <= maxTokens) {
      chunks.push({
        content: section.content,
        ordinal: ordinal++,
        headingPath: section.headingPath,
        startLine: section.startLine,
        endLine: section.endLine,
        tokens,
      })
      continue
    }
    const windows = windowSection(section, maxTokens, overlapTokens)
    for (const w of windows) {
      // Re-attach the heading prefix so downstream BM25 picks up the
      // canonical title even in overflow windows. The first window already
      // starts with the heading so we avoid duplicating it there.
      const contentWithHeading =
        w.startLine === section.startLine
          ? w.content
          : `${renderHeadingPrefix(section.headingPath)}\n\n${w.content}`
      chunks.push({
        content: contentWithHeading,
        ordinal: ordinal++,
        headingPath: section.headingPath,
        startLine: w.startLine,
        endLine: w.endLine,
        tokens: countTokens(contentWithHeading),
      })
    }
  }
  return chunks
}

export const chunkPlainText = (text: string, opts?: ChunkOptions): readonly Chunk[] => {
  const { maxTokens, overlapTokens } = normaliseOpts(opts)
  if (text.trim() === '') return []
  const window = maxTokens * 4
  const stride = Math.max(1, window - overlapTokens * 4)
  const chunks: Chunk[] = []
  let ordinal = 0
  let idx = 0
  // Track line offsets for start/end line reporting.
  const lineStarts: number[] = [0]
  for (let i = 0; i < text.length; i++) {
    if (text[i] === '\n') lineStarts.push(i + 1)
  }
  const lineForOffset = (offset: number): number => {
    // Binary search for largest lineStart <= offset.
    let lo = 0
    let hi = lineStarts.length - 1
    while (lo < hi) {
      const mid = (lo + hi + 1) >> 1
      if ((lineStarts[mid] ?? 0) <= offset) lo = mid
      else hi = mid - 1
    }
    return lo
  }
  while (idx < text.length) {
    const slice = text.slice(idx, idx + window)
    const startLine = lineForOffset(idx)
    const endLine = lineForOffset(Math.min(text.length, idx + window) - 1)
    chunks.push({
      content: slice,
      ordinal: ordinal++,
      headingPath: [],
      startLine,
      endLine,
      tokens: countTokens(slice),
    })
    if (idx + window >= text.length) break
    idx += stride
  }
  return chunks
}

/**
 * Detect markdown vs plain. The heuristic is deliberately loose: any ATX
 * heading, blockquote marker, or list line indicates markdown. Falls back
 * to plain when none of those appear.
 */
export const chunkAuto = (text: string, opts?: ChunkOptions): readonly Chunk[] => {
  if (looksLikeMarkdown(text)) return chunkMarkdown(text, opts)
  return chunkPlainText(text, opts)
}

export const looksLikeMarkdown = (text: string): boolean => {
  if (text === '') return false
  if (/^#{1,6}\s+\S/m.test(text)) return true
  if (/^>\s+\S/m.test(text)) return true
  if (/^\s*[-*+]\s+\S/m.test(text)) return true
  if (/^\s*\d+\.\s+\S/m.test(text)) return true
  return false
}
