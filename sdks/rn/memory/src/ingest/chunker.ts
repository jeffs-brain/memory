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

export const countTokens = (text: string): number => (text === '' ? 0 : Math.ceil(text.length / 4))

const normaliseOptions = (options: ChunkOptions | undefined): Required<ChunkOptions> => {
  const maxTokens = Math.max(MIN_MAX_TOKENS, options?.maxTokens ?? DEFAULT_MAX_TOKENS)
  const rawOverlap = options?.overlapTokens ?? DEFAULT_OVERLAP_TOKENS
  const overlapTokens = Math.max(0, Math.min(rawOverlap, Math.floor(maxTokens / 2)))
  return { maxTokens, overlapTokens }
}

const findHeadings = (lines: readonly string[]): readonly HeadingLine[] => {
  const headings: HeadingLine[] = []
  for (let index = 0; index < lines.length; index += 1) {
    const line = lines[index] ?? ''
    const atx = /^(#{1,6})\s+(.+?)\s*#*\s*$/.exec(line)
    if (atx !== null) {
      headings.push({
        line: index,
        level: (atx[1] ?? '').length,
        title: (atx[2] ?? '').trim(),
      })
      continue
    }

    const next = lines[index + 1]
    if (next !== undefined && line.trim() !== '' && /^[=-]{2,}\s*$/.test(next)) {
      headings.push({
        line: index,
        level: next.trimEnd().startsWith('=') ? 1 : 2,
        title: line.trim(),
      })
    }
  }
  return headings
}

const splitSections = (
  lines: readonly string[],
  headings: readonly HeadingLine[],
): readonly Section[] => {
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
  const first = headings[0]
  if (first !== undefined && first.line > 0) {
    const preamble = lines.slice(0, first.line)
    if (preamble.some((line) => line.trim() !== '')) {
      sections.push({
        headingPath: [],
        startLine: 0,
        endLine: first.line - 1,
        content: preamble.join('\n'),
      })
    }
  }

  const stack: HeadingLine[] = []
  for (let index = 0; index < headings.length; index += 1) {
    const heading = headings[index]
    if (heading === undefined) continue
    while (stack.length > 0 && (stack[stack.length - 1]?.level ?? 0) >= heading.level) {
      stack.pop()
    }
    stack.push(heading)
    const next = headings[index + 1]
    const endExclusive = next?.line ?? lines.length
    sections.push({
      headingPath: stack.map((item) => item.title),
      startLine: heading.line,
      endLine: endExclusive - 1,
      content: lines.slice(heading.line, endExclusive).join('\n'),
    })
  }

  return sections
}

const splitParagraphs = (text: string): readonly string[] => {
  if (text === '') return []
  const paragraphs: string[] = []
  let current: string[] = []
  for (const line of text.split('\n')) {
    if (line.trim() === '') {
      if (current.length > 0) {
        paragraphs.push(current.join('\n'))
        current = []
      }
      continue
    }
    current.push(line)
  }
  if (current.length > 0) paragraphs.push(current.join('\n'))
  return paragraphs
}

const windowSection = (
  section: Section,
  maxTokens: number,
  overlapTokens: number,
): readonly {
  readonly content: string
  readonly startLine: number
  readonly endLine: number
}[] => {
  const paragraphs = splitParagraphs(section.content)
  if (paragraphs.length === 0) {
    return [
      {
        content: section.content,
        startLine: section.startLine,
        endLine: section.endLine,
      },
    ]
  }

  const lines = section.content.split('\n')
  const rangedParagraphs: Array<{
    readonly content: string
    readonly startLine: number
    readonly endLine: number
    readonly tokens: number
  }> = []
  let cursor = section.startLine
  let lineIndex = 0

  for (const paragraph of paragraphs) {
    while (lineIndex < lines.length && (lines[lineIndex] ?? '').trim() === '') {
      lineIndex += 1
      cursor += 1
    }
    const paragraphLineCount = paragraph.split('\n').length
    rangedParagraphs.push({
      content: paragraph,
      startLine: cursor,
      endLine: cursor + paragraphLineCount - 1,
      tokens: countTokens(paragraph),
    })
    cursor += paragraphLineCount
    lineIndex += paragraphLineCount
  }

  const windows: Array<{
    readonly content: string
    readonly startLine: number
    readonly endLine: number
  }> = []
  let start = 0

  while (start < rangedParagraphs.length) {
    const batch: typeof rangedParagraphs = []
    let tokenCount = 0
    let end = start

    while (end < rangedParagraphs.length) {
      const paragraph = rangedParagraphs[end]
      if (paragraph === undefined) break
      if (paragraph.tokens > maxTokens) {
        if (batch.length === 0) {
          const step = maxTokens * 4
          const stride = Math.max(1, step - overlapTokens * 4)
          let offset = 0
          while (offset < paragraph.content.length) {
            const slice = paragraph.content.slice(offset, offset + step)
            windows.push({
              content: slice,
              startLine: paragraph.startLine,
              endLine: paragraph.endLine,
            })
            offset += stride
            if (offset + (step - stride) >= paragraph.content.length) break
          }
          end += 1
          start = end
          tokenCount = 0
          break
        }
        break
      }
      if (tokenCount + paragraph.tokens > maxTokens && batch.length > 0) break
      batch.push(paragraph)
      tokenCount += paragraph.tokens
      end += 1
    }

    if (batch.length === 0) continue

    const first = batch[0]
    const last = batch[batch.length - 1]
    if (first === undefined || last === undefined) {
      start = end
      continue
    }

    windows.push({
      content: batch.map((paragraph) => paragraph.content).join('\n\n'),
      startLine: first.startLine,
      endLine: last.endLine,
    })

    if (end >= rangedParagraphs.length) break
    if (overlapTokens === 0 || batch.length === 1) {
      start = end
      continue
    }

    let carried = 0
    let back = batch.length - 1
    while (back > 0 && carried + (batch[back]?.tokens ?? 0) <= overlapTokens) {
      carried += batch[back]?.tokens ?? 0
      back -= 1
    }
    const overlapCount = batch.length - 1 - back
    start = end - overlapCount
    if (start <= 0) start = end
  }

  return windows
}

const renderHeadingPrefix = (headingPath: readonly string[]): string => {
  if (headingPath.length === 0) return ''
  return headingPath.map((title, index) => `${'#'.repeat(index + 1)} ${title}`).join('\n')
}

export const chunkMarkdown = (text: string, options?: ChunkOptions): readonly Chunk[] => {
  const { maxTokens, overlapTokens } = normaliseOptions(options)
  if (text.trim() === '') return []

  const lines = text.split('\n')
  const headings = findHeadings(lines)
  const sections = splitSections(lines, headings)
  const chunks: Chunk[] = []
  let ordinal = 0

  for (const section of sections) {
    if (section.content.trim() === '') continue
    const tokens = countTokens(section.content)
    if (tokens <= maxTokens) {
      chunks.push({
        content: section.content,
        ordinal,
        headingPath: section.headingPath,
        startLine: section.startLine,
        endLine: section.endLine,
        tokens,
      })
      ordinal += 1
      continue
    }

    const windows = windowSection(section, maxTokens, overlapTokens)
    for (const window of windows) {
      const content =
        window.startLine === section.startLine
          ? window.content
          : `${renderHeadingPrefix(section.headingPath)}\n\n${window.content}`
      chunks.push({
        content,
        ordinal,
        headingPath: section.headingPath,
        startLine: window.startLine,
        endLine: window.endLine,
        tokens: countTokens(content),
      })
      ordinal += 1
    }
  }

  return chunks
}

export const chunkPlainText = (text: string, options?: ChunkOptions): readonly Chunk[] => {
  const { maxTokens, overlapTokens } = normaliseOptions(options)
  if (text.trim() === '') return []

  const window = maxTokens * 4
  const stride = Math.max(1, window - overlapTokens * 4)
  const chunks: Chunk[] = []
  const lineStarts = [0]
  for (let index = 0; index < text.length; index += 1) {
    if (text[index] === '\n') {
      lineStarts.push(index + 1)
    }
  }

  const lineForOffset = (offset: number): number => {
    let low = 0
    let high = lineStarts.length - 1
    while (low < high) {
      const mid = (low + high + 1) >> 1
      if ((lineStarts[mid] ?? 0) <= offset) low = mid
      else high = mid - 1
    }
    return low
  }

  let ordinal = 0
  let offset = 0
  while (offset < text.length) {
    const slice = text.slice(offset, offset + window)
    chunks.push({
      content: slice,
      ordinal,
      headingPath: [],
      startLine: lineForOffset(offset),
      endLine: lineForOffset(Math.min(text.length, offset + window) - 1),
      tokens: countTokens(slice),
    })
    ordinal += 1
    if (offset + window >= text.length) break
    offset += stride
  }

  return chunks
}

export const looksLikeMarkdown = (text: string): boolean => {
  if (text === '') return false
  if (/^#{1,6}\s+\S/m.test(text)) return true
  if (/^>\s+\S/m.test(text)) return true
  if (/^\s*[-*+]\s+\S/m.test(text)) return true
  if (/^\s*\d+\.\s+\S/m.test(text)) return true
  return false
}

export const chunkAuto = (text: string, options?: ChunkOptions): readonly Chunk[] => {
  return looksLikeMarkdown(text) ? chunkMarkdown(text, options) : chunkPlainText(text, options)
}
