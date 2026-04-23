export type SourceFetchLike = (input: string, init?: RequestInit) => Promise<Response>

export type UrlSource = {
  readonly content: string
  readonly mime: string
  readonly title?: string
  readonly filename?: string
  readonly url: string
  readonly metadata?: Readonly<Record<string, unknown>>
}

export type LoadUrlOptions = {
  readonly fetch?: SourceFetchLike
  readonly signal?: AbortSignal
  readonly maxBytes?: number
  readonly title?: string
}

const DEFAULT_MAX_BYTES = 10 * 1024 * 1024
const DROP_BLOCKS = ['script', 'style', 'nav', 'footer', 'aside', 'svg', 'noscript']

const defaultFetch = (): SourceFetchLike => {
  if (typeof globalThis.fetch === 'function') {
    return globalThis.fetch.bind(globalThis) as SourceFetchLike
  }
  throw new Error('loadUrl: no global fetch available; pass opts.fetch')
}

const stripBlock = (html: string, tag: string): string => {
  const expression = new RegExp(`<${tag}[\\s\\S]*?<\\/${tag}>`, 'gi')
  return html.replace(expression, '')
}

const stripTags = (html: string): string => html.replace(/<[^>]+>/g, '')

const decodeEntities = (text: string): string =>
  text
    .replace(/&nbsp;/gi, ' ')
    .replace(/&amp;/gi, '&')
    .replace(/&lt;/gi, '<')
    .replace(/&gt;/gi, '>')
    .replace(/&quot;/gi, '"')
    .replace(/&#39;/gi, "'")
    .replace(/&#(\d+);/g, (_match, code) => String.fromCodePoint(Number(code)))

const extractTitle = (html: string): string | undefined => {
  const match = /<title[^>]*>([\s\S]*?)<\/title>/i.exec(html)
  if (match === null) return undefined
  const raw = (match[1] ?? '').trim()
  return raw === '' ? undefined : decodeEntities(raw)
}

const filenameFromUrl = (url: string): string | undefined => {
  try {
    const pathname = new URL(url).pathname
    const filename = pathname.split('/').pop()
    return filename === undefined || filename === '' ? undefined : filename
  } catch {
    return undefined
  }
}

export const htmlToMarkdown = (html: string): string => {
  let out = html
  for (const tag of DROP_BLOCKS) {
    out = stripBlock(out, tag)
  }
  out = out.replace(/<h([1-6])[^>]*>([\s\S]*?)<\/h\1>/gi, (_match, level, inner) => {
    return `\n\n${'#'.repeat(Number(level))} ${stripTags(String(inner)).trim()}\n\n`
  })
  out = out.replace(/<li[^>]*>([\s\S]*?)<\/li>/gi, (_match, inner) => {
    return `\n- ${stripTags(String(inner)).trim()}`
  })
  out = out.replace(/<\/(ul|ol)>/gi, '\n')
  out = out.replace(/<blockquote[^>]*>([\s\S]*?)<\/blockquote>/gi, (_match, inner) => {
    const body = stripTags(String(inner)).trim()
    return body
      .split(/\n+/)
      .map((line) => `> ${line}`)
      .join('\n')
  })
  out = out.replace(/<pre[^>]*>([\s\S]*?)<\/pre>/gi, (_match, inner) => {
    return `\n\n\`\`\`\n${stripTags(String(inner)).trim()}\n\`\`\`\n\n`
  })
  out = out.replace(/<code[^>]*>([\s\S]*?)<\/code>/gi, (_match, inner) => {
    return `\`${stripTags(String(inner)).trim()}\``
  })
  out = out.replace(/<br\s*\/?>(\s*)/gi, '\n')
  out = out.replace(/<\/p>/gi, '\n\n')
  out = out.replace(/<p[^>]*>/gi, '')
  out = decodeEntities(stripTags(out))
  return out
    .replace(/\n{3,}/g, '\n\n')
    .split('\n')
    .map((line) => line.replace(/[\t ]+/g, ' ').replace(/\s+$/, ''))
    .join('\n')
    .trim()
}

export const loadUrl = async (url: string, options: LoadUrlOptions = {}): Promise<UrlSource> => {
  const fetchImpl = options.fetch ?? defaultFetch()
  const response = await fetchImpl(
    url,
    options.signal === undefined ? undefined : { signal: options.signal },
  )
  if (!response.ok) {
    throw new Error(`loadUrl: ${url} failed with ${response.status} ${response.statusText}`)
  }

  const arrayBuffer = await response.arrayBuffer()
  if (arrayBuffer.byteLength > (options.maxBytes ?? DEFAULT_MAX_BYTES)) {
    throw new Error(`loadUrl: ${url} exceeded maxBytes=${options.maxBytes ?? DEFAULT_MAX_BYTES}`)
  }

  const mime = response.headers.get('content-type')?.split(';')[0]?.trim() || 'text/plain'
  const text = new TextDecoder('utf-8').decode(arrayBuffer)
  const isHtml = mime.includes('html') || /<html[\s>]/i.test(text.slice(0, 512))
  const title = isHtml ? extractTitle(text) : undefined
  const filename = filenameFromUrl(url)

  if (!isHtml) {
    return {
      content: text,
      mime,
      ...(options.title !== undefined ? { title: options.title } : {}),
      ...(filename === undefined ? {} : { filename }),
      url,
    }
  }

  const derivedTitle = title ?? options.title
  const body = htmlToMarkdown(text)
  return {
    content:
      derivedTitle !== undefined && derivedTitle !== '' ? `# ${derivedTitle}\n\n${body}` : body,
    mime: 'text/markdown',
    ...(derivedTitle === undefined ? {} : { title: derivedTitle }),
    ...(filename === undefined ? {} : { filename }),
    url,
    metadata: {
      sourceMime: mime,
    },
  }
}
