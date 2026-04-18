import { describe, expect, it } from 'vitest'
import { detectSource } from './detect.js'
import { loadJsonTranscript } from './json-transcript.js'
import { loadSource } from './index.js'
import type { SourceFetchLike } from './types.js'
import { htmlToMarkdown, loadUrl } from './url.js'

const enc = (s: string): Buffer => Buffer.from(s, 'utf8')

describe('detectSource', () => {
  it('classifies markdown via magic bytes + extension', () => {
    expect(detectSource({ kind: 'bytes', bytes: enc('# hi\n\nbody'), filename: 'x.md' })).toBe('markdown')
  })
  it('classifies plain text', () => {
    expect(detectSource({ kind: 'bytes', bytes: enc('no markers here'), filename: 'x.txt' })).toBe('text')
  })
  it('classifies JSON transcript', () => {
    const payload = Buffer.from(JSON.stringify({ messages: [{ role: 'user', content: 'hi' }] }))
    expect(detectSource({ kind: 'bytes', bytes: payload })).toBe('json-transcript')
  })
  it('classifies PDF by magic bytes', () => {
    expect(detectSource({ kind: 'bytes', bytes: enc('%PDF-1.4\nrest') })).toBe('pdf')
  })
  it('classifies urls', () => {
    expect(detectSource({ kind: 'url', url: 'https://example.com' })).toBe('url')
    expect(detectSource({ kind: 'url', url: 'https://example.com/doc.pdf' })).toBe('pdf')
  })
})

describe('htmlToMarkdown', () => {
  it('strips script/style and keeps headings + paragraphs', () => {
    const html =
      '<html><head><style>body { color: red }</style></head>' +
      '<body><h1>Title</h1><p>First paragraph.</p><script>evil()</script></body></html>'
    const md = htmlToMarkdown(html)
    expect(md).toContain('# Title')
    expect(md).toContain('First paragraph.')
    expect(md).not.toContain('color: red')
    expect(md).not.toContain('evil()')
  })
})

describe('loadUrl', () => {
  it('fetches HTML and preserves the title as a heading', async () => {
    const stubFetch: SourceFetchLike = async () => ({
      ok: true,
      status: 200,
      statusText: 'OK',
      headers: { get: (name: string) => (name.toLowerCase() === 'content-type' ? 'text/html' : null) },
      arrayBuffer: async () =>
        new TextEncoder().encode(
          '<html><head><title>My Page</title></head><body><h2>Sub</h2><p>hello</p></body></html>',
        ).buffer,
      text: async () => '',
    })
    const loaded = await loadUrl('https://example.com/post', { fetch: stubFetch })
    const body = loaded.content.toString('utf8')
    expect(loaded.mime).toBe('text/markdown')
    expect(loaded.title).toBe('My Page')
    expect(body.startsWith('# My Page')).toBe(true)
    expect(body).toContain('hello')
  })
})

describe('loadJsonTranscript', () => {
  it('produces markdown with per-turn headings', async () => {
    const payload = Buffer.from(
      JSON.stringify({
        title: 'Chat',
        messages: [
          { role: 'user', content: 'Hi there' },
          { role: 'assistant', content: 'Hello friend' },
        ],
      }),
    )
    const loaded = await loadJsonTranscript(payload)
    const body = loaded.content.toString('utf8')
    expect(loaded.mime).toBe('text/markdown')
    expect(body).toContain('# Chat')
    expect(body).toContain('## User')
    expect(body).toContain('Hi there')
    expect(body).toContain('## Assistant')
    expect(body).toContain('Hello friend')
  })

  it('rejects non-messages JSON', async () => {
    await expect(loadJsonTranscript(Buffer.from('{"foo":1}'))).rejects.toThrow(/messages/)
  })
})

describe('loadSource dispatch', () => {
  it('routes markdown bytes through the markdown adapter', async () => {
    const loaded = await loadSource({ kind: 'bytes', bytes: enc('# Hello\n\nbody'), filename: 'x.md' })
    expect(loaded.mime).toBe('text/markdown')
  })

  it('routes Office-style binary bytes through markitdown when configured', async () => {
    const calls: Array<{ url: string; auth?: string; filename?: string; type?: string }> = []
    const stubFetch: SourceFetchLike = async (url, init) => {
      if (url.endsWith('/convert-file')) {
        const form = init?.body
        expect(form).toBeInstanceOf(FormData)
        const file = (form as FormData).get('file')
        expect(file).toBeInstanceOf(File)
        calls.push({
          url,
          ...(init?.headers?.authorization !== undefined
            ? { auth: init.headers.authorization }
            : {}),
          ...(file instanceof File ? { filename: file.name, type: file.type } : {}),
        })
        return {
          ok: true,
          status: 200,
          statusText: 'OK',
          headers: { get: () => 'application/json' },
          arrayBuffer: async () =>
            new TextEncoder().encode(
              JSON.stringify({
                markdown: '# Quarterly Report\n\nRevenue is up.',
                metadata: { source: 'uploaded_file' },
              }),
            ).buffer,
          text: async () =>
            JSON.stringify({
              markdown: '# Quarterly Report\n\nRevenue is up.',
              metadata: { source: 'uploaded_file' },
            }),
        }
      }
      throw new Error(`unexpected fetch ${url}`)
    }
    const loaded = await loadSource(
      {
        kind: 'bytes',
        bytes: Buffer.from('PK\x03\x04docx-binary', 'binary'),
        filename: 'report.docx',
        mime: 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
      },
      {
        fetch: stubFetch,
        markitdown: {
          url: 'http://markitdown.local',
          bearerToken: 'secret-token',
        },
      },
    )
    expect(loaded.mime).toBe('text/markdown')
    expect(loaded.content.toString('utf8')).toContain('Revenue is up.')
    expect(loaded.meta).toMatchObject({
      converted_by: 'markitdown-service',
      original_filename: 'report.docx',
    })
    expect(calls).toEqual([
      {
        url: 'http://markitdown.local/convert-file',
        auth: 'Bearer secret-token',
        filename: 'report.docx',
        type: 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
      },
    ])
  })

  it('bypasses markitdown for markdown bytes even when configured', async () => {
    const stubFetch: SourceFetchLike = async (url) => {
      throw new Error(`unexpected fetch ${url}`)
    }
    const loaded = await loadSource(
      { kind: 'bytes', bytes: enc('# Hello\n\nbody'), filename: 'x.md', mime: 'text/markdown' },
      {
        fetch: stubFetch,
        markitdown: {
          url: 'http://markitdown.local',
          bearerToken: 'secret-token',
        },
      },
    )
    expect(loaded.mime).toBe('text/markdown')
    expect(loaded.content.toString('utf8')).toBe('# Hello\n\nbody')
  })

  it('routes binary URL content back through markitdown-aware byte loading', async () => {
    const calls: string[] = []
    const officeBytes = Buffer.from('PK\x03\x04url-docx', 'binary')
    const stubFetch: SourceFetchLike = async (url, init) => {
      calls.push(url)
      if (url === 'https://example.com/report.docx') {
        return {
          ok: true,
          status: 200,
          statusText: 'OK',
          headers: {
            get(name: string) {
              if (name.toLowerCase() === 'content-type') {
                return 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
              }
              return null
            },
          },
          arrayBuffer: async () => officeBytes.buffer.slice(
            officeBytes.byteOffset,
            officeBytes.byteOffset + officeBytes.byteLength,
          ),
          text: async () => '',
        }
      }
      if (url === 'http://markitdown.local/convert-file') {
        expect(init?.headers?.authorization).toBe('Bearer secret-token')
        return {
          ok: true,
          status: 200,
          statusText: 'OK',
          headers: { get: () => 'application/json' },
          arrayBuffer: async () =>
            new TextEncoder().encode(
              JSON.stringify({ markdown: '# URL Report\n\nFetched through markitdown.' }),
            ).buffer,
          text: async () => JSON.stringify({ markdown: '# URL Report\n\nFetched through markitdown.' }),
        }
      }
      throw new Error(`unexpected fetch ${url}`)
    }
    const loaded = await loadSource(
      { kind: 'url', url: 'https://example.com/report.docx' },
      {
        fetch: stubFetch,
        markitdown: {
          url: 'http://markitdown.local',
          bearerToken: 'secret-token',
        },
      },
    )
    expect(loaded.mime).toBe('text/markdown')
    expect(loaded.content.toString('utf8')).toContain('Fetched through markitdown.')
    expect(loaded.meta).toMatchObject({
      url: 'https://example.com/report.docx',
      sourceMime: 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    })
    expect(calls).toEqual([
      'https://example.com/report.docx',
      'http://markitdown.local/convert-file',
    ])
  })

  it('routes URL inputs through loadUrl', async () => {
    const stubFetch: SourceFetchLike = async () => ({
      ok: true,
      status: 200,
      statusText: 'OK',
      headers: { get: () => 'text/html' },
      arrayBuffer: async () => new TextEncoder().encode('<html><body><p>hi</p></body></html>').buffer,
      text: async () => '',
    })
    const loaded = await loadSource({ kind: 'url', url: 'https://x.com' }, { fetch: stubFetch })
    expect(loaded.mime).toBe('text/markdown')
  })
})
