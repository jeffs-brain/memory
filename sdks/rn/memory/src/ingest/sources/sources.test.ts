import { describe, expect, it } from 'vitest'

import { detectSource } from './detect.js'
import { loadSource } from './index.js'
import { loadJsonTranscript } from './json-transcript.js'
import type { SourceFetchLike } from './types.js'
import { htmlToMarkdown, loadUrl } from './url.js'

const encode = (value: string): Uint8Array => new TextEncoder().encode(value)

describe('detectSource', () => {
  it('classifies markdown and JSON transcript payloads', () => {
    expect(detectSource({ kind: 'bytes', bytes: encode('# hi\n\nbody'), filename: 'x.md' })).toBe(
      'markdown',
    )
    expect(
      detectSource({
        kind: 'bytes',
        bytes: encode(JSON.stringify({ messages: [{ role: 'user', content: 'hi' }] })),
      }),
    ).toBe('json-transcript')
  })
})

describe('htmlToMarkdown', () => {
  it('strips script blocks and preserves useful text', () => {
    const html =
      '<html><head><style>body { color: red }</style></head>' +
      '<body><h1>Title</h1><p>First paragraph.</p><script>evil()</script></body></html>'
    const markdown = htmlToMarkdown(html)
    expect(markdown).toContain('# Title')
    expect(markdown).toContain('First paragraph.')
    expect(markdown).not.toContain('evil()')
  })
})

describe('URL and source loading', () => {
  it('fetches HTML sources and preserves the title heading', async () => {
    const fetch: SourceFetchLike = async () => ({
      ok: true,
      status: 200,
      statusText: 'OK',
      body: null,
      headers: {
        get: (name: string) => (name.toLowerCase() === 'content-type' ? 'text/html' : null),
      },
      arrayBuffer: async () =>
        encode(
          '<html><head><title>My Page</title></head><body><h2>Sub</h2><p>hello</p></body></html>',
        ).buffer,
      text: async () => '',
    })

    const loaded = await loadUrl('https://example.com/post', { fetch })
    expect(loaded.mime).toBe('text/markdown')
    expect(loaded.title).toBe('My Page')
    expect(loaded.content.startsWith('# My Page')).toBe(true)
    expect(loaded.content).toContain('hello')
  })

  it('loads transcripts and routes markdown through the source dispatcher', async () => {
    const transcript = await loadJsonTranscript(
      encode(
        JSON.stringify({
          title: 'Chat',
          messages: [
            { role: 'user', content: 'Hi there' },
            { role: 'assistant', content: 'Hello friend' },
          ],
        }),
      ),
    )
    expect(transcript.mime).toBe('text/markdown')
    expect(transcript.content).toContain('## User')
    expect(transcript.content).toContain('## Assistant')

    const markdown = await loadSource({
      kind: 'bytes',
      bytes: encode('# Hello\n\nbody'),
      filename: 'x.md',
    })
    expect(markdown.mime).toBe('text/markdown')
    expect(markdown.content).toBe('# Hello\n\nbody')
  })
})
