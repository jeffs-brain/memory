import { describe, expect, it } from 'vitest'

import { loadUrl } from './url.js'

describe('loadUrl', () => {
  it('rejects oversized responses from content-length before reading the body', async () => {
    const fetch = async () =>
      new Response(
        new ReadableStream<Uint8Array>({
          pull(controller) {
            controller.enqueue(new TextEncoder().encode('body'))
            controller.close()
          },
        }),
        {
          status: 200,
          headers: {
            'content-type': 'text/plain',
            'content-length': '100',
          },
        },
      )

    await expect(loadUrl('https://example.test/file.txt', { fetch, maxBytes: 8 })).rejects.toThrow(
      /exceeded maxBytes=8/,
    )
  })

  it('stops streaming once the configured byte cap is exceeded', async () => {
    const fetch = async () =>
      new Response(
        new ReadableStream<Uint8Array>({
          start(controller) {
            controller.enqueue(new TextEncoder().encode('1234'))
            controller.enqueue(new TextEncoder().encode('5678'))
            controller.close()
          },
        }),
        {
          status: 200,
          headers: {
            'content-type': 'text/plain',
          },
        },
      )

    await expect(loadUrl('https://example.test/file.txt', { fetch, maxBytes: 6 })).rejects.toThrow(
      /exceeded maxBytes=6/,
    )
  })
})
