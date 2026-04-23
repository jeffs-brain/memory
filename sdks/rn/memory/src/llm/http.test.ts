import { describe, expect, it } from 'vitest'

import { TransportError } from './errors.js'
import type { HttpClient } from './http.js'
import { postForText } from './http.js'

describe('http helpers', () => {
  it('times out stalled POST requests', async () => {
    const client: HttpClient = {
      fetch: async (_input, init) =>
        await new Promise<Response>((_resolve, reject) => {
          init?.signal?.addEventListener(
            'abort',
            () => reject(init.signal?.reason ?? new Error('aborted')),
            { once: true },
          )
        }),
    }

    await expect(
      postForText(
        client,
        'https://example.test/v1/chat/completions',
        { ok: true },
        { timeoutMs: 5 },
      ),
    ).rejects.toBeInstanceOf(TransportError)
  })
})
