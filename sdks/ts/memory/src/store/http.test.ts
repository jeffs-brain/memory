// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it, vi } from 'vitest'
import { ErrPayloadTooLarge } from './errors.js'
import { createHttpStore } from './http.js'
import { toPath } from './path.js'

describe('HttpStore', () => {
  it('rejects oversized batches before sending them', async () => {
    const fetch = vi.fn(async () => new Response(null, { status: 200 }))
    const store = createHttpStore({
      baseUrl: 'https://example.test',
      brainId: 'brain-a',
      fetch,
      bodyLimits: {
        batchDecodedBytes: 4,
        batchOpCount: 2,
      },
    })

    await expect(
      store.batch({ reason: 'test' }, async (batch) => {
        await batch.write(toPath('docs/one.txt'), Buffer.from('12345'))
      }),
    ).rejects.toBeInstanceOf(ErrPayloadTooLarge)
    expect(fetch).not.toHaveBeenCalled()
  })

  it('honours configured higher batch ceilings client-side', async () => {
    const fetch = vi.fn(async () => new Response(null, { status: 200 }))
    const store = createHttpStore({
      baseUrl: 'https://example.test',
      brainId: 'brain-a',
      fetch,
      bodyLimits: {
        batchDecodedBytes: 16,
      },
    })

    await store.batch({ reason: 'test' }, async (batch) => {
      await batch.write(toPath('docs/one.txt'), Buffer.from('1234567890'))
    })

    expect(fetch).toHaveBeenCalledTimes(1)
  })

  it('maps 413 responses to ErrPayloadTooLarge', async () => {
    const fetch = vi.fn(
      async () =>
        new Response(
          JSON.stringify({
            status: 413,
            title: 'Payload Too Large',
            code: 'payload_too_large',
            detail: 'batch payload exceeds 8 bytes after decode',
          }),
          {
            status: 413,
            headers: { 'content-type': 'application/problem+json' },
          },
        ),
    )
    const store = createHttpStore({
      baseUrl: 'https://example.test',
      brainId: 'brain-a',
      fetch,
    })

    await expect(store.write(toPath('docs/one.txt'), Buffer.from('ok'))).rejects.toBeInstanceOf(
      ErrPayloadTooLarge,
    )
  })
})
