// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from 'vitest'

import { startSse } from './sse.js'

describe('http/sse', () => {
  it('streams frames using the shared formatter', async () => {
    const session = startSse(undefined)

    expect(session.writer.sendRaw('ready', 'ok')).toBe(true)
    expect(session.writer.sendJson('change', { path: 'memory/a.md' })).toBe(true)
    session.writer.close()

    await expect(session.done).resolves.toBeUndefined()
    await expect(session.response.text()).resolves.toBe(
      'event: ready\ndata: ok\n\n' + 'event: change\ndata: {"path":"memory/a.md"}\n\n',
    )
  })
})
