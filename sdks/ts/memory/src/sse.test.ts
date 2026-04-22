// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it, vi } from 'vitest'

import { createSseHeartbeat, formatSseFrame } from './sse.js'

describe('sse utilities', () => {
  it('formats an SSE frame with optional id and multi-line data', () => {
    expect(
      formatSseFrame({
        event: 'change',
        id: '42',
        data: 'alpha\r\nbravo\ncharlie\rdelta',
      }),
    ).toBe('event: change\nid: 42\ndata: alpha\ndata: bravo\ndata: charlie\ndata: delta\n\n')
  })

  it('strips line breaks from event metadata', () => {
    expect(
      formatSseFrame({
        event: 'chan\ng\re',
        id: '1\r\n2',
        data: 'ok',
      }),
    ).toBe('event: change\nid: 12\ndata: ok\n\n')
  })

  it('ticks until stopped', () => {
    vi.useFakeTimers()
    try {
      const onHeartbeat = vi.fn()
      const stop = createSseHeartbeat(25_000, onHeartbeat)

      vi.advanceTimersByTime(50_000)
      expect(onHeartbeat).toHaveBeenCalledTimes(2)

      stop()
      vi.advanceTimersByTime(25_000)
      expect(onHeartbeat).toHaveBeenCalledTimes(2)
    } finally {
      vi.useRealTimers()
    }
  })

  it('rejects non-positive heartbeat intervals', () => {
    expect(() => createSseHeartbeat(0, () => undefined)).toThrow(RangeError)
  })
})
