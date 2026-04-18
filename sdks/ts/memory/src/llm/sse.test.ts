import { describe, expect, it } from 'vitest'
import { SSEParser, iterateSSE } from './sse.js'

describe('SSEParser', () => {
  it('parses a simple event', () => {
    const p = new SSEParser()
    const events = p.feed('event: ping\ndata: hello\n\n')
    expect(events).toEqual([{ event: 'ping', data: 'hello' }])
  })

  it('accumulates multi-line data fields with newline joiners', () => {
    const p = new SSEParser()
    const events = p.feed('data: line one\ndata: line two\n\n')
    expect(events).toEqual([{ event: '', data: 'line one\nline two' }])
  })

  it('ignores comment lines beginning with a colon', () => {
    const p = new SSEParser()
    const events = p.feed(': heartbeat\ndata: payload\n\n')
    expect(events).toEqual([{ event: '', data: 'payload' }])
  })

  it('strips a single leading space after the colon', () => {
    const p = new SSEParser()
    const events = p.feed('data:   three spaces\n\n')
    expect(events).toEqual([{ event: '', data: '  three spaces' }])
  })

  it('yields the [DONE] sentinel as-is', () => {
    const p = new SSEParser()
    const events = p.feed('data: [DONE]\n\n')
    expect(events).toEqual([{ event: '', data: '[DONE]' }])
  })

  it('handles events split across multiple feed calls', () => {
    const p = new SSEParser()
    expect(p.feed('event: message_start\ndata: {"a"')).toEqual([])
    expect(p.feed(':1}\n\nevent: ping\ndata: pong\n\n')).toEqual([
      { event: 'message_start', data: '{"a":1}' },
      { event: 'ping', data: 'pong' },
    ])
  })

  it('copes with CRLF terminators', () => {
    const p = new SSEParser()
    const events = p.feed('event: foo\r\ndata: bar\r\n\r\n')
    expect(events).toEqual([{ event: 'foo', data: 'bar' }])
  })

  it('flushes trailing events without a blank-line terminator', () => {
    const p = new SSEParser()
    expect(p.feed('event: foo\ndata: bar\n')).toEqual([])
    expect(p.flush()).toEqual([{ event: 'foo', data: 'bar' }])
  })
})

describe('iterateSSE', () => {
  it('yields events from a ReadableStream of Uint8Array chunks', async () => {
    const chunks = [
      'event: message_start\ndata: {"u":1}\n\n',
      'event: content_block_delta\ndata: {"text":"hi"}\n\n',
      'data: [DONE]\n\n',
    ]
    const stream = new ReadableStream<Uint8Array>({
      start(controller) {
        const encoder = new TextEncoder()
        for (const c of chunks) controller.enqueue(encoder.encode(c))
        controller.close()
      },
    })
    const out: Array<{ event: string; data: string }> = []
    for await (const evt of iterateSSE(stream)) out.push(evt)
    expect(out).toEqual([
      { event: 'message_start', data: '{"u":1}' },
      { event: 'content_block_delta', data: '{"text":"hi"}' },
      { event: '', data: '[DONE]' },
    ])
  })
})
