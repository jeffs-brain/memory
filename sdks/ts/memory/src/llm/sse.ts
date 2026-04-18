// SPDX-License-Identifier: Apache-2.0

/**
 * Minimal Server-Sent Events parser. Built for Anthropic / OpenAI /
 * Ollama streaming completions, not for full W3C conformance: we do
 * not track retry fields or ids.
 *
 * Feed chunks (UTF-8 strings) via {@link SSEParser.feed}. Events are
 * emitted when a blank line terminates an event block. Multi-line
 * `data:` fields accumulate with `\n` joiners, as the spec requires.
 * The sentinel `data: [DONE]` yields an event with data === '[DONE]'
 * so consumers can surface it as a terminal marker.
 *
 * Comments (lines beginning with ':') and unrecognised field names
 * are silently dropped.
 */

export type SSEEvent = {
  /** The `event:` field, or '' when none was provided. */
  event: string
  /** The accumulated `data:` field, with multiple data lines joined by `\n`. */
  data: string
}

export class SSEParser {
  private buffer = ''
  private currentEvent = ''
  private dataLines: string[] = []

  /**
   * Feed a chunk of SSE text. Returns every complete event since the
   * last feed. Partial events stay buffered until the next feed sees
   * their terminating blank line.
   */
  feed(chunk: string): readonly SSEEvent[] {
    this.buffer += chunk
    const out: SSEEvent[] = []

    // Split on any newline variant but keep the stream cursor advancing
    // only over fully-terminated lines; the tail (no trailing newline)
    // stays in the buffer.
    let idx: number
    while ((idx = this.indexOfLineEnd(this.buffer)) !== -1) {
      const line = this.buffer.slice(0, idx)
      // Consume the line plus its terminator (which may be 1 or 2 bytes).
      const term = this.buffer.startsWith('\r\n', idx) ? 2 : 1
      this.buffer = this.buffer.slice(idx + term)
      const evt = this.onLine(line)
      if (evt !== null) out.push(evt)
    }
    return out
  }

  /**
   * Flush any remaining buffered event. Call once after the upstream
   * reader has drained so trailing events without a blank-line
   * terminator are still surfaced.
   */
  flush(): readonly SSEEvent[] {
    const out: SSEEvent[] = []
    if (this.buffer.length > 0) {
      const evt = this.onLine(this.buffer)
      this.buffer = ''
      if (evt !== null) out.push(evt)
    }
    if (this.dataLines.length > 0) {
      out.push({ event: this.currentEvent, data: this.dataLines.join('\n') })
      this.dataLines = []
      this.currentEvent = ''
    }
    return out
  }

  private indexOfLineEnd(s: string): number {
    for (let i = 0; i < s.length; i++) {
      const c = s.charCodeAt(i)
      if (c === 10 /* \n */ || c === 13 /* \r */) return i
    }
    return -1
  }

  private onLine(line: string): SSEEvent | null {
    // Blank line terminates an event.
    if (line === '') {
      if (this.dataLines.length === 0 && this.currentEvent === '') return null
      const evt: SSEEvent = {
        event: this.currentEvent,
        data: this.dataLines.join('\n'),
      }
      this.currentEvent = ''
      this.dataLines = []
      return evt
    }

    // Comments.
    if (line.startsWith(':')) return null

    const colon = line.indexOf(':')
    let field: string
    let value: string
    if (colon === -1) {
      field = line
      value = ''
    } else {
      field = line.slice(0, colon)
      value = line.slice(colon + 1)
      if (value.startsWith(' ')) value = value.slice(1)
    }

    switch (field) {
      case 'event':
        this.currentEvent = value
        break
      case 'data':
        this.dataLines.push(value)
        break
      // 'id' and 'retry' are not needed by any caller here.
      default:
        break
    }
    return null
  }
}

/**
 * Iterate SSE events from a fetch response body. Parses on the fly and
 * yields each event as it completes. The caller is responsible for
 * cancellation via AbortSignal on the fetch request — this helper just
 * reads until the body closes.
 */
export async function* iterateSSE(
  body: ReadableStream<Uint8Array>,
): AsyncGenerator<SSEEvent, void, void> {
  const parser = new SSEParser()
  const reader = body.getReader()
  const decoder = new TextDecoder('utf-8')
  try {
    while (true) {
      const { value, done } = await reader.read()
      if (done) break
      const text = decoder.decode(value, { stream: true })
      for (const evt of parser.feed(text)) yield evt
    }
    const tail = decoder.decode()
    if (tail.length > 0) {
      for (const evt of parser.feed(tail)) yield evt
    }
    for (const evt of parser.flush()) yield evt
  } finally {
    reader.releaseLock()
  }
}
