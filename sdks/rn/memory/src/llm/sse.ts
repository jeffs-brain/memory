export type SSEEvent = {
  readonly event: string
  readonly data: string
}

export class SSEParser {
  private buffer = ''
  private currentEvent = ''
  private dataLines: string[] = []

  feed(chunk: string): readonly SSEEvent[] {
    this.buffer += chunk
    const out: SSEEvent[] = []

    for (;;) {
      const index = this.indexOfLineEnd(this.buffer)
      if (index === -1) break
      const line = this.buffer.slice(0, index)
      const terminator = this.buffer.startsWith('\r\n', index) ? 2 : 1
      this.buffer = this.buffer.slice(index + terminator)
      const event = this.onLine(line)
      if (event !== null) out.push(event)
    }

    return out
  }

  flush(): readonly SSEEvent[] {
    const out: SSEEvent[] = []
    if (this.buffer !== '') {
      const event = this.onLine(this.buffer)
      this.buffer = ''
      if (event !== null) out.push(event)
    }
    if (this.dataLines.length > 0) {
      out.push({ event: this.currentEvent, data: this.dataLines.join('\n') })
      this.currentEvent = ''
      this.dataLines = []
    }
    return out
  }

  private indexOfLineEnd(value: string): number {
    for (let index = 0; index < value.length; index += 1) {
      const code = value.charCodeAt(index)
      if (code === 10 || code === 13) return index
    }
    return -1
  }

  private onLine(line: string): SSEEvent | null {
    if (line === '') {
      if (this.currentEvent === '' && this.dataLines.length === 0) return null
      const event = {
        event: this.currentEvent,
        data: this.dataLines.join('\n'),
      }
      this.currentEvent = ''
      this.dataLines = []
      return event
    }

    if (line.startsWith(':')) return null

    const separator = line.indexOf(':')
    const field = separator === -1 ? line : line.slice(0, separator)
    let value = separator === -1 ? '' : line.slice(separator + 1)
    if (value.startsWith(' ')) value = value.slice(1)

    switch (field) {
      case 'event':
        this.currentEvent = value
        break
      case 'data':
        this.dataLines.push(value)
        break
      default:
        break
    }

    return null
  }
}

export async function* iterateSSE(
  body: ReadableStream<Uint8Array>,
): AsyncGenerator<SSEEvent, void, void> {
  const reader = body.getReader()
  const decoder = new TextDecoder('utf-8')
  const parser = new SSEParser()

  try {
    for (;;) {
      const { value, done } = await reader.read()
      if (done) break
      const text = decoder.decode(value, { stream: true })
      for (const event of parser.feed(text)) {
        yield event
      }
    }

    const tail = decoder.decode()
    if (tail !== '') {
      for (const event of parser.feed(tail)) {
        yield event
      }
    }

    for (const event of parser.flush()) {
      yield event
    }
  } finally {
    reader.releaseLock()
  }
}
