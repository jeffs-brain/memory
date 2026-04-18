// SPDX-License-Identifier: Apache-2.0

/**
 * SSE helper for the memory daemon. Returns a {@link Response} whose
 * body is a `ReadableStream` the handler can push frames onto via the
 * returned writer. The writer is safe to call from async code; closing
 * the abort signal released by the runtime drains the stream cleanly.
 */

export type SseWriter = {
  sendRaw: (event: string, data: string) => boolean
  sendJson: (event: string, payload: unknown) => boolean
  close: () => void
  readonly closed: boolean
}

export type SseSession = {
  readonly response: Response
  readonly writer: SseWriter
  /** Resolves when the client disconnects or the writer closes. */
  readonly done: Promise<void>
}

const encoder = new TextEncoder()

const formatFrame = (event: string, data: string): string => {
  const lines: string[] = []
  lines.push(`event: ${event}`)
  for (const line of data.split('\n')) {
    lines.push(`data: ${line}`)
  }
  lines.push('', '')
  return lines.join('\n')
}

/**
 * Start an SSE stream. The caller receives a Response pre-configured
 * with the correct headers plus a writer that pushes frames.
 *
 * @param signal Abort signal propagated by the runtime when the client
 *   disconnects. Closing the signal also closes the writer.
 */
export const startSse = (signal: AbortSignal | undefined): SseSession => {
  let controllerRef: ReadableStreamDefaultController<Uint8Array> | undefined
  let closed = false
  let resolveDone: (() => void) | undefined
  const done = new Promise<void>((resolve) => {
    resolveDone = resolve
  })

  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      controllerRef = controller
    },
    cancel() {
      closed = true
      resolveDone?.()
    },
  })

  const send = (event: string, data: string): boolean => {
    if (closed) return false
    try {
      controllerRef?.enqueue(encoder.encode(formatFrame(event, data)))
      return true
    } catch {
      closed = true
      resolveDone?.()
      return false
    }
  }

  const onAbort = (): void => {
    if (closed) return
    closed = true
    try {
      controllerRef?.close()
    } catch {
      /* ignore */
    }
    resolveDone?.()
  }
  if (signal !== undefined) {
    if (signal.aborted) onAbort()
    else signal.addEventListener('abort', onAbort, { once: true })
  }

  const writer: SseWriter = {
    sendRaw: (event, data) => send(event, data),
    sendJson: (event, payload) => send(event, JSON.stringify(payload)),
    close: () => onAbort(),
    get closed() {
      return closed
    },
  }

  const response = new Response(stream, {
    status: 200,
    headers: {
      'content-type': 'text/event-stream',
      'cache-control': 'no-cache',
      connection: 'keep-alive',
      'x-accel-buffering': 'no',
    },
  })

  return { response, writer, done }
}
