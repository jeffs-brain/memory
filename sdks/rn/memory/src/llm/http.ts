import { ProviderError, TransportError } from './errors.js'

export type FetchLike = (input: string | URL | Request, init?: RequestInit) => Promise<Response>

export type HttpClient = {
  readonly fetch: FetchLike
}

const DEFAULT_REQUEST_TIMEOUT_MS = 30_000

export const defaultHttpClient: HttpClient = {
  fetch: (input, init) => fetch(input as Parameters<typeof fetch>[0], init),
}

const createTimedSignal = (
  signal: AbortSignal | undefined,
  timeoutMs: number,
): {
  readonly signal: AbortSignal
  cleanup(): void
} => {
  const controller = new AbortController()
  const onAbort = (): void => controller.abort(signal?.reason)

  if (signal?.aborted === true) controller.abort(signal.reason)
  else if (signal !== undefined) signal.addEventListener('abort', onAbort, { once: true })

  const timer = setTimeout(() => controller.abort(new Error('request timed out')), timeoutMs)

  return {
    signal: controller.signal,
    cleanup: () => {
      clearTimeout(timer)
      if (signal !== undefined) signal.removeEventListener('abort', onAbort)
    },
  }
}

export const postForText = async (
  client: HttpClient,
  url: string,
  body: unknown,
  init: {
    readonly headers?: Record<string, string>
    readonly signal?: AbortSignal
    readonly timeoutMs?: number
  } = {},
): Promise<{ readonly response: Response; readonly text: string }> => {
  const request = createTimedSignal(init.signal, init.timeoutMs ?? DEFAULT_REQUEST_TIMEOUT_MS)
  let response: Response
  try {
    response = await client.fetch(url, {
      method: 'POST',
      headers: {
        'content-type': 'application/json',
        ...(init.headers ?? {}),
      },
      body: JSON.stringify(body),
      signal: request.signal,
    })
  } catch (error) {
    throw new TransportError(`POST ${url} transport error`, error)
  }

  let text = ''
  try {
    text = await response.text()
  } catch (error) {
    throw new TransportError(`POST ${url} failed to read body`, error)
  } finally {
    request.cleanup()
  }

  return { response, text }
}

export const postJSON = async <T>(
  client: HttpClient,
  url: string,
  body: unknown,
  init: {
    readonly headers?: Record<string, string>
    readonly signal?: AbortSignal
    readonly timeoutMs?: number
  } = {},
): Promise<T> => {
  const { response, text } = await postForText(client, url, body, init)
  if (!response.ok) {
    throw new ProviderError(
      `POST ${url} failed with status ${response.status}`,
      response.status,
      text,
    )
  }

  try {
    return JSON.parse(text) as T
  } catch (error) {
    throw new ProviderError(`failed to parse response from ${url}`, response.status, text, error)
  }
}

export const postForStream = async (
  client: HttpClient,
  url: string,
  body: unknown,
  init: {
    readonly headers?: Record<string, string>
    readonly signal?: AbortSignal
    readonly timeoutMs?: number
  } = {},
): Promise<Response> => {
  const request = createTimedSignal(init.signal, init.timeoutMs ?? DEFAULT_REQUEST_TIMEOUT_MS)
  let response: Response
  try {
    response = await client.fetch(url, {
      method: 'POST',
      headers: {
        'content-type': 'application/json',
        accept: 'text/event-stream',
        ...(init.headers ?? {}),
      },
      body: JSON.stringify(body),
      signal: request.signal,
    })
  } catch (error) {
    throw new TransportError(`POST ${url} transport error`, error)
  } finally {
    request.cleanup()
  }

  if (!response.ok) {
    let text = ''
    try {
      text = await response.text()
    } catch {
      // Ignore secondary read failures.
    }
    throw new ProviderError(
      `POST ${url} failed with status ${response.status}`,
      response.status,
      text,
    )
  }

  if (response.body === null) {
    throw new TransportError(`POST ${url} returned no body`)
  }

  return response
}
