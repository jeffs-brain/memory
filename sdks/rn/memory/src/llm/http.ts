import { ProviderError, TransportError } from './errors.js'

export type FetchLike = (input: string | URL | Request, init?: RequestInit) => Promise<Response>

export type HttpClient = {
  readonly fetch: FetchLike
}

export const defaultHttpClient: HttpClient = {
  fetch: (input, init) => fetch(input as Parameters<typeof fetch>[0], init),
}

export const postForText = async (
  client: HttpClient,
  url: string,
  body: unknown,
  init: {
    readonly headers?: Record<string, string>
    readonly signal?: AbortSignal
  } = {},
): Promise<{ readonly response: Response; readonly text: string }> => {
  let response: Response
  try {
    response = await client.fetch(url, {
      method: 'POST',
      headers: {
        'content-type': 'application/json',
        ...(init.headers ?? {}),
      },
      body: JSON.stringify(body),
      ...(init.signal === undefined ? {} : { signal: init.signal }),
    })
  } catch (error) {
    throw new TransportError(`POST ${url} transport error`, error)
  }

  let text = ''
  try {
    text = await response.text()
  } catch (error) {
    throw new TransportError(`POST ${url} failed to read body`, error)
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
  } = {},
): Promise<Response> => {
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
      ...(init.signal === undefined ? {} : { signal: init.signal }),
    })
  } catch (error) {
    throw new TransportError(`POST ${url} transport error`, error)
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
