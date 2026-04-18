/**
 * Shared fetch helpers. Keeps the provider files free of
 * boilerplate around timeouts, status classification, and error
 * mapping. Only what the ported layer needs; no retry middleware.
 */

import { ProviderError, TransportError } from './errors.js'

export type FetchLike = (
  input: string | URL | Request,
  init?: RequestInit,
) => Promise<Response>

export type HttpClient = {
  /** Injected fetch. Defaults to the global fetch when not provided. */
  fetch: FetchLike
}

export const defaultHttpClient: HttpClient = {
  fetch: (input, init) => fetch(input as unknown as Parameters<typeof fetch>[0], init),
}

/** POST JSON and return the parsed response body. Throws typed errors. */
export async function postJSON<T>(
  client: HttpClient,
  url: string,
  body: unknown,
  init: {
    headers?: Record<string, string>
    signal?: AbortSignal
  } = {},
): Promise<T> {
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
  } catch (err) {
    throw new ProviderError(
      `failed to parse response from ${url}`,
      response.status,
      text,
      err,
    )
  }
}

/** POST JSON and return the raw body so callers can inspect status. */
export async function postForText(
  client: HttpClient,
  url: string,
  body: unknown,
  init: {
    headers?: Record<string, string>
    signal?: AbortSignal
  } = {},
): Promise<{ response: Response; text: string }> {
  let response: Response
  try {
    response = await client.fetch(url, {
      method: 'POST',
      headers: {
        'content-type': 'application/json',
        ...(init.headers ?? {}),
      },
      body: JSON.stringify(body),
      ...(init.signal ? { signal: init.signal } : {}),
    })
  } catch (err) {
    throw new TransportError(`POST ${url} transport error`, err)
  }
  let text = ''
  try {
    text = await response.text()
  } catch (err) {
    throw new TransportError(`POST ${url} failed to read body`, err)
  }
  return { response, text }
}

/** POST JSON and return the streaming body. */
export async function postForStream(
  client: HttpClient,
  url: string,
  body: unknown,
  init: {
    headers?: Record<string, string>
    signal?: AbortSignal
  } = {},
): Promise<Response> {
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
      ...(init.signal ? { signal: init.signal } : {}),
    })
  } catch (err) {
    throw new TransportError(`POST ${url} transport error`, err)
  }
  if (!response.ok) {
    let errText = ''
    try {
      errText = await response.text()
    } catch {
      // Ignore; we still surface the status.
    }
    throw new ProviderError(
      `POST ${url} failed with status ${response.status}`,
      response.status,
      errText,
    )
  }
  if (response.body === null) {
    throw new TransportError(`POST ${url} returned no body`)
  }
  return response
}
