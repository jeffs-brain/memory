// SPDX-License-Identifier: Apache-2.0

/**
 * SSRF-safe outbound fetch for user- or model-supplied URLs.
 *
 * Every hop, including each redirect, is validated: the scheme must be
 * http or https, and every address the host resolves to must be public.
 * The connection is then pinned to the validated address through the
 * socket `lookup` hook, so a second DNS answer (rebinding) cannot swap
 * in an internal address between the check and the connect. Bodies are
 * capped while streaming and every request carries a timeout.
 *
 * The blocklist matches `go/knowledge/safefetch.go` and the Python SDK.
 */

import { lookup as dnsLookup } from 'node:dns/promises'
import type { IncomingMessage } from 'node:http'
import { request as httpRequest } from 'node:http'
import { request as httpsRequest } from 'node:https'
import type { LookupFunction } from 'node:net'
import { BlockList, isIP } from 'node:net'

export const DEFAULT_SAFE_FETCH_TIMEOUT_MS = 30_000
export const DEFAULT_SAFE_FETCH_MAX_BYTES = 50 * 1024 * 1024
export const DEFAULT_SAFE_FETCH_MAX_REDIRECTS = 5

/** The URL is malformed, uses a scheme other than http(s), or resolves
 *  to a non-public address. Maps to 400 at the HTTP layer. */
export class UnsafeUrlError extends Error {
  override readonly name = 'UnsafeUrlError'
}

/** The upstream could not be reached, answered with a non-2xx status,
 *  or sent a body over the limit. Maps to 502 at the HTTP layer. */
export class FetchFailedError extends Error {
  override readonly name = 'FetchFailedError'
  constructor(
    message: string,
    readonly status?: number,
  ) {
    super(message)
  }
}

export type Resolver = (host: string) => Promise<readonly string[]>

export type SafeFetchOptions = {
  readonly timeoutMs?: number
  readonly maxBytes?: number
  readonly maxRedirects?: number
  readonly signal?: AbortSignal
  readonly headers?: Readonly<Record<string, string>>
  /** Override DNS resolution. Tests only; production uses the system resolver. */
  readonly resolver?: Resolver
  /** Override the address policy. Tests only; production uses {@link isBlockedAddress}. */
  readonly isBlocked?: (ip: string) => boolean
}

export type SafeFetchResult = {
  readonly url: string
  readonly status: number
  readonly contentType: string
  readonly body: Buffer
}

const blocked = new BlockList()
for (const [net, prefix] of [
  ['0.0.0.0', 8],
  ['10.0.0.0', 8],
  ['100.64.0.0', 10],
  ['127.0.0.0', 8],
  ['169.254.0.0', 16],
  ['172.16.0.0', 12],
  ['192.0.0.0', 24],
  ['192.168.0.0', 16],
  ['198.18.0.0', 15],
  ['224.0.0.0', 4],
  ['240.0.0.0', 4],
] as const) {
  blocked.addSubnet(net, prefix, 'ipv4')
}
for (const [net, prefix] of [
  ['::', 128],
  ['::1', 128],
  ['fc00::', 7],
  ['fe80::', 10],
  ['fec0::', 10],
  ['ff00::', 8],
] as const) {
  blocked.addSubnet(net, prefix, 'ipv6')
}

const embeddedIpv4 = (ip: string): string | undefined => {
  const lower = ip.toLowerCase()
  for (const prefix of ['::ffff:', '64:ff9b::']) {
    if (!lower.startsWith(prefix)) continue
    const tail = lower.slice(prefix.length)
    if (isIP(tail) === 4) return tail
    const groups = tail.split(':')
    if (groups.length !== 2) return undefined
    const [hi, lo] = groups.map((g) => Number.parseInt(g, 16))
    if (hi === undefined || lo === undefined || Number.isNaN(hi) || Number.isNaN(lo)) {
      return undefined
    }
    return [hi >> 8, hi & 0xff, lo >> 8, lo & 0xff].join('.')
  }
  return undefined
}

/** True when `ip` is loopback, private, link-local, CGN, multicast,
 *  reserved or unspecified, including IPv4-mapped and NAT64 forms.
 *  Anything that is not an IP literal is treated as blocked. */
export const isBlockedAddress = (ip: string): boolean => {
  const clean = ip.startsWith('[') && ip.endsWith(']') ? ip.slice(1, -1) : ip
  const family = isIP(clean)
  if (family === 4) return blocked.check(clean, 'ipv4')
  if (family === 6) {
    const v4 = embeddedIpv4(clean)
    if (v4 !== undefined) return blocked.check(v4, 'ipv4')
    return blocked.check(clean, 'ipv6')
  }
  return true
}

const systemResolver: Resolver = async (host) => {
  if (isIP(host) !== 0) return [host]
  const records = await dnsLookup(host, { all: true, verbatim: true })
  return records.map((r) => r.address)
}

const parseExternalUrl = (raw: string): URL => {
  let url: URL
  try {
    url = new URL(raw.trim())
  } catch {
    throw new UnsafeUrlError('invalid URL')
  }
  if (url.protocol !== 'http:' && url.protocol !== 'https:') {
    throw new UnsafeUrlError(`unsupported scheme ${url.protocol} (only http and https allowed)`)
  }
  if (url.hostname === '') throw new UnsafeUrlError('URL missing host')
  if (url.username !== '' || url.password !== '') {
    throw new UnsafeUrlError('URL must not carry credentials')
  }
  return url
}

const bareHost = (url: URL): string =>
  url.hostname.startsWith('[') && url.hostname.endsWith(']')
    ? url.hostname.slice(1, -1)
    : url.hostname

/**
 * Resolve the URL's host and return the first address, refusing the
 * URL when any resolved address is non-public.
 */
export const resolvePublicAddress = async (
  url: URL,
  resolver: Resolver = systemResolver,
  isBlocked: (ip: string) => boolean = isBlockedAddress,
): Promise<string> => {
  const host = bareHost(url)
  let addresses: readonly string[]
  try {
    addresses = await resolver(host)
  } catch {
    throw new FetchFailedError(`DNS lookup failed for ${host}`)
  }
  const first = addresses[0]
  if (first === undefined) throw new FetchFailedError(`no DNS records for ${host}`)
  if (addresses.some(isBlocked)) {
    throw new UnsafeUrlError(`${host} resolves to a non-public address`)
  }
  return first
}

const pinnedLookup = (address: string): LookupFunction => {
  const family = isIP(address)
  return (_hostname, options, callback) => {
    if (typeof options === 'object' && options !== null && options.all === true) {
      callback(null, [{ address, family }])
      return
    }
    callback(null, address, family)
  }
}

type HopResponse = {
  readonly status: number
  readonly location: string | undefined
  readonly contentType: string
  readonly body: Buffer
}

const readCapped = (res: IncomingMessage, maxBytes: number): Promise<Buffer> =>
  new Promise((resolve, reject) => {
    const parts: Buffer[] = []
    let total = 0
    res.on('data', (chunk: Buffer) => {
      total += chunk.length
      if (total > maxBytes) {
        res.destroy()
        reject(new FetchFailedError(`response exceeds ${maxBytes} byte limit`))
        return
      }
      parts.push(chunk)
    })
    res.on('end', () => resolve(Buffer.concat(parts)))
    res.on('error', (err) => reject(new FetchFailedError(`reading response: ${err.message}`)))
  })

const fetchHop = (
  url: URL,
  address: string,
  opts: Required<Pick<SafeFetchOptions, 'maxBytes'>> & {
    readonly signal: AbortSignal
    readonly headers: Readonly<Record<string, string>>
  },
): Promise<HopResponse> =>
  new Promise((resolve, reject) => {
    const send = url.protocol === 'https:' ? httpsRequest : httpRequest
    const req = send(
      url,
      {
        method: 'GET',
        headers: opts.headers,
        lookup: pinnedLookup(address),
        signal: opts.signal,
      },
      (res) => {
        const status = res.statusCode ?? 0
        const location = res.headers.location
        const contentType = res.headers['content-type'] ?? ''
        if (status >= 300 && status < 400 && location !== undefined) {
          res.resume()
          resolve({ status, location, contentType, body: Buffer.alloc(0) })
          return
        }
        if (status < 200 || status >= 300) {
          res.resume()
          reject(new FetchFailedError(`upstream responded HTTP ${status}`, status))
          return
        }
        readCapped(res, opts.maxBytes).then(
          (body) => resolve({ status, location: undefined, contentType, body }),
          reject,
        )
      },
    )
    req.on('error', (err) => {
      if (err instanceof FetchFailedError || err instanceof UnsafeUrlError) {
        reject(err)
        return
      }
      const reason = opts.signal.aborted ? 'request timed out or was aborted' : err.message
      reject(new FetchFailedError(`fetch failed: ${reason}`))
    })
    req.end()
  })

/**
 * GET `raw` with the SSRF guard applied to every hop. Throws
 * {@link UnsafeUrlError} for refused URLs and {@link FetchFailedError}
 * for upstream failures.
 */
export const safeFetch = async (
  raw: string,
  options: SafeFetchOptions = {},
): Promise<SafeFetchResult> => {
  const timeout = AbortSignal.timeout(options.timeoutMs ?? DEFAULT_SAFE_FETCH_TIMEOUT_MS)
  const signal = options.signal !== undefined ? AbortSignal.any([options.signal, timeout]) : timeout
  const maxRedirects = options.maxRedirects ?? DEFAULT_SAFE_FETCH_MAX_REDIRECTS
  const hopOpts = {
    maxBytes: options.maxBytes ?? DEFAULT_SAFE_FETCH_MAX_BYTES,
    signal,
    headers: {
      accept: 'text/plain, text/markdown, text/html, application/pdf;q=0.9, */*;q=0.5',
      ...options.headers,
    },
  }
  let url = parseExternalUrl(raw)
  for (let hop = 0; ; hop++) {
    const address = await resolvePublicAddress(url, options.resolver, options.isBlocked)
    const res = await fetchHop(url, address, hopOpts)
    if (res.location === undefined) {
      return {
        url: url.toString(),
        status: res.status,
        contentType: res.contentType,
        body: res.body,
      }
    }
    if (hop >= maxRedirects) {
      throw new FetchFailedError(`stopped after ${maxRedirects} redirects`)
    }
    let next: URL
    try {
      next = new URL(res.location, url)
    } catch {
      throw new FetchFailedError('redirect to an invalid URL')
    }
    url = parseExternalUrl(next.toString())
  }
}
