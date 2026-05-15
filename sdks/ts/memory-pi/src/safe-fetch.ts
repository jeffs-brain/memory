// SPDX-License-Identifier: Apache-2.0

/**
 * SSRF guard for outbound URL fetches. `memory_ingest_url` is callable
 * from the model, so we must block schemes that bypass HTTP, and IPs
 * that reach loopback, link-local, private, CGN, or multicast ranges.
 *
 * Resolution is done up-front and the validated host string is handed
 * back so the caller can re-issue the fetch against the same URL it
 * already validated.
 */

import { isIP } from 'node:net'
import { promises as dns } from 'node:dns'

export const DEFAULT_FETCH_TIMEOUT_MS = 30_000

const ALLOWED_SCHEMES = new Set(['http:', 'https:'])

export class UnsafeUrlError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'UnsafeUrlError'
  }
}

const ipv4ToInt = (ip: string): number | undefined => {
  const parts = ip.split('.')
  if (parts.length !== 4) return undefined
  let n = 0
  for (const part of parts) {
    if (!/^\d+$/.test(part)) return undefined
    const v = Number(part)
    if (v < 0 || v > 255) return undefined
    n = (n << 8) | v
  }
  return n >>> 0
}

const matchesCidr = (n: number, base: number, mask: number): boolean =>
  ((n & mask) >>> 0) === ((base & mask) >>> 0)

export const isPrivateIPv4 = (ip: string): boolean => {
  const n = ipv4ToInt(ip)
  if (n === undefined) return false
  // 0.0.0.0/8 unspecified
  if (matchesCidr(n, 0x00000000, 0xff000000)) return true
  // 10.0.0.0/8
  if (matchesCidr(n, 0x0a000000, 0xff000000)) return true
  // 127.0.0.0/8 loopback
  if (matchesCidr(n, 0x7f000000, 0xff000000)) return true
  // 169.254.0.0/16 link-local
  if (matchesCidr(n, 0xa9fe0000, 0xffff0000)) return true
  // 172.16.0.0/12
  if (matchesCidr(n, 0xac100000, 0xfff00000)) return true
  // 192.168.0.0/16
  if (matchesCidr(n, 0xc0a80000, 0xffff0000)) return true
  // 100.64.0.0/10 carrier-grade NAT
  if (matchesCidr(n, 0x64400000, 0xffc00000)) return true
  // 224.0.0.0/4 multicast
  if (matchesCidr(n, 0xe0000000, 0xf0000000)) return true
  // 240.0.0.0/4 reserved
  if (matchesCidr(n, 0xf0000000, 0xf0000000)) return true
  return false
}

export const isPrivateIPv6 = (ip: string): boolean => {
  const lower = ip.toLowerCase()
  if (lower === '::' || lower === '::1') return true
  // IPv4-mapped IPv6 ::ffff:a.b.c.d -- delegate to IPv4 check.
  if (lower.startsWith('::ffff:')) {
    const v4 = lower.slice(7)
    if (v4.includes('.')) return isPrivateIPv4(v4)
  }
  // fc00::/7 unique local
  if (/^fc[0-9a-f]{2}:/.test(lower) || /^fd[0-9a-f]{2}:/.test(lower)) return true
  // fe80::/10 link-local (fe80 - febf)
  if (/^fe[89ab][0-9a-f]:/.test(lower)) return true
  // ff00::/8 multicast
  if (/^ff[0-9a-f]{2}:/.test(lower)) return true
  return false
}

const isPrivateAddress = (ip: string): boolean => {
  const family = isIP(ip)
  if (family === 4) return isPrivateIPv4(ip)
  if (family === 6) return isPrivateIPv6(ip)
  return false
}

export type ValidateUrlOptions = {
  readonly resolver?: (host: string) => Promise<readonly string[]>
}

export const defaultResolver = async (host: string): Promise<readonly string[]> => {
  const direct = isIP(host)
  if (direct !== 0) return [host]
  const records = await dns.lookup(host, { all: true, verbatim: true })
  return records.map((r) => r.address)
}

/**
 * Parse, validate scheme, resolve hostname, and reject any IP that
 * would reach a private or loopback range. Throws `UnsafeUrlError` on
 * any failure so callers can map it to a clear tool error.
 */
export const validateExternalUrl = async (
  raw: string,
  options: ValidateUrlOptions = {},
): Promise<URL> => {
  let url: URL
  try {
    url = new URL(raw)
  } catch (err) {
    const detail = err instanceof Error ? err.message : String(err)
    throw new UnsafeUrlError(`invalid URL: ${detail}`)
  }
  if (!ALLOWED_SCHEMES.has(url.protocol)) {
    throw new UnsafeUrlError(`disallowed scheme: ${url.protocol}`)
  }
  const host = url.hostname
  if (host === '') {
    throw new UnsafeUrlError('missing hostname')
  }
  // Strip IPv6 brackets so the resolver / net.isIP see a clean address.
  const cleanHost = host.startsWith('[') && host.endsWith(']') ? host.slice(1, -1) : host
  const resolve = options.resolver ?? defaultResolver
  let addresses: readonly string[]
  try {
    addresses = await resolve(cleanHost)
  } catch (err) {
    const detail = err instanceof Error ? err.message : String(err)
    throw new UnsafeUrlError(`DNS lookup failed for ${cleanHost}: ${detail}`)
  }
  if (addresses.length === 0) {
    throw new UnsafeUrlError(`no DNS records for ${cleanHost}`)
  }
  for (const addr of addresses) {
    if (isPrivateAddress(addr)) {
      throw new UnsafeUrlError(`refusing to fetch private/loopback address ${addr} (${cleanHost})`)
    }
  }
  return url
}

/**
 * Compose the caller's AbortSignal (if any) with a bounded timeout so
 * stuck fetches cannot block tool dispatch indefinitely.
 */
export const fetchSignalWithTimeout = (
  signal: AbortSignal | undefined,
  timeoutMs: number = DEFAULT_FETCH_TIMEOUT_MS,
): AbortSignal => {
  const timeout = AbortSignal.timeout(timeoutMs)
  if (signal === undefined) return timeout
  // AbortSignal.any (Node >= 20.3, Bun) merges signals so either source
  // can abort the fetch.
  const any = (AbortSignal as unknown as { any?: (s: AbortSignal[]) => AbortSignal }).any
  if (typeof any === 'function') return any([signal, timeout])
  // Older runtimes: fall back to honouring the caller's signal and
  // re-arm the timeout via a one-shot controller.
  const controller = new AbortController()
  const abort = (): void => controller.abort()
  signal.addEventListener('abort', abort, { once: true })
  timeout.addEventListener('abort', abort, { once: true })
  return controller.signal
}
