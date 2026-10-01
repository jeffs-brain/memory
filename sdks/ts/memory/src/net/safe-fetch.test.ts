// SPDX-License-Identifier: Apache-2.0

import { type Server, createServer } from 'node:http'
import type { AddressInfo } from 'node:net'
import { afterAll, beforeAll, describe, expect, it } from 'vitest'

import { FetchFailedError, UnsafeUrlError, isBlockedAddress, safeFetch } from './safe-fetch.js'

describe('isBlockedAddress', () => {
  it.each([
    '127.0.0.1',
    '127.8.8.8',
    '10.1.2.3',
    '172.16.0.1',
    '172.31.255.255',
    '192.168.1.1',
    '169.254.169.254',
    '100.64.0.1',
    '0.0.0.0',
    '192.0.0.8',
    '198.18.0.1',
    '224.0.0.1',
    '240.0.0.1',
    '255.255.255.255',
    '::',
    '::1',
    'fd00::1',
    'fe80::1',
    'fec0::1',
    'ff02::1',
    '::ffff:127.0.0.1',
    '::ffff:7f00:1',
    '64:ff9b::a00:1',
    '[::1]',
    'not-an-ip',
  ])('blocks %s', (ip) => {
    expect(isBlockedAddress(ip)).toBe(true)
  })

  it.each([
    '8.8.8.8',
    '1.1.1.1',
    '93.184.216.34',
    '2607:f8b0:4004:800::200e',
    '::ffff:8.8.8.8',
    '64:ff9b::808:808',
  ])('allows %s', (ip) => {
    expect(isBlockedAddress(ip)).toBe(false)
  })
})

describe('safeFetch', () => {
  let server: Server
  let port = 0
  // Test hosts: `public.test` is treated as public and pinned to the
  // local server; `internal.test` resolves to a private address.
  const resolver = async (host: string): Promise<readonly string[]> => {
    if (host === 'public.test') return ['127.0.0.1']
    if (host === 'internal.test') return ['10.0.0.1']
    if (host === 'mixed.test') return ['127.0.0.1', '10.0.0.1']
    throw new Error('NXDOMAIN')
  }
  const isBlocked = (ip: string): boolean => ip.startsWith('10.')

  beforeAll(async () => {
    server = createServer((req, res) => {
      switch (req.url) {
        case '/ok':
          res.writeHead(200, { 'content-type': 'text/markdown; charset=utf-8' })
          res.end('# hello')
          return
        case '/to-internal':
          res.writeHead(302, { location: 'http://internal.test/secret' })
          res.end()
          return
        case '/to-file':
          res.writeHead(302, { location: 'file:///etc/passwd' })
          res.end()
          return
        case '/relative':
          res.writeHead(301, { location: '/ok' })
          res.end()
          return
        case '/loop':
          res.writeHead(302, { location: '/loop' })
          res.end()
          return
        case '/big':
          res.writeHead(200, { 'content-type': 'text/plain' })
          res.end('x'.repeat(2048))
          return
        case '/slow':
          setTimeout(() => res.end('late'), 500)
          return
        default:
          res.writeHead(404)
          res.end('missing')
      }
    })
    await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve))
    port = (server.address() as AddressInfo).port
  })

  afterAll(async () => {
    await new Promise<void>((resolve) => server.close(() => resolve()))
  })

  const at = (path: string, host = 'public.test'): string => `http://${host}:${port}${path}`

  it('refuses loopback with the default policy', async () => {
    await expect(safeFetch(`http://127.0.0.1:${port}/ok`)).rejects.toBeInstanceOf(UnsafeUrlError)
  })

  it.each(['http://[::1]:1/ok', 'http://[fd00::1]/', 'http://[::ffff:127.0.0.1]/'])(
    'refuses the bracketed IPv6 literal %s',
    async (url) => {
      await expect(safeFetch(url)).rejects.toBeInstanceOf(UnsafeUrlError)
    },
  )

  it('refuses the cloud metadata address', async () => {
    await expect(safeFetch('http://169.254.169.254/latest/meta-data/')).rejects.toBeInstanceOf(
      UnsafeUrlError,
    )
  })

  it.each([
    'file:///etc/passwd',
    'ftp://example.com/x',
    'gopher://example.com/',
    'not a url',
    'http://user:pw@example.com/',
  ])('refuses %s', async (url) => {
    await expect(safeFetch(url)).rejects.toBeInstanceOf(UnsafeUrlError)
  })

  it('fetches a public URL through the pinned address', async () => {
    const res = await safeFetch(at('/ok'), { resolver, isBlocked })
    expect(res.status).toBe(200)
    expect(res.contentType).toBe('text/markdown; charset=utf-8')
    expect(res.body.toString('utf8')).toBe('# hello')
  })

  it('refuses a host when any resolved address is blocked', async () => {
    await expect(
      safeFetch(at('/ok', 'mixed.test'), { resolver, isBlocked }),
    ).rejects.toBeInstanceOf(UnsafeUrlError)
  })

  it('re-validates redirect targets', async () => {
    await expect(safeFetch(at('/to-internal'), { resolver, isBlocked })).rejects.toBeInstanceOf(
      UnsafeUrlError,
    )
    await expect(safeFetch(at('/to-file'), { resolver, isBlocked })).rejects.toBeInstanceOf(
      UnsafeUrlError,
    )
  })

  it('follows relative redirects', async () => {
    const res = await safeFetch(at('/relative'), { resolver, isBlocked })
    expect(res.body.toString('utf8')).toBe('# hello')
    expect(res.url).toBe(at('/ok'))
  })

  it('caps the redirect chain', async () => {
    await expect(safeFetch(at('/loop'), { resolver, isBlocked, maxRedirects: 2 })).rejects.toThrow(
      /redirects/,
    )
  })

  it('maps non-2xx to FetchFailedError with the status', async () => {
    const err = await safeFetch(at('/nope'), { resolver, isBlocked }).catch((e: unknown) => e)
    expect(err).toBeInstanceOf(FetchFailedError)
    expect((err as FetchFailedError).status).toBe(404)
  })

  it('caps the body size', async () => {
    await expect(safeFetch(at('/big'), { resolver, isBlocked, maxBytes: 1024 })).rejects.toThrow(
      /byte limit/,
    )
  })

  it('times out', async () => {
    await expect(
      safeFetch(at('/slow'), { resolver, isBlocked, timeoutMs: 50 }),
    ).rejects.toBeInstanceOf(FetchFailedError)
  })

  it('reports DNS failures as upstream errors', async () => {
    await expect(
      safeFetch(at('/ok', 'nowhere.test'), { resolver, isBlocked }),
    ).rejects.toBeInstanceOf(FetchFailedError)
  })
})
