// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from 'vitest'
import {
  UnsafeUrlError,
  isPrivateIPv4,
  isPrivateIPv6,
  validateExternalUrl,
} from '../src/safe-fetch.js'

const resolver = (host: string, addresses: readonly string[]) => async (h: string) => {
  if (h !== host) throw new Error(`unexpected host: ${h}`)
  return addresses
}

describe('isPrivateIPv4', () => {
  it.each([
    ['10.0.0.1', true],
    ['10.255.255.255', true],
    ['127.0.0.1', true],
    ['169.254.169.254', true],
    ['172.16.0.1', true],
    ['172.31.255.255', true],
    ['192.168.1.1', true],
    ['100.64.0.1', true],
    ['224.0.0.1', true],
    ['240.0.0.1', true],
    ['0.0.0.0', true],
    ['8.8.8.8', false],
    ['1.1.1.1', false],
    ['151.101.1.69', false],
    ['172.32.0.1', false],
    ['172.15.255.255', false],
    ['100.63.255.255', false],
    ['100.128.0.0', false],
  ])('classifies %s correctly', (ip, expected) => {
    expect(isPrivateIPv4(ip)).toBe(expected)
  })
})

describe('isPrivateIPv6', () => {
  it.each([
    ['::1', true],
    ['::', true],
    ['fe80::1', true],
    ['febf::1', true],
    ['fc00::1', true],
    ['fd12:3456::1', true],
    ['ff02::1', true],
    ['::ffff:127.0.0.1', true],
    ['::ffff:10.0.0.1', true],
    ['2001:4860:4860::8888', false],
    ['2606:4700:4700::1111', false],
  ])('classifies %s correctly', (ip, expected) => {
    expect(isPrivateIPv6(ip)).toBe(expected)
  })
})

describe('validateExternalUrl', () => {
  it('rejects non-http schemes', async () => {
    await expect(
      validateExternalUrl('file:///etc/passwd', { resolver: resolver('any', ['8.8.8.8']) }),
    ).rejects.toBeInstanceOf(UnsafeUrlError)
    await expect(
      validateExternalUrl('ftp://example.com/file', {
        resolver: resolver('example.com', ['8.8.8.8']),
      }),
    ).rejects.toBeInstanceOf(UnsafeUrlError)
  })

  it('rejects malformed URLs', async () => {
    await expect(validateExternalUrl('not-a-url')).rejects.toBeInstanceOf(UnsafeUrlError)
  })

  it('rejects when DNS resolves to a private address', async () => {
    await expect(
      validateExternalUrl('http://internal.example.com/secret', {
        resolver: resolver('internal.example.com', ['10.0.0.5']),
      }),
    ).rejects.toMatchObject({ name: 'UnsafeUrlError' })
  })

  it('rejects when DNS resolves to loopback', async () => {
    await expect(
      validateExternalUrl('http://localhost.example/', {
        resolver: resolver('localhost.example', ['127.0.0.1']),
      }),
    ).rejects.toMatchObject({ name: 'UnsafeUrlError' })
  })

  it('rejects link-local AWS metadata IP', async () => {
    await expect(
      validateExternalUrl('http://169.254.169.254/latest/meta-data/'),
    ).rejects.toBeInstanceOf(UnsafeUrlError)
  })

  it('rejects when any of the resolved addresses is private', async () => {
    await expect(
      validateExternalUrl('http://mixed.example.com/', {
        resolver: resolver('mixed.example.com', ['8.8.8.8', '192.168.1.1']),
      }),
    ).rejects.toBeInstanceOf(UnsafeUrlError)
  })

  it('accepts a public host with a public IP', async () => {
    const url = await validateExternalUrl('https://example.com/path?q=1', {
      resolver: resolver('example.com', ['151.101.1.69']),
    })
    expect(url.hostname).toBe('example.com')
    expect(url.protocol).toBe('https:')
  })

  it('strips IPv6 brackets before resolution', async () => {
    const url = await validateExternalUrl('http://[2001:4860:4860::8888]/', {
      resolver: resolver('2001:4860:4860::8888', ['2001:4860:4860::8888']),
    })
    expect(url.hostname).toBe('[2001:4860:4860::8888]')
  })
})
