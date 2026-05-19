// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from 'vitest'
import { createMemStore } from '../store/index.js'
import { SecureTokenStore, InvalidEncryptionKeyError, DecryptionError, timingSafeCompare } from './token.js'
import type { OAuth2Token } from './oauth2.js'

const makeToken = (overrides?: Partial<OAuth2Token>): OAuth2Token => ({
  accessToken: 'access-abc',
  refreshToken: 'refresh-def',
  expiresAt: new Date('2026-06-01T12:00:00Z'),
  tokenType: 'Bearer',
  scopes: ['read', 'write'],
  ...overrides,
})

const passphrase = Buffer.from('test-passphrase-at-least-16-bytes', 'utf8')

describe('SecureTokenStore', () => {
  it('saves and loads a token with encryption round-trip', async () => {
    const store = createMemStore()
    const ts = new SecureTokenStore(store, passphrase)

    await ts.save('slack', 'brain-1', makeToken())
    const loaded = await ts.load('slack', 'brain-1')

    expect(loaded).toBeDefined()
    expect(loaded!.accessToken).toBe('access-abc')
    expect(loaded!.refreshToken).toBe('refresh-def')
    expect(loaded!.expiresAt).toEqual(new Date('2026-06-01T12:00:00Z'))
    expect(loaded!.tokenType).toBe('Bearer')
    expect(loaded!.scopes).toEqual(['read', 'write'])
  })

  it('returns undefined for nonexistent token', async () => {
    const store = createMemStore()
    const ts = new SecureTokenStore(store, passphrase)

    const loaded = await ts.load('nonexistent', 'brain-1')
    expect(loaded).toBeUndefined()
  })

  it('deletes a stored token', async () => {
    const store = createMemStore()
    const ts = new SecureTokenStore(store, passphrase)

    await ts.save('slack', 'brain-1', makeToken())
    await ts.delete('slack', 'brain-1')
    const loaded = await ts.load('slack', 'brain-1')

    expect(loaded).toBeUndefined()
  })

  it('delete is no-op for nonexistent token', async () => {
    const store = createMemStore()
    const ts = new SecureTokenStore(store, passphrase)

    await expect(ts.delete('nonexistent', 'brain-1')).resolves.toBeUndefined()
  })

  it('isolates tokens across brains', async () => {
    const store = createMemStore()
    const ts = new SecureTokenStore(store, passphrase)

    await ts.save('slack', 'brain-a', makeToken({ accessToken: 'access-a' }))
    await ts.save('slack', 'brain-b', makeToken({ accessToken: 'access-b' }))

    const loadedA = await ts.load('slack', 'brain-a')
    const loadedB = await ts.load('slack', 'brain-b')

    expect(loadedA!.accessToken).toBe('access-a')
    expect(loadedB!.accessToken).toBe('access-b')
  })

  it('fails to decrypt with wrong key', async () => {
    const store = createMemStore()
    const ts1 = new SecureTokenStore(store, Buffer.from('passphrase-one-at-least-16-bytes', 'utf8'))
    await ts1.save('slack', 'brain-1', makeToken())

    const ts2 = new SecureTokenStore(store, Buffer.from('passphrase-two-at-least-16-bytes', 'utf8'))
    await expect(ts2.load('slack', 'brain-1')).rejects.toThrow(DecryptionError)
  })

  it('throws on short passphrase', () => {
    const store = createMemStore()
    expect(() => new SecureTokenStore(store, Buffer.from('short', 'utf8'))).toThrow(
      InvalidEncryptionKeyError,
    )
  })
})

describe('timingSafeCompare', () => {
  it('returns true for equal buffers', () => {
    const a = Buffer.from('hello')
    const b = Buffer.from('hello')
    expect(timingSafeCompare(a, b)).toBe(true)
  })

  it('returns false for different buffers', () => {
    const a = Buffer.from('hello')
    const b = Buffer.from('world')
    expect(timingSafeCompare(a, b)).toBe(false)
  })

  it('returns false for different length buffers', () => {
    const a = Buffer.from('hello')
    const b = Buffer.from('hi')
    expect(timingSafeCompare(a, b)).toBe(false)
  })
})
