// SPDX-License-Identifier: Apache-2.0

/**
 * SecureTokenStore: encrypts OAuth2 tokens at rest using AES-256-GCM.
 * Tokens are scoped by (connectorName, brainId) to prevent cross-brain
 * leakage. Stored in the brain's Store at:
 *
 *   connector/<name>/<brainId>/oauth-token.enc.json
 */

import { createCipheriv, createDecipheriv, randomBytes, createHash, timingSafeEqual } from 'node:crypto'
import type { Store } from '../store/index.js'
import { isNotFound, toPath } from '../store/index.js'
import type { OAuth2Token } from './oauth2.js'

/** Thrown when the encryption key is too short. */
export class InvalidEncryptionKeyError extends Error {
  constructor(message: string) {
    super(`Invalid encryption key: ${message}`)
    this.name = 'InvalidEncryptionKeyError'
  }
}

/** Thrown when decryption fails (wrong key, corrupted, or tampered). */
export class DecryptionError extends Error {
  constructor(message: string) {
    super(`Token decryption failed: ${message}`)
    this.name = 'DecryptionError'
  }
}

/** Minimum passphrase length in bytes. */
const MIN_PASSPHRASE_LENGTH = 16

/** AES-256-GCM configuration. */
const AES_ALGORITHM = 'aes-256-gcm' as const
const NONCE_LENGTH = 12
const AUTH_TAG_LENGTH = 16

/** JSON envelope stored at rest. */
type EncryptedEnvelope = {
  readonly nonce: string // hex-encoded
  readonly ciphertext: string // hex-encoded (includes auth tag)
  readonly tag: string // hex-encoded auth tag
}

/** JSON wire format for token serialisation. */
type SerialisedToken = {
  readonly accessToken: string
  readonly refreshToken: string
  readonly expiresAt: string
  readonly tokenType: string
  readonly scopes: readonly string[]
}

const serialiseToken = (token: OAuth2Token): string =>
  JSON.stringify({
    accessToken: token.accessToken,
    refreshToken: token.refreshToken,
    expiresAt: token.expiresAt.toISOString(),
    tokenType: token.tokenType,
    scopes: token.scopes,
  } satisfies SerialisedToken)

const deserialiseToken = (json: string): OAuth2Token => {
  const raw = JSON.parse(json) as SerialisedToken
  return {
    accessToken: raw.accessToken,
    refreshToken: raw.refreshToken,
    expiresAt: new Date(raw.expiresAt),
    tokenType: raw.tokenType,
    scopes: [...raw.scopes],
  }
}

/** Derive a 256-bit key from a passphrase using SHA-256. */
const deriveKey = (passphrase: Buffer): Buffer =>
  createHash('sha256').update(passphrase).digest()

/** Encrypt plaintext using AES-256-GCM. */
const encrypt = (key: Buffer, plaintext: Buffer): EncryptedEnvelope => {
  const nonce = randomBytes(NONCE_LENGTH)
  const cipher = createCipheriv(AES_ALGORITHM, key, nonce, { authTagLength: AUTH_TAG_LENGTH })
  const encrypted = Buffer.concat([cipher.update(plaintext), cipher.final()])
  const tag = cipher.getAuthTag()

  return {
    nonce: nonce.toString('hex'),
    ciphertext: encrypted.toString('hex'),
    tag: tag.toString('hex'),
  }
}

/** Decrypt ciphertext encrypted by encrypt(). */
const decrypt = (key: Buffer, envelope: EncryptedEnvelope): Buffer => {
  const nonce = Buffer.from(envelope.nonce, 'hex')
  const ciphertext = Buffer.from(envelope.ciphertext, 'hex')
  const tag = Buffer.from(envelope.tag, 'hex')

  const decipher = createDecipheriv(AES_ALGORITHM, key, nonce, { authTagLength: AUTH_TAG_LENGTH })
  decipher.setAuthTag(tag)

  try {
    return Buffer.concat([decipher.update(ciphertext), decipher.final()])
  } catch {
    throw new DecryptionError('authentication failed -- wrong key or tampered data')
  }
}

const tokenStorePath = (connectorName: string, brainId: string) =>
  toPath(`connector/${connectorName}/${brainId}/oauth-token.enc.json`)

/**
 * Pluggable token store interface. Implementations encrypt tokens at
 * rest and scope them by (connectorName, brainId).
 */
export type TokenStore = {
  /** Encrypt and persist a token. */
  save(connectorName: string, brainId: string, token: OAuth2Token): Promise<void>
  /** Retrieve and decrypt a token. Returns undefined when not found. */
  load(connectorName: string, brainId: string): Promise<OAuth2Token | undefined>
  /** Remove a stored token. No-op if not found. */
  delete(connectorName: string, brainId: string): Promise<void>
}

/**
 * SecureTokenStore encrypts OAuth2 tokens with AES-256-GCM and stores
 * them in the brain's Store.
 */
export class SecureTokenStore implements TokenStore {
  private readonly store: Store
  private readonly key: Buffer

  constructor(store: Store, passphrase: Buffer) {
    if (passphrase.length < MIN_PASSPHRASE_LENGTH) {
      throw new InvalidEncryptionKeyError(
        `passphrase must be at least ${MIN_PASSPHRASE_LENGTH} bytes, got ${passphrase.length}`,
      )
    }
    this.store = store
    this.key = deriveKey(passphrase)
  }

  async save(connectorName: string, brainId: string, token: OAuth2Token): Promise<void> {
    const plaintext = Buffer.from(serialiseToken(token), 'utf8')
    const envelope = encrypt(this.key, plaintext)
    const data = Buffer.from(JSON.stringify(envelope, null, 2), 'utf8')
    await this.store.write(tokenStorePath(connectorName, brainId), data)
  }

  async load(connectorName: string, brainId: string): Promise<OAuth2Token | undefined> {
    try {
      const data = await this.store.read(tokenStorePath(connectorName, brainId))
      const envelope = JSON.parse(data.toString('utf8')) as EncryptedEnvelope
      const plaintext = decrypt(this.key, envelope)
      return deserialiseToken(plaintext.toString('utf8'))
    } catch (err: unknown) {
      if (isNotFound(err)) return undefined
      throw err
    }
  }

  async delete(connectorName: string, brainId: string): Promise<void> {
    try {
      await this.store.delete(tokenStorePath(connectorName, brainId))
    } catch (err: unknown) {
      if (isNotFound(err)) return
      throw err
    }
  }
}

/**
 * Perform a timing-safe comparison of two buffers to prevent timing
 * attacks when comparing authentication tags or HMAC values.
 */
export const timingSafeCompare = (a: Buffer, b: Buffer): boolean => {
  if (a.length !== b.length) return false
  return timingSafeEqual(a, b)
}
