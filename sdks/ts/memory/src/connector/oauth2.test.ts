// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it, vi } from 'vitest'
import {
  OAuth2Client,
  InvalidOAuth2ConfigError,
  TokenRefreshError,
  isTokenExpired,
  type OAuth2Config,
  type OAuth2Token,
  type TokenExchanger,
} from './oauth2.js'

const validConfig = (): OAuth2Config => ({
  clientId: 'test-client-id',
  clientSecret: 'test-client-secret',
  authUrl: 'https://provider.example.com/auth',
  tokenUrl: 'https://provider.example.com/token',
  scopes: ['read', 'write'],
  redirectUri: 'https://app.example.com/callback',
})

const noopExchanger = (): TokenExchanger => ({
  exchange: vi.fn(),
  refresh: vi.fn(),
})

describe('OAuth2Client', () => {
  describe('constructor', () => {
    it('throws on invalid config', () => {
      expect(() => new OAuth2Client({} as OAuth2Config, noopExchanger())).toThrow(
        InvalidOAuth2ConfigError,
      )
    })

    it('accepts valid config', () => {
      expect(() => new OAuth2Client(validConfig(), noopExchanger())).not.toThrow()
    })
  })

  describe('authorisationUrl', () => {
    it('generates URL with all required parameters', () => {
      const client = new OAuth2Client(validConfig(), noopExchanger())
      const url = client.authorisationUrl('random-state')

      expect(url).toContain('https://provider.example.com/auth?')
      expect(url).toContain('client_id=test-client-id')
      expect(url).toContain('response_type=code')
      expect(url).toContain('state=random-state')
      expect(url).toContain('redirect_uri=')
      expect(url).toContain('scope=read+write')
    })
  })

  describe('exchangeCode', () => {
    it('delegates to exchanger and returns token', async () => {
      const expected: OAuth2Token = {
        accessToken: 'access-123',
        refreshToken: 'refresh-456',
        expiresAt: new Date(Date.now() + 3600_000),
        tokenType: 'Bearer',
        scopes: ['read'],
      }

      const exchanger: TokenExchanger = {
        exchange: vi.fn().mockResolvedValue(expected),
        refresh: vi.fn(),
      }

      const client = new OAuth2Client(validConfig(), exchanger)
      const token = await client.exchangeCode('auth-code-789')

      expect(exchanger.exchange).toHaveBeenCalledWith('auth-code-789')
      expect(token.accessToken).toBe('access-123')
      expect(token.refreshToken).toBe('refresh-456')
    })
  })

  describe('refreshToken', () => {
    it('refreshes an expired token', async () => {
      const refreshed: OAuth2Token = {
        accessToken: 'new-access',
        refreshToken: 'new-refresh',
        expiresAt: new Date(Date.now() + 3600_000),
        tokenType: 'Bearer',
        scopes: [],
      }

      const exchanger: TokenExchanger = {
        exchange: vi.fn(),
        refresh: vi.fn().mockResolvedValue(refreshed),
      }

      const client = new OAuth2Client(validConfig(), exchanger)
      const old: OAuth2Token = {
        accessToken: 'old-access',
        refreshToken: 'old-refresh',
        expiresAt: new Date(Date.now() - 3600_000),
        tokenType: 'Bearer',
        scopes: [],
      }

      const token = await client.refreshToken(old)
      expect(token.accessToken).toBe('new-access')
      expect(exchanger.refresh).toHaveBeenCalledWith('old-refresh')
    })

    it('preserves refresh token when provider does not return one', async () => {
      const exchanger: TokenExchanger = {
        exchange: vi.fn(),
        refresh: vi.fn().mockResolvedValue({
          accessToken: 'new-access',
          refreshToken: '',
          expiresAt: new Date(Date.now() + 3600_000),
          tokenType: 'Bearer',
          scopes: [],
        }),
      }

      const client = new OAuth2Client(validConfig(), exchanger)
      const old: OAuth2Token = {
        accessToken: 'old-access',
        refreshToken: 'old-refresh',
        expiresAt: new Date(Date.now() - 3600_000),
        tokenType: 'Bearer',
        scopes: [],
      }

      const token = await client.refreshToken(old)
      expect(token.refreshToken).toBe('old-refresh')
    })

    it('throws when no refresh token available', async () => {
      const client = new OAuth2Client(validConfig(), noopExchanger())
      const noRefresh: OAuth2Token = {
        accessToken: 'access',
        refreshToken: '',
        expiresAt: new Date(Date.now() - 3600_000),
        tokenType: 'Bearer',
        scopes: [],
      }

      await expect(client.refreshToken(noRefresh)).rejects.toThrow(TokenRefreshError)
    })

    it('propagates refresh failure', async () => {
      const exchanger: TokenExchanger = {
        exchange: vi.fn(),
        refresh: vi.fn().mockRejectedValue(new Error('401 unauthorised')),
      }

      const client = new OAuth2Client(validConfig(), exchanger)
      const old: OAuth2Token = {
        accessToken: 'access',
        refreshToken: 'refresh',
        expiresAt: new Date(Date.now() - 3600_000),
        tokenType: 'Bearer',
        scopes: [],
      }

      await expect(client.refreshToken(old)).rejects.toThrow()
    })
  })

  describe('validToken', () => {
    it('returns current token when not expired', async () => {
      const refreshFn = vi.fn()
      const exchanger: TokenExchanger = {
        exchange: vi.fn(),
        refresh: refreshFn,
      }

      const client = new OAuth2Client(validConfig(), exchanger)
      const valid: OAuth2Token = {
        accessToken: 'access',
        refreshToken: 'refresh',
        expiresAt: new Date(Date.now() + 30 * 60_000),
        tokenType: 'Bearer',
        scopes: [],
      }

      const token = await client.validToken(valid)
      expect(token.accessToken).toBe('access')
      expect(refreshFn).not.toHaveBeenCalled()
    })

    it('proactively refreshes token expiring within 5-minute buffer', async () => {
      const exchanger: TokenExchanger = {
        exchange: vi.fn(),
        refresh: vi.fn().mockResolvedValue({
          accessToken: 'refreshed',
          refreshToken: 'new-refresh',
          expiresAt: new Date(Date.now() + 3600_000),
          tokenType: 'Bearer',
          scopes: [],
        }),
      }

      const client = new OAuth2Client(validConfig(), exchanger)
      const expiring: OAuth2Token = {
        accessToken: 'old',
        refreshToken: 'refresh',
        expiresAt: new Date(Date.now() + 3 * 60_000), // 3 minutes
        tokenType: 'Bearer',
        scopes: [],
      }

      const token = await client.validToken(expiring)
      expect(token.accessToken).toBe('refreshed')
    })
  })
})

describe('isTokenExpired', () => {
  it('returns true for expired token', () => {
    const token: OAuth2Token = {
      accessToken: '',
      refreshToken: '',
      expiresAt: new Date(Date.now() - 3600_000),
      tokenType: '',
      scopes: [],
    }
    expect(isTokenExpired(token)).toBe(true)
  })

  it('returns true for token within buffer', () => {
    const token: OAuth2Token = {
      accessToken: '',
      refreshToken: '',
      expiresAt: new Date(Date.now() + 3 * 60_000), // 3 minutes
      tokenType: '',
      scopes: [],
    }
    expect(isTokenExpired(token)).toBe(true)
  })

  it('returns false for token with plenty of time', () => {
    const token: OAuth2Token = {
      accessToken: '',
      refreshToken: '',
      expiresAt: new Date(Date.now() + 30 * 60_000), // 30 minutes
      tokenType: '',
      scopes: [],
    }
    expect(isTokenExpired(token)).toBe(false)
  })
})
