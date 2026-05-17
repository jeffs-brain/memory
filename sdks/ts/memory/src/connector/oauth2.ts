// SPDX-License-Identifier: Apache-2.0

/**
 * OAuth2 authorization code flow helper. Handles authorization URL
 * generation, code exchange, and token refresh with a configurable
 * TokenExchanger for testability.
 */

/** OAuth2 configuration for an authorization code flow. */
export type OAuth2Config = {
  readonly clientId: string
  readonly clientSecret: string
  readonly authUrl: string
  readonly tokenUrl: string
  readonly scopes: readonly string[]
  readonly redirectUri: string
}

/** OAuth2 token pair with expiry information. */
export type OAuth2Token = {
  readonly accessToken: string
  readonly refreshToken: string
  readonly expiresAt: Date
  readonly tokenType: string
  readonly scopes: readonly string[]
}

/** Proactive refresh buffer: refresh 5 minutes before actual expiry. */
const TOKEN_EXPIRY_BUFFER_MS = 5 * 60 * 1000

/** Check whether a token has expired (including the 5-minute buffer). */
export const isTokenExpired = (token: OAuth2Token): boolean =>
  Date.now() + TOKEN_EXPIRY_BUFFER_MS > token.expiresAt.getTime()

/**
 * Abstracts the HTTP calls for OAuth2 token operations, allowing tests
 * to inject mocks without real HTTP.
 */
export type TokenExchanger = {
  /** Trade an authorization code for a token pair. */
  exchange(code: string): Promise<OAuth2Token>
  /** Use a refresh token to obtain a new access token. */
  refresh(refreshToken: string): Promise<OAuth2Token>
}

/** Thrown when the OAuth2 configuration is invalid. */
export class InvalidOAuth2ConfigError extends Error {
  constructor(message: string) {
    super(`Invalid OAuth2 config: ${message}`)
    this.name = 'InvalidOAuth2ConfigError'
  }
}

/** Thrown when token refresh fails. */
export class TokenRefreshError extends Error {
  constructor(message: string) {
    super(`Token refresh failed: ${message}`)
    this.name = 'TokenRefreshError'
  }
}

/** Validate that all required OAuth2 config fields are present. */
const validateOAuth2Config = (config: OAuth2Config): void => {
  const missing: string[] = []
  if (!config.clientId) missing.push('clientId')
  if (!config.clientSecret) missing.push('clientSecret')
  if (!config.authUrl) missing.push('authUrl')
  if (!config.tokenUrl) missing.push('tokenUrl')
  if (!config.redirectUri) missing.push('redirectUri')
  if (missing.length > 0) {
    throw new InvalidOAuth2ConfigError(`missing fields: ${missing.join(', ')}`)
  }
}

/**
 * OAuth2Client manages the authorization code flow including URL
 * generation, code exchange, and token refresh.
 */
export class OAuth2Client {
  private readonly config: OAuth2Config
  private readonly exchanger: TokenExchanger

  constructor(config: OAuth2Config, exchanger: TokenExchanger) {
    validateOAuth2Config(config)
    this.config = config
    this.exchanger = exchanger
  }

  /**
   * Generate the URL the user must visit to grant access. The state
   * parameter should be cryptographically random to prevent CSRF.
   */
  authorisationUrl(state: string): string {
    const params = new URLSearchParams({
      client_id: this.config.clientId,
      redirect_uri: this.config.redirectUri,
      response_type: 'code',
      state,
    })
    if (this.config.scopes.length > 0) {
      params.set('scope', this.config.scopes.join(' '))
    }
    return `${this.config.authUrl}?${params.toString()}`
  }

  /**
   * Exchange an authorization code for an access/refresh token pair.
   */
  async exchangeCode(code: string): Promise<OAuth2Token> {
    return this.exchanger.exchange(code)
  }

  /**
   * Refresh an expired access token using the refresh token.
   */
  async refreshToken(token: OAuth2Token): Promise<OAuth2Token> {
    if (!token.refreshToken) {
      throw new TokenRefreshError('no refresh token available')
    }
    const refreshed = await this.exchanger.refresh(token.refreshToken)
    // Preserve the refresh token if the provider did not issue a new one.
    if (!refreshed.refreshToken) {
      return { ...refreshed, refreshToken: token.refreshToken }
    }
    return refreshed
  }

  /**
   * Return the token if still valid, or refresh it if expired (or
   * within the 5-minute buffer).
   */
  async validToken(token: OAuth2Token): Promise<OAuth2Token> {
    if (!isTokenExpired(token)) {
      return token
    }
    return this.refreshToken(token)
  }

  /** Return the OAuth2 configuration (read-only). */
  getConfig(): OAuth2Config {
    return this.config
  }
}
