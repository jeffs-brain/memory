// SPDX-License-Identifier: Apache-2.0

/**
 * Barrel export for the connector framework and concrete connectors.
 */

export type {
  ConnectorConfig,
  ConnectorDocument,
  Connector,
  ConnectorFactory,
  SyncCursor,
  RateLimiterConfig,
  RateLimitHeaders,
  RateLimiter,
  PageResult,
  FetchPageFn,
  PaginatorConfig,
} from './types.js'

export { DEFAULT_POLL_INTERVAL } from './types.js'

export {
  ConnectorRegistry,
  ConnectorNotFoundError,
  ConnectorExistsError,
} from './registry.js'

export {
  OAuth2Client,
  InvalidOAuth2ConfigError,
  TokenRefreshError,
  isTokenExpired,
  type OAuth2Config,
  type OAuth2Token,
  type TokenExchanger,
} from './oauth2.js'

export {
  SecureTokenStore,
  InvalidEncryptionKeyError,
  DecryptionError,
  timingSafeCompare,
  type TokenStore,
} from './token.js'

export { createRateLimiter } from './rate-limiter.js'

export {
  paginate,
  collectPages,
  MaxPagesExceededError,
} from './paginator.js'

export { SyncStateManager } from './sync.js'

export type { SlackConnectorConfig } from './slack.js'

export {
  SlackConnector,
  createSlackConnector,
} from './slack.js'

export {
  convertMrkdwn,
  parseSlackTimestamp,
  formatDate,
  validateDownloadURL,
  readResponseWithLimit,
} from './slack_helpers.js'

export { GDriveConnector, createGDriveConnector } from './gdrive.js'
export type { GDriveConnectorConfig } from './gdrive.js'
