// SPDX-License-Identifier: Apache-2.0

/**
 * Google Drive connector implementing the Connector interface. Supports
 * full sync via files.list and incremental sync via the Changes API with
 * startPageToken cursors. Google-native formats (Docs, Sheets, Slides)
 * are exported to markdown, CSV, and plain text respectively.
 */

import type {
  Connector,
  ConnectorConfig,
  ConnectorDocument,
  HTTPClient,
  SyncCursor,
} from './types.js'

// -- Google Drive API constants -----------------------------------------------

/** Base endpoint for Drive files.list. */
const DRIVE_FILES_URL = 'https://www.googleapis.com/drive/v3/files'

/** Base endpoint for Drive changes.list. */
const DRIVE_CHANGES_URL = 'https://www.googleapis.com/drive/v3/changes'

/** Endpoint for fetching the initial changes start token. */
const DRIVE_START_PAGE_TOKEN_URL = 'https://www.googleapis.com/drive/v3/changes/startPageToken'

/** Fields requested for each file in list/changes responses. */
const DRIVE_FILE_FIELDS = 'id,name,mimeType,modifiedTime,size,parents,webViewLink'

/** Top-level response fields for files.list. */
const DRIVE_LIST_FIELDS = `nextPageToken,files(${DRIVE_FILE_FIELDS})`

/** Response fields for changes.list. */
const DRIVE_CHANGES_FIELDS =
  `nextPageToken,newStartPageToken,changes(fileId,removed,file(${DRIVE_FILE_FIELDS}))`

/** Number of results per API page. */
const DRIVE_DEFAULT_PAGE_SIZE = 100

/** Google-imposed limit on file exports (10 MB). */
const DRIVE_EXPORT_MAX_BYTES = 10 * 1024 * 1024

/** Default maximum file size to download (50 MB). */
const DEFAULT_MAX_FILE_SIZE = 50 * 1024 * 1024

/** Per-request timeout for Google API calls (30 seconds). */
const HTTP_REQUEST_TIMEOUT_MS = 30_000

/** Timeout for file download/export requests (5 minutes). */
const HTTP_DOWNLOAD_TIMEOUT_MS = 5 * 60 * 1000

/** Maximum response body length for error message truncation. */
const MAX_ERROR_BODY_LENGTH = 200

// -- Google MIME types --------------------------------------------------------

const MIME_GOOGLE_DOC = 'application/vnd.google-apps.document'
const MIME_GOOGLE_SHEET = 'application/vnd.google-apps.spreadsheet'
const MIME_GOOGLE_SLIDES = 'application/vnd.google-apps.presentation'
const MIME_GOOGLE_DRAWING = 'application/vnd.google-apps.drawing'
const MIME_GOOGLE_FOLDER = 'application/vnd.google-apps.folder'
const MIME_TEXT_MARKDOWN = 'text/markdown'
const MIME_TEXT_PLAIN = 'text/plain'
const MIME_TEXT_CSV = 'text/csv'
const MIME_IMAGE_PNG = 'image/png'

/** Default export format mapping for Google-native document types. */
const DEFAULT_EXPORT_FORMATS: Readonly<Record<string, string>> = {
  [MIME_GOOGLE_DOC]: MIME_TEXT_MARKDOWN,
  [MIME_GOOGLE_SHEET]: MIME_TEXT_CSV,
  [MIME_GOOGLE_SLIDES]: MIME_TEXT_PLAIN,
  [MIME_GOOGLE_DRAWING]: MIME_IMAGE_PNG,
}

/** Set of Google-native MIME types that require export instead of download. */
const GOOGLE_NATIVE_FORMATS: ReadonlySet<string> = new Set([
  MIME_GOOGLE_DOC,
  MIME_GOOGLE_SHEET,
  MIME_GOOGLE_SLIDES,
  MIME_GOOGLE_DRAWING,
])

// -- Drive API response types -------------------------------------------------

type DriveFile = {
  readonly id: string
  readonly name: string
  readonly mimeType: string
  readonly modifiedTime: string
  readonly size?: string | undefined
  readonly parents?: readonly string[] | undefined
  readonly webViewLink?: string | undefined
}

type DriveChange = {
  readonly fileId: string
  readonly file?: DriveFile | undefined
  readonly removed: boolean
}

type DriveListResponse = {
  readonly nextPageToken?: string | undefined
  readonly files: readonly DriveFile[]
}

type DriveChangesResponse = {
  readonly nextPageToken?: string | undefined
  readonly newStartPageToken?: string | undefined
  readonly changes: readonly DriveChange[]
}

type DriveStartPageTokenResponse = {
  readonly startPageToken: string
}

type DriveAPIError = {
  readonly error: {
    readonly message: string
    readonly errors?: readonly { readonly reason: string }[] | undefined
  }
}

// -- GDrive connector configuration -------------------------------------------

/**
 * Google Drive-specific connector settings. All optional fields
 * explicitly accept undefined to satisfy exactOptionalPropertyTypes.
 */
export type GDriveConnectorConfig = {
  readonly oauth2ClientId: string
  readonly oauth2ClientSecret: string
  readonly oauth2RedirectUri: string | undefined
  readonly accessToken: string | undefined
  readonly folderId: string | undefined
  readonly mimeTypeFilter: readonly string[] | undefined
  readonly includeSharedDrives: boolean
  readonly maxFileSize: number
  readonly exportFormats: Readonly<Record<string, string>> | undefined
}

// -- Implementation -----------------------------------------------------------

/**
 * Create a Google Drive connector with the given shared dependencies.
 * Call configure() before fetching documents.
 */
export const createGDriveConnector = (
  deps: ConnectorConfig,
  httpClient: HTTPClient,
): GDriveConnector => new GDriveConnector(deps, httpClient)

/**
 * GDriveConnector implements the Connector interface for Google Drive.
 * It supports full sync via files.list and incremental sync via the
 * Changes API with startPageToken cursors.
 */
export class GDriveConnector implements Connector {
  readonly name = 'gdrive' as const

  private readonly deps: ConnectorConfig
  private readonly httpClient: HTTPClient
  private driveConfig: GDriveConnectorConfig | undefined
  private accessToken = ''
  private exportFormats: Readonly<Record<string, string>> = DEFAULT_EXPORT_FORMATS

  constructor(deps: ConnectorConfig, httpClient: HTTPClient) {
    this.deps = deps
    this.httpClient = httpClient
  }

  /**
   * Validate and store Google Drive-specific settings.
   */
  async configure(rawConfig: Record<string, string>): Promise<void> {
    const clientId = rawConfig['oauth2_client_id'] ?? ''
    const clientSecret = rawConfig['oauth2_client_secret'] ?? ''

    if (clientId === '') {
      throw new Error('gdrive: oauth2_client_id is required')
    }
    if (clientSecret === '') {
      throw new Error('gdrive: oauth2_client_secret is required')
    }

    const mimeFilter = rawConfig['mime_type_filter']
    const maxFileSize = rawConfig['max_file_size']

    let parsedMaxFileSize = DEFAULT_MAX_FILE_SIZE
    if (maxFileSize !== undefined && maxFileSize !== '') {
      const parsed = Number(maxFileSize)
      if (Number.isNaN(parsed)) {
        throw new Error(`gdrive: invalid max_file_size "${maxFileSize}"`)
      }
      parsedMaxFileSize = parsed
    }

    const cfg: GDriveConnectorConfig = {
      oauth2ClientId: clientId,
      oauth2ClientSecret: clientSecret,
      oauth2RedirectUri: rawConfig['oauth2_redirect_uri'],
      accessToken: rawConfig['access_token'],
      folderId: rawConfig['folder_id'],
      mimeTypeFilter: mimeFilter ? mimeFilter.split(',') : undefined,
      includeSharedDrives: rawConfig['include_shared_drives'] === 'true',
      maxFileSize: parsedMaxFileSize,
      exportFormats: undefined,
    }

    this.driveConfig = cfg

    if (cfg.accessToken !== undefined) {
      this.accessToken = cfg.accessToken
    }

    this.exportFormats = { ...DEFAULT_EXPORT_FORMATS }
  }

  /**
   * Set the OAuth2 access token directly. Used when the caller manages
   * token lifecycle externally.
   */
  setAccessToken(token: string): void {
    this.accessToken = token
  }

  /**
   * Perform a full sync of all files from the configured Drive scope.
   */
  async *fetchAll(signal: AbortSignal): AsyncIterable<ConnectorDocument> {
    const cfg = this.getConfig()

    let pageToken: string | undefined
    do {
      const resp = await this.listFiles(cfg, signal, pageToken)

      for (const file of resp.files) {
        if (signal.aborted) return

        const doc = await this.fileToDocument(cfg, signal, file)
        if (doc !== undefined) {
          yield doc
        }
      }

      pageToken = resp.nextPageToken
    } while (pageToken)
  }

  /**
   * Perform an incremental sync using the Drive Changes API.
   */
  async *fetchSince(signal: AbortSignal, cursor: SyncCursor): AsyncIterable<ConnectorDocument> {
    const cfg = this.getConfig()

    let pageToken: string | undefined = cursor.value
    let processingChanges = true

    while (processingChanges) {
      if (signal.aborted) return

      const resp: DriveChangesResponse = await this.listChanges(cfg, signal, pageToken ?? '')

      for (const change of resp.changes) {
        if (signal.aborted) return

        const doc = this.changeToDocument(cfg, signal, change)
        if (doc instanceof Promise) {
          const resolved = await doc
          if (resolved !== undefined) {
            yield resolved
          }
        } else if (doc !== undefined) {
          yield doc
        }
      }

      if (resp.newStartPageToken) {
        yield {
          externalId: '__cursor_update__',
          content: '',
          mime: '',
          title: '',
          metadata: { new_start_page_token: resp.newStartPageToken },
          modifiedAt: new Date(),
        }
        processingChanges = false
      } else if (resp.nextPageToken) {
        pageToken = resp.nextPageToken
      } else {
        processingChanges = false
      }
    }
  }

  /**
   * Retrieve the current Changes API start page token. Used as the
   * initial cursor for incremental sync.
   */
  async getStartPageToken(signal: AbortSignal): Promise<string> {
    let reqUrl = DRIVE_START_PAGE_TOKEN_URL
    if (this.driveConfig?.includeSharedDrives) {
      reqUrl += '?supportsAllDrives=true'
    }

    const body = await this.doAPIGet(reqUrl, signal, HTTP_REQUEST_TIMEOUT_MS)
    const resp = JSON.parse(body) as DriveStartPageTokenResponse
    return resp.startPageToken
  }

  /**
   * Begin a continuous sync loop. Not implemented without P5-1
   * SyncStateManager integration.
   */
  async start(_signal: AbortSignal): Promise<void> {
    throw new Error('gdrive: continuous sync not yet supported without P5-1 SyncStateManager')
  }

  /** Halt the continuous sync loop. */
  async stop(): Promise<void> {
    // No-op until continuous sync is implemented.
  }

  // -- Private helpers --------------------------------------------------------

  /**
   * Return the validated config or throw. Replaces the asserts-this
   * pattern which conflicts with private properties under strict mode.
   */
  private getConfig(): GDriveConnectorConfig {
    const cfg = this.driveConfig
    if (cfg === undefined) {
      throw new Error('gdrive: connector not configured')
    }
    return cfg
  }

  private async listFiles(
    cfg: GDriveConnectorConfig,
    signal: AbortSignal,
    pageToken?: string | undefined,
  ): Promise<DriveListResponse> {
    const params = new URLSearchParams({
      pageSize: String(DRIVE_DEFAULT_PAGE_SIZE),
      fields: DRIVE_LIST_FIELDS,
    })

    if (pageToken) {
      params.set('pageToken', pageToken)
    }

    const query = buildFileQuery(cfg)
    if (query !== '') {
      params.set('q', query)
    }

    if (cfg.includeSharedDrives) {
      params.set('supportsAllDrives', 'true')
      params.set('includeItemsFromAllDrives', 'true')
    }

    const body = await this.doAPIGet(
      `${DRIVE_FILES_URL}?${params.toString()}`,
      signal,
      HTTP_REQUEST_TIMEOUT_MS,
    )
    return JSON.parse(body) as DriveListResponse
  }

  private async listChanges(
    cfg: GDriveConnectorConfig,
    signal: AbortSignal,
    pageToken: string,
  ): Promise<DriveChangesResponse> {
    const params = new URLSearchParams({
      pageToken,
      pageSize: String(DRIVE_DEFAULT_PAGE_SIZE),
      fields: DRIVE_CHANGES_FIELDS,
    })

    if (cfg.includeSharedDrives) {
      params.set('supportsAllDrives', 'true')
      params.set('includeItemsFromAllDrives', 'true')
    }

    const body = await this.doAPIGet(
      `${DRIVE_CHANGES_URL}?${params.toString()}`,
      signal,
      HTTP_REQUEST_TIMEOUT_MS,
    )
    return JSON.parse(body) as DriveChangesResponse
  }

  private async fileToDocument(
    cfg: GDriveConnectorConfig,
    signal: AbortSignal,
    file: DriveFile,
  ): Promise<ConnectorDocument | undefined> {
    if (file.mimeType === MIME_GOOGLE_FOLDER) {
      return undefined
    }

    const fileSize = parseFileSize(file.size)
    const maxSize = cfg.maxFileSize

    if (fileSize > maxSize && !GOOGLE_NATIVE_FORMATS.has(file.mimeType)) {
      this.deps.logger.warn('gdrive: file exceeds size limit, skipping', {
        fileId: file.id,
        name: file.name,
        size: fileSize,
        limit: maxSize,
      })
      return undefined
    }

    const { content, mime } = await this.downloadOrExport(signal, file)
    const modifiedAt = new Date(file.modifiedTime)

    return {
      externalId: file.id,
      content,
      mime,
      title: file.name,
      url: file.webViewLink ?? '',
      metadata: buildFileMetadata(file),
      modifiedAt,
    }
  }

  private changeToDocument(
    cfg: GDriveConnectorConfig,
    signal: AbortSignal,
    change: DriveChange,
  ): ConnectorDocument | Promise<ConnectorDocument | undefined> {
    if (change.removed || change.file === undefined) {
      return {
        externalId: change.fileId,
        content: '',
        mime: '',
        title: '',
        metadata: { source: 'gdrive' },
        modifiedAt: new Date(),
        deleted: true,
      }
    }

    return this.fileToDocument(cfg, signal, change.file)
  }

  private async downloadOrExport(
    signal: AbortSignal,
    file: DriveFile,
  ): Promise<{ content: string; mime: string }> {
    const exportMime = this.resolveExportFormat(file.mimeType)
    if (exportMime !== undefined) {
      return this.exportFile(signal, file.id, exportMime)
    }
    return this.downloadFile(signal, file.id, file.mimeType)
  }

  private resolveExportFormat(sourceMime: string): string | undefined {
    return this.exportFormats[sourceMime]
  }

  private async exportFile(
    signal: AbortSignal,
    fileId: string,
    targetMime: string,
  ): Promise<{ content: string; mime: string }> {
    const exportUrl = `${DRIVE_FILES_URL}/${encodeURIComponent(fileId)}/export?mimeType=${encodeURIComponent(targetMime)}`
    const body = await this.doAPIGet(exportUrl, signal, HTTP_DOWNLOAD_TIMEOUT_MS)

    if (body.length > DRIVE_EXPORT_MAX_BYTES) {
      throw new Error(`gdrive: export of ${fileId} exceeds ${DRIVE_EXPORT_MAX_BYTES} byte limit`)
    }

    return { content: body, mime: targetMime }
  }

  private async downloadFile(
    signal: AbortSignal,
    fileId: string,
    mime: string,
  ): Promise<{ content: string; mime: string }> {
    const downloadUrl = `${DRIVE_FILES_URL}/${encodeURIComponent(fileId)}?alt=media`
    const body = await this.doAPIGet(downloadUrl, signal, HTTP_DOWNLOAD_TIMEOUT_MS)
    return { content: body, mime }
  }

  private async doAPIGet(
    reqUrl: string,
    signal: AbortSignal,
    timeoutMs: number,
  ): Promise<string> {
    const controller = new AbortController()
    const timeoutId = setTimeout(() => controller.abort(), timeoutMs)

    // Compose abort: external signal or internal timeout.
    const onAbort = (): void => controller.abort()
    signal.addEventListener('abort', onAbort, { once: true })

    try {
      const resp = await this.httpClient.fetch(reqUrl, {
        method: 'GET',
        headers: { Authorization: `Bearer ${this.accessToken}` },
        signal: controller.signal,
      })

      const body = await resp.text()

      if (resp.status === 403 || resp.status === 429) {
        throw this.handleRateLimitError(body, resp.status)
      }

      if (resp.status < 200 || resp.status >= 300) {
        throw new Error(`gdrive: API returned status ${resp.status}: ${truncateBody(body)}`)
      }

      return body
    } finally {
      clearTimeout(timeoutId)
      signal.removeEventListener('abort', onAbort)
    }
  }

  private handleRateLimitError(body: string, statusCode: number): Error {
    try {
      const apiErr = JSON.parse(body) as DriveAPIError
      const reasons = apiErr.error.errors ?? []

      for (const entry of reasons) {
        if (entry.reason === 'rateLimitExceeded' || entry.reason === 'userRateLimitExceeded') {
          return new Error(`gdrive: rate limit exceeded: ${apiErr.error.message}`)
        }
      }

      return new Error(`gdrive: API error (status ${statusCode}): ${apiErr.error.message}`)
    } catch {
      return new Error(`gdrive: rate limit error (status ${statusCode}): ${truncateBody(body)}`)
    }
  }
}

// -- Pure helper functions ----------------------------------------------------

const parseFileSize = (sizeStr: string | undefined): number => {
  if (sizeStr === undefined || sizeStr === '') return 0
  const parsed = Number(sizeStr)
  return Number.isNaN(parsed) ? 0 : parsed
}

const buildFileQuery = (cfg: GDriveConnectorConfig): string => {
  const parts: string[] = []

  parts.push(`mimeType != '${MIME_GOOGLE_FOLDER}'`)

  if (cfg.folderId) {
    parts.push(`'${cfg.folderId}' in parents`)
  }

  if (cfg.mimeTypeFilter !== undefined && cfg.mimeTypeFilter.length > 0) {
    const mimeConditions = cfg.mimeTypeFilter.map(
      (mime) => `mimeType = '${mime}'`,
    )
    parts.push(`(${mimeConditions.join(' or ')})`)
  }

  parts.push('trashed = false')

  return parts.join(' and ')
}

const buildFileMetadata = (file: DriveFile): Readonly<Record<string, string>> => {
  const meta: Record<string, string> = {
    source: 'gdrive',
    mime_type: file.mimeType,
    file_id: file.id,
  }
  if (file.parents !== undefined && file.parents.length > 0) {
    const firstParent = file.parents[0]
    if (firstParent !== undefined) {
      meta['parent_id'] = firstParent
    }
  }
  if (file.size !== undefined && file.size !== '') {
    meta['size'] = file.size
  }
  return meta
}

const truncateBody = (body: string): string => {
  if (body.length <= MAX_ERROR_BODY_LENGTH) return body
  return body.slice(0, MAX_ERROR_BODY_LENGTH) + '...'
}
