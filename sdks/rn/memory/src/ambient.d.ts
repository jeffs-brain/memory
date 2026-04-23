declare module '@op-engineering/op-sqlite' {
  export type Scalar = string | number | boolean | null | ArrayBuffer | ArrayBufferView

  export type QueryResult = {
    readonly rowsAffected: number
    readonly rows: Array<Record<string, Scalar>>
  }

  export type DB = {
    executeSync(query: string, params?: readonly Scalar[]): QueryResult
    close(): void
  }

  export const open: (params: {
    name: string
    location?: string
    encryptionKey?: string
  }) => DB
}

declare module 'expo-file-system/legacy' {
  export const documentDirectory: string | null
  export const EncodingType: {
    readonly UTF8: string
  }

  export type FileInfo = {
    readonly exists: boolean
    readonly isDirectory: boolean
    readonly size?: number
    readonly modificationTime?: number
  }

  export type DownloadProgressData = {
    readonly totalBytesWritten: number
    readonly totalBytesExpectedToWrite: number
  }

  export type DownloadPauseState = {
    readonly resumeData?: string
  }

  export type DownloadResumable = {
    downloadAsync(): Promise<unknown>
    pauseAsync(): Promise<DownloadPauseState>
    resumeAsync(): Promise<unknown>
    savable(): DownloadPauseState
  }

  export const createDownloadResumable: (
    uri: string,
    fileUri: string,
    options?: Record<string, unknown>,
    callback?: (data: DownloadProgressData) => void,
    resumeData?: string,
  ) => DownloadResumable

  export const readAsStringAsync: (
    uri: string,
    options?: { readonly encoding?: string },
  ) => Promise<string>
  export const writeAsStringAsync: (
    uri: string,
    content: string,
    options?: { readonly encoding?: string },
  ) => Promise<void>
  export const makeDirectoryAsync: (
    uri: string,
    options?: { readonly intermediates?: boolean },
  ) => Promise<void>
  export const readDirectoryAsync: (uri: string) => Promise<string[]>
  export const deleteAsync: (
    uri: string,
    options?: { readonly idempotent?: boolean },
  ) => Promise<void>
  export const moveAsync: (options: { readonly from: string; readonly to: string }) => Promise<void>
  export const copyAsync: (options: { readonly from: string; readonly to: string }) => Promise<void>
  export const getInfoAsync: (
    uri: string,
    options?: { readonly size?: boolean },
  ) => Promise<FileInfo>
}

declare module '@react-native-community/netinfo' {
  export type NetInfoState = {
    readonly isConnected: boolean | null
    readonly isInternetReachable: boolean | null
  }

  export const addEventListener: (listener: (state: NetInfoState) => void) => () => void
  export const fetch: () => Promise<NetInfoState>
}

declare module 'better-sqlite3' {
  type BetterStatement = {
    all(...params: readonly unknown[]): unknown[]
    get(...params: readonly unknown[]): unknown
    run(...params: readonly unknown[]): unknown
  }

  type BetterDatabase = {
    pragma(query: string): void
    exec(sql: string): void
    prepare(sql: string): BetterStatement
    transaction<T>(fn: () => T): () => T
    loadExtension(path: string): void
    close(): void
  }

  const BetterSqlite: {
    new (path: string, options?: { readonly allowExtension?: boolean }): BetterDatabase
  }

  export default BetterSqlite
}

declare module 'sqlite-vec' {
  export const getLoadablePath: () => string
}
