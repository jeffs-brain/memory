export type DownloadProgress = {
  readonly downloadedBytes: number
  readonly totalBytes?: number
}

export type ModelDownloadTransport = {
  download(args: {
    readonly modelId: string
    readonly url: string
    readonly destination: string
    readonly headers?: Readonly<Record<string, string>>
    readonly onProgress?: (progress: DownloadProgress) => void
  }): Promise<void>
  cancel(modelId: string): void
}

export const createExpoResumableDownloadTransport = async (): Promise<ModelDownloadTransport> => {
  const fs = await import('expo-file-system/legacy')
  const tasks = new Map<string, ReturnType<typeof fs.createDownloadResumable>>()

  return {
    download: async ({ modelId, url, destination, headers, onProgress }) => {
      const task = fs.createDownloadResumable(
        url,
        destination,
        headers === undefined ? {} : { headers },
        onProgress === undefined
          ? undefined
          : (progress) => {
              onProgress({
                downloadedBytes: progress.totalBytesWritten,
                totalBytes: progress.totalBytesExpectedToWrite,
              })
            },
      )
      tasks.set(modelId, task)
      try {
        await task.downloadAsync()
      } finally {
        tasks.delete(modelId)
      }
    },
    cancel: (modelId) => {
      const task = tasks.get(modelId)
      if (task === undefined) return
      void task.pauseAsync().finally(() => {
        tasks.delete(modelId)
      })
    },
  }
}
