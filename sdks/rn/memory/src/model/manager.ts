import type { FileAdapter } from '../store/index.js'
import { joinPath } from '../store/index.js'
import type { ModelDownloadTransport } from './download.js'
import { DEFAULT_MODEL_MANIFESTS, type ModelManifest } from './registry.js'

export type DownloadedModel = {
  readonly manifest: ModelManifest
  readonly path: string
  readonly sizeBytes: number
}

export type ModelManager = {
  listAvailable(): readonly ModelManifest[]
  listDownloaded(): Promise<readonly DownloadedModel[]>
  download(modelId: string, onProgress?: (pct: number) => void): Promise<void>
  cancelDownload(modelId: string): void
  deleteModel(modelId: string): Promise<void>
  getModelPath(modelId: string): string | null
  storageUsed(): Promise<number>
}

export const createModelManager = (args: {
  readonly root: string
  readonly adapter: FileAdapter
  readonly downloadTransport: ModelDownloadTransport
  readonly manifests?: readonly ModelManifest[]
}): ModelManager => {
  const manifests = args.manifests ?? DEFAULT_MODEL_MANIFESTS
  const manifestMap = new Map(manifests.map((manifest) => [manifest.id, manifest] as const))

  const resolveModelPath = (manifest: ModelManifest): string => {
    return args.adapter.join(args.root, joinPath(manifest.id, manifest.filename))
  }

  return {
    listAvailable: () => manifests,
    listDownloaded: async () => {
      const downloaded: DownloadedModel[] = []
      for (const manifest of manifests) {
        const path = resolveModelPath(manifest)
        if (!(await args.adapter.exists(path))) continue
        const info = await args.adapter.stat(path)
        downloaded.push({
          manifest,
          path,
          sizeBytes: info.size,
        })
      }
      return downloaded
    },
    download: async (modelId, onProgress) => {
      const manifest = manifestMap.get(modelId)
      if (manifest === undefined) {
        throw new Error(`model manager: unknown model ${modelId}`)
      }
      if (manifest.downloadUrl === undefined || manifest.downloadUrl === '') {
        throw new Error(`model manager: manifest ${modelId} has no download URL`)
      }
      await args.adapter.ensureDirectory(args.adapter.join(args.root, manifest.id))
      await args.downloadTransport.download({
        modelId,
        url: manifest.downloadUrl,
        destination: resolveModelPath(manifest),
        ...(onProgress === undefined
          ? {}
          : {
              onProgress: (progress: {
                readonly downloadedBytes: number
                readonly totalBytes?: number
              }) => {
                const total = progress.totalBytes ?? manifest.sizeBytes
                if (total <= 0) return
                onProgress(Math.min(100, Math.round((progress.downloadedBytes / total) * 100)))
              },
            }),
      })
    },
    cancelDownload: (modelId) => {
      args.downloadTransport.cancel(modelId)
    },
    deleteModel: async (modelId) => {
      const manifest = manifestMap.get(modelId)
      if (manifest === undefined) return
      const path = resolveModelPath(manifest)
      if (await args.adapter.exists(path)) {
        await args.adapter.delete(path)
      }
      const directory = args.adapter.join(args.root, manifest.id)
      if (await args.adapter.exists(directory)) {
        await args.adapter.delete(directory)
      }
    },
    getModelPath: (modelId) => {
      const manifest = manifestMap.get(modelId)
      if (manifest === undefined) return null
      return resolveModelPath(manifest)
    },
    storageUsed: async () => {
      const downloaded = await (async () => {
        let total = 0
        for (const manifest of manifests) {
          const path = resolveModelPath(manifest)
          if (!(await args.adapter.exists(path))) continue
          total += (await args.adapter.stat(path)).size
        }
        return total
      })()
      return downloaded
    },
  }
}
