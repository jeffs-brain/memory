import type { FileAdapter } from '../store/index.js'
import { joinPath } from '../store/index.js'
import type { ModelDownloadTransport } from './download.js'
import {
  DEFAULT_MODEL_MANIFESTS,
  type ModelFileManifest,
  type ModelManifest,
  listModelFiles,
  modelBundleSizeBytes,
} from './registry.js'

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

  const resolveModelPath = (manifest: ModelManifest, filename: string): string => {
    return args.adapter.join(args.root, joinPath(manifest.id, filename))
  }

  const hasAuthorisationHeader = (
    headers: Readonly<Record<string, string>> | undefined,
  ): boolean => {
    if (headers === undefined) return false
    return Object.entries(headers).some(
      ([key, value]) => key.toLowerCase() === 'authorization' && value.trim() !== '',
    )
  }

  const assertDownloadable = (
    manifest: ModelManifest,
    files: readonly ModelFileManifest[],
  ): void => {
    for (const file of files) {
      if (file.downloadUrl === undefined || file.downloadUrl === '') {
        throw new Error(
          `model manager: manifest ${manifest.id} has no download URL; use createUpstreamModelManifests() or createHostedModelManifests()`,
        )
      }
      if (file.source?.provider === 'huggingface' && file.source.gated === true) {
        if (!hasAuthorisationHeader(file.downloadHeaders)) {
          throw new Error(
            `model manager: manifest ${manifest.id} points to a gated Hugging Face artefact; pass createUpstreamModelManifests({ huggingFaceToken }) or mirror it with createHostedModelManifests()`,
          )
        }
      }
    }
  }

  const sumExistingModelBytes = async (manifest: ModelManifest): Promise<number> => {
    let total = 0
    for (const file of listModelFiles(manifest)) {
      const path = resolveModelPath(manifest, file.filename)
      if (!(await args.adapter.exists(path))) continue
      total += (await args.adapter.stat(path)).size
    }
    return total
  }

  return {
    listAvailable: () => manifests,
    listDownloaded: async () => {
      const downloaded: DownloadedModel[] = []
      for (const manifest of manifests) {
        const path = resolveModelPath(manifest, manifest.filename)
        if (!(await args.adapter.exists(path))) continue
        downloaded.push({
          manifest,
          path,
          sizeBytes: await sumExistingModelBytes(manifest),
        })
      }
      return downloaded
    },
    download: async (modelId, onProgress) => {
      const manifest = manifestMap.get(modelId)
      if (manifest === undefined) {
        throw new Error(`model manager: unknown model ${modelId}`)
      }

      const files = listModelFiles(manifest)
      assertDownloadable(manifest, files)

      await args.adapter.ensureDirectory(args.adapter.join(args.root, manifest.id))

      const totalBytes = modelBundleSizeBytes(manifest)
      let completedBytes = 0

      for (const file of files) {
        await args.downloadTransport.download({
          modelId,
          url: file.downloadUrl ?? '',
          destination: resolveModelPath(manifest, file.filename),
          ...(file.downloadHeaders === undefined ? {} : { headers: file.downloadHeaders }),
          ...(onProgress === undefined
            ? {}
            : {
                onProgress: (progress: {
                  readonly downloadedBytes: number
                  readonly totalBytes?: number
                }) => {
                  const fileTotal = progress.totalBytes ?? file.sizeBytes
                  if (fileTotal <= 0 || totalBytes <= 0) return
                  const absolute = completedBytes + Math.min(progress.downloadedBytes, fileTotal)
                  onProgress(Math.min(100, Math.round((absolute / totalBytes) * 100)))
                },
              }),
        })
        completedBytes += file.sizeBytes
        if (onProgress !== undefined && totalBytes > 0) {
          onProgress(Math.min(100, Math.round((completedBytes / totalBytes) * 100)))
        }
      }
    },
    cancelDownload: (modelId) => {
      args.downloadTransport.cancel(modelId)
    },
    deleteModel: async (modelId) => {
      const manifest = manifestMap.get(modelId)
      if (manifest === undefined) return
      const path = resolveModelPath(manifest, manifest.filename)
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
      return resolveModelPath(manifest, manifest.filename)
    },
    storageUsed: async () => {
      const downloaded = await (async () => {
        let total = 0
        for (const manifest of manifests) {
          total += await sumExistingModelBytes(manifest)
        }
        return total
      })()
      return downloaded
    },
  }
}
