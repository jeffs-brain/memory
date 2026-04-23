import type { FileAdapter, PortableFileInfo } from './index.js'

const stripTrailingSlash = (value: string): string => {
  if (value === '/') return value
  return value.endsWith('/') ? value.slice(0, -1) : value
}

const joinUri = (root: string, path: string): string => {
  const base = stripTrailingSlash(root)
  return path === '' ? base : `${base}/${path.replace(/^\/+/, '')}`
}

const basename = (uri: string): string => {
  const trimmed = stripTrailingSlash(uri)
  const index = trimmed.lastIndexOf('/')
  return index === -1 ? trimmed : trimmed.slice(index + 1)
}

export const createExpoFileAdapter = async (): Promise<FileAdapter> => {
  const fs = await import('expo-file-system/legacy')

  const toInfo = async (uri: string): Promise<PortableFileInfo> => {
    const info = await fs.getInfoAsync(uri, { size: true })
    return {
      uri,
      size: info.size ?? 0,
      modTime: new Date((info.modificationTime ?? Date.now() / 1000) * 1000),
      isDir: info.isDirectory,
    }
  }

  return {
    kind: 'expo-file-system',
    join: (root, path) => joinUri(root, path),
    ensureDirectory: async (uri) => {
      await fs.makeDirectoryAsync(uri, { intermediates: true })
    },
    readText: async (uri) => await fs.readAsStringAsync(uri, { encoding: fs.EncodingType.UTF8 }),
    writeText: async (uri, content) => {
      await fs.writeAsStringAsync(uri, content, { encoding: fs.EncodingType.UTF8 })
    },
    appendText: async (uri, content) => {
      const existing = await fs
        .readAsStringAsync(uri, { encoding: fs.EncodingType.UTF8 })
        .catch(() => '')
      await fs.writeAsStringAsync(uri, `${existing}${content}`, { encoding: fs.EncodingType.UTF8 })
    },
    delete: async (uri) => {
      await fs.deleteAsync(uri)
    },
    rename: async (from, to) => {
      await fs.moveAsync({ from, to })
    },
    exists: async (uri) => {
      const info = await fs.getInfoAsync(uri)
      return info.exists
    },
    stat: async (uri) => await toInfo(uri),
    list: async (uri) => {
      const names = await fs.readDirectoryAsync(uri)
      const entries = await Promise.all(names.map((name) => toInfo(joinUri(uri, name))))
      return entries
    },
    basename,
  }
}

export const resolveExpoDocumentDirectory = async (): Promise<string> => {
  const fs = await import('expo-file-system/legacy')
  if (fs.documentDirectory === null) {
    throw new Error('expo-file-system: documentDirectory unavailable')
  }
  return fs.documentDirectory
}
