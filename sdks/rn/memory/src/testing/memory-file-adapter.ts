import type { FileAdapter, PortableFileInfo } from '../store/index.js'

type Entry =
  | {
      readonly kind: 'dir'
      modTime: Date
    }
  | {
      readonly kind: 'file'
      modTime: Date
      content: string
    }

const normaliseUri = (uri: string): string => {
  if (uri === '') return '/'
  let output = uri.replace(/\/+/g, '/')
  if (!output.startsWith('/')) output = `/${output}`
  if (output.length > 1 && output.endsWith('/')) output = output.slice(0, -1)
  return output
}

const parentUri = (uri: string): string => {
  const value = normaliseUri(uri)
  if (value === '/') return '/'
  const index = value.lastIndexOf('/')
  return index <= 0 ? '/' : value.slice(0, index)
}

const basename = (uri: string): string => {
  const value = normaliseUri(uri)
  if (value === '/') return ''
  const index = value.lastIndexOf('/')
  return index === -1 ? value : value.slice(index + 1)
}

export const createMemoryFileAdapter = (): FileAdapter => {
  const entries = new Map<string, Entry>()
  entries.set('/', { kind: 'dir', modTime: new Date() })

  const ensureDirectory = async (uri: string): Promise<void> => {
    const target = normaliseUri(uri)
    const parts = target.split('/').filter((part) => part.length > 0)
    let current = '/'
    for (const part of parts) {
      current = current === '/' ? `/${part}` : `${current}/${part}`
      const existing = entries.get(current)
      if (existing === undefined) {
        entries.set(current, { kind: 'dir', modTime: new Date() })
        continue
      }
      if (existing.kind !== 'dir') {
        throw new Error(`not a directory: ${current}`)
      }
    }
  }

  const stat = async (uri: string): Promise<PortableFileInfo> => {
    const target = normaliseUri(uri)
    const entry = entries.get(target)
    if (entry === undefined) throw new Error(`missing entry: ${target}`)
    return {
      uri: target,
      size: entry.kind === 'dir' ? 0 : entry.content.length,
      modTime: entry.modTime,
      isDir: entry.kind === 'dir',
    }
  }

  return {
    kind: 'memory-file-adapter',
    join: (root, path) => normaliseUri(path === '' ? root : `${normaliseUri(root)}/${path}`),
    ensureDirectory,
    readText: async (uri) => {
      const target = normaliseUri(uri)
      const entry = entries.get(target)
      if (entry === undefined || entry.kind !== 'file') throw new Error(`missing file: ${target}`)
      return entry.content
    },
    writeText: async (uri, content) => {
      const target = normaliseUri(uri)
      await ensureDirectory(parentUri(target))
      entries.set(target, { kind: 'file', content, modTime: new Date() })
    },
    appendText: async (uri, content) => {
      const target = normaliseUri(uri)
      await ensureDirectory(parentUri(target))
      const existing = entries.get(target)
      if (existing === undefined || existing.kind !== 'file') {
        entries.set(target, { kind: 'file', content, modTime: new Date() })
        return
      }
      entries.set(target, {
        kind: 'file',
        content: `${existing.content}${content}`,
        modTime: new Date(),
      })
    },
    delete: async (uri) => {
      const target = normaliseUri(uri)
      const existing = entries.get(target)
      if (existing === undefined) throw new Error(`missing entry: ${target}`)
      entries.delete(target)
      if (existing.kind === 'dir') {
        for (const key of [...entries.keys()]) {
          if (key.startsWith(`${target}/`)) entries.delete(key)
        }
      }
    },
    rename: async (from, to) => {
      const source = normaliseUri(from)
      const target = normaliseUri(to)
      const existing = entries.get(source)
      if (existing === undefined) throw new Error(`missing entry: ${source}`)
      await ensureDirectory(parentUri(target))
      entries.delete(source)
      entries.set(target, existing)
      if (existing.kind === 'dir') {
        const descendants = [...entries.entries()].filter(([key]) => key.startsWith(`${source}/`))
        for (const [key, value] of descendants) {
          entries.delete(key)
          entries.set(key.replace(source, target), value)
        }
      }
    },
    exists: async (uri) => entries.has(normaliseUri(uri)),
    stat,
    list: async (uri) => {
      const target = normaliseUri(uri)
      const prefix = target === '/' ? '/' : `${target}/`
      const out: PortableFileInfo[] = []
      const seen = new Set<string>()
      for (const key of entries.keys()) {
        if (!key.startsWith(prefix) || key === target) continue
        const rest = key.slice(prefix.length)
        if (rest === '' || rest.includes('/')) continue
        if (seen.has(key)) continue
        seen.add(key)
        out.push(await stat(key))
      }
      out.sort((left, right) => left.uri.localeCompare(right.uri))
      return out
    },
    basename,
  }
}
