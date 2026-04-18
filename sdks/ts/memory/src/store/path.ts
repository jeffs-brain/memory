import { ErrInvalidPath } from './errors.js'

export type Path = string & { readonly __brand: 'BrainPath' }

export const validatePath = (p: string): void => {
  if (p === '') throw new ErrInvalidPath('empty path')
  if (p.includes('\0')) throw new ErrInvalidPath(`contains null byte: ${JSON.stringify(p)}`)
  if (p.includes('\\')) throw new ErrInvalidPath(`contains backslash: ${p}`)
  if (p.startsWith('/')) throw new ErrInvalidPath(`leading slash: ${p}`)
  if (p.endsWith('/')) throw new ErrInvalidPath(`trailing slash: ${p}`)
  const cleaned = cleanPosix(p)
  if (cleaned !== p) throw new ErrInvalidPath(`non-canonical: ${p}`)
  for (const part of p.split('/')) {
    if (part === '..') throw new ErrInvalidPath(`contains ..: ${p}`)
    if (part === '.') throw new ErrInvalidPath(`contains .: ${p}`)
    if (part === '') throw new ErrInvalidPath(`contains empty segment: ${p}`)
  }
}

export const toPath = (p: string): Path => {
  validatePath(p)
  return p as Path
}

export const isGenerated = (p: Path | string): boolean => {
  const base = lastSegment(p)
  return base.startsWith('_')
}

export const lastSegment = (p: string): string => {
  const idx = p.lastIndexOf('/')
  return idx === -1 ? p : p.slice(idx + 1)
}

export const validatePathSegment = (segment: string): void => {
  if (segment === '') throw new ErrInvalidPath('empty path segment')
  if (segment.includes('\0')) {
    throw new ErrInvalidPath(`path segment contains null byte: ${JSON.stringify(segment)}`)
  }
  if (segment.includes('/')) throw new ErrInvalidPath(`path segment contains slash: ${segment}`)
  if (segment.includes('\\')) {
    throw new ErrInvalidPath(`path segment contains backslash: ${segment}`)
  }
  if (segment === '.' || segment === '..') {
    throw new ErrInvalidPath(`path segment is not canonical: ${segment}`)
  }
}

export const joinPath = (...parts: readonly string[]): Path => {
  const joined = parts.filter((s) => s.length > 0).join('/')
  return toPath(cleanPosix(joined))
}

export const pathUnder = (p: string, dir: string, recursive: boolean): boolean => {
  if (dir === '') return true
  if (p === dir) return true
  if (!(p.length > dir.length && p.startsWith(dir) && p[dir.length] === '/')) return false
  if (recursive) return true
  const rest = p.slice(dir.length + 1)
  return !rest.includes('/')
}

// cleanPosix mirrors Go's path.Clean: collapse ".", "..", and duplicate
// separators. Kept local to avoid pulling in node:path which normalises
// to platform separators on Windows.
const cleanPosix = (p: string): string => {
  if (p === '') return '.'
  const rooted = p.startsWith('/')
  const segments = p.split('/')
  const stack: string[] = []
  for (const seg of segments) {
    if (seg === '' || seg === '.') continue
    if (seg === '..') {
      if (stack.length > 0 && stack[stack.length - 1] !== '..') {
        stack.pop()
      } else if (!rooted) {
        stack.push('..')
      }
      continue
    }
    stack.push(seg)
  }
  const out = stack.join('/')
  if (rooted) return `/${out}`
  return out === '' ? '.' : out
}

// matchGlob applies simple shell-glob semantics over a single base name —
// the same subset Go's path.Match supports: `*` matches any run of
// non-separator characters, `?` matches a single character, `[...]`
// matches one of the enclosed characters.
export const matchGlob = (pattern: string, name: string): boolean => {
  return matchGlobAt(pattern, 0, name, 0)
}

const matchGlobAt = (pattern: string, pi: number, name: string, ni: number): boolean => {
  let patternIndex = pi
  let nameIndex = ni
  while (patternIndex < pattern.length) {
    const pc = pattern[patternIndex]
    if (pc === '*') {
      // Collapse repeated stars.
      while (patternIndex < pattern.length && pattern[patternIndex] === '*') patternIndex++
      if (patternIndex === pattern.length) return true
      for (let k = nameIndex; k <= name.length; k++) {
        if (matchGlobAt(pattern, patternIndex, name, k)) return true
      }
      return false
    }
    if (pc === '?') {
      if (nameIndex >= name.length) return false
      patternIndex++
      nameIndex++
      continue
    }
    if (pc === '[') {
      const close = pattern.indexOf(']', patternIndex + 1)
      if (close === -1 || nameIndex >= name.length) return false
      const set = pattern.slice(patternIndex + 1, close)
      const target = name[nameIndex] as string
      let negate = false
      let idx = 0
      if (set[0] === '!' || set[0] === '^') {
        negate = true
        idx = 1
      }
      let matched = false
      while (idx < set.length) {
        const a = set[idx] as string
        if (idx + 2 < set.length && set[idx + 1] === '-') {
          const b = set[idx + 2] as string
          if (target >= a && target <= b) matched = true
          idx += 3
        } else {
          if (target === a) matched = true
          idx += 1
        }
      }
      if (matched === negate) return false
      patternIndex = close + 1
      nameIndex += 1
      continue
    }
    if (nameIndex >= name.length || name[nameIndex] !== pc) return false
    patternIndex++
    nameIndex++
  }
  return nameIndex === name.length
}
