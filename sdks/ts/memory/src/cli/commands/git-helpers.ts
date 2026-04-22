// SPDX-License-Identifier: Apache-2.0

import { execFile as execFileCallback } from 'node:child_process'
import * as nodeFs from 'node:fs'
import { access, readFile, readdir, rm, stat, writeFile } from 'node:fs/promises'
import { basename, join } from 'node:path'
import { promisify } from 'node:util'
import * as git from 'isomorphic-git'
import { DEFAULT_GIT_SIGNATURE, buildGitExecEnv } from '../../store/git-author.js'
import { readGitLog, validatePath } from '../../store/index.js'
import { CliError } from '../config.js'

const execFile = promisify(execFileCallback)

export const DEFAULT_REMOTE = 'origin'
const MAX_REASON_FILTER_MULTIPLIER = 5

const LOCKFILES = new Set([
  '.ds_store',
  'thumbs.db',
  'bun.lock',
  'bun.lockb',
  'package-lock.json',
  'pnpm-lock.yaml',
  'yarn.lock',
])

const JUNK_SEGMENTS = new Set([
  '.next',
  '.turbo',
  'build',
  'coverage',
  'dist',
  'node_modules',
  'vendor',
])

const GENERATED_SUFFIXES = ['.map', '.min.css', '.min.js']

export type GitChange = {
  readonly code: string
  readonly label: string
  readonly path: string
}

export type BrainFile = {
  readonly path: string
  readonly abs: string
  readonly size: number
  readonly modTime: Date
}

export type VerifyCheck = {
  readonly name: string
  readonly ok: boolean
  readonly detail: string
}

export type CleanReport = {
  readonly junkDirs: readonly string[]
  readonly lockfiles: readonly string[]
  readonly generated: readonly string[]
  readonly oversized: readonly string[]
  readonly totalFound: number
}

export type GitLogEntry = {
  readonly oid: string
  readonly message: string
  readonly reason: string
  readonly rest: string
  readonly relativeTime: string
  readonly primaryFile: string
  readonly fileCount: number
  readonly authorName: string
  readonly authorEmail: string
  readonly timestamp: number
}

export const writeJson = (payload: Record<string, unknown>): void => {
  process.stdout.write(`${JSON.stringify(payload)}\n`)
}

export const resolveBranchArg = async (
  brainDir: string,
  raw: string | undefined,
): Promise<string> => {
  if (raw !== undefined && raw !== '') return raw
  const current = await git.currentBranch({ fs: nodeFs, dir: brainDir, fullname: false })
  return current ?? 'main'
}

export const requireGitBrain = async (brainDir: string): Promise<void> => {
  try {
    await access(join(brainDir, '.git'))
  } catch {
    throw new CliError(`brain is not git-backed: ${brainDir}`)
  }
}

export const runGit = async (brainDir: string, args: readonly string[]): Promise<string> => {
  try {
    const env = await buildGitExecEnv(brainDir, args, DEFAULT_GIT_SIGNATURE)
    const { stdout } = await execFile('git', ['-C', brainDir, ...args], {
      encoding: 'utf8',
      ...(env !== undefined ? { env } : {}),
    })
    return stdout
  } catch (err) {
    throw new CliError(`git ${args.join(' ')}: ${describeExecFailure(err)}`)
  }
}

export const tryReadAheadBehind = async (
  brainDir: string,
  branch: string,
): Promise<{ readonly ahead: number | null; readonly behind: number | null }> => {
  try {
    const raw = await runGit(brainDir, [
      'rev-list',
      '--left-right',
      '--count',
      `HEAD...origin/${branch}`,
    ])
    const [aheadRaw, behindRaw] = raw.trim().split(/\s+/, 2)
    const ahead = Number.parseInt(aheadRaw ?? '0', 10)
    const behind = Number.parseInt(behindRaw ?? '0', 10)
    if (!Number.isFinite(ahead) || !Number.isFinite(behind)) {
      return { ahead: null, behind: null }
    }
    return { ahead, behind }
  } catch {
    return { ahead: null, behind: null }
  }
}

export const readRemoteUrl = async (
  brainDir: string,
  remote: string = DEFAULT_REMOTE,
): Promise<string | null> => {
  const value = await git.getConfig({
    fs: nodeFs,
    dir: brainDir,
    path: `remote.${remote}.url`,
  })
  return typeof value === 'string' && value.trim() !== '' ? value : null
}

export const readDiff = async (brainDir: string): Promise<readonly GitChange[]> => {
  const raw = await runGit(brainDir, ['status', '--porcelain'])
  const changes: GitChange[] = []
  for (const line of raw.split('\n')) {
    if (line.trim() === '' || line.length < 4) continue
    const code = line.slice(0, 2)
    const path = line.slice(3).trim()
    changes.push({
      code,
      label: statusLabel(code),
      path: normalisePorcelainPath(path),
    })
  }
  return changes
}

export const summariseDiff = (
  changes: readonly GitChange[],
): {
  readonly total: number
  readonly modified: number
  readonly added: number
  readonly deleted: number
  readonly conflicted: number
  readonly other: number
} => {
  let modified = 0
  let added = 0
  let deleted = 0
  let conflicted = 0
  let other = 0
  for (const change of changes) {
    switch (change.label) {
      case 'Modified':
        modified++
        break
      case 'Added':
        added++
        break
      case 'Deleted':
        deleted++
        break
      case 'Conflicted':
        conflicted++
        break
      default:
        other++
        break
    }
  }
  return {
    total: changes.length,
    modified,
    added,
    deleted,
    conflicted,
    other,
  }
}

export const statusLabel = (code: string): string => {
  switch (true) {
    case code === '??' || code === 'A ' || code === ' A':
      return 'Added'
    case code === 'D ' || code === ' D':
      return 'Deleted'
    case code === 'M ' || code === ' M' || code === 'MM':
      return 'Modified'
    case code === 'R ' || code.startsWith('R'):
      return 'Renamed'
    case code === 'C ' || code.startsWith('C'):
      return 'Copied'
    case code.includes('U') || code === 'DD' || code === 'AA':
      return 'Conflicted'
    default:
      return 'Changed'
  }
}

export const parseReason = (
  subject: string,
): { readonly reason: string; readonly rest: string } => {
  const trimmed = subject.trim()
  if (!trimmed.startsWith('[')) {
    return { reason: '', rest: trimmed }
  }
  const end = trimmed.indexOf(']')
  if (end < 0) {
    return { reason: '', rest: trimmed }
  }
  return {
    reason: trimmed.slice(1, end),
    rest: trimmed.slice(end + 1).trim(),
  }
}

export const relativeTime = (now: Date, when: Date): string => {
  const deltaMs = now.getTime() - when.getTime()
  if (deltaMs < 60_000) return 'just now'
  if (deltaMs < 3_600_000) return `${String(Math.floor(deltaMs / 60_000))}m ago`
  if (deltaMs < 86_400_000) return `${String(Math.floor(deltaMs / 3_600_000))}h ago`
  if (deltaMs < 30 * 86_400_000) return `${String(Math.floor(deltaMs / 86_400_000))}d ago`
  if (deltaMs < 365 * 86_400_000) return `${String(Math.floor(deltaMs / (30 * 86_400_000)))}mo ago`
  return `${String(Math.floor(deltaMs / (365 * 86_400_000)))}y ago`
}

export const readFormattedLog = async (args: {
  readonly brainDir: string
  readonly limit: number
  readonly reasonFilter: string
}): Promise<readonly GitLogEntry[]> => {
  const fetchDepth =
    args.reasonFilter === ''
      ? args.limit
      : Math.max(args.limit * MAX_REASON_FILTER_MULTIPLIER, args.limit)
  const now = new Date()
  const out: GitLogEntry[] = []
  for (const entry of await readGitLog(args.brainDir, fetchDepth)) {
    const { reason, rest } = parseReason(entry.message)
    if (args.reasonFilter !== '' && reason !== args.reasonFilter) continue
    const files = await listChangedFiles(args.brainDir, entry.oid)
    out.push({
      oid: entry.oid,
      message: entry.message,
      reason,
      rest,
      relativeTime: relativeTime(now, new Date(entry.timestamp * 1000)),
      primaryFile: files[0] ?? '',
      fileCount: files.length,
      authorName: entry.authorName,
      authorEmail: entry.authorEmail,
      timestamp: entry.timestamp,
    })
    if (out.length >= args.limit) break
  }
  return out
}

export const listConflictedFiles = async (brainDir: string): Promise<readonly string[]> => {
  const changes = await readDiff(brainDir)
  return changes.filter((change) => change.label === 'Conflicted').map((change) => change.path)
}

export const listChangedFiles = async (
  brainDir: string,
  oid: string,
): Promise<readonly string[]> => {
  const raw = await runGit(brainDir, [
    'diff-tree',
    '--root',
    '--no-commit-id',
    '--name-only',
    '-r',
    oid,
  ])
  return raw
    .split('\n')
    .map((line) => line.trim())
    .filter((line) => line !== '')
}

export const abortMergeOrRebase = async (brainDir: string): Promise<void> => {
  const gitDir = join(brainDir, '.git')
  if (await exists(join(gitDir, 'rebase-merge'))) {
    await runGit(brainDir, ['rebase', '--abort'])
    return
  }
  if (await exists(join(gitDir, 'rebase-apply'))) {
    await runGit(brainDir, ['rebase', '--abort'])
    return
  }
  await runGit(brainDir, ['merge', '--abort'])
}

export const batchResolve = async (
  brainDir: string,
  files: readonly string[],
  side: 'ours' | 'theirs',
): Promise<number> => {
  for (const file of files) {
    await runGit(brainDir, ['checkout', `--${side}`, '--', file])
    await runGit(brainDir, ['add', '--', file])
  }
  return files.length
}

export const autoResolve = async (brainDir: string, files: readonly string[]): Promise<number> => {
  let resolved = 0
  for (const file of files) {
    if (file === '_log.md') {
      await resolveLogConcat(brainDir, file)
      resolved++
      continue
    }
    if (basename(file).startsWith('_')) {
      await runGit(brainDir, ['checkout', '--theirs', '--', file])
      await runGit(brainDir, ['add', '--', file])
      resolved++
    }
  }
  return resolved
}

export const listBrainFiles = async (
  brainDir: string,
  prefixes: readonly string[] = [''],
): Promise<readonly BrainFile[]> => {
  const out: BrainFile[] = []
  const seen = new Set<string>()
  for (const prefix of prefixes) {
    const abs = prefix === '' ? brainDir : join(brainDir, prefix)
    await walkFiles(brainDir, abs, prefix, out, seen)
  }
  out.sort((left, right) => left.path.localeCompare(right.path))
  return out
}

export const readSectionStats = async (
  brainDir: string,
): Promise<
  readonly {
    readonly name: string
    readonly count: number
    readonly bytes: number
  }[]
> => {
  const sections = [
    { name: 'memory', prefixes: ['memory'] },
    { name: 'reflections', prefixes: ['reflections'] },
    { name: 'wiki', prefixes: ['wiki'] },
    { name: 'drafts', prefixes: ['drafts'] },
    { name: 'raw/documents', prefixes: ['raw/documents'] },
  ] as const
  const out: Array<{ name: string; count: number; bytes: number }> = []
  for (const section of sections) {
    const files = await listBrainFiles(brainDir, section.prefixes)
    out.push({
      name: section.name,
      count: files.length,
      bytes: files.reduce((sum, file) => sum + file.size, 0),
    })
  }
  return out
}

export const runVerifyChecks = async (brainDir: string): Promise<readonly VerifyCheck[]> => {
  const checks: VerifyCheck[] = []
  const branch = await git.currentBranch({ fs: nodeFs, dir: brainDir, fullname: false })
  checks.push({
    name: 'branch',
    ok: branch !== null,
    detail: branch ?? 'no HEAD branch',
  })

  const staged = (await runGit(brainDir, ['diff', '--cached', '--name-only'])).trim()
  checks.push({
    name: 'git-index',
    ok: staged === '',
    detail: staged === '' ? 'clean' : `${String(staged.split('\n').length)} staged file(s)`,
  })

  const conflicts = await listConflictedFiles(brainDir)
  checks.push({
    name: 'conflicts',
    ok: conflicts.length === 0,
    detail: conflicts.length === 0 ? 'none' : `${String(conflicts.length)} conflicted file(s)`,
  })

  let recursiveOk = true
  let files: readonly BrainFile[] = []
  try {
    files = await listBrainFiles(brainDir)
  } catch (err) {
    recursiveOk = false
    checks.push({
      name: 'recursive-list',
      ok: false,
      detail: err instanceof Error ? err.message : String(err),
    })
  }
  if (recursiveOk) {
    checks.push({
      name: 'recursive-list',
      ok: true,
      detail: `${String(files.length)} file(s)`,
    })
  }

  let invalid = 0
  for (const file of files) {
    try {
      validatePath(file.path)
    } catch {
      invalid++
    }
  }
  checks.push({
    name: 'path-validation',
    ok: invalid === 0,
    detail:
      invalid === 0
        ? `${String(files.length)} valid path(s)`
        : `${String(invalid)} invalid path(s)`,
  })
  return checks
}

export const readCleanReport = (files: readonly BrainFile[], maxSizeBytes: number): CleanReport => {
  const junkDirs: string[] = []
  const lockfiles: string[] = []
  const generated: string[] = []
  const oversized: string[] = []
  for (const file of files) {
    const lower = file.path.toLowerCase()
    const parts = lower.split('/')
    const leaf = parts[parts.length - 1] ?? lower
    if (parts.some((part) => JUNK_SEGMENTS.has(part))) {
      junkDirs.push(file.path)
      continue
    }
    if (LOCKFILES.has(leaf)) {
      lockfiles.push(file.path)
      continue
    }
    if (GENERATED_SUFFIXES.some((suffix) => leaf.endsWith(suffix))) {
      generated.push(file.path)
      continue
    }
    if (file.size > maxSizeBytes) {
      oversized.push(file.path)
    }
  }
  return {
    junkDirs,
    lockfiles,
    generated,
    oversized,
    totalFound: junkDirs.length + lockfiles.length + generated.length + oversized.length,
  }
}

export const scopePrefixes = (scopeRaw: string): readonly string[] => {
  const scope = scopeRaw.trim().toLowerCase()
  switch (scope) {
    case '':
    case 'all':
      return ['memory', 'reflections', 'wiki', 'drafts', 'raw/documents']
    case 'raw':
    case 'raw/documents':
      return ['raw/documents']
    case 'wiki':
      return ['wiki', 'drafts']
    case 'drafts':
      return ['drafts']
    case 'memory':
      return ['memory', 'reflections']
    case 'reflections':
      return ['reflections']
    default:
      throw new CliError(
        `unknown scope '${scopeRaw}'; expected all|raw|wiki|drafts|memory|reflections`,
      )
  }
}

export const deletePaths = async (brainDir: string, paths: readonly string[]): Promise<void> => {
  const unique = [...new Set(paths)].sort()
  for (const file of unique) {
    try {
      await rm(join(brainDir, file), { force: true, recursive: false })
    } catch {
      // Best effort on direct filesystem cleanup before staging.
    }
  }
}

export const stageAndCommit = async (args: {
  readonly brainDir: string
  readonly paths: readonly string[]
  readonly message: string
}): Promise<boolean> => {
  const unique = [...new Set(args.paths)].sort()
  if (unique.length === 0) return false
  await runGit(args.brainDir, ['add', '-A', '--', ...unique])
  const diff = (
    await runGit(args.brainDir, ['diff', '--cached', '--name-only', '--', ...unique])
  ).trim()
  if (diff === '') return false
  await runGit(args.brainDir, ['commit', '-m', args.message, '--', ...unique])
  return true
}

const walkFiles = async (
  brainDir: string,
  currentAbs: string,
  currentRel: string,
  out: BrainFile[],
  seen: Set<string>,
): Promise<void> => {
  let entries: nodeFs.Dirent[]
  try {
    entries = await readdir(currentAbs, { withFileTypes: true })
  } catch (err) {
    if ((err as NodeJS.ErrnoException).code === 'ENOENT') return
    throw err
  }
  for (const entry of entries) {
    if (entry.name === '.git') continue
    const rel = currentRel === '' ? entry.name : `${currentRel}/${entry.name}`
    const abs = join(currentAbs, entry.name)
    if (entry.isDirectory()) {
      await walkFiles(brainDir, abs, rel, out, seen)
      continue
    }
    if (!entry.isFile()) continue
    if (seen.has(rel)) continue
    const info = await stat(abs)
    seen.add(rel)
    out.push({
      path: rel,
      abs,
      size: info.size,
      modTime: info.mtime,
    })
  }
}

const resolveLogConcat = async (brainDir: string, file: string): Promise<void> => {
  const theirs = await readGitBlob(brainDir, `:3:${file}`)
  const ours = await readGitBlob(brainDir, `:2:${file}`)
  const theirsLines = theirs.split('\n')
  const seen = new Set(theirsLines)
  const merged = [...theirsLines]
  for (const line of ours.split('\n')) {
    if (seen.has(line)) continue
    merged.push(line)
    seen.add(line)
  }
  await writeFile(join(brainDir, file), merged.join('\n'), 'utf8')
  await runGit(brainDir, ['add', '--', file])
}

const readGitBlob = async (brainDir: string, spec: string): Promise<string> => {
  const raw = await runGit(brainDir, ['show', spec])
  return raw
}

const normalisePorcelainPath = (path: string): string => {
  const arrow = ' -> '
  if (!path.includes(arrow)) return path
  return path
}

const exists = async (path: string): Promise<boolean> => {
  try {
    await access(path)
    return true
  } catch {
    return false
  }
}

const describeExecFailure = (err: unknown): string => {
  if (!(err instanceof Error)) return String(err)
  const failure = err as Error & {
    readonly stderr?: string | Buffer
    readonly stdout?: string | Buffer
  }
  if (typeof failure.stderr === 'string' && failure.stderr.trim() !== '') {
    return failure.stderr.trim()
  }
  if (Buffer.isBuffer(failure.stderr) && failure.stderr.length > 0) {
    return failure.stderr.toString('utf8').trim()
  }
  return err.message
}
