// SPDX-License-Identifier: Apache-2.0

import { execFile as execFileCallback } from 'node:child_process'
import * as nodeFs from 'node:fs'
import { access, mkdir, readdir } from 'node:fs/promises'
import { basename, dirname, isAbsolute, join, resolve } from 'node:path'
import { promisify } from 'node:util'
import * as git from 'isomorphic-git'
import { ErrConflict, ErrReadOnly, StoreError } from './errors.js'
import { type FsStore, createFsStore } from './fsstore.js'
import type {
  Batch,
  BatchOptions,
  ChangeEvent,
  EventSink,
  FileInfo,
  ListOpts,
  Store,
  Unsubscribe,
} from './index.js'
import type { Path } from './path.js'

export type GitSignature = {
  readonly name: string
  readonly email: string
}

export type GitSignPayload = {
  readonly payload: string
}

export type GitSignFn = (payload: GitSignPayload) => Promise<string>

export type GitStoreOptions = {
  readonly dir: string
  readonly branch?: string
  readonly init?: boolean
  readonly defaultAuthor?: GitSignature
  readonly sign?: GitSignFn
  readonly remoteUrl?: string
  readonly autoPush?: boolean
}

const DEFAULT_AUTHOR: GitSignature = {
  name: 'jeffs-brain',
  email: 'noreply@jeffsbrain.com',
}

const DEFAULT_BRANCH = 'main'
const DEFAULT_REMOTE = 'origin'
const AUTO_STASH_MESSAGE = 'jeffs-brain-sync-autostash'
const REMOTE_FETCH_SPEC = '+refs/heads/*:refs/remotes/origin/*'
const execFile = promisify(execFileCallback)

export const createGitStore = async (opts: GitStoreOptions): Promise<GitStore> => {
  const abs = isAbsolute(opts.dir) ? opts.dir : resolve(opts.dir)
  await mkdir(abs, { recursive: true })

  const gitdir = join(abs, '.git')
  const hasRepo = await exists(gitdir)
  const requestedBranch = opts.branch ?? DEFAULT_BRANCH
  const defaultAuthor = opts.defaultAuthor ?? DEFAULT_AUTHOR
  const remoteUrl = cleanRemoteUrl(opts.remoteUrl)

  if (!hasRepo) {
    if (opts.init !== true) {
      throw new StoreError(`gitstore: no git repo at ${abs}; pass { init: true } to create one`)
    }
    if (remoteUrl !== undefined && (await isDirectoryEmpty(abs))) {
      const remoteState = await inspectRemoteBranch(remoteUrl, requestedBranch)
      if (remoteState === 'present') {
        await cloneRepository(abs, remoteUrl, requestedBranch)
      } else if (remoteState === 'missing-branch') {
        throw new StoreError(
          `gitstore: remote ${remoteUrl} does not have branch ${requestedBranch}`,
        )
      } else {
        await initialiseRepository({
          dir: abs,
          branch: requestedBranch,
          defaultAuthor,
          ...(opts.sign !== undefined ? { sign: opts.sign } : {}),
          remoteUrl,
        })
      }
    } else {
      await initialiseRepository({
        dir: abs,
        branch: requestedBranch,
        defaultAuthor,
        ...(opts.sign !== undefined ? { sign: opts.sign } : {}),
        ...(remoteUrl !== undefined ? { remoteUrl } : {}),
      })
    }
  } else if (remoteUrl !== undefined) {
    await configureOrigin(abs, remoteUrl)
  }

  const branch = await resolveCurrentBranch(abs, requestedBranch)
  const fs = await createFsStore({ root: abs })
  const store = new GitStore(abs, branch, defaultAuthor, fs, opts.sign, opts.autoPush === true)
  await store.openSync()
  return store
}

// isomorphic-git only invokes onSign when `signingKey` is truthy. Callers of
// gitstore supply a single `sign` callback that owns the key material itself
// (SSH agent, KMS, GPG, whatever) — the placeholder key here is never used by
// the callback, it just arms the signing path inside isomorphic-git.
const CALLBACK_SIGN_KEY_PLACEHOLDER = 'jeffs-brain-callback-sign-key'

type IsoSignCallback = (args: {
  payload: string
  secretKey: string
}) => Promise<{ signature: string }>

const wrapSign = (sign: GitSignFn): IsoSignCallback => {
  return async ({ payload }) => {
    const signature = await sign({ payload })
    return { signature }
  }
}

const signOptions = (
  sign: GitSignFn | undefined,
): { onSign: IsoSignCallback; signingKey: string } | Record<string, never> => {
  if (sign === undefined) return {}
  return { onSign: wrapSign(sign), signingKey: CALLBACK_SIGN_KEY_PLACEHOLDER }
}

export class GitStore implements Store {
  private closed = false
  private readonly sinks = new Map<number, EventSink>()
  private nextSinkId = 0
  // forwards fsstore events to gitstore subscribers. Kept so close can unhook.
  private readonly fsUnsubscribe: Unsubscribe
  // serialises commits so two concurrent batches do not stamp over each other's
  // index state. The underlying fsstore already serialises working-tree
  // mutations, so we only need to coordinate git index writes.
  private commitChain: Promise<unknown> = Promise.resolve()

  constructor(
    readonly root: string,
    readonly branch: string,
    private readonly defaultAuthor: GitSignature,
    private readonly fs: FsStore,
    private readonly sign?: GitSignFn,
    private readonly autoPush: boolean = false,
  ) {
    this.fsUnsubscribe = fs.subscribe((evt) => this.forwardEvent(evt))
  }

  async read(p: Path): Promise<Buffer> {
    this.ensureOpen()
    return this.fs.read(p)
  }

  async exists(p: Path): Promise<boolean> {
    this.ensureOpen()
    return this.fs.exists(p)
  }

  async stat(p: Path): Promise<FileInfo> {
    this.ensureOpen()
    return this.fs.stat(p)
  }

  async list(dir: Path | '', opts: ListOpts = {}): Promise<FileInfo[]> {
    this.ensureOpen()
    return this.fs.list(dir, opts)
  }

  async write(p: Path, content: Buffer): Promise<void> {
    await this.batch({ reason: 'write', message: `write ${p}` }, async (b) => {
      await b.write(p, content)
    })
  }

  async append(p: Path, content: Buffer): Promise<void> {
    await this.batch({ reason: 'append', message: `append ${p}` }, async (b) => {
      await b.append(p, content)
    })
  }

  async delete(p: Path): Promise<void> {
    await this.batch({ reason: 'delete', message: `delete ${p}` }, async (b) => {
      await b.delete(p)
    })
  }

  async rename(src: Path, dst: Path): Promise<void> {
    await this.batch({ reason: 'rename', message: `rename ${src} -> ${dst}` }, async (b) => {
      await b.rename(src, dst)
    })
  }

  async batch(opts: BatchOptions, fn: (batch: Batch) => Promise<void>): Promise<void> {
    this.ensureOpen()
    return this.serialiseCommit(async () => {
      const touched = new Set<Path>()
      await this.fs.batch(opts, async (inner) => {
        const wrapped = new TrackingBatch(inner, touched)
        await fn(wrapped)
      })
      if (touched.size === 0) return
      await this.commitTouched(touched, opts)
      if (this.autoPush) {
        await this.pushWithRetry(DEFAULT_REMOTE, this.branch)
      }
    })
  }

  subscribe(sink: EventSink): Unsubscribe {
    const id = this.nextSinkId++
    this.sinks.set(id, sink)
    return () => {
      this.sinks.delete(id)
    }
  }

  localPath(p: Path): string | undefined {
    return this.fs.localPath(p)
  }

  async close(): Promise<void> {
    if (this.closed) return
    this.closed = true
    this.fsUnsubscribe()
    this.sinks.clear()
    await this.fs.close()
  }

  async push(remote: string = DEFAULT_REMOTE, branch: string = this.branch): Promise<void> {
    this.ensureOpen()
    await this.serialiseCommit(async () => {
      await this.pushWithRetry(remote, branch)
    })
  }

  async pull(remote: string = DEFAULT_REMOTE, branch: string = this.branch): Promise<void> {
    this.ensureOpen()
    await this.serialiseCommit(async () => {
      await this.pullOnce(remote, branch)
    })
  }

  async sync(remote: string = DEFAULT_REMOTE, branch: string = this.branch): Promise<void> {
    this.ensureOpen()
    await this.serialiseCommit(async () => {
      await this.pullOnce(remote, branch)
      await this.pushOnce(remote, branch)
    })
  }

  private ensureOpen(): void {
    if (this.closed) throw new ErrReadOnly()
  }

  private forwardEvent(evt: ChangeEvent): void {
    for (const sink of this.sinks.values()) sink(evt)
  }

  private async serialiseCommit<T>(fn: () => Promise<T>): Promise<T> {
    const prior = this.commitChain.catch(() => undefined)
    const next = prior.then(fn)
    this.commitChain = next.catch(() => undefined)
    return next
  }

  async openSync(): Promise<void> {
    if (!(await this.hasRemote(DEFAULT_REMOTE))) return
    try {
      await this.pullOnce(DEFAULT_REMOTE, this.branch)
    } catch {
      // Best effort only on open.
    }
    try {
      await this.recoverPendingPush(DEFAULT_REMOTE, this.branch)
    } catch {
      // The commit is durable locally; explicit sync remains available.
    }
  }

  private async commitTouched(touched: Set<Path>, opts: BatchOptions): Promise<void> {
    for (const p of touched) {
      const abs = this.fs.resolve(p)
      if (await exists(abs)) {
        await git.add({ fs: nodeFs, dir: this.root, filepath: p })
      } else {
        // Remove from the index; swallow benign errors for entries that are
        // not tracked (path was created and deleted within the same batch,
        // or never committed in the first place).
        try {
          await git.remove({ fs: nodeFs, dir: this.root, filepath: p })
        } catch (err) {
          if (!isNotInIndex(err)) throw err
        }
      }
    }

    const author = {
      name: opts.author !== undefined && opts.author !== '' ? opts.author : this.defaultAuthor.name,
      email: opts.email !== undefined && opts.email !== '' ? opts.email : this.defaultAuthor.email,
    }
    const message = buildCommitMessage(opts)
    try {
      await git.commit({
        fs: nodeFs,
        dir: this.root,
        message,
        author,
        committer: author,
        ...signOptions(this.sign),
      })
    } catch (err) {
      if (isEmptyCommit(err)) return
      throw err
    }
  }

  private async recoverPendingPush(remote: string, branch: string): Promise<void> {
    if (!(await this.hasLocalCommitsAhead(remote, branch))) return
    await this.pushOnce(remote, branch)
  }

  private async hasRemote(remote: string): Promise<boolean> {
    return (await readRemoteUrl(this.root, remote)) !== undefined
  }

  private async requireRemote(remote: string): Promise<void> {
    if (await this.hasRemote(remote)) return
    throw new StoreError(`gitstore: no remote configured for ${remote}`)
  }

  private async pullOnce(remote: string, branch: string): Promise<void> {
    await this.requireRemote(remote)
    try {
      await runGit(this.root, ['fetch', remote])
    } catch (err) {
      throw new StoreError(`gitstore: fetch ${remote}: ${describeGitFailure(err)}`, err)
    }

    if (!(await hasRemoteBranch(this.root, remote, branch))) return

    const stashed = await this.autoStashIfDirty()
    try {
      await runGit(this.root, ['rebase', `${remote}/${branch}`])
    } catch (err) {
      await abortRebase(this.root)
      if (stashed) {
        await restoreStashAfterAbort(this.root)
      }
      if (isRebaseConflict(err)) {
        throw new ErrConflict(`git pull conflicted while rebasing on ${remote}/${branch}`, err)
      }
      throw new StoreError(`gitstore: rebase ${remote}/${branch}: ${describeGitFailure(err)}`, err)
    }

    if (!stashed) return
    try {
      await runGit(this.root, ['stash', 'pop'])
    } catch (err) {
      throw new ErrConflict('git pull restored remote changes but local stash pop conflicted', err)
    }
  }

  private async autoStashIfDirty(): Promise<boolean> {
    const status = await runGit(this.root, ['status', '--porcelain'])
    if (status.trim() === '') return false
    try {
      await runGit(this.root, [
        'stash',
        'push',
        '--include-untracked',
        '--message',
        AUTO_STASH_MESSAGE,
      ])
    } catch (err) {
      throw new StoreError(`gitstore: stash push failed: ${describeGitFailure(err)}`, err)
    }
    return true
  }

  private async pushOnce(remote: string, branch: string): Promise<void> {
    await this.requireRemote(remote)
    try {
      await runGit(this.root, ['push', '--set-upstream', remote, branch])
    } catch (err) {
      if (isNonFastForward(err)) {
        throw new ErrConflict(`git push rejected by ${remote}/${branch}`, err)
      }
      throw new StoreError(`gitstore: push ${remote}/${branch}: ${describeGitFailure(err)}`, err)
    }
  }

  private async pushWithRetry(remote: string, branch: string): Promise<void> {
    try {
      await this.pushOnce(remote, branch)
    } catch (err) {
      if (!(err instanceof ErrConflict)) throw err
      await this.pullOnce(remote, branch)
      await this.pushOnce(remote, branch)
    }
  }

  private async hasLocalCommitsAhead(remote: string, branch: string): Promise<boolean> {
    try {
      const counts = await runGit(this.root, [
        'rev-list',
        '--left-right',
        '--count',
        `HEAD...${remote}/${branch}`,
      ])
      const [aheadRaw] = counts.trim().split(/\s+/, 2)
      const ahead = Number.parseInt(aheadRaw ?? '0', 10)
      return Number.isFinite(ahead) && ahead > 0
    } catch (err) {
      if (isMissingRemoteReference(err)) return false
      throw new StoreError(
        `gitstore: ahead check for ${remote}/${branch}: ${describeGitFailure(err)}`,
        err,
      )
    }
  }
}

const buildCommitMessage = (opts: BatchOptions): string => {
  const reason = opts.reason === '' ? 'write' : opts.reason
  const head = `[${reason}]`
  if (opts.message !== undefined && opts.message !== '') {
    return `${head} ${opts.message}`
  }
  return head
}

class TrackingBatch implements Batch {
  constructor(
    private readonly inner: Batch,
    private readonly touched: Set<Path>,
  ) {}

  read(path: Path): Promise<Buffer> {
    return this.inner.read(path)
  }

  async write(path: Path, content: Buffer): Promise<void> {
    await this.inner.write(path, content)
    this.touched.add(path)
  }

  async append(path: Path, content: Buffer): Promise<void> {
    await this.inner.append(path, content)
    this.touched.add(path)
  }

  async delete(path: Path): Promise<void> {
    await this.inner.delete(path)
    this.touched.add(path)
  }

  async rename(src: Path, dst: Path): Promise<void> {
    await this.inner.rename(src, dst)
    this.touched.add(src)
    this.touched.add(dst)
  }

  exists(path: Path): Promise<boolean> {
    return this.inner.exists(path)
  }

  stat(path: Path): Promise<FileInfo> {
    return this.inner.stat(path)
  }

  list(dir: Path | '', opts?: ListOpts): Promise<FileInfo[]> {
    return this.inner.list(dir, opts)
  }
}

const exists = async (abs: string): Promise<boolean> => {
  try {
    await access(abs)
    return true
  } catch {
    return false
  }
}

const isDirectoryEmpty = async (dir: string): Promise<boolean> => {
  const entries = await readdir(dir)
  return entries.length === 0
}

const cleanRemoteUrl = (value: string | undefined): string | undefined => {
  if (value === undefined) return undefined
  const trimmed = value.trim()
  return trimmed === '' ? undefined : trimmed
}

const initialiseRepository = async (args: {
  readonly dir: string
  readonly branch: string
  readonly defaultAuthor: GitSignature
  readonly sign?: GitSignFn
  readonly remoteUrl?: string
}): Promise<void> => {
  await git.init({ fs: nodeFs, dir: args.dir, defaultBranch: args.branch })
  if (args.remoteUrl !== undefined) {
    await configureOrigin(args.dir, args.remoteUrl)
  }
  await git.commit({
    fs: nodeFs,
    dir: args.dir,
    message: '[init] jeffs-brain gitstore initialised',
    author: { name: args.defaultAuthor.name, email: args.defaultAuthor.email },
    committer: { name: args.defaultAuthor.name, email: args.defaultAuthor.email },
    ...signOptions(args.sign),
  })
}

const resolveCurrentBranch = async (dir: string, fallback: string): Promise<string> => {
  const current = await git.currentBranch({ fs: nodeFs, dir, fullname: false })
  return current ?? fallback
}

const configureOrigin = async (dir: string, remoteUrl: string): Promise<void> => {
  await git.setConfig({
    fs: nodeFs,
    dir,
    path: `remote.${DEFAULT_REMOTE}.url`,
    value: remoteUrl,
  })
  await git.setConfig({
    fs: nodeFs,
    dir,
    path: `remote.${DEFAULT_REMOTE}.fetch`,
    value: REMOTE_FETCH_SPEC,
  })
}

const readRemoteUrl = async (dir: string, remote: string): Promise<string | undefined> => {
  const value = await git.getConfig({
    fs: nodeFs,
    dir,
    path: `remote.${remote}.url`,
  })
  return typeof value === 'string' && value.trim() !== '' ? value : undefined
}

const cloneRepository = async (dir: string, remoteUrl: string, branch: string): Promise<void> => {
  const parent = dirname(dir)
  const target = basename(dir)
  try {
    await execFile('git', ['clone', '--branch', branch, '--single-branch', remoteUrl, target], {
      cwd: parent,
      encoding: 'utf8',
    })
  } catch (err) {
    throw new StoreError(`gitstore: clone ${remoteUrl}: ${describeGitFailure(err)}`, err)
  }
}

const inspectRemoteBranch = async (
  remoteUrl: string,
  branch: string,
): Promise<'present' | 'missing-branch' | 'empty'> => {
  const allHeads = await lsRemote(remoteUrl)
  if (allHeads.trim() === '') return 'empty'
  const branchHeads = await lsRemote(remoteUrl, branch)
  return branchHeads.trim() === '' ? 'missing-branch' : 'present'
}

const lsRemote = async (remoteUrl: string, branch?: string): Promise<string> => {
  try {
    const args =
      branch === undefined
        ? ['ls-remote', '--heads', remoteUrl]
        : ['ls-remote', '--heads', remoteUrl, branch]
    const { stdout } = await execFile('git', args, { encoding: 'utf8' })
    return stdout
  } catch (err) {
    throw new StoreError(`gitstore: ls-remote ${remoteUrl}: ${describeGitFailure(err)}`, err)
  }
}

const runGit = async (cwd: string, args: readonly string[]): Promise<string> => {
  const { stdout } = await execFile('git', [...args], { cwd, encoding: 'utf8' })
  return stdout
}

const hasRemoteBranch = async (cwd: string, remote: string, branch: string): Promise<boolean> => {
  try {
    await execFile('git', ['show-ref', '--verify', '--quiet', `refs/remotes/${remote}/${branch}`], {
      cwd,
      encoding: 'utf8',
    })
    return true
  } catch (err) {
    if (isExitCode(err, 1)) return false
    throw err
  }
}

const abortRebase = async (cwd: string): Promise<void> => {
  try {
    await runGit(cwd, ['rebase', '--abort'])
  } catch {
    // Best effort only.
  }
}

const restoreStashAfterAbort = async (cwd: string): Promise<void> => {
  try {
    await runGit(cwd, ['stash', 'pop'])
  } catch {
    // The stash remains available for manual recovery.
  }
}

const isNotInIndex = (err: unknown): boolean => {
  if (!(err instanceof Error)) return false
  const code = (err as { code?: string }).code
  if (code === 'NotFoundError') return true
  return err.message.includes('not found') || err.message.includes('Could not find')
}

const isEmptyCommit = (err: unknown): boolean => {
  if (!(err instanceof Error)) return false
  return (
    err.message.includes('No changes to commit') ||
    err.message.includes('no changes') ||
    err.message.includes('nothing to commit')
  )
}

const isNonFastForward = (err: unknown): boolean => {
  const text = describeGitFailure(err).toLowerCase()
  return (
    text.includes('non-fast-forward') || text.includes('fetch first') || text.includes('stale info')
  )
}

const isRebaseConflict = (err: unknown): boolean => {
  const text = describeGitFailure(err).toLowerCase()
  return (
    text.includes('could not apply') ||
    text.includes('merge conflict') ||
    text.includes('conflict') ||
    text.includes('patch failed')
  )
}

const isMissingRemoteReference = (err: unknown): boolean => {
  const text = describeGitFailure(err).toLowerCase()
  return (
    text.includes('ambiguous argument') ||
    text.includes('unknown revision') ||
    text.includes('bad revision') ||
    text.includes('unknown ref')
  )
}

type ExecFailure = Error & {
  readonly code?: string | number
  readonly stderr?: string | Buffer
  readonly stdout?: string | Buffer
}

const isExitCode = (err: unknown, code: number): boolean => {
  if (!(err instanceof Error)) return false
  return (err as ExecFailure).code === code
}

const describeGitFailure = (err: unknown): string => {
  if (!(err instanceof Error)) return String(err)
  const failure = err as ExecFailure
  const stderr =
    typeof failure.stderr === 'string'
      ? failure.stderr.trim()
      : Buffer.isBuffer(failure.stderr)
        ? failure.stderr.toString('utf8').trim()
        : ''
  if (stderr !== '') return stderr
  return err.message
}

// ---- internal helpers exposed for gitstore-specific tests ----

export type CommitSummary = {
  readonly oid: string
  readonly message: string
  readonly authorName: string
  readonly authorEmail: string
  readonly parents: readonly string[]
  readonly timestamp: number
}

export const readGitLog = async (dir: string, depth?: number): Promise<CommitSummary[]> => {
  const log = await git.log({ fs: nodeFs, dir, ...(depth !== undefined ? { depth } : {}) })
  return log.map((entry) => ({
    oid: entry.oid,
    message: entry.commit.message,
    authorName: entry.commit.author.name,
    authorEmail: entry.commit.author.email,
    parents: entry.commit.parent,
    timestamp: entry.commit.author.timestamp,
  }))
}

export const listCommitFiles = async (dir: string, oid: string): Promise<string[]> => {
  const { commit } = await git.readCommit({ fs: nodeFs, dir, oid })
  const out: string[] = []
  await walkCommitTree(dir, commit.tree, '', out)
  out.sort()
  return out
}

const walkCommitTree = async (
  dir: string,
  oid: string,
  prefix: string,
  out: string[],
): Promise<void> => {
  const { tree } = await git.readTree({ fs: nodeFs, dir, oid })
  for (const entry of tree) {
    const full = prefix === '' ? entry.path : `${prefix}/${entry.path}`
    if (entry.type === 'tree') {
      await walkCommitTree(dir, entry.oid, full, out)
    } else if (entry.type === 'blob') {
      out.push(full)
    }
  }
}
