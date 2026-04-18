import { execFile as execFileCallback } from 'node:child_process'
import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { promisify } from 'node:util'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'
import { createGitStore, listCommitFiles, readGitLog, type GitStore } from './gitstore.js'
import { toPath, type Path } from './path.js'

const buf = (s: string): Buffer => Buffer.from(s, 'utf8')
const p = (s: string): Path => toPath(s)
const execFile = promisify(execFileCallback)

const runGit = async (args: readonly string[], cwd?: string): Promise<string> => {
  const { stdout } = await execFile('git', [...args], {
    ...(cwd !== undefined ? { cwd } : {}),
    encoding: 'utf8',
  })
  return stdout
}

describe('gitstore specifics', () => {
  let dir: string
  let store: GitStore
  const extraDirs: string[] = []

  beforeEach(async () => {
    dir = await mkdtemp(join(tmpdir(), 'jbmem-gitstore-spec-'))
    store = await createGitStore({ dir, init: true })
  })

  afterEach(async () => {
    await store.close()
    await rm(dir, { recursive: true, force: true })
    while (extraDirs.length > 0) {
      const extra = extraDirs.pop()
      if (extra !== undefined) {
        await rm(extra, { recursive: true, force: true })
      }
    }
  })

  const makeRemote = async (): Promise<string> => {
    const remote = await mkdtemp(join(tmpdir(), 'jbmem-gitstore-remote-'))
    extraDirs.push(remote)
    await runGit(['init', '--bare', '--initial-branch=main', remote])
    return remote
  }

  const cloneRemote = async (remote: string): Promise<string> => {
    const clone = await mkdtemp(join(tmpdir(), 'jbmem-gitstore-clone-'))
    extraDirs.push(clone)
    await runGit(['clone', remote, clone])
    await runGit(['config', 'user.name', 'Remote Tester'], clone)
    await runGit(['config', 'user.email', 'remote@example.com'], clone)
    return clone
  }

  const reopenWithRemote = async (
    remoteUrl: string,
    opts: { readonly autoPush?: boolean } = {},
  ): Promise<void> => {
    await store.close()
    store = await createGitStore({
      dir,
      init: true,
      remoteUrl,
      ...(opts.autoPush === true ? { autoPush: true } : {}),
    })
  }

  const commitAndPushClone = async (clone: string, message: string): Promise<void> => {
    await runGit(['add', '.'], clone)
    await runGit(['commit', '-m', message], clone)
    await runGit(['push'], clone)
  }

  it('init: true creates the repo with an initial commit', async () => {
    const log = await readGitLog(dir)
    expect(log.length).toBeGreaterThanOrEqual(1)
    // Oldest commit is the bootstrap.
    const first = log[log.length - 1]
    expect(first?.message).toContain('[init]')
  })

  it('each batch produces exactly one commit', async () => {
    const before = (await readGitLog(dir)).length
    await store.batch({ reason: 'extract' }, async (b) => {
      await b.write(p('memory/a.md'), buf('one'))
      await b.write(p('memory/b.md'), buf('two'))
      await b.write(p('memory/c.md'), buf('three'))
    })
    const after = (await readGitLog(dir)).length
    expect(after - before).toBe(1)
  })

  it('author + email come from BatchOptions', async () => {
    await store.batch(
      { reason: 'reflect', author: 'Alex J', email: 'alex@lleverage.ai' },
      async (b) => {
        await b.write(p('memory/note.md'), buf('body'))
      },
    )
    const [head] = await readGitLog(dir, 1)
    expect(head?.authorName).toBe('Alex J')
    expect(head?.authorEmail).toBe('alex@lleverage.ai')
  })

  it('default author falls back to jeffs-brain/noreply', async () => {
    await store.batch({ reason: 'write' }, async (b) => {
      await b.write(p('memory/default.md'), buf('x'))
    })
    const [head] = await readGitLog(dir, 1)
    expect(head?.authorName).toBe('jeffs-brain')
    expect(head?.authorEmail).toBe('noreply@jeffsbrain.com')
  })

  it('commit message embeds reason and message', async () => {
    await store.batch({ reason: 'consolidate', message: 'daily roll-up' }, async (b) => {
      await b.write(p('memory/summary.md'), buf('s'))
    })
    const [head] = await readGitLog(dir, 1)
    expect(head?.message.trim()).toBe('[consolidate] daily roll-up')
  })

  it('deleting a file removes it from the commit tree', async () => {
    await store.write(p('memory/victim.md'), buf('to-be-gone'))
    const afterWrite = await readGitLog(dir, 1)
    const writtenTree = await listCommitFiles(dir, afterWrite[0]!.oid)
    expect(writtenTree).toContain('memory/victim.md')

    await store.delete(p('memory/victim.md'))
    const afterDelete = await readGitLog(dir, 1)
    const deletedTree = await listCommitFiles(dir, afterDelete[0]!.oid)
    expect(deletedTree).not.toContain('memory/victim.md')
  })

  it('concurrent writes serialise without corrupting the tree', async () => {
    await Promise.all(
      Array.from({ length: 10 }, (_, i) =>
        store.write(p(`memory/race-${i}.md`), buf(`v${i}`)),
      ),
    )
    for (let i = 0; i < 10; i++) {
      expect((await store.read(p(`memory/race-${i}.md`))).toString()).toBe(`v${i}`)
    }
    // Every write is its own commit on top of the bootstrap; tree at HEAD must
    // contain all ten files.
    const [head] = await readGitLog(dir, 1)
    const tree = await listCommitFiles(dir, head!.oid)
    for (let i = 0; i < 10; i++) {
      expect(tree).toContain(`memory/race-${i}.md`)
    }
  })

  it('init: false against a fresh directory refuses to create the repo', async () => {
    const empty = await mkdtemp(join(tmpdir(), 'jbmem-gitstore-empty-'))
    try {
      await expect(createGitStore({ dir: empty })).rejects.toThrow(/no git repo/)
    } finally {
      await rm(empty, { recursive: true, force: true })
    }
  })

  it('sign callback is invoked on each commit', async () => {
    const signed = await mkdtemp(join(tmpdir(), 'jbmem-gitstore-sign-'))
    const signatures: string[] = []
    const signed_store = await createGitStore({
      dir: signed,
      init: true,
      sign: async ({ payload }) => {
        signatures.push(payload.slice(0, 20))
        return `-----BEGIN FAKE SIG-----\n${payload.length}\n-----END FAKE SIG-----\n`
      },
    })
    try {
      await signed_store.batch({ reason: 'write', message: 'hi' }, async (b) => {
        await b.write(p('memory/s.md'), buf('body'))
      })
      expect(signatures.length).toBeGreaterThanOrEqual(1)
    } finally {
      await signed_store.close()
      await rm(signed, { recursive: true, force: true })
    }
  })

  it('push, pull, and sync fail clearly when no remote is configured', async () => {
    await expect(store.push()).rejects.toThrow(/no remote configured/)
    await expect(store.pull()).rejects.toThrow(/no remote configured/)
    await expect(store.sync()).rejects.toThrow(/no remote configured/)
  })

  it('pushes local commits to a configured remote', async () => {
    const remote = await makeRemote()
    await reopenWithRemote(remote)

    await store.write(p('memory/pushed.md'), buf('hello remote'))
    await store.push()

    const clone = await cloneRemote(remote)
    expect(await readFile(join(clone, 'memory', 'pushed.md'), 'utf8')).toBe('hello remote')
  })

  it('pulls remote commits into the local checkout', async () => {
    const remote = await makeRemote()
    await reopenWithRemote(remote)

    await store.write(p('memory/local.md'), buf('seed'))
    await store.push()

    const clone = await cloneRemote(remote)
    await mkdir(join(clone, 'memory'), { recursive: true })
    await writeFile(join(clone, 'memory', 'remote.md'), 'from remote', 'utf8')
    await commitAndPushClone(clone, 'add remote file')

    await store.pull()

    expect((await store.read(p('memory/remote.md'))).toString('utf8')).toBe('from remote')
  })

  it('autostashes tracked and untracked local changes during pull', async () => {
    const remote = await makeRemote()
    await reopenWithRemote(remote)

    await store.write(p('memory/local.md'), buf('clean'))
    await store.push()

    await writeFile(join(dir, 'memory', 'local.md'), 'dirty local edit', 'utf8')
    const untracked = join(dir, 'scratch.tmp')
    await writeFile(untracked, 'left alone', 'utf8')

    const clone = await cloneRemote(remote)
    await mkdir(join(clone, 'memory'), { recursive: true })
    await writeFile(join(clone, 'memory', 'remote.md'), 'arrived remotely', 'utf8')
    await commitAndPushClone(clone, 'push remote update')

    await store.pull()

    expect(await readFile(join(dir, 'memory', 'local.md'), 'utf8')).toBe('dirty local edit')
    expect(await readFile(untracked, 'utf8')).toBe('left alone')
    expect((await store.read(p('memory/remote.md'))).toString('utf8')).toBe('arrived remotely')
  })

  it('retries an explicit push after a non-fast-forward rejection', async () => {
    const remote = await makeRemote()
    await reopenWithRemote(remote)

    await store.write(p('memory/base.md'), buf('base'))
    await store.push()

    await store.write(p('memory/local.md'), buf('local only'))

    const clone = await cloneRemote(remote)
    await mkdir(join(clone, 'memory'), { recursive: true })
    await writeFile(join(clone, 'memory', 'remote.md'), 'remote only', 'utf8')
    await commitAndPushClone(clone, 'advance remote')

    await store.push()

    const fresh = await cloneRemote(remote)
    expect(await readFile(join(fresh, 'memory', 'local.md'), 'utf8')).toBe('local only')
    expect(await readFile(join(fresh, 'memory', 'remote.md'), 'utf8')).toBe('remote only')
  })

  it('surfaces a rebase conflict when explicit push cannot be retried cleanly', async () => {
    const remote = await makeRemote()
    await reopenWithRemote(remote)

    await store.write(p('memory/conflict.md'), buf('base'))
    await store.push()

    const clone = await cloneRemote(remote)

    await writeFile(join(clone, 'memory', 'conflict.md'), 'remote change', 'utf8')
    await commitAndPushClone(clone, 'remote conflict')

    await store.write(p('memory/conflict.md'), buf('local change'))

    await expect(store.push()).rejects.toThrow(/pull conflicted while rebasing/)

    expect((await store.read(p('memory/conflict.md'))).toString('utf8')).toBe('local change')

    const fresh = await cloneRemote(remote)
    expect(await readFile(join(fresh, 'memory', 'conflict.md'), 'utf8')).toBe('remote change')
  })

  it('recovers a pending local push when reopened', async () => {
    const remote = await makeRemote()
    await reopenWithRemote(remote)

    await store.write(p('memory/seed.md'), buf('seed'))
    await store.push()

    await store.write(p('memory/pending.md'), buf('pending'))
    await store.close()

    store = await createGitStore({ dir, remoteUrl: remote })

    const clone = await cloneRemote(remote)
    expect(await readFile(join(clone, 'memory', 'pending.md'), 'utf8')).toBe('pending')
  })

  it('auto-pushes writes when enabled', async () => {
    const remote = await makeRemote()
    await reopenWithRemote(remote, { autoPush: true })

    await store.write(p('memory/auto.md'), buf('auto pushed'))

    const clone = await cloneRemote(remote)
    expect(await readFile(join(clone, 'memory', 'auto.md'), 'utf8')).toBe('auto pushed')
  })
})
