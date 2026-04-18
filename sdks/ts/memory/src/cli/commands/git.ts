// SPDX-License-Identifier: Apache-2.0

import * as nodeFs from 'node:fs'
import { defineCommand } from 'citty'
import * as git from 'isomorphic-git'
import { openBrain } from '../brain.js'
import { CliError, CliUsageError, resolveBrainDir } from '../config.js'
import {
  DEFAULT_REMOTE,
  abortMergeOrRebase,
  autoResolve,
  batchResolve,
  deletePaths,
  listBrainFiles,
  listChangedFiles,
  listConflictedFiles,
  parseReason,
  readCleanReport,
  readDiff,
  readFormattedLog,
  readRemoteUrl,
  readSectionStats,
  requireGitBrain,
  resolveBranchArg,
  runGit,
  runVerifyChecks,
  scopePrefixes,
  stageAndCommit,
  summariseDiff,
  tryReadAheadBehind,
  writeJson,
} from './git-helpers.js'
import { readGitLog } from '../../store/gitstore.js'

const asBool = (value: unknown): boolean => value === true || value === 'true'

const parsePositiveInt = (raw: unknown, fallback: number): number => {
  const parsed = typeof raw === 'string' ? Number.parseInt(raw, 10) : fallback
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback
}

const parsePositiveNumber = (raw: unknown, fallback: number): number => {
  const parsed = typeof raw === 'string' ? Number.parseFloat(raw) : fallback
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback
}

const runStoreGitOp = async (args: {
  readonly brain?: string
  readonly remote?: string
  readonly branch?: string
  readonly op: 'pull' | 'push' | 'sync'
}): Promise<void> => {
  const brainDir = resolveBrainDir(args.brain)
  const remote = args.remote !== undefined && args.remote !== '' ? args.remote : DEFAULT_REMOTE
  const branch = await resolveBranchArg(brainDir, args.branch)
  const store = await openBrain(brainDir)
  try {
    if (args.op === 'pull') {
      await store.pull(remote, branch)
    } else if (args.op === 'push') {
      await store.push(remote, branch)
    } else {
      await store.sync(remote, branch)
    }
  } catch (err) {
    throw new CliError(`git ${args.op}: ${err instanceof Error ? err.message : String(err)}`)
  } finally {
    await store.close()
  }
  writeJson({
    brain: brainDir,
    remote,
    branch,
    operation: args.op,
    ok: true,
  })
}

const pullCommand = defineCommand({
  meta: {
    name: 'pull',
    description: 'Fetch and rebase a git-backed brain from its remote',
  },
  args: {
    brain: { type: 'string', description: 'Brain directory' },
    remote: { type: 'string', description: 'Remote name', default: DEFAULT_REMOTE },
    branch: { type: 'string', description: 'Branch name' },
  },
  run: async ({ args }) => {
    await runStoreGitOp({
      op: 'pull',
      ...(typeof args.brain === 'string' ? { brain: args.brain } : {}),
      ...(typeof args.remote === 'string' ? { remote: args.remote } : {}),
      ...(typeof args.branch === 'string' ? { branch: args.branch } : {}),
    })
  },
})

const pushCommand = defineCommand({
  meta: {
    name: 'push',
    description: 'Push local brain commits to the remote',
  },
  args: {
    brain: { type: 'string', description: 'Brain directory' },
    remote: { type: 'string', description: 'Remote name', default: DEFAULT_REMOTE },
    branch: { type: 'string', description: 'Branch name' },
  },
  run: async ({ args }) => {
    await runStoreGitOp({
      op: 'push',
      ...(typeof args.brain === 'string' ? { brain: args.brain } : {}),
      ...(typeof args.remote === 'string' ? { remote: args.remote } : {}),
      ...(typeof args.branch === 'string' ? { branch: args.branch } : {}),
    })
  },
})

const syncCommand = defineCommand({
  meta: {
    name: 'sync',
    description: 'Pull then push a git-backed brain',
  },
  args: {
    brain: { type: 'string', description: 'Brain directory' },
    remote: { type: 'string', description: 'Remote name', default: DEFAULT_REMOTE },
    branch: { type: 'string', description: 'Branch name' },
  },
  run: async ({ args }) => {
    await runStoreGitOp({
      op: 'sync',
      ...(typeof args.brain === 'string' ? { brain: args.brain } : {}),
      ...(typeof args.remote === 'string' ? { remote: args.remote } : {}),
      ...(typeof args.branch === 'string' ? { branch: args.branch } : {}),
    })
  },
})

const statusCommand = defineCommand({
  meta: {
    name: 'status',
    description: 'Show git status, recent commits, and sync state for a brain',
  },
  args: {
    brain: { type: 'string', description: 'Brain directory' },
    depth: { type: 'string', description: 'Number of commits to show', default: '5' },
  },
  run: async ({ args }) => {
    const brainDir = resolveBrainDir(typeof args.brain === 'string' ? args.brain : undefined)
    await requireGitBrain(brainDir)
    const depth = parsePositiveInt(args.depth, 5)
    const branch = (await git.currentBranch({ fs: nodeFs, dir: brainDir, fullname: false })) ?? null
    const remoteUrl = await readRemoteUrl(brainDir)
    const changes = await readDiff(brainDir)
    const { ahead, behind } = branch === null ? { ahead: null, behind: null } : await tryReadAheadBehind(brainDir, branch)
    writeJson({
      brain: brainDir,
      operation: 'status',
      branch,
      remote: remoteUrl,
      dirty: changes.length,
      ahead,
      behind,
      changes,
      commits: await readGitLog(brainDir, depth),
    })
  },
})

const diffCommand = defineCommand({
  meta: {
    name: 'diff',
    description: 'Show uncommitted working-tree changes',
  },
  args: {
    brain: { type: 'string', description: 'Brain directory' },
    stat: { type: 'boolean', description: 'Show summary only', default: false },
  },
  run: async ({ args }) => {
    const brainDir = resolveBrainDir(typeof args.brain === 'string' ? args.brain : undefined)
    await requireGitBrain(brainDir)
    const changes = await readDiff(brainDir)
    const summary = summariseDiff(changes)
    writeJson({
      brain: brainDir,
      operation: 'diff',
      clean: summary.total === 0,
      summary,
      ...(asBool(args.stat) ? {} : { changes }),
    })
  },
})

const logCommand = defineCommand({
  meta: {
    name: 'log',
    description: 'Show formatted brain commit history',
  },
  args: {
    brain: { type: 'string', description: 'Brain directory' },
    limit: { type: 'string', description: 'Number of commits to show', default: '20' },
    reason: { type: 'string', description: 'Filter by [reason] tag' },
  },
  run: async ({ args }) => {
    const brainDir = resolveBrainDir(typeof args.brain === 'string' ? args.brain : undefined)
    await requireGitBrain(brainDir)
    const limit = parsePositiveInt(args.limit, 20)
    const reasonFilter = typeof args.reason === 'string' ? args.reason.trim() : ''
    writeJson({
      brain: brainDir,
      operation: 'log',
      entries: await readFormattedLog({ brainDir, limit, reasonFilter }),
    })
  },
})

const showCommand = defineCommand({
  meta: {
    name: 'show',
    description: 'Show a single commit with touched files',
  },
  args: {
    brain: { type: 'string', description: 'Brain directory' },
    commit: { type: 'string', description: 'Commit oid' },
  },
  run: async ({ args }) => {
    const brainDir = resolveBrainDir(typeof args.brain === 'string' ? args.brain : undefined)
    await requireGitBrain(brainDir)
    const oid = typeof args.commit === 'string' ? args.commit.trim() : ''
    if (oid === '') throw new CliUsageError('git show: pass --commit <oid>')
    try {
      const { commit } = await git.readCommit({ fs: nodeFs, dir: brainDir, oid })
      const { reason, rest } = parseReason(commit.message)
      writeJson({
        brain: brainDir,
        operation: 'show',
        commit: {
          oid,
          message: commit.message,
          reason,
          rest,
          authorName: commit.author.name,
          authorEmail: commit.author.email,
          committerName: commit.committer.name,
          committerEmail: commit.committer.email,
          timestamp: commit.author.timestamp,
          parents: commit.parent,
          files: await listChangedFiles(brainDir, oid),
        },
      })
    } catch (err) {
      throw new CliError(`git show: ${err instanceof Error ? err.message : String(err)}`)
    }
  },
})

const filesCommand = defineCommand({
  meta: {
    name: 'files',
    description: 'List files touched by a commit',
  },
  args: {
    brain: { type: 'string', description: 'Brain directory' },
    commit: { type: 'string', description: 'Commit oid' },
  },
  run: async ({ args }) => {
    const brainDir = resolveBrainDir(typeof args.brain === 'string' ? args.brain : undefined)
    await requireGitBrain(brainDir)
    const oid = typeof args.commit === 'string' ? args.commit.trim() : ''
    if (oid === '') throw new CliUsageError('git files: pass --commit <oid>')
    writeJson({
      brain: brainDir,
      operation: 'files',
      commit: oid,
      files: await listChangedFiles(brainDir, oid),
    })
  },
})

const verifyCommand = defineCommand({
  meta: {
    name: 'verify',
    description: 'Run read-only health checks against the git-backed brain',
  },
  args: {
    brain: { type: 'string', description: 'Brain directory' },
  },
  run: async ({ args }) => {
    const brainDir = resolveBrainDir(typeof args.brain === 'string' ? args.brain : undefined)
    await requireGitBrain(brainDir)
    const checks = await runVerifyChecks(brainDir)
    const ok = checks.every((check) => check.ok)
    writeJson({
      brain: brainDir,
      operation: 'verify',
      ok,
      checks,
    })
    if (!ok) {
      throw new CliError('git verify: one or more checks failed')
    }
  },
})

const statsCommand = defineCommand({
  meta: {
    name: 'stats',
    description: 'Show content and sync statistics for the brain',
  },
  args: {
    brain: { type: 'string', description: 'Brain directory' },
  },
  run: async ({ args }) => {
    const brainDir = resolveBrainDir(typeof args.brain === 'string' ? args.brain : undefined)
    await requireGitBrain(brainDir)
    const branch = (await git.currentBranch({ fs: nodeFs, dir: brainDir, fullname: false })) ?? null
    const remoteUrl = await readRemoteUrl(brainDir)
    const changes = await readDiff(brainDir)
    const sections = await readSectionStats(brainDir)
    const totals = sections.reduce(
      (acc, section) => ({
        count: acc.count + section.count,
        bytes: acc.bytes + section.bytes,
      }),
      { count: 0, bytes: 0 },
    )
    const { ahead, behind } = branch === null ? { ahead: null, behind: null } : await tryReadAheadBehind(brainDir, branch)
    writeJson({
      brain: brainDir,
      operation: 'stats',
      branch,
      remote: remoteUrl,
      ahead,
      behind,
      dirty: changes.length,
      sections,
      total: totals,
    })
  },
})

const resolveCommand = defineCommand({
  meta: {
    name: 'resolve',
    description: 'Resolve merge or rebase conflicts in the brain working tree',
  },
  args: {
    brain: { type: 'string', description: 'Brain directory' },
    theirs: { type: 'boolean', description: 'Accept theirs for all conflicts', default: false },
    ours: { type: 'boolean', description: 'Accept ours for all conflicts', default: false },
    auto: { type: 'boolean', description: 'Use simple auto-resolution heuristics', default: false },
    abort: { type: 'boolean', description: 'Abort an in-progress merge or rebase', default: false },
  },
  run: async ({ args }) => {
    const brainDir = resolveBrainDir(typeof args.brain === 'string' ? args.brain : undefined)
    await requireGitBrain(brainDir)
    const theirs = asBool(args.theirs)
    const ours = asBool(args.ours)
    const auto = asBool(args.auto)
    const abort = asBool(args.abort)
    const selected = Number(theirs) + Number(ours) + Number(auto)
    if (abort && selected > 0) {
      throw new CliUsageError('git resolve: --abort cannot be combined with resolution flags')
    }
    if (selected > 1) {
      throw new CliUsageError('git resolve: choose only one of --theirs, --ours, or --auto')
    }
    if (abort) {
      await abortMergeOrRebase(brainDir)
      writeJson({
        brain: brainDir,
        operation: 'resolve',
        aborted: true,
      })
      return
    }

    const conflicted = await listConflictedFiles(brainDir)
    if (conflicted.length === 0) {
      writeJson({
        brain: brainDir,
        operation: 'resolve',
        resolved: 0,
        remaining: 0,
        remainingFiles: [],
        committed: false,
      })
      return
    }

    let resolved = 0
    if (theirs) {
      resolved = await batchResolve(brainDir, conflicted, 'theirs')
    } else if (ours) {
      resolved = await batchResolve(brainDir, conflicted, 'ours')
    } else if (auto) {
      resolved = await autoResolve(brainDir, conflicted)
    }

    const remainingFiles = await listConflictedFiles(brainDir)
    const committed = remainingFiles.length === 0 && resolved > 0
    if (committed) {
      await runGit(brainDir, ['commit', '-m', `[resolve] resolved ${String(resolved)} conflict(s)`])
    }
    writeJson({
      brain: brainDir,
      operation: 'resolve',
      resolved,
      remaining: remainingFiles.length,
      remainingFiles,
      committed,
    })
  },
})

const resetCommand = defineCommand({
  meta: {
    name: 'reset',
    description: 'Delete brain content in the selected scope and commit the reset',
  },
  args: {
    brain: { type: 'string', description: 'Brain directory' },
    scope: { type: 'string', description: 'Reset scope', default: 'all' },
    confirm: { type: 'boolean', description: 'Required for destructive reset', default: false },
  },
  run: async ({ args }) => {
    if (!asBool(args.confirm)) {
      throw new CliUsageError('git reset: pass --confirm to apply the reset')
    }
    const brainDir = resolveBrainDir(typeof args.brain === 'string' ? args.brain : undefined)
    await requireGitBrain(brainDir)
    const scope = typeof args.scope === 'string' ? args.scope : 'all'
    const prefixes = scopePrefixes(scope)
    const files = await listBrainFiles(brainDir, prefixes)
    const paths = files.map((file) => file.path)
    await deletePaths(brainDir, paths)
    const committed = await stageAndCommit({
      brainDir,
      paths,
      message: `[reset] reset ${scope} (${String(paths.length)} files)`,
    })
    writeJson({
      brain: brainDir,
      operation: 'reset',
      scope,
      deleted: paths.length,
      committed,
      files: paths,
    })
  },
})

const cleanCommand = defineCommand({
  meta: {
    name: 'clean',
    description: 'Find and optionally remove junk files from the brain',
  },
  args: {
    brain: { type: 'string', description: 'Brain directory' },
    scope: { type: 'string', description: 'Clean scope', default: 'ingested' },
    apply: { type: 'boolean', description: 'Delete matching files', default: false },
    'max-size-mb': {
      type: 'string',
      description: 'Oversized file threshold in MiB',
      default: '5',
    },
  },
  run: async ({ args }) => {
    const brainDir = resolveBrainDir(typeof args.brain === 'string' ? args.brain : undefined)
    await requireGitBrain(brainDir)
    const scope = typeof args.scope === 'string' ? args.scope : 'ingested'
    const prefixes = scopePrefixes(scope)
    const maxSizeBytes = Math.floor(parsePositiveNumber(args['max-size-mb'], 5) * 1024 * 1024)
    const report = readCleanReport(await listBrainFiles(brainDir, prefixes), maxSizeBytes)
    const paths = [
      ...report.junkDirs,
      ...report.lockfiles,
      ...report.generated,
      ...report.oversized,
    ]
    let committed = false
    if (asBool(args.apply) && paths.length > 0) {
      await deletePaths(brainDir, paths)
      committed = await stageAndCommit({
        brainDir,
        paths,
        message: `[clean] clean ${scope} (${String(paths.length)} files)`,
      })
    }
    writeJson({
      brain: brainDir,
      operation: 'clean',
      scope,
      apply: asBool(args.apply),
      committed,
      ...report,
    })
  },
})

export const gitCommand = defineCommand({
  meta: { name: 'git', description: 'Git operations on a brain' },
  subCommands: {
    pull: pullCommand,
    push: pushCommand,
    sync: syncCommand,
    status: statusCommand,
    diff: diffCommand,
    log: logCommand,
    show: showCommand,
    files: filesCommand,
    verify: verifyCommand,
    stats: statsCommand,
    resolve: resolveCommand,
    reset: resetCommand,
    clean: cleanCommand,
  },
})
