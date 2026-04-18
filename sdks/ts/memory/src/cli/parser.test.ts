// SPDX-License-Identifier: Apache-2.0

/**
 * Parser smoke tests for the memory CLI.
 *
 * We exercise only citty's argument parsing — each subcommand is asked to
 * parse a representative argv and we assert the shape of the result. No
 * side effects, no handlers invoked. If these pass the CLI surface is at
 * least internally consistent; integration coverage lives in the
 * sibling `integration.test.ts`.
 */

import { parseArgs } from 'citty'
import type { ArgsDef, CommandDef } from 'citty'
import { describe, expect, it } from 'vitest'
import {
  aclCommand,
  consolidateCommand,
  evalCommand,
  extractCommand,
  gitCommand,
  ingestCommand,
  initCommand,
  reflectCommand,
  searchCommand,
  serveCommand,
} from './commands/index.js'

type AnyCommand = CommandDef<ArgsDef>

const argsOf = (cmd: AnyCommand): ArgsDef => {
  const raw = cmd.args
  if (typeof raw === 'function' || raw === undefined) return {} as ArgsDef
  return raw as ArgsDef
}

const pickSub = (parent: AnyCommand, name: string): AnyCommand => {
  const subs = parent.subCommands
  if (subs === undefined || typeof subs === 'function') {
    throw new Error('expected subCommands')
  }
  const entry = (subs as Record<string, AnyCommand>)[name]
  if (entry === undefined) throw new Error(`no subcommand ${name}`)
  return entry
}

const asAny = (cmd: unknown): AnyCommand => cmd as AnyCommand

describe('memory arg parsing', () => {
  it('parses `init <path>`', () => {
    const parsed = parseArgs(['/tmp/brain'], argsOf(asAny(initCommand)))
    expect(parsed['path']).toBe('/tmp/brain')
  })

  it('parses `ingest <file> --brain <path>`', () => {
    const parsed = parseArgs(
      ['notes.md', '--brain', '/tmp/b'],
      argsOf(asAny(ingestCommand)),
    )
    expect(parsed['file']).toBe('notes.md')
    expect(parsed['brain']).toBe('/tmp/b')
  })

  it('parses `search` flags, defaulting mode to hybrid', () => {
    const parsed = parseArgs(
      ['hello world', '--mode', 'bm25', '--rerank', '--json'],
      argsOf(asAny(searchCommand)),
    )
    expect(parsed['query']).toBe('hello world')
    expect(parsed['mode']).toBe('bm25')
    expect(parsed['rerank']).toBe(true)
    expect(parsed['json']).toBe(true)

    const dflt = parseArgs(['q'], argsOf(asAny(searchCommand)))
    expect(dflt['mode']).toBe('hybrid')
    expect(dflt['rerank']).toBe(false)
  })

  it('parses `extract --from <msgs>`', () => {
    const parsed = parseArgs(
      ['--from', 'messages.json', '--brain', '/tmp/b'],
      argsOf(asAny(extractCommand)),
    )
    expect(parsed['from']).toBe('messages.json')
    expect(parsed['brain']).toBe('/tmp/b')
  })

  it('parses `reflect --session <id> --from <msgs>`', () => {
    const parsed = parseArgs(
      ['--session', 's-1', '--from', 'm.json'],
      argsOf(asAny(reflectCommand)),
    )
    expect(parsed['session']).toBe('s-1')
    expect(parsed['from']).toBe('m.json')
  })

  it('parses `consolidate --brain <path>`', () => {
    const parsed = parseArgs(['--brain', '/tmp/b'], argsOf(asAny(consolidateCommand)))
    expect(parsed['brain']).toBe('/tmp/b')
  })

  it('parses `eval lme fetch`, `run`, `score`, `compare`, and `doctor`', () => {
    const lme = pickSub(asAny(evalCommand), 'lme')

    const fetch = pickSub(lme, 'fetch')
    const fetched = parseArgs(
      ['--bundle', 'cleaned', '--splits', 'oracle,s', '--skipRepo'],
      argsOf(fetch),
    )
    expect(fetched['bundle']).toBe('cleaned')
    expect(fetched['splits']).toBe('oracle,s')
    expect(fetched['skipRepo']).toBe(true)

    const run = pickSub(lme, 'run')
    const runParsed = parseArgs(
      [
        '--dataset',
        'lme.jsonl',
        '--outDir',
        '/tmp/out',
        '--ingestMode',
        'agentic',
        '--judgeConcurrency',
        '6',
        '--readerBudgetChars',
        '12000',
        '--categories',
        'single-session-preference,multi-session',
        '--retrievalMode',
        'semantic',
        '--rerank',
      ],
      argsOf(run),
    )
    expect(runParsed['dataset']).toBe('lme.jsonl')
    expect(runParsed['outDir']).toBe('/tmp/out')
    expect(runParsed['ingestMode']).toBe('agentic')
    expect(runParsed['judgeConcurrency']).toBe('6')
    expect(runParsed['readerBudgetChars']).toBe('12000')
    expect(runParsed['categories']).toBe('single-session-preference,multi-session')
    expect(runParsed['retrievalMode']).toBe('semantic')
    expect(runParsed['rerank']).toBe(true)

    const score = pickSub(lme, 'score')
    const scoreParsed = parseArgs(
      ['--report', '/tmp/out/latest/report.json', '--official'],
      argsOf(score),
    )
    expect(scoreParsed['report']).toBe('/tmp/out/latest/report.json')
    expect(scoreParsed['official']).toBe(true)

    const compare = pickSub(lme, 'compare')
    const compareParsed = parseArgs(
      ['--left', '/tmp/a.json', '--right', '/tmp/b.json'],
      argsOf(compare),
    )
    expect(compareParsed['left']).toBe('/tmp/a.json')
    expect(compareParsed['right']).toBe('/tmp/b.json')

    const doctor = pickSub(lme, 'doctor')
    const doctorParsed = parseArgs(
      ['--dataset', 'lme.jsonl', '--outDir', '/tmp/out', '--official'],
      argsOf(doctor),
    )
    expect(doctorParsed['dataset']).toBe('lme.jsonl')
    expect(doctorParsed['outDir']).toBe('/tmp/out')
    expect(doctorParsed['official']).toBe(true)
  })

  it('parses `serve --port 9999`', () => {
    const parsed = parseArgs(['--port', '9999'], argsOf(asAny(serveCommand)))
    expect(parsed['port']).toBe('9999')
  })

  it('parses `acl grant` subcommand args', () => {
    const grant = pickSub(asAny(aclCommand), 'grant')
    const parsed = parseArgs(
      [
        '--subject',
        'user:alice',
        '--role',
        'reader',
        '--resource',
        'brain:main',
      ],
      argsOf(grant),
    )
    expect(parsed['subject']).toBe('user:alice')
    expect(parsed['role']).toBe('reader')
    expect(parsed['resource']).toBe('brain:main')
  })

  it('parses `acl revoke` and `acl list` subcommands', () => {
    const revoke = pickSub(asAny(aclCommand), 'revoke')
    const parsed = parseArgs(
      [
        '--subject',
        'user:alice',
        '--role',
        'reader',
        '--resource',
        'brain:main',
      ],
      argsOf(revoke),
    )
    expect(parsed['subject']).toBe('user:alice')

    const list = pickSub(asAny(aclCommand), 'list')
    const listParsed = parseArgs(['--brain', '/tmp/b'], argsOf(list))
    expect(listParsed['brain']).toBe('/tmp/b')
  })

  it('parses the git operator subcommands', () => {
    const status = pickSub(asAny(gitCommand), 'status')
    const parsed = parseArgs(['--depth', '3'], argsOf(status))
    expect(parsed['depth']).toBe('3')

    const diff = pickSub(asAny(gitCommand), 'diff')
    const diffParsed = parseArgs(['--stat'], argsOf(diff))
    expect(diffParsed['stat']).toBe(true)

    const log = pickSub(asAny(gitCommand), 'log')
    const logParsed = parseArgs(['--limit', '7', '--reason', 'compile'], argsOf(log))
    expect(logParsed['limit']).toBe('7')
    expect(logParsed['reason']).toBe('compile')

    const show = pickSub(asAny(gitCommand), 'show')
    const showParsed = parseArgs(['--commit', 'abc123'], argsOf(show))
    expect(showParsed['commit']).toBe('abc123')

    const files = pickSub(asAny(gitCommand), 'files')
    const filesParsed = parseArgs(['--commit', 'def456'], argsOf(files))
    expect(filesParsed['commit']).toBe('def456')

    const verify = pickSub(asAny(gitCommand), 'verify')
    const verifyParsed = parseArgs([], argsOf(verify))
    expect(verifyParsed['brain']).toBeUndefined()

    const stats = pickSub(asAny(gitCommand), 'stats')
    const statsParsed = parseArgs(['--brain', '/tmp/b'], argsOf(stats))
    expect(statsParsed['brain']).toBe('/tmp/b')

    const resolve = pickSub(asAny(gitCommand), 'resolve')
    const resolveParsed = parseArgs(['--auto'], argsOf(resolve))
    expect(resolveParsed['auto']).toBe(true)

    const reset = pickSub(asAny(gitCommand), 'reset')
    const resetParsed = parseArgs(['--scope', 'wiki', '--confirm'], argsOf(reset))
    expect(resetParsed['scope']).toBe('wiki')
    expect(resetParsed['confirm']).toBe(true)

    const clean = pickSub(asAny(gitCommand), 'clean')
    const cleanParsed = parseArgs(['--apply', '--max-size-mb', '8'], argsOf(clean))
    expect(cleanParsed['apply']).toBe(true)
    expect(cleanParsed['max-size-mb']).toBe('8')

    const pull = pickSub(asAny(gitCommand), 'pull')
    const pullParsed = parseArgs(['--brain', '/tmp/b', '--branch', 'dev'], argsOf(pull))
    expect(pullParsed['brain']).toBe('/tmp/b')
    expect(pullParsed['branch']).toBe('dev')
    expect(pullParsed['remote']).toBe('origin')

    const push = pickSub(asAny(gitCommand), 'push')
    const pushParsed = parseArgs(['--remote', 'upstream'], argsOf(push))
    expect(pushParsed['remote']).toBe('upstream')

    const sync = pickSub(asAny(gitCommand), 'sync')
    const syncParsed = parseArgs(['--brain', '/tmp/b'], argsOf(sync))
    expect(syncParsed['brain']).toBe('/tmp/b')
    expect(syncParsed['remote']).toBe('origin')
  })
})
