// SPDX-License-Identifier: Apache-2.0

import { readFileSync } from 'node:fs'
import path from 'node:path'
import { describe, expect, it } from 'vitest'
import type { InstallConfig } from '../src/types.js'
import {
  writeAgent,
  writeClaudeCode,
  writeClaudeDesktop,
  writeCursor,
  writeWindsurf,
  writeZed,
} from '../src/writers/index.js'
import { backupPath } from '../src/writers/shared.js'
import type { Fs } from '../src/writers/shared.js'

const FIXTURES = path.resolve(import.meta.dirname, 'fixtures')
const fixture = (name: string): string => readFileSync(path.join(FIXTURES, name), 'utf8')

type MemFsState = {
  files: Map<string, string>
  dirs: Set<string>
  clock: { next: Date }
}

const makeMemFs = (seed: Record<string, string> = {}): { fs: Fs; state: MemFsState } => {
  const files = new Map(Object.entries(seed))
  const dirs = new Set<string>()
  for (const p of Object.keys(seed)) {
    dirs.add(path.dirname(p))
  }
  const state: MemFsState = {
    files,
    dirs,
    clock: { next: new Date('2026-04-18T09:30:15Z') },
  }
  const fs: Fs = {
    exists: (p) => files.has(p) || dirs.has(p),
    read: (p) => {
      const v = files.get(p)
      if (v === undefined) throw new Error(`ENOENT ${p}`)
      return v
    },
    write: (p, data) => {
      files.set(p, data)
      dirs.add(path.dirname(p))
    },
    rename: (from, to) => {
      const v = files.get(from)
      if (v === undefined) throw new Error(`ENOENT ${from}`)
      files.set(to, v)
      files.delete(from)
    },
    mkdir: (p) => {
      dirs.add(p)
    },
    now: () => state.clock.next,
  }
  return { fs, state }
}

const localConfig = (overrides: Partial<InstallConfig> = {}): InstallConfig => ({
  agents: ['claude-code'],
  mode: 'local',
  storage: '/home/alex/.jeffs-brain',
  endpoint: 'https://api.jeffsbrain.com',
  dryRun: false,
  ...overrides,
})

const hostedConfig = (): InstallConfig => ({
  agents: ['claude-code'],
  mode: 'hosted',
  storage: '/home/alex/.jeffs-brain',
  endpoint: 'https://api.jeffsbrain.com',
  token: 'jbp_test_token',
  dryRun: false,
})

describe('writeClaudeCode', () => {
  it('creates a new config when the file does not exist', () => {
    const { fs, state } = makeMemFs()
    const target = '/home/alex/.claude/claude.json'
    const outcome = writeClaudeCode(target, localConfig(), fs)
    expect(outcome.created).toBe(true)
    expect(outcome.backup).toBeUndefined()
    expect(outcome.smokeTest).toContain('claude mcp add jeffs-brain')
    const written = JSON.parse(state.files.get(target) ?? '{}')
    expect(written.mcpServers['jeffs-brain'].env.JB_HOME).toBe('/home/alex/.jeffs-brain')
    expect(written.mcpServers['jeffs-brain'].args).toEqual(['-y', '@jeffs-brain/memory-mcp'])
  })

  it('merges without clobbering existing servers and creates a backup', () => {
    const target = '/home/alex/.claude/claude.json'
    const { fs, state } = makeMemFs({ [target]: fixture('claude-code.existing.json') })
    const outcome = writeClaudeCode(target, localConfig(), fs)
    expect(outcome.created).toBe(false)
    expect(outcome.backup).toBeDefined()
    const written = JSON.parse(state.files.get(target) ?? '{}')
    expect(Object.keys(written.mcpServers).sort()).toEqual(['jeffs-brain', 'other-tool'])
    expect(written.telemetry).toEqual({ enabled: false })
    expect(written.mcpServers['other-tool'].command).toBe('node')
  })

  it('is idempotent: a second run produces equivalent content plus a second backup', () => {
    const target = '/home/alex/.claude/claude.json'
    const { fs, state } = makeMemFs({ [target]: fixture('claude-code.existing.json') })
    state.clock.next = new Date('2026-04-18T09:30:15Z')
    writeClaudeCode(target, localConfig(), fs)
    const afterFirst = state.files.get(target)
    state.clock.next = new Date('2026-04-18T10:15:00Z')
    const outcome = writeClaudeCode(target, localConfig(), fs)
    const afterSecond = state.files.get(target)
    expect(afterFirst).toBe(afterSecond)
    expect(outcome.backup).toContain('.jbpre-20260418101500.bak')
    const backups = [...state.files.keys()].filter((p) => p.includes('.jbpre-'))
    expect(backups.length).toBe(2)
  })
})

describe('writeClaudeDesktop', () => {
  it('writes into an empty config seeded with {}', () => {
    const target = '/Users/alex/Library/Application Support/Claude/claude_desktop_config.json'
    const { fs, state } = makeMemFs({ [target]: fixture('claude-desktop.empty.json') })
    const outcome = writeClaudeDesktop(target, hostedConfig(), fs)
    expect(outcome.created).toBe(false)
    expect(outcome.backup).toBeDefined()
    const written = JSON.parse(state.files.get(target) ?? '{}')
    expect(written.mcpServers['jeffs-brain'].env.JB_TOKEN).toBe('jbp_test_token')
    expect(written.mcpServers['jeffs-brain'].env.JB_ENDPOINT).toBe('https://api.jeffsbrain.com')
    expect(written.mcpServers['jeffs-brain'].env.JB_HOME).toBeUndefined()
  })
})

describe('writeCursor', () => {
  it('merges into an empty mcpServers object', () => {
    const target = '/home/alex/.cursor/mcp.json'
    const { fs, state } = makeMemFs({ [target]: fixture('cursor.empty.json') })
    writeCursor(target, localConfig(), fs)
    const written = JSON.parse(state.files.get(target) ?? '{}')
    expect(Object.keys(written.mcpServers)).toEqual(['jeffs-brain'])
  })

  it('creates parent dirs when config does not exist', () => {
    const target = '/home/alex/.cursor/mcp.json'
    const { fs, state } = makeMemFs()
    writeCursor(target, localConfig(), fs)
    expect(state.dirs.has('/home/alex/.cursor')).toBe(true)
  })
})

describe('writeWindsurf', () => {
  it('overwrites an existing jeffs-brain entry with a backup', () => {
    const target = '/home/alex/.codeium/windsurf/mcp_config.json'
    const { fs, state } = makeMemFs({ [target]: fixture('windsurf.existing.json') })
    const outcome = writeWindsurf(target, localConfig(), fs)
    expect(outcome.backup).toBeDefined()
    const written = JSON.parse(state.files.get(target) ?? '{}')
    expect(written.mcpServers['jeffs-brain'].env.JB_HOME).toBe('/home/alex/.jeffs-brain')
    const backup = JSON.parse(state.files.get(outcome.backup ?? '') ?? '{}')
    expect(backup.mcpServers['jeffs-brain'].env.JB_HOME).toBe('/old/path')
  })
})

describe('writeZed', () => {
  it('uses context_servers with a custom source shape, preserving other servers', () => {
    const target = '/home/alex/.config/zed/settings.json'
    const { fs, state } = makeMemFs({ [target]: fixture('zed.existing.json') })
    writeZed(target, localConfig(), fs)
    const written = JSON.parse(state.files.get(target) ?? '{}')
    expect(written.theme).toBe('One Dark')
    expect(Object.keys(written.context_servers).sort()).toEqual(['jeffs-brain', 'some-server'])
    expect(written.context_servers['jeffs-brain'].source).toBe('custom')
    expect(written.context_servers['jeffs-brain'].env.JB_HOME).toBe('/home/alex/.jeffs-brain')
    expect(written.mcpServers).toBeUndefined()
  })
})

describe('writeAgent dispatch', () => {
  it('routes each agent id to the correct writer', () => {
    const target = '/tmp/any.json'
    const { fs } = makeMemFs()
    const outcome = writeAgent('cursor', target, localConfig(), fs)
    expect(outcome.agent).toBe('cursor')
  })
})

describe('backupPath', () => {
  it('formats the UTC timestamp with zero padding', () => {
    const p = backupPath('/x/y.json', new Date('2026-01-02T03:04:05Z'))
    expect(p).toBe('/x/y.json.jbpre-20260102030405.bak')
  })
})
