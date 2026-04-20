// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from 'vitest'
import { detectAgents, resolveAgentPath } from '../src/detect.js'

const FAKE_HOME = '/tmp/jb-fake-home'

describe('detectAgents', () => {
  it('returns a record per agent, marking non-existent configs as exists:false', () => {
    const results = detectAgents({
      home: FAKE_HOME,
      platform: 'linux',
      fileExists: () => false,
    })
    expect(results.map((r) => r.agent)).toEqual([
      'claude-code',
      'claude-desktop',
      'cursor',
      'windsurf',
      'zed',
    ])
    expect(results.every((r) => r.exists === false)).toBe(true)
  })

  it('marks agents as detected when their candidate paths exist', () => {
    const detectedPaths = new Set([
      `${FAKE_HOME}/.claude/claude.json`,
      `${FAKE_HOME}/.codeium/windsurf/mcp_config.json`,
    ])
    const results = detectAgents({
      home: FAKE_HOME,
      platform: 'linux',
      fileExists: (p) => detectedPaths.has(p),
    })
    const byAgent = Object.fromEntries(results.map((r) => [r.agent, r]))
    expect(byAgent['claude-code']?.exists).toBe(true)
    expect(byAgent.windsurf?.exists).toBe(true)
    expect(byAgent.cursor?.exists).toBe(false)
    expect(byAgent.zed?.exists).toBe(false)
  })

  it('picks the XDG config path for Claude Code when the dotfile variant is missing', () => {
    const results = detectAgents({
      home: FAKE_HOME,
      platform: 'linux',
      fileExists: (p) => p === `${FAKE_HOME}/.config/claude/claude.json`,
    })
    const claudeCode = results.find((r) => r.agent === 'claude-code')
    expect(claudeCode?.exists).toBe(true)
    expect(claudeCode?.path).toBe(`${FAKE_HOME}/.config/claude/claude.json`)
  })

  it('uses macOS-specific Claude Desktop path on darwin', () => {
    const results = detectAgents({
      home: FAKE_HOME,
      platform: 'darwin',
      fileExists: () => false,
    })
    const desktop = results.find((r) => r.agent === 'claude-desktop')
    expect(desktop?.path).toBe(
      `${FAKE_HOME}/Library/Application Support/Claude/claude_desktop_config.json`,
    )
  })

  it('uses Linux config path for Claude Desktop on linux', () => {
    const results = detectAgents({
      home: FAKE_HOME,
      platform: 'linux',
      fileExists: () => false,
    })
    const desktop = results.find((r) => r.agent === 'claude-desktop')
    expect(desktop?.path).toBe(`${FAKE_HOME}/.config/Claude/claude_desktop_config.json`)
  })
})

describe('resolveAgentPath', () => {
  it('returns the first candidate when nothing exists', () => {
    const p = resolveAgentPath('cursor', FAKE_HOME, 'linux', () => false)
    expect(p).toBe(`${FAKE_HOME}/.cursor/mcp.json`)
  })

  it('returns the first existing candidate when one matches', () => {
    const p = resolveAgentPath(
      'cursor',
      FAKE_HOME,
      'linux',
      (path) => path === `${FAKE_HOME}/.config/cursor/mcp.json`,
    )
    expect(p).toBe(`${FAKE_HOME}/.config/cursor/mcp.json`)
  })
})
