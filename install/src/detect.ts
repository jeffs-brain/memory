// SPDX-License-Identifier: Apache-2.0

/**
 * Agent config detection. Given a HOME directory and platform, resolves the
 * list of candidate config paths per supported agent and returns a detection
 * result indicating which actually exist on disk.
 */

import { existsSync } from 'node:fs'
import path from 'node:path'
import type { AgentId, AgentMeta, DetectionResult } from './types.js'

const joinHome = (home: string, ...parts: ReadonlyArray<string>): string =>
  path.join(home, ...parts)

export const AGENT_META: Readonly<Record<AgentId, AgentMeta>> = {
  'claude-code': {
    id: 'claude-code',
    label: 'Claude Code',
    candidatePaths: (home) => [
      joinHome(home, '.claude', 'claude.json'),
      joinHome(home, '.config', 'claude', 'claude.json'),
    ],
    smokeTest: (storage) =>
      `claude mcp add jeffs-brain --env JB_HOME=${storage} -- npx -y @jeffs-brain/memory-mcp`,
  },
  'claude-desktop': {
    id: 'claude-desktop',
    label: 'Claude Desktop',
    candidatePaths: (home, platform) => {
      if (platform === 'darwin') {
        return [
          joinHome(
            home,
            'Library',
            'Application Support',
            'Claude',
            'claude_desktop_config.json',
          ),
        ]
      }
      if (platform === 'win32') {
        const appData = process.env.APPDATA ?? joinHome(home, 'AppData', 'Roaming')
        return [path.join(appData, 'Claude', 'claude_desktop_config.json')]
      }
      return [joinHome(home, '.config', 'Claude', 'claude_desktop_config.json')]
    },
    smokeTest: () =>
      'Restart Claude Desktop, then ask: "What MCP tools are available?" You should see memory_remember, memory_recall, memory_search.',
  },
  cursor: {
    id: 'cursor',
    label: 'Cursor',
    candidatePaths: (home) => [
      joinHome(home, '.cursor', 'mcp.json'),
      joinHome(home, '.config', 'cursor', 'mcp.json'),
    ],
    smokeTest: () =>
      'Reload Cursor (Cmd/Ctrl+Shift+P -> "Reload Window"), then open the MCP panel and confirm jeffs-brain is listed.',
  },
  windsurf: {
    id: 'windsurf',
    label: 'Windsurf',
    candidatePaths: (home) => [joinHome(home, '.codeium', 'windsurf', 'mcp_config.json')],
    smokeTest: () =>
      'Restart Windsurf, open Cascade settings -> MCP, confirm jeffs-brain is green.',
  },
  zed: {
    id: 'zed',
    label: 'Zed',
    candidatePaths: (home) => [joinHome(home, '.config', 'zed', 'settings.json')],
    smokeTest: () =>
      'Restart Zed (cmd+q / file->quit), then run `zed: context server status` from the command palette.',
  },
}

export type DetectOptions = {
  readonly home: string
  readonly platform?: NodeJS.Platform
  readonly fileExists?: (p: string) => boolean
}

export const detectAgents = (opts: DetectOptions): ReadonlyArray<DetectionResult> => {
  const platform = opts.platform ?? process.platform
  const fileExists = opts.fileExists ?? existsSync
  const results: Array<DetectionResult> = []
  for (const meta of Object.values(AGENT_META)) {
    const candidates = meta.candidatePaths(opts.home, platform)
    const match = candidates.find((p) => fileExists(p))
    const resolved = match ?? candidates[0]
    if (!resolved) continue
    results.push({ agent: meta.id, path: resolved, exists: Boolean(match) })
  }
  return results
}

export const resolveAgentPath = (
  agent: AgentId,
  home: string,
  platform: NodeJS.Platform = process.platform,
  fileExists: (p: string) => boolean = existsSync,
): string => {
  const meta = AGENT_META[agent]
  const candidates = meta.candidatePaths(home, platform)
  const match = candidates.find((p) => fileExists(p))
  const resolved = match ?? candidates[0]
  if (!resolved) {
    throw new Error(`No candidate paths configured for agent ${agent}`)
  }
  return resolved
}
