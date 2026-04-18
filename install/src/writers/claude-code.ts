// SPDX-License-Identifier: Apache-2.0

/**
 * Writer adapter for Claude Code. Claude Code stores MCP servers under
 * `mcpServers` in `~/.claude/claude.json` (or the XDG variant).
 */

import { AGENT_META } from '../detect.js'
import type { InstallConfig, WriteOutcome } from '../types.js'
import {
  buildServerSpec,
  type Fs,
  isObject,
  nodeFs,
  readJsonOrEmpty,
  toPlainSpec,
  writeJsonWithBackup,
} from './shared.js'

export const writeClaudeCode = (
  target: string,
  config: InstallConfig,
  fs: Fs = nodeFs,
): WriteOutcome => {
  const current = readJsonOrEmpty(fs, target)
  const servers = isObject(current.mcpServers) ? { ...current.mcpServers } : {}
  servers['jeffs-brain'] = toPlainSpec(buildServerSpec(config))
  const next = { ...current, mcpServers: servers }
  const result = writeJsonWithBackup(fs, target, next)
  return {
    agent: 'claude-code',
    path: target,
    created: result.created,
    ...(result.backup ? { backup: result.backup } : {}),
    smokeTest: AGENT_META['claude-code'].smokeTest(config.storage),
  }
}
