// SPDX-License-Identifier: Apache-2.0

/**
 * Writer adapter for Cursor. Cursor uses an `mcpServers` top-level object in
 * `~/.cursor/mcp.json` (matching the MCP reference shape).
 */

import { AGENT_META } from '../detect.js'
import type { InstallConfig, WriteOutcome } from '../types.js'
import {
  type Fs,
  buildServerSpec,
  isObject,
  nodeFs,
  readJsonOrEmpty,
  toPlainSpec,
  writeJsonWithBackup,
} from './shared.js'

export const writeCursor = (
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
    agent: 'cursor',
    path: target,
    created: result.created,
    ...(result.backup ? { backup: result.backup } : {}),
    smokeTest: AGENT_META.cursor.smokeTest(config.storage),
  }
}
