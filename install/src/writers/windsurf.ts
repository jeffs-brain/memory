// SPDX-License-Identifier: Apache-2.0

/**
 * Writer adapter for Windsurf. Stores MCP servers in
 * `~/.codeium/windsurf/mcp_config.json` under `mcpServers`.
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

export const writeWindsurf = (
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
    agent: 'windsurf',
    path: target,
    created: result.created,
    ...(result.backup ? { backup: result.backup } : {}),
    smokeTest: AGENT_META.windsurf.smokeTest(config.storage),
  }
}
