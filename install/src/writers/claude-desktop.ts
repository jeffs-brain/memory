// SPDX-License-Identifier: Apache-2.0

/**
 * Writer adapter for Claude Desktop. Uses the same `mcpServers` key as
 * Claude Code but lives at a per-OS path handled by the detector.
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

export const writeClaudeDesktop = (
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
    agent: 'claude-desktop',
    path: target,
    created: result.created,
    ...(result.backup ? { backup: result.backup } : {}),
    smokeTest: AGENT_META['claude-desktop'].smokeTest(config.storage),
  }
}
