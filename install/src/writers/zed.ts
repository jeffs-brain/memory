// SPDX-License-Identifier: Apache-2.0

/**
 * Writer adapter for Zed. Zed's `settings.json` puts MCP servers under the
 * `context_servers` key with a `source` field; we write the `custom` shape.
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

export const writeZed = (
  target: string,
  config: InstallConfig,
  fs: Fs = nodeFs,
): WriteOutcome => {
  const current = readJsonOrEmpty(fs, target)
  const servers = isObject(current.context_servers)
    ? { ...current.context_servers }
    : {}
  const spec = toPlainSpec(buildServerSpec(config))
  servers['jeffs-brain'] = {
    source: 'custom',
    command: spec.command,
    args: spec.args,
    env: spec.env,
  }
  const next = { ...current, context_servers: servers }
  const result = writeJsonWithBackup(fs, target, next)
  return {
    agent: 'zed',
    path: target,
    created: result.created,
    ...(result.backup ? { backup: result.backup } : {}),
    smokeTest: AGENT_META.zed.smokeTest(config.storage),
  }
}
