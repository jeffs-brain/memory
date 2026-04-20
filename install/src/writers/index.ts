// SPDX-License-Identifier: Apache-2.0

import type { AgentId, InstallConfig, WriteOutcome } from '../types.js'
import { writeClaudeCode } from './claude-code.js'
import { writeClaudeDesktop } from './claude-desktop.js'
import { writeCursor } from './cursor.js'
import type { Fs } from './shared.js'
import { nodeFs } from './shared.js'
import { writeWindsurf } from './windsurf.js'
import { writeZed } from './zed.js'

export { buildServerSpec, backupPath, type Fs, nodeFs } from './shared.js'
export { writeClaudeCode, writeClaudeDesktop, writeCursor, writeWindsurf, writeZed }

export const WRITERS: Readonly<
  Record<AgentId, (target: string, config: InstallConfig, fs?: Fs) => WriteOutcome>
> = {
  'claude-code': writeClaudeCode,
  'claude-desktop': writeClaudeDesktop,
  cursor: writeCursor,
  windsurf: writeWindsurf,
  zed: writeZed,
}

export const writeAgent = (
  agent: AgentId,
  target: string,
  config: InstallConfig,
  fs: Fs = nodeFs,
): WriteOutcome => WRITERS[agent](target, config, fs)
