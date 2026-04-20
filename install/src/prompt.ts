// SPDX-License-Identifier: Apache-2.0

/**
 * Interactive prompts for the install orchestrator. Uses @inquirer/prompts
 * so non-TTY callers can bail out via checks at the call site.
 */

import { checkbox, input, select } from '@inquirer/prompts'
import { AGENT_META } from './detect.js'
import { obtainHostedToken } from './oauth.js'
import {
  type AgentId,
  DEFAULT_ENDPOINT,
  type DetectionResult,
  type InstallConfig,
  type Mode,
} from './types.js'

export type PromptInput = {
  readonly detections: ReadonlyArray<DetectionResult>
  readonly defaultStorage: string
  readonly envToken?: string | undefined
}

export const runInteractivePrompts = async (opts: PromptInput): Promise<InstallConfig> => {
  const choices = opts.detections.map((d) => ({
    name: `${AGENT_META[d.agent].label}${d.exists ? '' : '  (no config found - will be created)'}`,
    value: d.agent,
    checked: d.exists,
  }))

  const agents = await checkbox<AgentId>({
    message: 'Which agents should we wire up?',
    choices,
    required: true,
  })

  const mode = await select<Mode>({
    message: 'Run mode?',
    choices: [
      { name: 'Local (FsStore + sqlite, no API key)', value: 'local' },
      { name: 'Hosted (requires JB_TOKEN)', value: 'hosted' },
    ],
    default: 'local',
  })

  const storage = await input({
    message: 'Storage path for JB_HOME:',
    default: opts.defaultStorage,
    validate: (value) => (value.trim().length > 0 ? true : 'Storage path required'),
  })

  let token: string | undefined
  let endpoint = DEFAULT_ENDPOINT
  if (mode === 'hosted') {
    const result = await obtainHostedToken(opts.envToken)
    token = result.token
    endpoint = await input({
      message: 'Hosted endpoint:',
      default: DEFAULT_ENDPOINT,
    })
  }

  return {
    agents,
    mode,
    storage: storage.trim(),
    endpoint: endpoint.trim(),
    ...(token ? { token } : {}),
    dryRun: false,
  }
}
