// SPDX-License-Identifier: Apache-2.0

/**
 * Resolve runtime configuration from process env. Two modes:
 *
 *  - `hosted`: `JB_TOKEN` is set, server talks to the remote platform via
 *    `HttpStore`. Endpoint defaults to `https://api.jeffsbrain.com` and
 *    can be overridden with `JB_ENDPOINT`.
 *  - `local`: no token, server runs against `JB_HOME` (default
 *    `$HOME/.jeffs-brain`). Uses FsStore + sqlite-backed search + Ollama
 *    auto-detect on localhost.
 */

import { homedir } from 'node:os'
import { join } from 'node:path'

export type LocalConfig = {
  readonly kind: 'local'
  readonly brainRoot: string
  readonly defaultBrain: string | undefined
  readonly ollamaBaseUrl: string
}

export type HostedConfig = {
  readonly kind: 'hosted'
  readonly endpoint: string
  readonly token: string
  readonly defaultBrain: string | undefined
}

export type ConfigMode = LocalConfig | HostedConfig

const DEFAULT_ENDPOINT = 'https://api.jeffsbrain.com'
const DEFAULT_OLLAMA_BASE_URL = 'http://localhost:11434'

export const resolveConfig = (env: NodeJS.ProcessEnv): ConfigMode => {
  const defaultBrain = env.JB_BRAIN ?? undefined
  const token = env.JB_TOKEN
  if (token !== undefined && token !== '') {
    return {
      kind: 'hosted',
      endpoint: env.JB_ENDPOINT ?? DEFAULT_ENDPOINT,
      token,
      defaultBrain,
    }
  }
  const home = env.JB_HOME ?? join(env.HOME ?? homedir(), '.jeffs-brain')
  return {
    kind: 'local',
    brainRoot: home,
    defaultBrain,
    ollamaBaseUrl: env.OLLAMA_HOST ?? DEFAULT_OLLAMA_BASE_URL,
  }
}
