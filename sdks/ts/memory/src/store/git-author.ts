// SPDX-License-Identifier: Apache-2.0

import { execFile as execFileCallback } from 'node:child_process'
import { promisify } from 'node:util'

const execFile = promisify(execFileCallback)

export type GitSignature = {
  readonly name: string
  readonly email: string
}

export const DEFAULT_GIT_SIGNATURE: GitSignature = {
  name: 'jeffs-brain',
  email: 'noreply@jeffsbrain.com',
}

const NEEDS_IDENTITY = new Set(['commit', 'rebase', 'stash'])

const readGitConfigValue = async (cwd: string, key: string): Promise<string | undefined> => {
  try {
    const { stdout } = await execFile('git', ['config', '--get', key], {
      cwd,
      encoding: 'utf8',
    })
    const value = stdout.trim()
    return value === '' ? undefined : value
  } catch {
    return undefined
  }
}

const envValue = (...keys: readonly string[]): string | undefined => {
  for (const key of keys) {
    const value = process.env[key]
    if (value !== undefined && value.trim() !== '') return value
  }
  return undefined
}

export const buildGitExecEnv = async (
  cwd: string,
  args: readonly string[],
  fallback: GitSignature = DEFAULT_GIT_SIGNATURE,
): Promise<NodeJS.ProcessEnv | undefined> => {
  const command = args[0]
  if (command === undefined || !NEEDS_IDENTITY.has(command)) return undefined

  const name =
    envValue('GIT_AUTHOR_NAME', 'GIT_COMMITTER_NAME') ??
    (await readGitConfigValue(cwd, 'user.name')) ??
    fallback.name
  const email =
    envValue('GIT_AUTHOR_EMAIL', 'GIT_COMMITTER_EMAIL') ??
    (await readGitConfigValue(cwd, 'user.email')) ??
    fallback.email

  return {
    ...process.env,
    ...(envValue('GIT_AUTHOR_NAME') === undefined ? { GIT_AUTHOR_NAME: name } : {}),
    ...(envValue('GIT_COMMITTER_NAME') === undefined ? { GIT_COMMITTER_NAME: name } : {}),
    ...(envValue('GIT_AUTHOR_EMAIL') === undefined ? { GIT_AUTHOR_EMAIL: email } : {}),
    ...(envValue('GIT_COMMITTER_EMAIL') === undefined ? { GIT_COMMITTER_EMAIL: email } : {}),
  }
}
