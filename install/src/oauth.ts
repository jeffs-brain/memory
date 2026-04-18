// SPDX-License-Identifier: Apache-2.0

/**
 * Stub OAuth device-flow for hosted mode. Today this is a plain prompt that
 * reads a pasted `JB_TOKEN`. Kept behind an explicit function so the real
 * device-flow implementation can slot in without touching the CLI surface.
 *
 * TODO: replace with proper device-flow against platform backend and bind
 * the resulting token to the OS keychain (macOS Keychain, libsecret on
 * Linux, Credential Manager on Windows).
 */

import { password } from '@inquirer/prompts'

export type DeviceFlowResult = {
  readonly token: string
  readonly source: 'env' | 'prompt'
}

export const obtainHostedToken = async (
  envToken: string | undefined,
): Promise<DeviceFlowResult> => {
  if (envToken && envToken.trim().length > 0) {
    return { token: envToken.trim(), source: 'env' }
  }
  const token = await password({
    message: 'Paste your JB_TOKEN (device-flow OAuth lands in a later release):',
    mask: true,
    validate: (value) => (value.trim().length > 0 ? true : 'Token cannot be empty'),
  })
  return { token: token.trim(), source: 'prompt' }
}
