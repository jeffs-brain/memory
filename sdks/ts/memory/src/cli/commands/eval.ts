// SPDX-License-Identifier: Apache-2.0

import { defineCommand } from 'citty'
import { lmeCommand } from './lme.js'

export const evalCommand = defineCommand({
  meta: {
    name: 'eval',
    description: 'Evaluation commands',
  },
  subCommands: {
    lme: lmeCommand,
  },
})
