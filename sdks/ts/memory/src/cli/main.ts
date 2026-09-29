// SPDX-License-Identifier: Apache-2.0

/**
 * Root command definition for the memory CLI. Kept as its own module so
 * tests can import `rootCommand` without triggering `runMain`.
 */

import { readFileSync } from 'node:fs'
import { defineCommand } from 'citty'
import {
  aclCommand,
  consolidateCommand,
  evalCommand,
  extractCommand,
  gitCommand,
  ingestCommand,
  initCommand,
  reflectCommand,
  searchCommand,
  serveCommand,
} from './commands/index.js'
import { CliError, CliUsageError } from './config.js'

/** Read `version` from package.json at `relative` to this module. */
const readPackageVersion = (relative: string): string => {
  const raw: unknown = JSON.parse(readFileSync(new URL(relative, import.meta.url), 'utf8'))
  if (
    typeof raw === 'object' &&
    raw !== null &&
    'version' in raw &&
    typeof raw.version === 'string'
  ) {
    return raw.version
  }
  throw new Error(`package.json at ${relative} has no version`)
}

/** The published package version, read so it can never drift. */
export const CLI_VERSION = readPackageVersion('../../package.json')

export const rootCommand = defineCommand({
  meta: {
    name: 'memory',
    version: CLI_VERSION,
    description: 'Slim CLI for @jeffs-brain/memory',
  },
  subCommands: {
    init: initCommand,
    ingest: ingestCommand,
    search: searchCommand,
    extract: extractCommand,
    reflect: reflectCommand,
    consolidate: consolidateCommand,
    eval: evalCommand,
    serve: serveCommand,
    acl: aclCommand,
    git: gitCommand,
  },
})

export const exitCodeForError = (err: unknown): number => {
  if (err instanceof CliUsageError) return 2
  if (err instanceof CliError) return 1
  return 1
}

export { CliError, CliUsageError }
