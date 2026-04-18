/**
 * Root command definition for the jbmem CLI. Kept as its own module so
 * tests can import `rootCommand` without triggering `runMain`.
 */

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

export const rootCommand = defineCommand({
  meta: {
    name: 'jbmem',
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
