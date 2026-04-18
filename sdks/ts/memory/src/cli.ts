#!/usr/bin/env bun
/**
 * `jbmem` CLI entry point. Compiled to `dist/cli.js` and wired as the
 * `jbmem` bin in package.json.
 *
 * Exit codes (see §Style in the CLI spec):
 *   0 - success
 *   1 - expected runtime failure (network down, brain missing, etc.)
 *   2 - usage error (bad flags, missing required positional)
 */

import { runMain } from 'citty'
import { exitCodeForError, rootCommand } from './cli/main.js'

runMain(rootCommand).catch((err: unknown) => {
  const message = err instanceof Error ? err.message : String(err)
  process.stderr.write(`jbmem: ${message}\n`)
  process.exit(exitCodeForError(err))
})
