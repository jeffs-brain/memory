// SPDX-License-Identifier: Apache-2.0

/**
 * Public surface of the memory CLI. Re-exports the root command plus the
 * helpers library consumers need to embed the CLI in a custom runner.
 */

export { rootCommand, exitCodeForError, CliError, CliUsageError } from './main.js'
