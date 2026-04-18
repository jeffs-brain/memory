/**
 * Public surface of the jbmem CLI. Re-exports the root command plus the
 * helpers library consumers need to embed the CLI in a custom runner.
 */

export { rootCommand, exitCodeForError, CliError, CliUsageError } from './main.js'
