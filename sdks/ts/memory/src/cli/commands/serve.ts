import { defineCommand } from 'citty'
import { openBrain } from '../brain.js'
import {
  CliUsageError,
  buildEmbedder,
  buildProvider,
  embedderFromEnv,
  providerFromEnv,
  resolveBrainDir,
} from '../config.js'
import { startServer } from '../server.js'

const DEFAULT_PORT = 7300

export const serveCommand = defineCommand({
  meta: {
    name: 'serve',
    description: 'Start a minimal HTTP server exposing the brain',
  },
  args: {
    brain: {
      type: 'string',
      description: 'Brain directory (overrides JBMEM_BRAIN)',
    },
    port: {
      type: 'string',
      description: 'Port to listen on',
      default: String(DEFAULT_PORT),
    },
    host: {
      type: 'string',
      description: 'Host to bind',
      default: '127.0.0.1',
    },
  },
  run: async ({ args }) => {
    const port = parsePort(args.port)
    const brainDir = resolveBrainDir(typeof args.brain === 'string' ? args.brain : undefined)
    const embedderSettings = embedderFromEnv()
    const embedder = embedderSettings !== undefined ? buildEmbedder(embedderSettings) : undefined
    const providerEnv = process.env.JBMEM_PROVIDER
    const provider =
      providerEnv !== undefined && providerEnv !== '' ? buildProvider(providerFromEnv()) : undefined
    const store = await openBrain(brainDir)
    const server = await startServer({
      store,
      port,
      hostname: typeof args.host === 'string' ? args.host : '127.0.0.1',
      ...(embedder !== undefined ? { embedder } : {}),
      ...(provider !== undefined ? { provider } : {}),
    })
    process.stderr.write(`jbmem serve: listening on ${server.url}\n`)
    const shutdown = async (): Promise<void> => {
      process.stderr.write('jbmem serve: shutting down\n')
      await server.stop()
      await store.close()
      process.exit(0)
    }
    process.once('SIGINT', () => {
      void shutdown()
    })
    process.once('SIGTERM', () => {
      void shutdown()
    })
    // Keep the event loop alive until a signal arrives.
    await new Promise(() => undefined)
  },
})

const parsePort = (raw: unknown): number => {
  const str = typeof raw === 'string' ? raw : String(DEFAULT_PORT)
  const n = Number.parseInt(str, 10)
  if (!Number.isFinite(n) || n <= 0 || n > 65535) {
    throw new CliUsageError(`serve: invalid --port '${str}'`)
  }
  return n
}
