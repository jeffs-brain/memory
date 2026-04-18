#!/usr/bin/env node
// SPDX-License-Identifier: Apache-2.0

/**
 * Entry point for `npx @jeffs-brain/install`. Detects MCP-capable agents,
 * prompts (or reads flags) for config, then wires the `jeffs-brain` MCP
 * server into each agent's config file idempotently.
 */

import { homedir } from 'node:os'
import { defineCommand, runMain } from 'citty'
import { z } from 'zod'
import { AGENT_META, detectAgents, resolveAgentPath } from './detect.js'
import { runInteractivePrompts } from './prompt.js'
import { obtainHostedToken } from './oauth.js'
import {
  type AgentId,
  ALL_AGENTS,
  DEFAULT_ENDPOINT,
  DEFAULT_STORAGE,
  type InstallConfig,
  type Mode,
  type WriteOutcome,
} from './types.js'
import { writeAgent } from './writers/index.js'

const agentSchema = z.enum([
  'claude-code',
  'claude-desktop',
  'cursor',
  'windsurf',
  'zed',
])

const modeSchema = z.enum(['local', 'hosted'])

const parseAgentList = (raw: string): ReadonlyArray<AgentId> => {
  const tokens = raw
    .split(',')
    .map((t) => t.trim())
    .filter((t) => t.length > 0)
  const agents: Array<AgentId> = []
  for (const token of tokens) {
    const parsed = agentSchema.safeParse(token)
    if (!parsed.success) {
      throw new Error(
        `Unknown agent '${token}'. Supported: ${ALL_AGENTS.join(', ')}`,
      )
    }
    agents.push(parsed.data)
  }
  if (agents.length === 0) {
    throw new Error('At least one --agents entry required')
  }
  return agents
}

const expandHome = (p: string): string => {
  if (p === '~' || p.startsWith('~/')) return `${homedir()}${p.slice(1)}`
  return p
}

export type RunOptions = {
  readonly agents?: string
  readonly mode?: string
  readonly storage?: string
  readonly endpoint?: string
  readonly token?: string
  readonly nonInteractive?: boolean
  readonly dryRun?: boolean
  readonly home?: string
}

const formatOutcome = (outcome: WriteOutcome): string => {
  const label = AGENT_META[outcome.agent].label
  const note = outcome.created ? 'created new config' : 'merged into existing config'
  const backup = outcome.backup ? `\n    backup: ${outcome.backup}` : ''
  return `  ${label}\n    path:   ${outcome.path}\n    status: ${note}${backup}\n    smoke:  ${outcome.smokeTest}`
}

export const run = async (opts: RunOptions): Promise<ReadonlyArray<WriteOutcome>> => {
  const home = opts.home ?? homedir()
  const nonInteractive =
    Boolean(opts.nonInteractive) || Boolean(opts.agents) || !process.stdin.isTTY

  let config: InstallConfig
  if (nonInteractive) {
    const agents = parseAgentList(opts.agents ?? '')
    const mode = (modeSchema.parse(opts.mode ?? 'local') as Mode)
    const storage = expandHome(opts.storage ?? DEFAULT_STORAGE)
    const endpoint = opts.endpoint ?? DEFAULT_ENDPOINT
    let token: string | undefined
    if (mode === 'hosted') {
      const envToken = opts.token ?? process.env.JB_TOKEN
      if (!envToken || envToken.trim().length === 0) {
        throw new Error('Hosted mode requires --token or JB_TOKEN env var in non-interactive runs')
      }
      const result = await obtainHostedToken(envToken)
      token = result.token
    }
    config = {
      agents,
      mode,
      storage,
      endpoint,
      ...(token ? { token } : {}),
      dryRun: Boolean(opts.dryRun),
    }
  } else {
    const detections = detectAgents({ home })
    const defaultStorage = expandHome(opts.storage ?? DEFAULT_STORAGE)
    config = await runInteractivePrompts({
      detections,
      defaultStorage,
      ...(opts.token ? { envToken: opts.token } : {}),
    })
  }

  if (config.dryRun) {
    process.stdout.write('Dry run - no files will be written.\n')
  }

  const outcomes: Array<WriteOutcome> = []
  for (const agent of config.agents) {
    const target = resolveAgentPath(agent, home)
    if (config.dryRun) {
      outcomes.push({
        agent,
        path: target,
        created: false,
        smokeTest: AGENT_META[agent].smokeTest(config.storage),
      })
      continue
    }
    const outcome = writeAgent(agent, target, config)
    outcomes.push(outcome)
  }

  process.stdout.write(`\nWired jeffs-brain into ${outcomes.length} agent(s):\n`)
  for (const o of outcomes) {
    process.stdout.write(`${formatOutcome(o)}\n`)
  }
  process.stdout.write(
    '\nTip: restart each agent after install so it picks up the new MCP server.\n',
  )
  return outcomes
}

export const rootCommand = defineCommand({
  meta: {
    name: 'jeffs-brain-install',
    description:
      'Wire the @jeffs-brain/memory-mcp server into Claude Code, Claude Desktop, Cursor, Windsurf, and Zed configs.',
  },
  args: {
    agents: {
      type: 'string',
      description: 'Comma-separated list of agents (non-interactive)',
    },
    mode: {
      type: 'string',
      description: 'local | hosted (default: local)',
    },
    storage: {
      type: 'string',
      description: `Storage path for JB_HOME (default: ${DEFAULT_STORAGE})`,
    },
    endpoint: {
      type: 'string',
      description: `Hosted endpoint (default: ${DEFAULT_ENDPOINT})`,
    },
    token: {
      type: 'string',
      description: 'JB_TOKEN for hosted mode (or set JB_TOKEN env var)',
    },
    'non-interactive': {
      type: 'boolean',
      description: 'Skip prompts; requires --agents',
    },
    'dry-run': {
      type: 'boolean',
      description: 'Show what would be written without touching files',
    },
  },
  run: async ({ args }) => {
    await run({
      ...(typeof args.agents === 'string' ? { agents: args.agents } : {}),
      ...(typeof args.mode === 'string' ? { mode: args.mode } : {}),
      ...(typeof args.storage === 'string' ? { storage: args.storage } : {}),
      ...(typeof args.endpoint === 'string' ? { endpoint: args.endpoint } : {}),
      ...(typeof args.token === 'string' ? { token: args.token } : {}),
      nonInteractive: Boolean(args['non-interactive']),
      dryRun: Boolean(args['dry-run']),
    })
  },
})

const isEntry = (() => {
  try {
    const invoked = process.argv[1]
    if (!invoked) return false
    const self = new URL(import.meta.url).pathname
    return invoked === self || invoked.endsWith('/index.js') || invoked.endsWith('jeffs-brain-install')
  } catch {
    return false
  }
})()

if (isEntry) {
  runMain(rootCommand).catch((err: unknown) => {
    const message = err instanceof Error ? err.message : String(err)
    process.stderr.write(`jeffs-brain-install: ${message}\n`)
    process.exit(1)
  })
}
