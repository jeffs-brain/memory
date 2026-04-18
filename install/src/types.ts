// SPDX-License-Identifier: Apache-2.0

/**
 * Shared types for the `@jeffs-brain/install` orchestrator. Kept in one place
 * so the writer adapters, detector, prompts and CLI entry can agree on the
 * shape of what flows between them.
 */

export type AgentId =
  | 'claude-code'
  | 'claude-desktop'
  | 'cursor'
  | 'windsurf'
  | 'zed'

export type Mode = 'local' | 'hosted'

export type InstallConfig = {
  readonly agents: ReadonlyArray<AgentId>
  readonly mode: Mode
  readonly storage: string
  readonly endpoint: string
  readonly token?: string
  readonly dryRun: boolean
}

export type McpServerSpec = {
  readonly command: string
  readonly args: ReadonlyArray<string>
  readonly env: Readonly<Record<string, string>>
}

export type DetectionResult = {
  readonly agent: AgentId
  readonly path: string
  readonly exists: boolean
}

export type WriteOutcome = {
  readonly agent: AgentId
  readonly path: string
  readonly created: boolean
  readonly backup?: string
  readonly smokeTest: string
}

export type AgentMeta = {
  readonly id: AgentId
  readonly label: string
  readonly candidatePaths: (home: string, platform: NodeJS.Platform) => ReadonlyArray<string>
  readonly smokeTest: (storage: string) => string
}

export const ALL_AGENTS: ReadonlyArray<AgentId> = [
  'claude-code',
  'claude-desktop',
  'cursor',
  'windsurf',
  'zed',
]

export const DEFAULT_ENDPOINT = 'https://api.jeffsbrain.com'
export const DEFAULT_STORAGE = '~/.jeffs-brain'
export const MCP_PACKAGE = '@jeffs-brain/memory-mcp'
