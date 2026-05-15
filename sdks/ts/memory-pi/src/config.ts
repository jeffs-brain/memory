import { z } from 'zod'

const StoreConfigSchema = z.discriminatedUnion('kind', [
  z.object({ kind: z.literal('auto') }),
  z.object({ kind: z.literal('fs') }),
  z.object({ kind: z.literal('git'), remote: z.string().optional() }),
  z.object({
    kind: z.literal('http'),
    endpoint: z.string().url(),
    token: z.string(),
  }),
])

const EmbedderConfigSchema = z.discriminatedUnion('kind', [
  z.object({ kind: z.literal('auto') }),
  z.object({
    kind: z.literal('ollama'),
    baseUrl: z.string().optional(),
    model: z.string().optional(),
  }),
  z.object({
    kind: z.literal('openai'),
    apiKey: z.string(),
    model: z.string().optional(),
  }),
  z.object({ kind: z.literal('tei'), endpoint: z.string().url() }),
  z.object({ kind: z.literal('off') }),
])

const ProviderConfigSchema = z.discriminatedUnion('kind', [
  z.object({ kind: z.literal('auto') }),
  z.object({
    kind: z.literal('openai'),
    apiKey: z.string(),
    model: z.string().optional(),
  }),
  z.object({
    kind: z.literal('anthropic'),
    apiKey: z.string(),
    model: z.string().optional(),
  }),
  z.object({
    kind: z.literal('ollama'),
    baseUrl: z.string().optional(),
    model: z.string().optional(),
  }),
])

const RerankerConfigSchema = z.discriminatedUnion('kind', [
  z.object({ kind: z.literal('auto') }),
  z.object({ kind: z.literal('llm') }),
  z.object({ kind: z.literal('tei'), endpoint: z.string().url() }),
  z.object({ kind: z.literal('off') }),
])

export const MEMORY_TOOL_NAMES = [
  'remember',
  'recall',
  'search',
  'ask',
  'ingest_file',
  'ingest_url',
  'extract',
  'reflect',
  'consolidate',
  'create_brain',
  'list_brains',
] as const

export type MemoryToolName = (typeof MEMORY_TOOL_NAMES)[number]

export const MemoryExtensionConfigSchema = z
  .object({
    brainRoot: z.string().optional(),
    brainId: z.string().optional(),
    // When true, `brainRoot` is treated as the brain itself and the
    // `brainId` is NOT joined into a subdirectory. Wiki / memory / raw
    // content is expected to live directly under `brainRoot`. Useful
    // for single-brain hosts that already manage one brain per identity
    // on disk and do not want memory-pi to nest its own subdirectory
    // inside the git working tree.
    flatLayout: z.boolean().optional(),
    // Override the on-disk path for the search index. When unset the
    // index lives at `<resolvedBrainRoot>/search.sqlite`. Hosts that
    // keep the brain content git-backed typically point this at a
    // machine-local state directory so the FTS sqlite never enters the
    // working tree.
    searchIndexPath: z.string().optional(),
    // When the search index is empty on boot, walk these subdirectories
    // of `brainRoot` and ingest every markdown file found. Only honoured
    // when `flatLayout` is true. Defaults to ['wiki', 'memory', 'raw'].
    bootstrapScanDirs: z.array(z.string()).optional(),
    // Override the path to sqlite-vec's loadable extension (e.g.
    // `vec0.so`). sqlite-vec normally resolves this via
    // `import.meta.resolve('sqlite-vec-linux-x64/vec0.so')`, which fails
    // inside a bun-compiled single-file binary where the platform package
    // is not on disk. Hosts that compile a binary should copy `vec0.so`
    // (or the platform-equivalent .dylib/.dll) next to the executable
    // and point this option at it.
    vectorExtensionPath: z.string().optional(),
    store: StoreConfigSchema.optional(),
    embedder: EmbedderConfigSchema.optional(),
    provider: ProviderConfigSchema.optional(),
    reranker: RerankerConfigSchema.optional(),
    recall: z
      .object({
        onPrompt: z.boolean().optional(),
        topK: z.number().int().positive().optional(),
        minScore: z.number().min(0).max(1).optional(),
        scope: z.string().optional(),
        fallbackScopes: z.array(z.string()).optional(),
        cacheFriendly: z.boolean().optional(),
      })
      .optional(),
    extract: z
      .object({
        onTurnEnd: z.boolean().optional(),
        minMessages: z.number().int().nonnegative().optional(),
        contextualise: z.boolean().optional(),
      })
      .optional(),
    reflect: z
      .object({
        onSessionEnd: z.boolean().optional(),
      })
      .optional(),
    consolidate: z
      .object({
        schedule: z.enum(['manual', 'session', '@daily']).optional(),
      })
      .optional(),
    tools: z
      .object({
        expose: z.array(z.enum(MEMORY_TOOL_NAMES)).optional(),
      })
      .optional(),
    acl: z
      .object({
        actorId: z.string().optional(),
        provider: z
          .union([
            z.literal('rbac'),
            z.object({ kind: z.literal('openfga'), endpoint: z.string().url() }),
          ])
          .optional(),
      })
      .optional(),
  })
  .partial()

export type MemoryExtensionConfig = z.infer<typeof MemoryExtensionConfigSchema>
