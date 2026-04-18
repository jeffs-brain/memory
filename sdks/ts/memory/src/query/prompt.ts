// SPDX-License-Identifier: Apache-2.0

/**
 * Distillation prompt and LLM caller. The system prompt is ported
 * verbatim from apps/jeff/internal/query/prompt.go in the jeff
 * repository so the TS SDK stays aligned with the canonical behaviour.
 * When jeff updates the prompt, bump `DISTILL_PROMPT_VERSION` and
 * mirror the change here.
 */

import type { Provider } from '../llm/index.js'

export const DISTILL_PROMPT_VERSION = 1

/**
 * Verbatim copy of `distillSystemPrompt` from
 * apps/jeff/internal/query/prompt.go. Do not paraphrase.
 */
export const DISTILL_SYSTEM_PROMPT = `You are a search query distiller. Given a raw user message (which may be a huge error paste, a vague question, or a multi-part request), produce structured search queries that will retrieve the most relevant information from a knowledge base.

Respond with ONLY a JSON array of query objects:
[{"text": "concise search query", "domain": "optional domain hint", "entities": ["extracted entities"], "recency_bias": "recent|historical|", "confidence": 0.0-1.0}]

Rules:
- Extract the actual question from noise (error logs, pasted code, etc.)
- Split multi-intent queries into separate query objects
- Expand abbreviations and jargon where possible
- Resolve anaphoric references ("it", "that") using context if available
- Maximum 3 queries per input
- Each query text should be 5-30 words, focused and searchable
- Set confidence to 0.0-1.0 based on how certain you are the rewrite captures the intent`

const DEFAULT_MAX_TOKENS = 256
const DEFAULT_TEMPERATURE = 0

/**
 * callDistillLLM issues a single completion against the supplied
 * provider using the jeff distillation prompt. The raw user message is
 * sent verbatim as the user turn; the caller is responsible for any
 * truncation upstream.
 */
export async function callDistillLLM(
  provider: Provider,
  raw: string,
  model?: string,
  signal?: AbortSignal,
): Promise<string> {
  const resp = await provider.complete(
    {
      model: model ?? '',
      messages: [
        { role: 'system', content: DISTILL_SYSTEM_PROMPT },
        { role: 'user', content: raw },
      ],
      temperature: DEFAULT_TEMPERATURE,
      maxTokens: DEFAULT_MAX_TOKENS,
    },
    signal,
  )
  return (resp.content ?? '').trim()
}
