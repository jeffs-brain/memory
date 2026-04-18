/**
 * LME judge. The prompts are copied verbatim from the official
 * LongMemEval codebase (xiaowu0162/LongMemEval) so verdict semantics stay
 * compatible with the Go harness and the paper. The judge itself is
 * injectable; the harness ships a provider-backed default that wraps the
 * supplied Provider with the canonical category-aware prompt.
 */

import type { Provider } from '../llm/index.js'
import type { JudgeFn, JudgeVerdict, LMEExample } from './types.js'

export const JUDGE_PROMPT_VERSION = 6

// --- Prompts (verbatim from the Go judge_prompt.go) -----------------------

export const JUDGE_PROMPT_STANDARD = `I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no.

Question: %s

Correct Answer: %s

Model Response: %s

Is the model response correct? Answer yes or no only.`

export const JUDGE_PROMPT_TEMPORAL = `I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct.

Question: %s

Correct Answer: %s

Model Response: %s

Is the model response correct? Answer yes or no only.`

export const JUDGE_PROMPT_KNOWLEDGE_UPDATE = `I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer.

Question: %s

Correct Answer: %s

Model Response: %s

Is the model response correct? Answer yes or no only.`

export const JUDGE_PROMPT_PREFERENCE = `I will give you a question, a rubric for desired personalized response, and a response from a model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user's personal information correctly.

Question: %s

Rubric: %s

Model Response: %s

Is the model response correct? Answer yes or no only.`

export const JUDGE_PROMPT_ABSTENTION = `I will give you an unanswerable question, an explanation, and a response from a model. Please answer yes if the model correctly identifies the question as unanswerable. The model could say that the information is incomplete, or some other information is given but the asked information is not.

Question: %s

Explanation: %s

Model Response: %s

Does the model correctly identify the question as unanswerable? Answer yes or no only.`

/** Abstention detection: mirrors evaluate_qa.py (id contains "_abs"). */
export const isAbstention = (id: string): boolean => id.includes('_abs')

export const judgePromptForCategory = (
  category: string,
  abstention: boolean,
): string => {
  if (abstention) return JUDGE_PROMPT_ABSTENTION
  switch (category) {
    case 'temporal-reasoning':
      return JUDGE_PROMPT_TEMPORAL
    case 'knowledge-update':
      return JUDGE_PROMPT_KNOWLEDGE_UPDATE
    case 'single-session-preference':
      return JUDGE_PROMPT_PREFERENCE
    default:
      return JUDGE_PROMPT_STANDARD
  }
}

/** Render a judge prompt with the canonical positional substitutions. */
export const formatJudgePrompt = (args: {
  readonly category: string
  readonly abstention: boolean
  readonly question: string
  readonly groundTruth: string
  readonly response: string
  readonly questionDate?: string
}): string => {
  const template = judgePromptForCategory(args.category, args.abstention)
  const body = sprintf3(template, args.question, args.groundTruth, args.response)
  if (args.questionDate === undefined || args.questionDate === '') return body
  return `Question date: ${args.questionDate}\n\n${body}`
}

// Minimal `%s`-only formatter that preserves the Go behaviour without
// dragging in a full printf implementation.
const sprintf3 = (tmpl: string, a: string, b: string, c: string): string => {
  const parts = tmpl.split('%s')
  if (parts.length !== 4) {
    throw new Error('judge template expected exactly three %s placeholders')
  }
  return parts[0] + a + parts[1] + b + parts[2] + c + parts[3]
}

/** Parse the LME yes/no judge response. Official rule: substring match. */
export const parseYesNo = (
  raw: string,
  abstention: boolean,
): { verdict: JudgeVerdict; rationale: string } => {
  const lower = raw.trim().toLowerCase()
  if (lower === '') {
    return { verdict: 'error', rationale: 'empty judge response' }
  }
  if (lower.includes('yes')) {
    return {
      verdict: abstention ? 'abstain_correct' : 'correct',
      rationale: raw.trim(),
    }
  }
  if (lower.includes('no')) {
    return {
      verdict: abstention ? 'abstain_incorrect' : 'incorrect',
      rationale: raw.trim(),
    }
  }
  return { verdict: 'error', rationale: `unparseable verdict: ${raw.trim()}` }
}

export type ProviderJudgeOpts = {
  readonly provider: Provider
  readonly model?: string
  readonly maxTokens?: number
  readonly budgetChars?: number
}

/**
 * Default provider-backed judge. Sends the official category-aware
 * prompt to the supplied Provider and maps the yes/no response onto the
 * canonical verdict set.
 */
export const createProviderJudge = (opts: ProviderJudgeOpts): JudgeFn => {
  const budget = opts.budgetChars ?? 40_000
  return async ({ example, predicted }) => {
    const abstention = isAbstention(example.id)
    const prompt = formatJudgePrompt({
      category: example.category,
      abstention,
      question: example.question,
      groundTruth: example.answer,
      response: predicted.length > budget ? predicted.slice(0, budget) : predicted,
      ...(example.questionDate !== undefined ? { questionDate: example.questionDate } : {}),
    })
    const resp = await opts.provider.complete({
      ...(opts.model !== undefined ? { model: opts.model } : {}),
      messages: [{ role: 'user', content: prompt }],
      maxTokens: opts.maxTokens ?? 256,
      temperature: 0,
    })
    const parsed = parseYesNo(resp.content, abstention)
    return {
      verdict: parsed.verdict,
      rationale: parsed.rationale,
      rawResponse: resp.content,
    }
  }
}

/**
 * Convenience judge that always returns a single verdict. Useful as a
 * sentinel in tests, but rarely in production.
 */
export const createStaticJudge = (verdict: JudgeVerdict, rationale = 'static'): JudgeFn => {
  return async () => ({ verdict, rationale })
}

/** Exact-match fallback used when the judge call is absent or fails. */
export const exactMatchVerdict = (example: LMEExample, predicted: string): JudgeVerdict => {
  const a = example.answer.trim().toLowerCase()
  const b = predicted.trim().toLowerCase()
  if (a === '' || b === '') return 'incorrect'
  const abstention = isAbstention(example.id)
  if (b.includes(a)) return abstention ? 'abstain_correct' : 'correct'
  return abstention ? 'abstain_incorrect' : 'incorrect'
}
