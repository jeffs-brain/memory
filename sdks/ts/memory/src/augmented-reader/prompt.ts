// SPDX-License-Identifier: Apache-2.0

import { readerTodayAnchor } from '../query/index.js'

export const READER_AUGMENTED_MAX_TOKENS = 800
export const READER_AUGMENTED_TEMPERATURE = 0.0

export const READER_USER_TEMPLATE = `I will give you several history chats between you and a user. Please answer the question based on the relevant chat history. Answer the question step by step: first extract all the relevant information, and then reason over the information to get the answer.

Resolving conflicting information:
- Each fact is tagged with a date. When the same topic appears with different values on different dates, prefer the value from the most recent session date.
- Treat explicit supersession phrases as hard overrides regardless of how often the old value appears: "now", "currently", "most recently", "actually", "correction", "I updated", "I changed", "no longer".
- Do not vote by frequency. One later correction outweighs any number of earlier mentions.
- Never use a fact dated after the current date.
- When the question names a specific item, event, place, or descriptor, prefer the fact that matches that target most directly. Do not substitute a broader category match or a different example from the same topic.
- A direct statement of the full usual value outranks a newer note about only one segment, leg, or example from that routine unless the newer note explicitly says the full value changed.
- For habit and routine questions ("usually", "normally", "every week", "on Saturdays", "on weekdays"), prefer explicit habitual statements over isolated single-day examples.
- Do not let an example note about a narrower segment override the whole routine. For example, a "30-minute morning commute" note does not replace a direct statement of a "45-minute daily commute to work".
- When one fact names the event and another fact gives the associated submission, booking, or join date for that same event or venue, combine them if the connection is explicit in the retrieved facts.

Enumeration and counting:
- When the question asks to list, count, enumerate, or total ("how many", "list", "which", "what are all", "total", "in total"), return every matching item you find across the retrieved facts, one per line, each tagged with its session date. Then state the count or total explicitly at the end.
- Do not summarise into a single sentence when the question demands a list.
- Add numeric values across sessions when the question asks for a total (hours, days, money, items). Show the arithmetic.
- When both atomic event facts and retrospective roll-up summaries are present, prefer the atomic event facts and avoid double counting the roll-up.
- Treat first-person past-tense purchases, gifts, sales, earnings, completions, or submissions as confirmed historical events even when they appear inside a planning or advice conversation. Exclude only clearly hypothetical or planned amounts.
- If a spending or earnings question does not explicitly restrict the timeframe ("today", "this time", "most recent", "current"), include all confirmed historical amounts for the same subject across sessions.
- For totals over named items, sum only the facts that match those named items directly. Do not add alternative purchases, adjacent examples, or broader category summaries unless the note clearly says they refer to the same item.
- When a total names multiple specific items, people, or occasions, every named part must be supported directly. If any named part is missing or lacks an amount, do not return a partial total. State that the information provided is not enough.
- When the question names a singular item plus another category, choose the single best-matching fact for that singular item. Do not combine multiple different handbags, flights, meals, or other same-category purchases unless the question explicitly asks for all of them.
- When multiple notes appear to describe the same purchase, gift, booking, or transaction, count it once. Prefer the most direct transactional fact over recap notes, budget summaries, tracker entries, or assistant bookkeeping.
- For "spent", "cost", and "total amount" questions, prefer direct transactional facts over plans, budgets, broad summaries, or calculations that only restate the same purchase.

Preference-sensitive questions:
- When the user asks for ideas, advice, inspiration, or recommendations, anchor the answer in explicit prior preferences, recent projects, recurring habits, and stated dislikes from the retrieved facts.
- Avoid generic suggestions when the history already contains concrete tastes or recent examples. Reuse those specifics directly in the answer.
- Infer durable preferences from concrete desired features or liked attributes even when the earlier example was tied to a different city, venue, or product.
- When concrete amenities or features are present, prefer them over generic travel style or budget signals.
- Ignore unrelated hostel, budget, or solo-travel examples when the retrieved facts already contain a clearer accommodation-feature preference and the question does not ask about price.
- When the question asks for a specific or exact previously recommended item, answer with the narrowest directly supported set from the retrieved facts. Do not widen the answer with adjacent frameworks, resource catalogues, or loosely related examples.

Unanswerable questions:
- If the retrieved facts do not directly answer the question, state that clearly in the first sentence.
- Keep the extraction step brief and limited to the missing subject. Do not narrate your search process.
- Do not pad the answer with near-miss facts about a different city, person, product, or date unless they directly explain why the requested fact is unavailable.
- End with a direct abstention that the information provided is not enough to answer the question.

Temporal reasoning:
- Today is %TODAY% (this is the current date). Resolve relative references ("recently", "last week", "a few days ago", "this month") against this anchor.
- For date-arithmetic questions ("how many days between X and Y"), first extract each event's ISO date from the fact tags, then compute the difference in days.

History Chats:

%CONTEXT%

Current Date: %DATE%
Question: %QUESTION%
Answer (step by step):`

export const buildAugmentedReaderPrompt = (
  question: string,
  renderedEvidence: string,
  questionDate: string | undefined,
): string => {
  const today = readerTodayAnchor(questionDate)
  const date = questionDate !== undefined && questionDate !== '' ? questionDate : 'unknown'
  return READER_USER_TEMPLATE.replace('%CONTEXT%', renderedEvidence)
    .replace('%TODAY%', today)
    .replace('%DATE%', date)
    .replace('%QUESTION%', question)
}
