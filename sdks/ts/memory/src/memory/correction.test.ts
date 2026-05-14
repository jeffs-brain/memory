// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from 'vitest'
import {
  buildCorrectionReminder,
  buildCorrectionReminderWithOptions,
  detectCorrection,
} from './correction.js'

describe('correction detection', () => {
  it('detects correction-shaped user turns', () => {
    const cases = [
      "No, it's actually 7am, not 8am.",
      "That's wrong, the meeting is on Tuesday.",
      'You got that wrong, the project is called Roo not Roe.',
      "Actually, it's blue not red.",
      'Stop saying I live in London, I live in Amersfoort.',
      "Wrong, it's Anthropic not OpenAI.",
      'I never said I wanted Slack messages, I wanted Telegram.',
      "That's not what I meant, please update memory.",
      'Please forget that I work at Google.',
      "Correction: my wife's name is Nadia, not Nadya.",
      "You're wrong about my role.",
      "Actually it's a Pi 4B, not a Pi 5.",
      "I didn't say that. I never use docker compose for this.",
      'It is actually 4 children, not 3.',
    ]

    for (const input of cases) {
      const correction = detectCorrection(input)
      expect(correction, input).toBeDefined()
      expect(correction?.snippet).not.toBe('')
    }
  })

  it('ignores common false positives', () => {
    const cases = [
      "No problem, I'll handle it.",
      'No idea what that means.',
      'No worries, take your time.',
      'You got the wrong end of the stick earlier.',
      "There's nothing wrong with the build.",
      "Not wrong per se, just not what I'd pick.",
      'No, before that I asked you to set up Slack.',
      "I think you're right about the route.",
      "Yes, that's right.",
      'Can you fix the bug in app.go?',
      'What is the status of the deploy?',
      'I see no problem with that approach.',
      'No, but seriously can you try again?',
    ]

    for (const input of cases) {
      expect(detectCorrection(input), input).toBeUndefined()
    }
  })

  it('builds a default memory reminder', () => {
    const reminder = buildCorrectionReminder('Actually it is 7am not 8am.')
    for (const expected of [
      'memory_search',
      'memory_update',
      'memory_remove',
      'memory_create',
      'Mention',
    ]) {
      expect(reminder).toContain(expected)
    }
  })

  it('builds a reminder with custom tool names', () => {
    const reminder = buildCorrectionReminderWithOptions('Wrong, it is Tuesday.', {
      searchTool: 'lookup',
      updateTool: 'patch',
      removeTool: 'retire',
      createTool: 'create',
    })
    for (const expected of ['lookup', 'patch', 'retire', 'create']) {
      expect(reminder).toContain(expected)
    }
    expect(reminder).not.toContain('Mention')
  })
})
