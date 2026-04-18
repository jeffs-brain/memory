// SPDX-License-Identifier: Apache-2.0

import { defineConfig } from 'vitest/config'

export default defineConfig({
  test: {
    name: '@jeffs-brain/install',
    include: ['test/**/*.test.ts'],
  },
})
