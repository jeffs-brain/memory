import { defineConfig } from 'vitest/config'

export default defineConfig({
  test: {
    name: '@jeffs-brain/memory-pi',
    include: ['test/**/*.test.ts'],
  },
})
