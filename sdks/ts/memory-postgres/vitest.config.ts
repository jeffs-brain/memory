import { defineConfig } from 'vitest/config'

export default defineConfig({
  test: {
    name: '@jeffs-brain/memory-postgres',
    include: ['src/**/*.test.ts'],
  },
})
