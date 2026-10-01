import { defineConfig } from 'vitest/config'

export default defineConfig({
  test: {
    projects: [
      './sdks/ts/memory/vitest.config.ts',
      './sdks/ts/memory-postgres/vitest.config.ts',
      './sdks/ts/memory-openfga/vitest.config.ts',
      './sdks/ts/memory-pi/vitest.config.ts',
      './mcp/ts/vitest.config.ts',
      './install/vitest.config.ts',
    ],
  },
})
