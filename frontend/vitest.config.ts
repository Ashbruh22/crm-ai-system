import { defineConfig } from "vitest/config";

/**
 * Unit tests only. The Playwright specs under e2e/ use the same `.spec.ts`
 * suffix and would otherwise be collected here and fail — they need a browser
 * and a running service, which is `npm run e2e`.
 */
export default defineConfig({
  test: {
    include: ["src/**/*.{test,spec}.{ts,tsx}"],
    exclude: ["e2e/**", "node_modules/**", "dist/**"],
    environment: "node",
  },
});
