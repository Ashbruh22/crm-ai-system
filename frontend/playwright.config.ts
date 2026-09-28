import { defineConfig, devices } from "@playwright/test";

/**
 * Smoke test config (spec section 11).
 *
 * Requires the ML service already running and reachable at VITE_API_URL — the
 * point of the test is that the real stack updates live, so mocking the API
 * would test nothing. The dev server is started here; the backend is not,
 * because it needs Redis >= 5 and a migrated database.
 */
export default defineConfig({
  testDir: "./e2e",
  timeout: 90_000,
  expect: { timeout: 30_000 },
  fullyParallel: false,
  retries: 0,
  reporter: [["list"]],
  use: {
    baseURL: "http://localhost:5173",
    ...devices["Desktop Chrome"],
    trace: "retain-on-failure",
  },
  webServer: {
    command: "npm run dev",
    url: "http://localhost:5173",
    reuseExistingServer: true,
    timeout: 60_000,
  },
});
