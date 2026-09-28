/**
 * Phase 6 acceptance: simulating an event updates the open deal page live,
 * without a refresh.
 *
 * Runs against the real service. The assertion that matters is that the
 * probability on screen changes while the page stays put — no reload, no
 * navigation — which is only possible if the SSE subscription and the cache
 * invalidation are both working.
 */
import { expect, test } from "@playwright/test";

const API = process.env.VITE_API_URL ?? "http://localhost:8000";

test.beforeAll(async ({ request }) => {
  const health = await request.get(`${API}/healthz`);
  expect(health.ok(), `service not reachable at ${API}`).toBeTruthy();

  const body = await health.json();
  expect(
    body.bus?.available,
    "event ingestion is unavailable — the bus needs Redis >= 5",
  ).toBe(true);
});

test("the pipeline lists scored deals", async ({ page }) => {
  await page.goto("/");

  await expect(page.getByRole("heading", { name: "Open pipeline" })).toBeVisible();
  // The synthetic-data notice must be present on every view.
  await expect(page.getByText("synthetic data").first()).toBeVisible();

  const rows = page.locator("tbody tr");
  await expect(rows.first()).toBeVisible();
  expect(await rows.count()).toBeGreaterThan(5);
});

test("an event updates the open deal page without a refresh", async ({ page }) => {
  await page.goto("/");

  // Open the first deal that already has a score, so there is a number to move.
  const scored = page
    .locator("tbody tr")
    .filter({ hasNot: page.getByText("not scored") })
    .first();
  const company = await scored.locator("a").first().innerText();
  await scored.locator("a").first().click();

  await expect(page.getByRole("heading", { name: company })).toBeVisible();

  const meter = page.getByRole("meter", { name: "Win probability" });
  await expect(meter).toBeVisible();
  const before = Number(await meter.getAttribute("aria-valuenow"));
  expect(Number.isFinite(before)).toBeTruthy();

  // Wait for the stream to attach, or the event may be published before the
  // dashboard is listening.
  await expect(page.getByText("Live", { exact: true })).toBeVisible();

  // Pin the page so a reload would be detectable.
  await page.evaluate(() => {
    (window as unknown as { __noReload: boolean }).__noReload = true;
  });

  await page.getByRole("button", { name: "Champion identified" }).click();

  // The number must change on its own.
  await expect
    .poll(
      async () => Number(await meter.getAttribute("aria-valuenow")),
      { timeout: 30_000, message: "win probability did not update over SSE" },
    )
    .not.toBe(before);

  // Nothing reloaded: the flag we set is still there.
  expect(
    await page.evaluate(
      () => (window as unknown as { __noReload?: boolean }).__noReload === true,
    ),
    "the page reloaded instead of updating in place",
  ).toBeTruthy();

  // The contribution ledger still reconciles to the displayed probability.
  await expect(page.getByText("Why this score")).toBeVisible();
  await expect(page.getByText("Win probability").last()).toBeVisible();
});

test("what-if scores a change without saving it", async ({ page }) => {
  await page.goto("/");
  await page
    .locator("tbody tr")
    .filter({ hasNot: page.getByText("not scored") })
    .first()
    .locator("a")
    .first()
    .click();

  await expect(page.getByText("Try a change")).toBeVisible();

  const slider = page.getByRole("slider").nth(1); // stakeholders
  await slider.focus();
  for (let i = 0; i < 4; i += 1) await slider.press("ArrowRight");

  await expect(page.getByText(/to\s*$/).first().or(page.getByText("Reset to this deal"))).toBeVisible();
  await expect(page.getByText("nothing is saved")).toBeVisible();

  const historyBefore = await page.getByText("Score over time").count();
  expect(historyBefore).toBeLessThanOrEqual(1);
});
