/**
 * Browser tests for viewer/live.html — the live record/stream/play round trip.
 *
 * The page is a working demonstration that a wurld file can be *streamed*: a
 * WurldRecorder encodes frames and emits chunks, those chunks go straight into a
 * WurldLivePlayer that has seen nothing else, and the player rebuilds geometry
 * from the bytes alone. When recording stops the accumulated chunks must also be
 * a valid finished file.
 *
 * That round trip is covered on the Python side, but in the browser it existed
 * only as something a human clicked. These tests drive the same buttons and read
 * the same two panels a person would.
 *
 * What makes it worth testing rather than just rendering: it exercises the SPEC
 * §9 live layout end to end — poses emitted ahead of the Clusters that carry
 * them, and a consolidated table on finish — across three independent readers
 * (the streaming player, the pose table, and a buffered decode of the result).
 * A regression in any of them shows up here as a number that stops moving.
 *
 * Checked by breaking the code each panel depends on and confirming the right
 * test fails:
 *   cutting `player.feed(c)` in live.html      -> `recording streams chunks…`
 *   suppressing WURLD_FRAMES in WurldRecorder  -> `stopping finalises a file…`
 */
import { test, expect } from '@playwright/test';

const LIVE = '/viewer/live.html';

/** The stats panel is `label   value` lines; pull the numbers out. */
function parsePanel(text) {
  const out = {};
  for (const line of text.split('\n')) {
    const m = line.match(/^(.+?)\s{2,}(.+)$/);
    if (m) out[m[1].trim()] = m[2].trim();
  }
  return out;
}

const num = (s) => Number(String(s ?? '').replace(/[^0-9.]/g, '')) || 0;

async function stats(page) {
  return parsePanel(await page.locator('#stats').textContent());
}

test.beforeEach(async ({ page }) => {
  const errors = [];
  page.on('pageerror', (e) => errors.push(String(e)));
  page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()); });
  page.__errors = errors;
  await page.goto(LIVE);
});

test('a page whose codec failed to load says so instead of going quiet', async ({ page }) => {
  await page.route('**/chromapakz.js', (r) => r.abort());
  await page.goto(LIVE);
  await expect(page.locator('#stats'))
    .toContainText('failed to load its codec', { timeout: 10_000 });
});

test('idle until started', async ({ page }) => {
  await expect(page.locator('#start')).toBeEnabled();
  await expect(page.locator('#stop')).toBeDisabled();
  expect(await page.locator('#stats').textContent()).toContain('idle');
});

test('recording streams chunks a player rebuilds from bytes alone', async ({ page }) => {
  await page.locator('#start').click();
  await expect(page.locator('#stop')).toBeEnabled();
  await expect(page.locator('#start')).toBeDisabled();

  // Wait for the *player* to have decoded frames, not merely for the recorder to
  // have produced them: the point of the page is that bytes crossing the seam
  // are enough on their own.
  await expect
    .poll(async () => num((await stats(page))['player frames']), { timeout: 45_000 })
    .toBeGreaterThan(0);

  const s = await stats(page);
  expect(num(s['recorded frames']), 'recorder ran').toBeGreaterThan(0);
  expect(num(s['stream chunks']), 'chunks were emitted').toBeGreaterThan(0);
  expect(num(s['stream bytes']), 'and carried payload').toBeGreaterThan(0);
  // Poses ride as WURLD_POSES chunks ahead of their Clusters (SPEC §9 live
  // form). If they only appeared in the final table this would stay at zero.
  expect(num(s['player poses']), 'poses arrived mid-stream').toBeGreaterThan(0);

  // The player cannot be ahead of the recorder; it may legitimately lag.
  expect(num(s['player frames'])).toBeLessThanOrEqual(num(s['recorded frames']));

  await page.locator('#stop').click();
});

test('stopping finalises a file that decodes and carries its poses', async ({ page }) => {
  await page.locator('#start').click();

  // Enough frames that the file spans more than a single cluster.
  await expect
    .poll(async () => num((await stats(page))['recorded frames']), { timeout: 45_000 })
    .toBeGreaterThanOrEqual(12);

  await page.locator('#stop').click();

  // finish() runs after the loop unwinds, so wait for the verdict to fill in.
  await expect
    .poll(async () => (await page.locator('#verdict').textContent()).includes('buffered decode'),
          { timeout: 45_000 })
    .toBe(true);

  const v = parsePanel(await page.locator('#verdict').textContent());
  expect(v['buffered decode'], 'the finalised bytes decode as an ordinary file')
    .toContain('✓');
  expect(num(v['buffered decode']), 'and yield frames').toBeGreaterThan(0);
  expect(num(v['poses in table']), 'the consolidated WURLD_FRAMES table is present')
    .toBeGreaterThan(0);
  expect(num(v['file size'])).toBeGreaterThan(0);

  // The table describes the recording that actually happened. Read the recorder
  // count *after* stopping: the loop keeps going until the click lands, so a
  // count sampled beforehand is only a lower bound and the table rightly exceeds it.
  const recorded = num((await stats(page))['recorded frames']);
  expect(num(v['poses in table'])).toBeLessThanOrEqual(recorded);
  expect(num(v['poses in table'])).toBeGreaterThanOrEqual(num(v['buffered decode']));

  expect(page.__errors, 'no page errors during a full round trip').toEqual([]);
});

test('the finalised file is offered as a download', async ({ page }) => {
  await page.locator('#start').click();
  await expect
    .poll(async () => num((await stats(page))['recorded frames']), { timeout: 45_000 })
    .toBeGreaterThanOrEqual(4);
  await page.locator('#stop').click();

  const link = page.locator('#verdict a[download]');
  await expect(link).toBeVisible({ timeout: 45_000 });
  // The suffix the format now recommends; it is what a user ends up with on disk.
  await expect(link).toHaveAttribute('download', /\.wurld\.webm$/);
});

test('stopping returns the controls to their idle state', async ({ page }) => {
  await page.locator('#start').click();
  await expect(page.locator('#stop')).toBeEnabled();
  await page.locator('#stop').click();
  await expect(page.locator('#start')).toBeEnabled();
  await expect(page.locator('#stop')).toBeDisabled();
});
