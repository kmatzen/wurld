/**
 * Opening a local file without reading all of it.
 *
 * A batch-written wurld file puts the whole WURLD document — calibration and
 * every pose — in front of the first Cluster, and its Cues at the end. So a
 * viewer can show everything except pixels after reading a few tens of KB, then
 * decode clusters as the user reaches them. The remote path always did this;
 * the local path used to read and decode the entire file up front, which on a
 * 2.6 MB / 300-frame capture meant fifteen seconds before the page was usable.
 *
 * The risk in going lazy is not speed but correctness: a frame assembled from a
 * spliced header plus one cluster must be the same frame the whole-file decode
 * produces, especially at cluster boundaries. That is what most of this file
 * checks.
 *
 * fixtures/multicluster.wurld.webm is 90 frames across 3 clusters — enough that
 * the lazy path engages and boundaries exist. The conformance vectors are all
 * single-cluster and deliberately tiny, so they cannot exercise this.
 */
import { test, expect } from '@playwright/test';

const VIEWER = '/viewer/index.html';
const FIXTURE = '/test/viewer/fixtures/multicluster.wurld.webm';
const FRAMES = 90;

async function drop(page, url) {
  await page.evaluate(async (u) => {
    const bytes = new Uint8Array(await (await fetch(u)).arrayBuffer());
    const dt = new DataTransfer();
    dt.items.add(new File([bytes], u.split('/').pop(), { type: 'video/webm' }));
    window.dispatchEvent(new DragEvent('drop', {
      dataTransfer: dt, bubbles: true, cancelable: true,
    }));
  }, url);
}

const usable = (page) =>
  expect.poll(async () => page.evaluate(() => document.getElementById('frameinfo').textContent),
              { timeout: 20_000 }).toContain(`1/${FRAMES}`);

/** Move to a frame and wait until it is actually on screen. */
async function show(page, i) {
  await page.evaluate((n) => {
    const s = document.getElementById('scrub');
    s.value = n; s.dispatchEvent(new Event('input', { bubbles: true }));
  }, i);
  await expect
    .poll(async () => page.evaluate(() => document.getElementById('frameinfo').textContent),
          { timeout: 30_000 })
    .toMatch(new RegExp(`frame ${i + 1}/${FRAMES}(?!.*fetching)`));
}

/** A cheap digest of what the panes are displaying. */
const panes = (page) => page.evaluate(() => {
  const hash = (id) => {
    const c = document.getElementById(id);
    if (!c.width) return 'empty';
    const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
    let h = 2166136261;
    for (let k = 0; k < d.length; k += 31) { h ^= d[k]; h = Math.imul(h, 16777619); }
    return (h >>> 0).toString(16);
  };
  return { rgb: hash('rgbPane'), depth: hash('depthPane') };
});

test('a multi-cluster file opens lazily and says so', async ({ page }) => {
  await page.goto(VIEWER);
  await drop(page, FIXTURE);
  await usable(page);

  const meta = await page.locator('#meta').textContent();
  expect(meta, 'the metadata line reports the lazy read')
    .toMatch(/clusters decode on demand|range reads/);
  // The whole timeline is navigable immediately, before the pixels exist.
  expect(await page.evaluate(() => Number(document.getElementById('scrub').max)))
    .toBe(FRAMES - 1);
});

test('lazy and whole-file decoding agree, including across cluster boundaries', async ({ page }) => {
  // The one that matters. ?lazy=0 forces the old path; the two must not differ.
  const sample = [0, 1, 29, 30, 31, 45, 59, 60, 89];

  const digestsFor = async (url) => {
    await page.goto(url);
    await drop(page, FIXTURE);
    await usable(page);
    const out = [];
    for (const i of sample) {
      await show(page, i);
      out.push({ i, ...(await panes(page)) });
    }
    return out;
  };

  const lazy = await digestsFor(VIEWER);
  const eager = await digestsFor(`${VIEWER}?lazy=0`);

  for (let n = 0; n < sample.length; n++) {
    expect(lazy[n], `frame ${sample[n]} must decode identically either way`)
      .toEqual(eager[n]);
  }
  // Guard against the comparison being vacuous: the frames must differ from
  // each other, or two identical blanks would "agree" perfectly.
  expect(new Set(lazy.map((d) => d.rgb)).size,
         'sampled frames should not all be the same image').toBeGreaterThan(1);
});

test('poses come from the header, so they are all present immediately', async ({ page }) => {
  // The point of the layout: the document in front of the Clusters carries every
  // pose, so the exports are complete before a single pixel has been decoded.
  await page.goto(VIEWER);
  await drop(page, FIXTURE);
  await usable(page);

  const [download] = await Promise.all([
    page.waitForEvent('download', { timeout: 20_000 }),
    page.locator('#expPoses').click(),
  ]);
  const chunks = [];
  for await (const c of await download.createReadStream()) chunks.push(c);
  const lines = Buffer.concat(chunks).toString('utf8').trim().split('\n');
  expect(lines.length - 1, 'every pose, with almost no bytes read').toBe(FRAMES);
});

test('a file with no Cues still opens, by reading all of it', async ({ page }) => {
  // Streaming-written files have no Cues by construction (SPEC §9 forbids them
  // there, since interleaving tags moves the Clusters they point at). Those
  // cannot be indexed, so the whole-file path has to remain correct.
  await page.goto(VIEWER);
  await drop(page, '/conformance/vectors/v02_depth.wurld.webm');
  await expect
    .poll(async () => page.evaluate(() => Number(document.getElementById('scrub').max)),
          { timeout: 20_000 })
    .toBe(3);
  const p = await panes(page);
  expect(p.rgb).not.toBe('empty');
  expect(p.depth).not.toBe('empty');
});
