/**
 * Browser tests for viewer/index.html.
 *
 * Why a real browser: the viewer decodes through chromapakz, which needs
 * WebCodecs, and it draws into 2D canvases and WebGL. jsdom has none of those,
 * and the page is a single module script with no exported seam, so the only
 * honest way to test it is to run it and look at what a user would see.
 *
 * Everything here asserts on **observable** state — the frame counter, the
 * scrubber, the metadata line, and the actual pixels in the panes — rather than
 * on internals.
 *
 * Each test was checked against the code it guards, by reverting the fix and
 * confirming it fails — a test that has never failed proves nothing. Against the
 * pre-fix viewer, `depth pane clears` and `cannot be decoded is reported` fail.
 * Against chromapakz 0.9.0, `an RGB-only capture loads at all` fails. The
 * remaining cases are characterisation, not regression: they pin behaviour that
 * was already correct, and are marked where that is so.
 *
 * Fixtures are the committed conformance vectors, so this adds no binaries and
 * exercises the files the format already promises to read:
 *   v02_depth        4 frames, depth signal
 *   v01_minimal      4 frames, RGB only (signals: []) — no depth
 *   v10_single_frame 1 frame,  RGB only
 *   v05_stereo       4 frames, two display streams
 */
import { test, expect } from '@playwright/test';

const VIEWER = '/viewer/index.html';
const V = (name) => `/conformance/vectors/${name}.wurld.webm`;

/** Drop a file into the page exactly as a user would, and wait for it to settle. */
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

/** What the user can actually see. */
async function view(page) {
  return page.evaluate(() => {
    const opaque = (id) => {
      const c = document.getElementById(id);
      if (!c.width || !c.height) return 0;
      const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
      let n = 0;
      for (let i = 3; i < d.length; i += 4) if (d[i] !== 0) n++;
      return n;
    };
    return {
      frameinfo: document.getElementById('frameinfo').textContent,
      scrubMax: Number(document.getElementById('scrub').max),
      rgbPixels: opaque('rgbPane'),
      depthPixels: opaque('depthPane'),
      meta: document.getElementById('meta').textContent,
      depthExportEnabled: !document.getElementById('expDepth').disabled,
      streamPickerVisible: !document.getElementById('streamPick').hidden,
    };
  });
}

/** Poll until the page reports the expected frame count, so tests never race the decoder. */
async function waitForFrames(page, n) {
  await expect
    .poll(async () => (await view(page)).scrubMax, { timeout: 15_000 })
    .toBe(n - 1);
}

test.beforeEach(async ({ page }) => {
  const errors = [];
  page.on('pageerror', (e) => errors.push(String(e)));
  page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()); });
  page.__errors = errors;
  await page.goto(VIEWER);
});

test('a depth capture loads: pixels, counter, and the depth export', async ({ page }) => {
  await drop(page, V('v02_depth'));
  await waitForFrames(page, 4);

  const v = await view(page);
  expect(v.frameinfo).toContain('1/4');
  expect(v.rgbPixels).toBeGreaterThan(0);
  expect(v.depthPixels).toBeGreaterThan(0);
  expect(v.depthExportEnabled).toBe(true);
  expect(page.__errors, 'no page errors').toEqual([]);
});

test('an RGB-only capture loads at all', async ({ page }) => {
  // Regression: chromapakz's normalizeMetadata rejected an empty signals[], so
  // createDecoder threw and the page silently rendered nothing. Needs
  // chromapakz >= 0.9.1. This is the failure the user reported as "the viewer
  // ignores my file".
  await drop(page, V('v01_minimal'));
  await waitForFrames(page, 4);

  const v = await view(page);
  expect(v.rgbPixels, 'RGB-only file must still draw its pixels').toBeGreaterThan(0);
  expect(v.depthPixels, 'it has no depth, so the depth pane stays empty').toBe(0);
  expect(v.depthExportEnabled).toBe(false);
  expect(page.__errors, 'no page errors').toEqual([]);
});

test('opening a file with no depth clears the previous file\'s depth pane', async ({ page }) => {
  // The reported bug. drawPanes only ever *wrote* to a pane, so a pane with
  // nothing to show kept whatever was painted into it last.
  await drop(page, V('v02_depth'));
  await waitForFrames(page, 4);
  expect((await view(page)).depthPixels).toBeGreaterThan(0);

  await drop(page, V('v01_minimal'));
  await waitForFrames(page, 4);

  const v = await view(page);
  expect(v.depthPixels, 'the old depth map must not survive the swap').toBe(0);
  expect(v.depthExportEnabled, 'and the export it fed must go with it').toBe(false);
  expect(v.rgbPixels).toBeGreaterThan(0);
});

test('opening a shorter file resets the counter and the scrubber', async ({ page }) => {
  // Characterisation, not regression: `state.cur` did survive a load, but
  // ingestFrame calls setFrame(0) on the first decoded frame, so the stale value
  // never reached the screen on this path. Pinned because the reset is now
  // explicit in forgetPreviousFile() and should stay that way — the value does
  // reach the screen when a load fails, which the next test covers.
  await drop(page, V('v02_depth'));
  await waitForFrames(page, 4);

  // Park at the last frame, which does not exist in the file we open next.
  await page.evaluate(() => {
    const s = document.getElementById('scrub');
    s.value = 3;
    s.dispatchEvent(new Event('input', { bubbles: true }));
  });
  expect((await view(page)).frameinfo).toContain('4/4');

  await drop(page, V('v10_single_frame'));
  await waitForFrames(page, 1);

  const v = await view(page);
  expect(v.frameinfo, 'must not read "frame 4/1"').toContain('1/1');
  expect(v.scrubMax).toBe(0);
});

test('a file that cannot be decoded is reported, not silently ignored', async ({ page }) => {
  await drop(page, V('v02_depth'));
  await waitForFrames(page, 4);

  // Not a WebM at all. Previously this threw out of the drop handler into an
  // unhandled rejection: no message, and every pane kept showing the old file,
  // so the drop looked like it had been ignored.
  await page.evaluate(() => {
    const dt = new DataTransfer();
    dt.items.add(new File([new Uint8Array(4096).fill(0x41)], 'rubbish.webm',
                          { type: 'video/webm' }));
    window.dispatchEvent(new DragEvent('drop', {
      dataTransfer: dt, bubbles: true, cancelable: true,
    }));
  });

  await expect.poll(async () => (await view(page)).meta, { timeout: 15_000 })
    .toContain('could not read');

  const v = await view(page);
  expect(v.meta).toContain('rubbish.webm');
  expect(v.rgbPixels, 'the previous file must not be left on screen').toBe(0);
  expect(v.depthPixels).toBe(0);
  expect(v.frameinfo, 'nor its frame counter').toBe('');
});

test('a stereo file offers a choice of stream', async ({ page }) => {
  // SPEC §4.4: stream ids are camera ids. Showing only the primary without
  // saying so hides half the file.
  await drop(page, V('v05_stereo'));
  await waitForFrames(page, 4);

  const v = await view(page);
  expect(v.streamPickerVisible).toBe(true);
  expect(v.rgbPixels).toBeGreaterThan(0);
});

test('a page whose decoder failed to load says so', async ({ page }) => {
  // Reported as "the file dialog is not working". A module script that fails to
  // load takes every handler with it: the page renders normally, the buttons are
  // all present, and clicking any of them does nothing — including the one that
  // opens the file picker. The only trace is a console error nobody sees.
  // Commonest causes are no `npm install` from a checkout, or the CDN blocked by
  // an extension or proxy on the hosted copy.
  await page.route('**/chromapakz.js', (r) => r.abort());
  await page.goto(VIEWER);

  const note = page.locator('#drop .small');
  await expect(note).toContainText('failed to load its decoder', { timeout: 10_000 });

  // And the symptom really is what was reported: the picker never opens.
  let chooserOpened = false;
  page.on('filechooser', () => { chooserOpened = true; });
  await page.locator('#pick').click();
  await page.waitForTimeout(500);
  expect(chooserOpened, 'a dead page cannot open the picker — hence the message').toBe(false);
});

test('a healthy page shows no failure notice', async ({ page }) => {
  // The guard must not cry wolf on a slow but working load.
  await page.goto(VIEWER);
  await page.waitForTimeout(2500);
  await expect(page.locator('#drop .small')).not.toContainText('failed to load');
});

test('the drop overlay goes away once a file is open', async ({ page }) => {
  expect(await page.locator('#drop').isVisible()).toBe(true);
  await drop(page, V('v02_depth'));
  await waitForFrames(page, 4);
  expect(await page.locator('#drop').isVisible()).toBe(false);
});
