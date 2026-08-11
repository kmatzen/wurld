/**
 * The viewer's export buttons: poses.csv, imu_<id>.csv, wurld.json, depth.npy.
 *
 * These are the paths where a defect is quietest. A wrong pose column or a
 * dropped row produces a file that opens fine in a spreadsheet and is simply
 * wrong, and an earlier sweep of the exporters turned up seven defects, none of
 * which had failed a test.
 *
 * Correctness is judged against the conformance corpus's own `.expected.json`,
 * not against what the viewer parsed. Comparing the export to the viewer's
 * in-memory state would pass even if both were wrong in the same way; the
 * expectation files are the format's declared truth, produced independently.
 *
 * Downloads are captured through Playwright's download event, so the assertions
 * are on the bytes a user actually ends up with.
 *
 * Each check was made to fail by breaking the exporter it guards:
 *   emitting unposed frames            -> `omits frames the producer could not localise`
 *   writing the quaternion as xyzw     -> `holds exactly the posed frames, with the declared values`
 *   skipping the depth dequantisation  -> `depth.npy is a real npy of metric depth`
 * The middle one is the realistic mistake: ROS orders quaternions xyzw and
 * wurld orders them wxyz, and a CSV with the columns swapped looks perfectly fine.
 */
import { test, expect } from '@playwright/test';

const VIEWER = '/viewer/index.html';
const V = (n) => `/conformance/vectors/${n}.wurld.webm`;
const EXPECTED = (n) => `/conformance/vectors/${n}.expected.json`;

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

/** Load a vector and wait for its frames to arrive. */
async function open(page, name, frames) {
  await drop(page, V(name));
  await expect
    .poll(async () => page.evaluate(() => Number(document.getElementById('scrub').max)),
          { timeout: 20_000 })
    .toBe(frames - 1);
}

/** Click an export and return what the browser downloaded, as text. */
async function grab(page, buttonId) {
  const [download] = await Promise.all([
    page.waitForEvent('download', { timeout: 20_000 }),
    page.locator(`#${buttonId}`).click(),
  ]);
  const stream = await download.createReadStream();
  const chunks = [];
  for await (const c of stream) chunks.push(c);
  return { name: download.suggestedFilename(), buf: Buffer.concat(chunks) };
}

/** The corpus's declared truth for a vector. */
async function expected(page, name) {
  return page.evaluate((u) => fetch(u).then((r) => r.json()), EXPECTED(name));
}

const rows = (csv) => csv.trim().split('\n');
const near = (a, b, tol = 1e-6) => Math.abs(a - b) <= tol;

test('poses.csv holds exactly the posed frames, with the declared values', async ({ page }) => {
  await page.goto(VIEWER);
  await open(page, 'v02_depth', 4);

  const { name, buf } = await grab(page, 'expPoses');
  expect(name).toBe('poses.csv');

  const lines = rows(buf.toString('utf8'));
  expect(lines[0]).toBe('i,t,qw,qx,qy,qz,tx,ty,tz,camera');

  const truth = (await expected(page, 'v02_depth')).frames
    .filter((f) => f.pose_valid !== false);
  expect(lines.length - 1, 'one row per posed frame').toBe(truth.length);

  lines.slice(1).forEach((line, k) => {
    const c = line.split(',');
    const want = truth[k];
    expect(Number(c[0])).toBe(want.i);
    expect(near(Number(c[1]), want.t)).toBe(true);
    [0, 1, 2, 3].forEach((j) =>
      expect(near(Number(c[2 + j]), want.q_wxyz[j]), `q[${j}] of frame ${want.i}`).toBe(true));
    [0, 1, 2].forEach((j) =>
      expect(near(Number(c[6 + j]), want.tr[j]), `tr[${j}] of frame ${want.i}`).toBe(true));
    expect(c[9]).toBe(want.camera ?? '0');
  });
});

test('poses.csv omits frames the producer could not localise', async ({ page }) => {
  // v04_unposed carries 6 frames of which 2 have pose_valid false. Emitting a
  // row for those — with zeros, or with the previous pose — would be a file that
  // silently claims the camera was somewhere it never was.
  await page.goto(VIEWER);
  await open(page, 'v04_unposed', 6);

  const truth = await expected(page, 'v04_unposed');
  const posed = truth.frames.filter((f) => f.pose_valid !== false);
  expect(posed.length, 'fixture really does contain unposed frames').toBeLessThan(truth.frames.length);

  const { buf } = await grab(page, 'expPoses');
  const lines = rows(buf.toString('utf8'));
  expect(lines.length - 1).toBe(posed.length);

  const emitted = lines.slice(1).map((l) => Number(l.split(',')[0]));
  expect(emitted).toEqual(posed.map((f) => f.i));
  for (const f of truth.frames.filter((x) => x.pose_valid === false)) {
    expect(emitted, `frame ${f.i} is unposed and must not appear`).not.toContain(f.i);
  }
});

test('poses.csv works when the poses live only in the binary table', async ({ page }) => {
  // The button exists precisely for this file: ffmpeg cannot see a binary pose
  // table at all, so an export that only understood the JSON array would hand
  // back a header and nothing else.
  await page.goto(VIEWER);
  await open(page, 'v03_binary_frames', 6);

  const truth = (await expected(page, 'v03_binary_frames')).frames
    .filter((f) => f.pose_valid !== false);
  const { buf } = await grab(page, 'expPoses');
  const lines = rows(buf.toString('utf8'));

  expect(lines.length - 1, 'binary table must still export').toBe(truth.length);
  expect(Number(lines[1].split(',')[0])).toBe(truth[0].i);
});

test('wurld.json is the document, and parses', async ({ page }) => {
  await page.goto(VIEWER);
  await open(page, 'v02_depth', 4);

  const { name, buf } = await grab(page, 'expMeta');
  expect(name).toBe('wurld.json');

  const doc = JSON.parse(buf.toString('utf8'));
  const truth = await expected(page, 'v02_depth');
  expect(Object.keys(doc.cameras).sort()).toEqual(Object.keys(truth.cameras).sort());
  expect(doc.world.metric_scale).toBe(truth.world.metric_scale);
  const sig = (doc.signals ?? []).find((s) => s.role === 'depth');
  expect(sig, 'the depth signal survives the round trip').toBeTruthy();
});

test('imu_<id>.csv carries every sample, and is hidden when there is none', async ({ page }) => {
  await page.goto(VIEWER);

  await open(page, 'v02_depth', 4);
  expect(await page.locator('#expImu').isHidden(), 'no IMU in this file').toBe(true);

  await open(page, 'v06_rig_imu', 4);
  expect(await page.locator('#expImu').isHidden()).toBe(false);

  const { name, buf } = await grab(page, 'expImu');
  expect(name).toBe('imu_imu0.csv');

  const lines = rows(buf.toString('utf8'));
  expect(lines[0]).toBe('t,gx,gy,gz,ax,ay,az');

  const truth = (await expected(page, 'v06_rig_imu')).imu.imu0;
  expect(lines.length - 1, 'IMU runs at its own rate, not the frame rate')
    .toBe(truth.length);

  const first = lines[1].split(',').map(Number);
  truth[0].forEach((want, j) => {
    // float32 in the record, so compare at that precision.
    expect(near(first[j], want, 1e-5), `imu column ${j}`).toBe(true);
  });
});

test('depth.npy is a real npy of metric depth for the current frame', async ({ page }) => {
  await page.goto(VIEWER);
  await open(page, 'v02_depth', 4);

  const { name, buf } = await grab(page, 'expDepth');
  expect(name, 'named for the frame it came from').toMatch(/^depth_\d{5}\.npy$/);

  // npy v1.0: magic, version, then a little-endian u16 header length.
  expect(buf.subarray(0, 6).toString('latin1')).toBe('\x93NUMPY');
  expect(buf[6]).toBe(1);
  expect(buf[7]).toBe(0);
  const headLen = buf.readUInt16LE(8);
  const header = buf.subarray(10, 10 + headLen).toString('latin1');
  expect(header).toContain("'descr': '<f4'");
  expect(header).toContain("'fortran_order': False");
  expect((10 + headLen) % 64, 'numpy requires the data to start 64-byte aligned').toBe(0);

  const cam = Object.values((await expected(page, 'v02_depth')).cameras)[0];
  expect(header).toContain(`'shape': (${cam.height}, ${cam.width})`);

  const data = buf.subarray(10 + headLen);
  expect(data.length, 'float32 per pixel').toBe(cam.height * cam.width * 4);

  // Values are metres. The vector is a synthetic scene a little over a metre
  // out, so anything finite should be a plausible distance rather than raw
  // uint16 codes — which is what a missing dequantisation step would leave.
  const finite = [];
  for (let i = 0; i < data.length; i += 4) {
    const v = data.readFloatLE(i);
    if (Number.isFinite(v)) finite.push(v);
  }
  expect(finite.length, 'some pixels have a return').toBeGreaterThan(0);
  const max = Math.max(...finite);
  expect(max, 'metres, not uint16 codes').toBeLessThan(100);
  expect(Math.min(...finite)).toBeGreaterThan(0);
});

test('exports that a file cannot support are disabled', async ({ page }) => {
  // v01_minimal has poses and calibration but no signals, so there is no depth
  // to export. An enabled button that yields nothing is worse than a dead one.
  await page.goto(VIEWER);
  await open(page, 'v01_minimal', 4);

  expect(await page.locator('#expDepth').isDisabled(), 'no depth signal').toBe(true);
  expect(await page.locator('#expPoses').isDisabled(), 'but it does have poses').toBe(false);
  expect(await page.locator('#expMeta').isDisabled()).toBe(false);
  expect(await page.locator('#expImu').isHidden()).toBe(true);
});
