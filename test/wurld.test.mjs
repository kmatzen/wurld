// Parity checks for the JavaScript implementation. The binary records here
// must stay byte-identical to what wurld/container.py writes — that identity is
// the whole point of the format, so it is asserted rather than assumed.
import assert from 'node:assert/strict';
import test from 'node:test';
import {
  ID, packFrames, unpackFrames, FRAME_RECORD_SIZE, buildTags,
  collectTagsInRange, readVint, cat, unpackImu, IMU_RECORD_SIZE,
  readDocument, resolvePoses, readImuStreams, readWurldTags,
} from '../viewer/wurld.js';
import { readFileSync, existsSync } from 'node:fs';

// readWurldTags expects a whole file (it descends through Segment); these
// fixtures are bare Tags elements, so collect over the buffer directly.
const tagsOf = bytes => collectTagsInRange(bytes, 0, bytes.length);

test('frame records are 45 bytes and round-trip', () => {
  const frames = [
    { i: 0, t: 1.5, camera: '0', q_wxyz: [0.5, 0.5, 0.5, 0.5], tr: [1.25, -2.5, 3.0] },
    { i: 1, t: 2.5, pose_valid: false },
  ];
  const packed = packFrames(frames, ['0']);
  assert.equal(FRAME_RECORD_SIZE, 45);
  assert.equal(packed.length, 2 * 45, 'record size drift vs SPEC §7');

  const back = unpackFrames(packed, ['0']);
  assert.equal(back.length, 2);
  assert.equal(back[0].t, 1.5);
  assert.equal(back[0].camera, '0');
  assert.deepEqual(Array.from(back[0].tr), [1.25, -2.5, 3.0]);
  assert.equal(back[1].pose_valid, false);
});

test('camera index resolves through the supplied key table', () => {
  const frames = [
    { i: 0, t: 0, camera: 'left', q_wxyz: [1, 0, 0, 0], tr: [0, 0, 0] },
    { i: 1, t: 0.1, camera: 'right', q_wxyz: [1, 0, 0, 0], tr: [0, 0, 0] },
  ];
  const keys = ['left', 'right'];
  const back = unpackFrames(packFrames(frames, keys), keys);
  assert.deepEqual(back.map(f => f.camera), ['left', 'right']);
});

test('tags round-trip, and binary tags stay binary', () => {
  const payload = packFrames(
    [{ i: 0, t: 0, camera: '0', q_wxyz: [1, 0, 0, 0], tr: [0, 0, 0] }], ['0']);
  const bytes = buildTags({ WURLD: '{"format":"wurld"}', WURLD_FRAMES: payload });
  const tags = tagsOf(bytes);
  assert.equal(typeof tags.WURLD, 'string');
  assert.ok(tags.WURLD_FRAMES instanceof Uint8Array);
  assert.equal(tags.WURLD_FRAMES.length, 45);
});

test('repeated binary tags concatenate in file order', () => {
  const a = packFrames([{ i: 0, t: 0, camera: '0', q_wxyz: [1, 0, 0, 0], tr: [0, 0, 0] }], ['0']);
  const b = packFrames([{ i: 1, t: 0.1, camera: '0', q_wxyz: [1, 0, 0, 0], tr: [0, 0, 0] }], ['0']);
  const bytes = cat([buildTags({ WURLD_POSES: a }), buildTags({ WURLD_POSES: b })]);
  const tags = tagsOf(bytes);
  assert.equal(tags.WURLD_POSES.length, 90, 'chunks must concatenate, not replace');
  assert.deepEqual(unpackFrames(tags.WURLD_POSES, ['0']).map(f => f.i), [0, 1]);
});

test('vints decode to their documented widths', () => {
  assert.equal(readVint(new Uint8Array([0x81]), 0)[0], 1);        // 1-byte vint
  assert.equal(readVint(new Uint8Array([0x40, 0x01]), 0)[0], 1);  // 2-byte, same value
  assert.equal(readVint(new Uint8Array([0x40, 0x01]), 0)[2], 2);  // width reported
  assert.equal(ID.CLUSTER, 0x1f43b675);
});


// The viewer exports IMU as CSV because ffmpeg cannot reach these tags at all
// (EXTRACTING.md). Layout must stay lockstep with wurld/container.py's
// _IMU_RECORD = struct.Struct("<d3f3f").
test('unpackImu reads the 32-byte record layout', () => {
  const buf = new Uint8Array(2 * IMU_RECORD_SIZE);
  const dv = new DataView(buf.buffer);
  dv.setFloat64(0, 1.5, true);
  [0.1, 0.2, 0.3, 0.0, 0.0, 9.81].forEach((v, k) => dv.setFloat32(8 + 4 * k, v, true));
  dv.setFloat64(IMU_RECORD_SIZE, 1.505, true);

  const got = unpackImu(buf);
  assert.equal(got.length, 2);
  assert.equal(got[0].t, 1.5);
  assert.equal(got[1].t, 1.505);
  assert.ok(Math.abs(got[0].gyro[1] - 0.2) < 1e-6);
  assert.ok(Math.abs(got[0].accel[2] - 9.81) < 1e-5);
});

test('unpackImu rejects a truncated buffer rather than inventing a sample', () => {
  assert.throws(() => unpackImu(new Uint8Array(IMU_RECORD_SIZE + 7)), /multiple of 32/);
});


// ---------------------------------------------------------------------------
// The conformance corpus (conformance/vectors) is the shared definition of
// correct across the Python, JS and C++ readers. These check the JS entry
// points against it directly; tests/test_conformance.py does the full
// cross-implementation diff.

const VECTORS = new URL('../conformance/vectors/', import.meta.url);
const vector = name => new URL(name, VECTORS);
const haveVectors = existsSync(vector('index.json'));

test('readDocument resolves a binary pose table', { skip: !haveVectors && 'no vectors' },
  () => {
    const bytes = new Uint8Array(readFileSync(vector('v03_binary_frames.wl.webm')));
    const { doc } = readDocument(bytes);
    const want = JSON.parse(readFileSync(vector('v03_binary_frames.expected.json')));
    assert.equal(doc.frames.length, want.frames.length);
    assert.ok(doc.frames.length > 0);
    for (let i = 0; i < want.frames.length; i++) {
      assert.equal(doc.frames[i].i, want.frames[i].i);
      assert.ok(Math.abs(doc.frames[i].tr[0] - want.frames[i].tr[0]) < 1e-6);
    }
  });

test('resolvePoses works from tags a caller already holds',
  { skip: !haveVectors && 'no vectors' }, () => {
    // This is exactly how the viewer uses it: tags first (possibly from ranged
    // reads), poses resolved after. Keeping the viewer on this path is why the
    // chain lives in the library instead of in the page.
    const bytes = new Uint8Array(readFileSync(vector('v03_binary_frames.wl.webm')));
    const tags = readWurldTags(bytes);
    const doc = JSON.parse(tags.WURLD);
    assert.ok(!doc.frames || doc.frames.length === 0, 'fixture should store poses in binary');
    const frames = resolvePoses(doc, tags);
    assert.ok(frames.length > 0, 'binary table must resolve to poses');
    assert.equal(frames, doc.frames);
  });

test('unposed frames survive as unposed', { skip: !haveVectors && 'no vectors' }, () => {
  const bytes = new Uint8Array(readFileSync(vector('v04_unposed.wl.webm')));
  const { doc } = readDocument(bytes);
  const lost = doc.frames.filter(f => f.pose_valid === false).map(f => f.i);
  assert.deepEqual(lost, [1, 4]);
  // Absent, not identity: a substituted pose is a camera that never existed.
  for (const f of doc.frames) {
    if (f.pose_valid === false) assert.equal(f.tr, undefined);
  }
});

test('IMU streams unpack from the document declaration',
  { skip: !haveVectors && 'no vectors' }, () => {
    const bytes = new Uint8Array(readFileSync(vector('v06_rig_imu.wl.webm')));
    const { doc, imu, tags } = readDocument(bytes);
    assert.deepEqual(Object.keys(imu), ['imu0']);
    assert.ok(imu.imu0.length > doc.frames.length);
    assert.deepEqual(Object.keys(readImuStreams(doc, tags)), ['imu0']);
  });

test('readDocument rejects a file with no WURLD tag', () => {
  assert.throws(() => readDocument(new Uint8Array([0x1a, 0x45, 0xdf, 0xa3, 0x00])),
                /no WURLD tag|Unexpected|EBML|vint/i);
});

test('unpackFrames rejects corruption instead of inventing a camera', () => {
  const buf = new Uint8Array(FRAME_RECORD_SIZE);
  new DataView(buf.buffer).setUint32(4, 7, true);      // camera index 7
  assert.throws(() => unpackFrames(buf, ['only-one']), /camera index 7/);
  assert.throws(() => unpackFrames(new Uint8Array(FRAME_RECORD_SIZE + 5), ['a']),
                /not a multiple of 45/);
});


test("the README's JavaScript quickstart runs on every file shape",
  { skip: !haveVectors && 'no vectors' }, () => {
    // It previously hand-rolled the precedence chain as
    //   unpackFrames(tags.WURLD_FRAMES ?? tags.WURLD_POSES, keys)
    // which throws on a file whose poses are in the JSON array — most files.
    for (const name of ['v01_minimal', 'v03_binary_frames', 'v06_rig_imu']) {
      const bytes = new Uint8Array(readFileSync(vector(`${name}.wl.webm`)));
      const { doc, imu, rgbStreams } = readDocument(bytes);
      assert.ok(doc.frames.length > 0, `${name}: no poses resolved`);
      assert.ok(typeof doc.frames[0].t === 'number');
      assert.ok(doc.cameras && Object.keys(doc.cameras).length > 0);
      assert.ok(Array.isArray(rgbStreams));
      assert.equal(typeof imu, 'object');
    }
  });
