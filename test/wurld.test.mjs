// Parity checks for the JavaScript implementation. The binary records here
// must stay byte-identical to what wurld/container.py writes — that identity is
// the whole point of the format, so it is asserted rather than assumed.
import assert from 'node:assert/strict';
import test from 'node:test';
import {
  ID, packFrames, unpackFrames, FRAME_RECORD_SIZE, buildTags,
  collectTagsInRange, readVint, cat, unpackImu, IMU_RECORD_SIZE,
} from '../viewer/wurld.js';

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
