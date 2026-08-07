// Parity checks for the JavaScript implementation. The binary records here
// must stay byte-identical to what wurld/container.py writes — that identity is
// the whole point of the format, so it is asserted rather than assumed.
import assert from 'node:assert/strict';
import test from 'node:test';
import {
  ID, packFrames, unpackFrames, FRAME_RECORD_SIZE, buildTags,
  collectTagsInRange, readVint, cat,
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
