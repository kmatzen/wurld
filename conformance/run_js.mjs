// Conformance runner for the JavaScript reader.
//
// Reads one vector and prints its parse in the corpus's normalised shape. The
// comparison lives in the harness (tests/test_conformance.py) so all three
// implementations are judged by one piece of code rather than three.
//
//   node conformance/run_js.mjs vectors/v03_binary_frames.wl.webm

import { readFileSync } from 'node:fs';
import { readDocument } from '../viewer/wurld.js';

const path = process.argv[2];
if (!path) {
  console.error('usage: run_js.mjs <vector.wl.webm>');
  process.exit(2);
}

try {
  const { doc, imu, rgbStreams } = readDocument(new Uint8Array(readFileSync(path)));

  const frames = (doc.frames ?? []).map(f => {
    const valid = f.pose_valid !== false;
    const rec = { i: f.i, t: f.t, pose_valid: valid };
    if (valid) {
      rec.camera = f.camera ?? '0';
      rec.q_wxyz = Array.from(f.q_wxyz);
      rec.tr = Array.from(f.tr);
    }
    return rec;
  });

  const out = {
    // Declared so the harness can tell "not supported" from "silently missing".
    supports: ['cameras', 'frames', 'signals', 'world', 'rigs', 'imu', 'rgb_streams'],
    cameras: Object.fromEntries(Object.entries(doc.cameras ?? {}).map(([k, c]) => [
      k, { model: c.model, width: c.width, height: c.height, params: Array.from(c.params) },
    ])),
    frames,
    signals: (doc.signals ?? []).map(s => ({
      id: s.id, role: s.role, value_map: s.value_map ?? {},
    })),
    world: doc.world ?? {},
    rigs: doc.rigs ?? {},
    imu: Object.fromEntries(Object.entries(imu).map(([id, samples]) => [
      id, samples.map(s => [s.t, ...s.gyro, ...s.accel]),
    ])),
    rgb_streams: rgbStreams,
  };
  process.stdout.write(JSON.stringify(out) + '\n');
} catch (err) {
  console.error(String(err && err.message ? err.message : err));
  process.exit(1);
}
