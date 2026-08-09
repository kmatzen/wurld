# wurld

> **wurld** — the *World's Unbroken Record of Localization & Depth*
**Posed sensor video in one playable WebM** — RGB video + per-frame camera pose +
intrinsics + timestamps + bit-exact metric depth (and confidence / object IDs), in a
single file that an ordinary video player treats as plain RGB.

wurld is the missing interchange layer between the things that *produce* posed
RGBD video (phone capture apps, SLAM systems, robots, synthetic renderers) and the
things that *consume* it (NeRF/splat training, robot learning, world-model pipelines,
SLAM evaluation). Today every producer invents a directory layout — COLMAP dirs,
`transforms.json`, TUM text files, per-app zip formats — and every consumer maintains a
zoo of parsers. wurld replaces the zoo with one canonical-convention container:

- **Payload**: [chromapakz](https://github.com/kmatzen/chromapakz) tracks — VP9 RGB +
  lossless, bit-exact uint16 planes for depth / confidence / IDs. Decodes natively in
  the browser via WebCodecs.
- **Metadata**: one Matroska `WURLD` tag holding cameras (COLMAP-style models),
  per-frame poses and sensor timestamps, signal semantics (how uint16 maps to meters),
  and world info (metric scale, gravity). See [SPEC.md](SPEC.md).
- **Conventions are fixed, not declared**: RDF camera axes (OpenCV/COLMAP),
  camera-to-world, wxyz quaternions, meters, seconds. Converters normalize on write;
  consumers never branch on axis flags.

## Install

```sh
pip install wurld                 # Python reader/writer, converters, CLI
npm install wurld-core            # JavaScript: same format, byte-identical records
```

Optional Python extras: `wurld[record3d]` for `.r3d` imports, `wurld[mcap]` for
Foxglove export, `wurld[dev]` for the test suite.

## Quickstart

```sh
# working from a checkout instead: pip install -e ".[dev]"

# write a synthetic demo sequence (analytic RGBD + orbit poses)
wurld demo demo.wl.webm

# inspect
wurld info demo.wl.webm

# convert real datasets (auto-detects TUM / transforms.json / COLMAP / Stray Scanner)
wurld convert path/to/rgbd_dataset_freiburg1_desk desk.wl.webm
wurld convert path/to/transforms.json scene.wl.webm
wurld convert path/to/colmap_project scene.wl.webm --images path/to/images
wurld convert path/to/stray_capture scan.wl.webm       # needs ffmpeg on PATH

# extract back out
wurld extract demo.wl.webm out/ --format tum        # or transforms | colmap

# check a file against the spec (exit 1 on a MUST violation)
wurld validate scene.wl.webm
```

`validate` turns SPEC's normative requirements into executable checks, so anyone
writing a producer can confirm their output before shipping it. Findings name the
section they come from — pose-table ordering and precedence, camera/video
resolution agreement, quaternion normalisation, timestamp monotonicity, value-map
sanity, rig and IMU references — with MUST violations as errors and SHOULD as
warnings.

Python API:

```python
import wurld as wl

seq = wl.read("demo.wl.webm")
seq.rgb                  # (T, H, W, 4) uint8, lazily decoded
seq.depth_meters(0)      # (H, W) float, NaN = invalid
seq.c2w(0)               # 4x4 camera-to-world (RDF, meters)
seq.K("0")               # 3x3 intrinsics (K("0", frame_index=i) honors overrides)
seq.frames[0].t          # sensor timestamp (authoritative, may be non-uniform)
seq.rigs                 # camera-to-rig calibration; seq.rig_c2w(i, "1") derives poses
seq.imu["imu0"].samples  # (N, 7) [t, gyro xyz, accel xyz]
```

Long sequences (>10k frames) automatically pack poses into a binary frame table
(45 bytes/frame; SPEC §7) — `wl.write(..., frames_format="binary")` forces it.

JavaScript API (`wurld-core`, no dependencies — works in browsers and Node):

```js
import { readWurldTags, unpackFrames, WurldRecorder, StreamSplitter } from 'wurld-core';

const tags = readWurldTags(bytes);            // SimpleTags: strings last-win, binaries concatenate
const doc = JSON.parse(tags.WURLD);           // cameras, signals, conventions
const keys = doc.frames_binary?.cameras ?? Object.keys(doc.cameras).sort();
const frames = unpackFrames(tags.WURLD_FRAMES ?? tags.WURLD_POSES, keys);
frames[0].t;                                  // seconds; .q_wxyz, .tr, .pose_valid
```

The 45-byte records are byte-identical to the Python writer's — the parity suite
asserts it rather than assuming it. Pose precedence is `WURLD_FRAMES`, then
`WURLD_POSES` chunks, then the JSON array (SPEC §9).

Writing live (`WurldRecorder`) needs a chromapakz encoder, which you supply — it
is an optional peer dependency, so reading costs you no native install.

## Browser viewer

```sh
npm install                      # fetches the chromapakz decoder for the demo
python3 -m http.server 8000      # from the repo root
# open http://localhost:8000/viewer/index.html  and drop a .wl.webm
# (or .../viewer/index.html?src=../demo.wl.webm)
```

Zero-install, WebCodecs-decoded: seekable playback with per-frame point-cloud
reprojection, camera trajectory + frustum, RGB and metric-depth panes, and the same
file playing in a plain `<video>` element.

Hosted, with a sample loaded — nothing to clone or install:
**[kmatzen.com/wurld](https://kmatzen.com/wurld/)**

## No install at all

The metadata lives in standard Matroska tags, so standard tools read it.
`ffprobe` prints the whole document — cameras, conventions, signal semantics,
and (for batch-written files) every pose:

```sh
ffprobe -v error -show_entries format_tags=WURLD -of default=nw=1:nk=1 scene.wl.webm
```

[EXTRACTING.md](EXTRACTING.md) is the cookbook: poses to CSV, intrinsics, RGB
frames, and depth in metres using nothing but ffmpeg and arithmetic — including
the exact triangle-fold and inverse-depth formulas, checked against the
reference reader to 1.5 µm. It is also honest about the two things that route
does not reach (binary pose tables and IMU streams) and what to do instead.

## Use cases

[USE_CASES.md](USE_CASES.md) surveys what people actually build with posed RGBD
— feed-forward reconstruction, splat and NeRF training, SLAM benchmarking, robot
rigs, dataset distribution — maps each to the format, and is explicit about the
cases where wurld is the wrong tool (multi-agent scenes, camera-less LiDAR
sweeps, geospatial survey, non-rigid capture). Seven scenarios ship as runnable,
tested examples under `examples/`.

It also carries the measured playback matrix and the HDR plans. In short: VLC,
Chrome and ffmpeg play these files and pick the right track; **QuickTime and iOS
Photos cannot open them at all**, because AVFoundation has no WebM demuxer. That
is a deliberate trade — VP9 lossless is what makes bit-exact depth possible —
and desktop viewing is VLC or IINA.

## ROS 2

`wurld ros2 export` writes a real **rosbag2** — CDR-encoded `sensor_msgs` and
`tf2_msgs`, so `ros2 bag play` works and ROS nodes can subscribe. (The older
`wurld convert --to mcap` writes Foxglove jsonschema channels, which Foxglove
Studio reads but no ROS node can.)

```bash
wurld ros2 export scene.wl.webm ./bag          # mcap storage; --storage sqlite3 also
wurld ros2 import ./bag out.wl.webm
```

```
/camera/<id>/image_raw        sensor_msgs/msg/Image        rgb8
/camera/<id>/camera_info      sensor_msgs/msg/CameraInfo
/camera/<id>/depth/image_raw  sensor_msgs/msg/Image        32FC1, metres
/tf                           tf2_msgs/msg/TFMessage       world -> *_optical_frame
/imu/<id>                     sensor_msgs/msg/Imu
```

Two conventions this gets right, both of which fail silently when got wrong:
ROS quaternions are **xyzw** where wurld's are wxyz, and ROS *optical* frames are
RDF (REP 145) — the same convention wurld uses — so `world -> <cam>_optical_frame`
is `c2w` with no axis conversion. The `_optical_frame` suffix is what tells a
consumer not to apply one.

Depth goes out as `32FC1` metres rather than `16UC1` millimetres so NaN survives;
in the 16-bit convention 0 means both "no return" and "at the sensor".

**Fidelity is one-way.** wurld → rosbag2 is exact. The return leg is not: a bag
carries no quantization range, so depth is requantized (measured under 5 µm on a
1.5–2 m scene) and images re-encode through lossy VP9 (mean |Δ| ≈ 3/255). Good
for moving between ecosystems, not for archival round trips.

Needs `pip install 'wurld[ros2]'`, which does **not** require ROS itself.

## Conformance corpus

Three readers — Python, JavaScript, C++ — is three chances to disagree.
`conformance/vectors/` is 88 KiB of small files paired with the parse a reader
must produce, and one harness judges all three against it:

```bash
pytest tests/test_conformance.py     # python, javascript and c++
```

Expectations are generated from intent rather than captured from a reader
(`conformance/generate.py`), so the corpus cannot enshrine a bug all three
happen to share — which a golden file dumped from the Python reader would.
The vectors cover JSON and binary pose tables, unposed frames, stereo streams,
rigs, IMU, distorted camera models, `float16_bits`, unicode, and a single-frame
file. If you write a fourth reader, this is the definition of correct.

## C++ reader

`cpp/include/wurld.hpp` is a single-header C++17 reader with **no dependencies**
— calibration, per-frame poses, timestamps, signal descriptors, rigs and IMU:

```cpp
#include "wurld.hpp"
auto doc = wurld::read("scene.wl.webm");
for (const auto& f : doc.frames)
    if (f.pose_valid) use(f.c2w());          // row-major 4x4, camera-to-world
```

It does **not** decode pixels or depth; that needs libvpx and chromapakz, and
the point of the header is that it drops onto a robot without dragging a codec
stack along. `doc.cluster_start` reports where the Clusters begin for consumers
that do want to hand bytes to a decoder. Traversal seeks over payloads and reads
only Tags, so opening a 10 GB file costs a handful of seeks.

Agreement with the Python reader is the actual requirement, so it is tested that
way: `tests/test_cpp_reader.py` diffs both readers field-by-field over JSON and
binary pose tables, rigs, IMU, unposed frames and the example files. A second
implementation that only agrees with itself is not a second implementation.

```bash
cmake -S cpp -B cpp/build && cmake --build cpp/build
./cpp/build/wurld_info scene.wl.webm
```

### Writing from C++

`cpp/include/wurld_write.hpp` attaches a metadata layer to an already-encoded
WebM. That split is the point: a robot already has chromapakz for encoding, and
pulling libvpx into this header would destroy the zero-dependency property that
made the reader worth having.

```cpp
#include "wurld_write.hpp"
wurld::WriteDoc doc;
doc.cameras["0"] = {"PINHOLE", 640, 480, {525, 525, 320, 240}};
doc.frames.push_back({0, 0.0, "0", true, {1,0,0,0}, {0,0,0}});
doc.world_json = R"({"metric_scale":true})";
wurld::write_file("encoded.webm", "out.wl.webm", doc);   // or wurld::attach(bytes, doc)
```

Inserting tags shifts every Cluster, so Cues are rebuilt at the new offsets and
the SeekHead is regenerated — a carried-over Cues element seeks into the middle
of a Cluster, which presents as corrupt video rather than as a muxing bug.
Re-attaching replaces the previous wurld tags instead of accumulating them,
while leaving foreign tags (chromapakz's own) untouched.

The bar is equality with Python, not merely readability: `pack_frames` and
`pack_imu` must emit byte-identical buffers, and a C++-written file must satisfy
`wurld validate` and read identically through all three readers
(`tests/test_cpp_writer.py`).

## Collections: a corpus as a dataset

One file is one sequence; training is ten thousand of them. A **collection**
(SPEC §14) is a manifest plus the files it names — a sidecar, not a new
container, so every member stays an ordinary playable wurld file.

```bash
wurld index captures/ -o collection.json     # headers only; no pixels decoded
wurld collection collection.json --members
```

```python
from wurld import Collection
c = Collection.read("collection.json")
c.locate(12345)                  # global frame -> (member, frame within it)
for item in c.iter_frames(fields=("rgb", "depth"), shard=(0, 4), shuffle=7):
    ...
```

Indexing reads each member's header and stops at the first Cluster: a file with
100x the pixels of another, at the same frame count, indexes for the same ~8 KiB.
Sharding keeps whole files together so each member decodes once, and yields every
frame exactly once — asserted across ranks x DataLoader workers, because a bad
split does not crash, it quietly trains on duplicates.

With PyTorch (`pip install 'wurld[torch]'`):

```python
from wurld.integrations.torch_data import WurldIterableDataset
ds = WurldIterableDataset("collection.json", fields=("rgb", "depth"))
loader = DataLoader(ds, batch_size=8, num_workers=4)     # shards itself
```

Verify before you train — the manifest caches frame counts, and global indexing
is computed from them, so a member that was re-exported one frame shorter shifts
every index after it while `locate()` keeps returning an answer:

```bash
wurld collection collection.json --verify     # exits 1 on drift
```

It re-reads headers, which is the same cheap read that built the manifest.
`--checksum` additionally compares recorded hashes, which is the only way to see
a change that leaves the header identical.

Measured at 10,000 members (`scripts/bench_collection.py`): index 1.9 s, manifest
3.4 MiB, `locate()` 0.46 µs, verify 1.0 s, streaming 31,700 frames/s for metadata
and 8,400 with pixels — peak RSS 58.7 MB, flat. Streaming holds a Cluster, not a
file: see USE_CASES scenario 7 for what measuring that turned up.

A collection asserts nothing about how members relate in space or time; that is
the separate scene-manifest concern reserved in SPEC §11.

## Layout

- `SPEC.md` — the format (v1.2)
- `wurld/` — Python reference implementation (container, conventions, EBML tag
  layer, converters, CLI, synthetic test scene)
- `tests/` — round-trip suite (`pytest`): bit-exact depth, pose fidelity, converter
  round trips, COLMAP binary parsing, validation errors
- `viewer/` — single-file browser viewer
- `cpp/` — dependency-free single-header C++17 metadata reader + `wurld_info`
- `conformance/` — cross-implementation test vectors and their generator
- `wurld/converters/ros2.py` — rosbag2 bridge (real ROS 2 messages)
- `wurld/collection.py` — manifests, global indexing, sharded streaming
- `wurld/integrations/` — nerfstudio DataParser, PyTorch datasets
- `examples/` — runnable scenario walkthroughs (see USE_CASES.md)
- `LANDSCAPE.md` — the market research that motivated this project
- `USE_CASES.md` — pipelines, scenarios, and the format's limits

## Status / verified

- 29/29 tests pass: depth round-trips bit-exactly through VP9; poses, timestamps, and
  intrinsics survive every converter round trip; TUM native-unit depth is bit-exact
  end to end; binary frame tables, rigs, IMU streams, and both Stray resampling
  policies covered.
- `ffmpeg`/`ffprobe` read and fully decode wurld files with zero warnings (all
  three VP9 streams); the WURLD tags are invisible to standard demuxers.
- Viewer verified in Chrome (native WebCodecs decode, no WASM), including binary
  frame tables.

## 1.0

SPEC declared **1.0** (frozen semantics; 1.x additions are additive-only, and
§11 codifies file scope: one rig, one clock). chromapakz pinned to the released
**0.4.0** — no more git installs. LICENSE (MIT), CHANGELOG, CI (Python 3.11/3.13
on Linux+macOS plus a viewer parity job), and packaging validated (sdist+wheel
build clean, twine-checked, fresh-venv install + CLI smoke). Multi-RGB pixel
storage remains gated on the ChromaPakZ #47 design review; phone importers and
WurldCam remain fixture/harness-validated pending a real capture.

## v0.9: bounded-memory iteration, trim, auto-lazy

- **`Sequence.iter_frames(start, stop)`**: memory-bounded streaming decode —
  one Cluster (~1 s) decoded at a time, so hour-long files iterate without
  holding the take in RAM. Pre-cadence files fall back to a full decode with a
  logged warning.
- **`wurld trim in out --frames 30:60`**: cut a range into a new file —
  poses rebased, signals sliced bit-exactly, quantization specs preserved,
  in-range IMU kept, provenance noted in the world description.
- **Viewer auto-lazy**: the probe's `Content-Range` total switches large files
  (>32 MB) to on-demand cluster scrubbing automatically; `?lazy=1|0` forces
  either way.
- **Multi-RGB design proposal** filed as
  [ChromaPakZ #47](https://github.com/kmatzen/ChromaPakZ/issues/47) — the
  schema/ABI sketch for true stereo pixel storage, awaiting maintainer review
  before implementation.

## v0.8: partial decode, EuRoC stereo+IMU, release plumbing

- **`Sequence.fetch_frames(indices)`**: local partial decode — reading 3 frames
  of a 10k-frame file no longer decodes the other clusters (same cluster-splice
  machinery as remote access, bit-exact parity tested).
- **EuRoC MAV importer** (`wurld convert MH_01_easy out.wl.webm`): the first
  real exercise of rigs + IMU together — cam0 posed video via
  `T_WB @ T_BS`, both cameras' OPENCV calibration plus a `body` rig
  (camera-to-body extrinsics; `rig_c2w` derives cam1 poses), imu0 with
  imu-to-cam0 extrinsics and measured rate. Dependency-free mini-YAML parser
  for `sensor.yaml`. (Video carried cam0 only at 0.8; both eyes ship as display
  streams now — see Unreleased.)
- **MCAP `/tf`**: FrameTransform world→camera per posed frame, so Foxglove's 3D
  panel places the moving camera correctly.
- **ChromaPakZ #45 merged** (signal keyframe cadence) and release PR
  [#46 (0.4.0)](https://github.com/kmatzen/ChromaPakZ/pull/46) staged — once
  published, wurld pins `chromapakz>=0.4.0` instead of git installs. iOS
  vendor libs rebuilt from merged main.

## v0.7: random video access over ranges

- **Cluster-independent decode** (enabled by
  [ChromaPakZ #45](https://github.com/kmatzen/ChromaPakZ/pull/45), pending merge):
  signal tracks now keyframe at the RGB cadence, so `[header + Cluster k]`
  decodes that second of footage bit-exactly in isolation (+1.0% file size).
- **`wurld.remote.fetch_frames(fetch, indices)`**: random access to any
  frames of a remote file — SeekHead → Cues → only the touched Clusters are
  fetched and spliced-decoded. Verified: bit-exact vs full decode, untouched
  clusters never downloaded, and a clear error for files written before the
  keyframe cadence.
- **Viewer lazy scrubbing** (`?lazy=1&src=…`): poses render from the header
  instantly; video clusters fetch on demand as you scrub. Verified in Chrome:
  jumping frame 1 → 81 of the demo took **4 ranged requests total** with the
  middle cluster never downloaded.
- **WurldCam confidence**: on-device recordings now carry ARKit confidence
  as a second lossless signal (0/1/2 labels), verified device-free through the
  macOS harness. (Note: `build-native.sh` needs a chromapakz checkout with
  #45 for cluster-independent app recordings.)

## v0.6: on-device wurld recording, ffmpeg-native metadata

- **WurldCam v2**: the iOS app now records **wurld directly on-device** —
  libvpx + the chromapakz C core cross-compiled for iOS
  (`ios/scripts/build-native.sh`), Swift bindings over the `dc_stream_*`
  streaming ABI, and a Swift port of the pose-weaving `StreamWriter`
  (`.wl.webm` / `.r3d` toggle in the UI). The recording pipeline is verified
  without a device: `ios/scripts/verify-pipeline.sh` compiles the app's own
  writer+encoder for macOS, records a synthetic take, and validates it with the
  Python reader (poses at f32 precision vs analytic ground truth, exact f64
  timestamps, NaN-invalid depth preserved, chunked layout + consolidated table,
  ffmpeg-clean). ARKit-on-hardware remains the one untested link.
- **ffmpeg needs no patch**: ffprobe already surfaces the complete WURLD
  JSON document as a format tag —
  `ffprobe -show_entries format_tags=WURLD -of json scene.wl.webm` yields
  cameras, conventions, and (for JSON-frame files) every pose, in any
  ffmpeg-based tool. Binary pose tables/IMU tags don't surface; everything else
  does. The roadmap's "ffmpeg demuxer patch" is retired as unnecessary.

## v0.5: Record3D, MCAP/Foxglove, ranged viewer, nerfstudio

- **Record3D importer** (`wurld convert capture.r3d out.wl.webm`): the .r3d zip
  layout confirmed against the app author's own snippets and four community
  parsers — column-major K, scalar-last ARKit quaternions, LZFSE float32-meters
  depth (NaN invalid, float16 export variant handled), 0/1/2 confidence. Needs
  `pip install wurld[record3d]`.
- **MCAP export** (`wurld extract scene.wl.webm out.mcap --format mcap`):
  Foxglove-ready jsonschema channels (`/camera/pose`, `/camera/image` jpeg,
  `/camera/depth` 16UC1 bit-exact codes, `/camera/calibration`, `/imu/<id>`) plus
  the full WURLD document as an MCAP metadata record. Needs `[mcap]` extra.
- **Progressive ranged viewer**: `viewer/index.html?src=...` now loads via Range
  requests — trajectory and all poses render from the ~18KB header before any
  video byte, then video streams through the network decoder. Falls back to full
  download when the server lacks ranges. `scripts/range_server.py` is a
  range-capable dev server (python's builtin lacks Range).
- **nerfstudio DataParser** (`wurld.integrations.nerfstudio_parser`): reads a
  .wl.webm directly (frames extracted to a `<file>.cache/` beside it), poses
  converted to nerfstudio's convention, metric depth via
  `depth_unit_scale_factor` — verified against a live nerfstudio 1.1.5 install.

## v0.4: range-request access + Polycam

- **SeekHead** (SPEC §9.1): batch files begin with a fixed-width SeekHead covering
  the header elements, the first Cluster, and Cues. Combined with the
  metadata-first layout, a static file on S3/CDN serves calibration and every pose
  in at most two ranged reads.
- **`wurld.remote`**: `fetch_header(http_fetcher(url))` pulls all metadata +
  poses without downloading video — verified <2% of file bytes on a 60-frame
  sequence; `cues_offset`/`header_extent` expose what a video-seeking client needs
  next.
- **Polycam raw importer**: keyframes/cameras JSON (`fx/fy/cx/cy`, `t_00..t_23`
  ARKit c2w), microsecond-timestamp stems (sorted numerically), 16-bit mm depth,
  0/127/255 confidence labels, `corrected_cameras`/`corrected_images` preferred
  with `corrected=False` opt-out, same `--at depth|rgb` resolution policy as
  Stray. *Like Stray: validated against synthetic fixtures built from polyform's
  published schema — a real capture to confirm conventions is welcome.*

## v0.3: fully streamable playback and recording

- **Streaming layout** (SPEC §9): all wurld metadata — calibration *and* every
  pose — now lands **before the first Cluster**; Cues are rebuilt for the new
  offsets. A progressive reader has full pose data before the first video byte;
  ffmpeg still decodes and *seeks* cleanly.
- **Live recording**: `viewer/wurld.js` provides `WurldRecorder` (wraps
  chromapakz's streaming encoder, weaves `WURLD_POSES` chunks before each
  Cluster, consolidates a pose table on finish) and `WurldLivePlayer`
  (extracts pose tags from the byte stream, forwards clean video bytes to the
  network decoder). A crash-truncated recording is still a valid, fully-posed file
  up to its last flushed chunk.
- **`viewer/live.html`**: record → stream → play in one page — the player consumes
  only the byte stream and reconstructs the point cloud live (verified in Chrome:
  46/46 frames + poses received during recording; finalized file passes buffered
  decode).
- **Python `wurld.stream.StreamReader`**: incremental parser for live/growing
  streams; batch `wl.read()` also accepts crash-truncated live files via chunk
  concatenation.
- **Python live recording — `wurld.StreamWriter`** (chromapakz ≥ 0.4.0):

  ```python
  w = wl.StreamWriter(out.write, cameras={"0": cam}, has_rgb=True,
                      signal_meta=[wl.SignalMeta("depth", "depth",
                          {"type": "inverse_depth", "near": 0.4, "far": 12.0})])
  w.add_frame(frame, rgb=rgba, signals={"depth": {"float": z}})   # per capture tick
  w.add_imu("imu0", samples)                                      # any cadence
  w.finish()
  ```

  Verified: bit-exact depth through the live path, IMU chunk concatenation,
  progressive parse parity, crash-truncation survival, and ffmpeg-clean output.
  Pose-only takes (RGB + poses, no depth) work too — enabled upstream by
  [ChromaPakZ #44](https://github.com/kmatzen/ChromaPakZ/pull/44).

## v0.2

- **Binary frame tables** (`WURLD_FRAMES` TagBinary, 45 B/frame) for 10^6+-frame
  sequences; automatic beyond 10k frames.
- **Multi-camera rigs**: camera-to-rig calibration block + `rig_c2w()` derivation.
- **Per-frame intrinsics overrides** (zoom / autofocus drift).
- **IMU streams**: packed 32 B/sample gyro+accel tags with extrinsics.
- **Stray Scanner importer** (first real capture-app source; ffmpeg-based, ARKit
  RUB→RDF pose conversion, honest RGB/depth resolution policy). *Validated against
  synthetic fixtures — a real capture to confirm conventions is welcome.*

## Roadmap (v1.0)

Real-device validation (one WurldCam capture + one third-party-app capture
confirms every convention); LeRobot depth-backend PR upstream (staged
privately); ChromaPakZ #46 merge + 0.4.0 publish, then pin here. Multi-RGB
tracks per the #47 design have since landed, and the EuRoC importer now stores
both cameras' pixels with SPEC §4.4 binding frame camera ids to RGB streams.
