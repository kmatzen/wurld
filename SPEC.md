# wurld — posed sensor video, v1.1

> **wurld** — the World's Unbroken Record of Localization & Depth.

Wurld is a container profile for **posed sensor video**: RGB video + per-frame
camera pose + intrinsics + timestamps + bit-exact auxiliary signals (metric depth,
confidence, object/semantic IDs) in **one ordinary `.webm` file**. A legacy player
shows plain RGB video; reconstruction, robot-learning, and world-model pipelines read
poses and lossless 16-bit signals from the same file.

Wurld is a **container layer**: pixel payloads are carried by
[chromapakz](https://github.com/kmatzen/chromapakz) tracks (VP9; RGB + bit-exact
uint16 planes). Wurld adds the *sensor semantics*: who took the pictures, from
where, when, and what the numbers mean.

## 1. Design rules

1. **One file, ordinary video.** The container is Matroska/WebM. Everything wurld
   adds is metadata that spec-compliant demuxers skip. The file plays as RGB video in
   Chrome, VLC, ffmpeg.
2. **Canonical conventions, not declared conventions.** Every wurld file uses the
   same coordinate conventions (§3). Converters normalize *on write*. Consumers never
   branch on axis flags. The conventions block exists in the JSON for self-description
   and versioning, but in 1.0 its values are fixed.
3. **Bit-exact or honestly labeled.** Integer signals (raw sensor depth, IDs) round-trip
   bit-exactly via chromapakz lossless tracks. Value maps (§6) declare how integers map
   to physical quantities. Nothing is silently rescaled.
4. **Timestamps are data.** Video-track timing is presentation timing; per-frame sensor
   timestamps (float64 seconds, possibly non-uniform) live in the metadata and are
   authoritative.

## 2. Storage mechanism

A wurld file is a chromapakz WebM with one additional top-level Matroska element
inside the Segment:

- A `Tags` element (ID `0x1254C367`) containing one `Tag` with a `SimpleTag`:
  - `TagName` (0x45A3) = `WURLD`
  - `TagString` (0x4487) = UTF-8 JSON document (§4)

This mirrors chromapakz's own `CHROMAPAKZ` SimpleTag. Element placement follows the
streaming layout (§9): batch writers put wurld Tags in the header region before
the first Cluster (rebuilding Cues for the shifted offsets); live recorders emit the
header tag first and chunk tags between Clusters. Appending a Tags element after the
Clusters (rewriting only the Segment size vint) also yields a valid file — early
writers did — but forfeits streamable metadata. Readers scan all top-level Tags
elements regardless of position.

Recommended file suffix: `.wl.webm` (plain `.webm` also valid).

## 3. Canonical conventions (fixed in 1.0)

| Aspect | Convention |
|---|---|
| Camera axes | **RDF**: +X right, +Y down, +Z forward (OpenCV / COLMAP) |
| Pose direction | **camera-to-world** (c2w): `X_world = R * X_cam + t` |
| Rotation | unit quaternion, **wxyz** order (field is named `q_wxyz`) |
| Translation | meters (field `tr`), world frame |
| Timestamps | float64 **seconds** (field `t`); epoch is arbitrary but monotonic per file |
| Image origin | pixel (0,0) is top-left; `cx, cy` in pixel units, center-of-pixel at integer coordinates |
| World frame | arbitrary orientation, described by `world` block (§5); gravity direction optional but recommended |

Conversion helpers for other conventions (OpenGL/Blender RUB axes, world-to-camera,
xyzw quaternions) belong in libraries, not in files.

## 4. The WURLD JSON document

```json
{
  "format": "wurld",
  "version": "1.1",
  "conventions": {
    "camera_axes": "RDF",
    "pose_direction": "camera_to_world",
    "quaternion_order": "wxyz",
    "units": "meters",
    "timestamp_units": "seconds"
  },
  "world": {
    "metric_scale": true,
    "gravity_in_world": [0.0, 0.0, -1.0],
    "description": "free text: origin/frame provenance"
  },
  "cameras": {
    "0": {
      "model": "PINHOLE",
      "width": 640, "height": 480,
      "params": [525.0, 525.0, 319.5, 239.5]
    }
  },
  "signals": [
    { "id": "depth", "role": "depth",
      "value_map": { "type": "linear", "scale": 0.0002, "offset": 0.0, "invalid": 0 } },
    { "id": "confidence", "role": "confidence",
      "value_map": { "type": "labels", "labels": { "0": "low", "1": "medium", "2": "high" } } }
  ],
  "frames": [
    { "i": 0, "t": 1305031102.175304, "camera": "0",
      "q_wxyz": [0.6132, -0.5962, 0.3311, -0.3986],
      "tr": [1.3405, 0.6266, 1.6575] }
  ]
}
```

### 4.1 `cameras`

Keyed by string camera id. `model` and `params` follow **COLMAP camera model naming**:

| model | params |
|---|---|
| `SIMPLE_PINHOLE` | `[f, cx, cy]` |
| `PINHOLE` | `[fx, fy, cx, cy]` |
| `SIMPLE_RADIAL` | `[f, cx, cy, k]` |
| `RADIAL` | `[f, cx, cy, k1, k2]` |
| `OPENCV` | `[fx, fy, cx, cy, k1, k2, p1, p2]` |
| `OPENCV_FISHEYE` | `[fx, fy, cx, cy, k1, k2, k3, k4]` |

`width`/`height` are the *calibrated* resolution and MUST equal the video track
resolution (a `scale` extension is reserved for a future revision).

### 4.2 `frames`

One entry per video frame, in presentation order. Fields:

- `i` (int, required): frame index into the video/signal tracks.
- `t` (float64, required): sensor timestamp, seconds. Monotonic non-decreasing.
- `camera` (string, required): key into `cameras`.
- `q_wxyz`, `tr` (required unless `pose_valid` is false): camera-to-world rotation and
  translation.
- `pose_valid` (bool, default true): false marks tracking-lost frames; `q_wxyz`/`tr`
  may then be omitted.

Frames MAY be sparser than the video (e.g. poses at keyframes only); consumers MUST NOT
interpolate unless they choose to. Frames MUST NOT be denser than the video track.

### 4.3 `signals`

Each entry binds a chromapakz signal id to a semantic role:

- `role`: `depth` | `confidence` | `object_id` | `semantic_id` | `normal_packed` | `custom`
- `value_map` (§6) defines the uint16 → physical mapping.
- For `role: depth`, values map to **metric depth along the camera +Z axis** (not ray
  length), in meters.

### 4.4 Display streams and cameras (v1.2)

A file MAY store pixels for several cameras — a stereo rig recording both eyes.
**Stream ids are camera ids**: a display stream named `cam1` carries the pixels
`cameras["cam1"]` calibrates. That binding is the whole mechanism; without it a
reader cannot tell which intrinsics apply to which pixels.

- The first declared stream is the **primary**. It keeps the underlying
  container's track 1 and the name `rgb`, so readers that predate multi-stream
  support decode it and ignore the rest.
- A file with a **single** stream carries the conventional id `rgb` and binds to
  its sole camera implicitly; the id-is-camera-id rule applies from two streams
  up, where a reader would otherwise have nothing to match on.
- Every stream shares the file's width, height and frame grid. Unsynchronised
  rigs are out of scope (§11: one rig, one clock).
- A declared camera MAY have no stream of its own: its pose still derives from
  `rigs` (§8.1), it simply has no recorded pixels. The reverse MUST NOT happen —
  a stream whose id is not a declared camera is invalid.
- **Poses stay single-camera by default.** Frames name one camera and the rest
  derive through `rigs`, because a rigid rig's extrinsics should be stated once
  rather than restated per frame, where they can drift. Per-camera poses remain
  expressible via the frame record's camera field for non-rigid setups.

Writers SHOULD calibrate every camera at the shared resolution (§4.1 requires
equality with the video track).

### 4.5 HDR display track (v1.2)

The lossy display stream MAY be HDR: 10-bit, BT.2020, PQ or HLG, with the
container carrying the colour signalling that makes a player treat it as HDR
rather than washed-out SDR. Readers see `uint16` codes (0..1023) instead of
`uint8`.

This is **display-referred** — absolute nits through a transfer curve — and is a
different thing from a `float16_bits` signal (§6.1), which is **scene-referred**
linear radiance. A file MAY carry both: the display stream is what a player
shows, the signal is the data. Neither substitutes for the other, and a consumer
that needs radiance MUST NOT read it off the display track.

HDR applies to all of a file's display streams or none.

## 5. `world` block

- `metric_scale` (bool): true when translations are in true meters (ARKit, TUM, robot
  odometry). false for reconstructions with arbitrary scale (COLMAP SfM). Consumers
  requiring metric data MUST check this.
- `gravity_in_world` (unit 3-vector or null): direction gravity points, world frame.
- `description` (string): provenance free text.

## 6. Value maps

```
linear:        value = scale * raw + offset          (raw == invalid → no data)
inverse_depth: value = 1 / (a * raw + b)             (chromapakz near/far quantization;
                                                      a, b derived from near/far/levels)
labels:        raw is categorical; labels maps raw → name
identity:      raw is the value (IDs, packed encodings)
float16_bits:  raw IS an IEEE 754 binary16, reinterpreted (v1.2)
```

When a chromapakz signal already carries a `quant` spec (e.g. `inverse-depth`), the
wurld `value_map` MUST agree with it; the chromapakz spec is authoritative for
decode, wurld's copy is for consumers that read metadata only.

### 6.1 `float16_bits` (v1.2)

Not a quantization. The uint16 code is the half-float's bit pattern, so a
lossless signal track carries scene-referred HDR — an EXR half channel, one
signal per colour channel — with no range, no scale factor and no loss.

Readers MUST reinterpret the code as IEEE 754 binary16 rather than converting
it. Every bit pattern denotes a value, including NaN, ±Inf, −0.0 and denormals,
so a `float16_bits` value map MUST NOT declare an `invalid` code: absence is
expressed as NaN, which is itself a bit pattern that round-trips.

This is *scene-referred* data — linear radiance, unbounded above 1.0 — and is
distinct from an HDR *display* track, which is display-referred (PQ or HLG,
absolute nits) and lives in the lossy RGB stream. A file may carry both; they
answer different questions and neither substitutes for the other.

Writers SHOULD set `role: "custom"` unless a more specific role applies, and
SHOULD name channels so their correspondence is unambiguous (`hdr_r`, `hdr_g`,
`hdr_b`). float32 sources do not fit a 16-bit code and are out of scope.

## 7. Binary frame table (v0.2)

For long sequences, the `frames` array MAY be replaced by a packed binary table in a
second SimpleTag in the same Tags element:

- `TagName` = `WURLD_FRAMES`, payload in **TagBinary** (0x4485), not TagString.
- The JSON document then carries `"frames": []` and a descriptor:

```json
"frames_binary": { "version": 1, "count": 108000, "cameras": ["0", "1"] }
```

Record format v1 — little-endian, 45 bytes per frame, `count` records:

| offset | type | field |
|---|---|---|
| 0 | u32 | `i` (video frame index) |
| 4 | u32 | camera index into `frames_binary.cameras` |
| 8 | f64 | `t` (seconds) |
| 16 | 4 × f32 | `q_wxyz` |
| 32 | 3 × f32 | `tr` (meters) |
| 44 | u8 | flags: bit 0 = `pose_valid` |

When both a non-empty JSON `frames` array and a `WURLD_FRAMES` tag are present,
the binary table is authoritative. Per-frame intrinsics overrides (§8.2) require the
JSON form; writers MUST NOT use the binary table with overridden frames.

## 8. Rigs, per-frame intrinsics, IMU (v0.2)

### 8.1 `rigs`

Calibrated multi-camera rigs are described as **camera-to-rig** transforms:

```json
"rigs": {
  "rig0": {
    "cameras": { "0": { "q_wxyz": [1,0,0,0], "tr": [0,0,0] },
                 "1": { "q_wxyz": [1,0,0,0], "tr": [0.12,0,0] } },
    "description": "stereo pair, left camera is rig origin"
  }
}
```

Frame poses remain camera-to-world for the frame's own camera; rigs are calibration
metadata letting consumers derive the other cameras' poses
(`c2w_other = c2w_frame @ inv(rig[cam_frame]) @ rig[cam_other]`).

### 8.2 Per-frame intrinsics override

A JSON frame entry MAY carry `"params": [...]` — a full replacement for its camera's
`params` (same model, same length) for that frame only (zoom, autofocus drift).

### 8.3 IMU streams

```json
"imu": {
  "imu0": { "rate_hz": 200.0, "count": 720000,
            "extrinsics": { "q_wxyz": [1,0,0,0], "tr": [0,0,0] },
            "description": "device IMU, extrinsics = imu-to-camera \"0\"" }
}
```

Samples live in one SimpleTag per stream, `TagName` = `WURLD_IMU_<id>`, TagBinary
payload of `count` little-endian 32-byte records:

| offset | type | field |
|---|---|---|
| 0 | f64 | `t` (seconds, same clock as frame timestamps) |
| 8 | 3 × f32 | gyro x,y,z (rad/s) |
| 20 | 3 × f32 | accel x,y,z (m/s², includes gravity) |

`extrinsics` is the imu-to-camera transform for the camera named in `description`
convention `"0"` unless stated; axes follow the canonical camera convention (§3).

## 9. Streaming layout (v0.3)

The v0.2 layout (all metadata in one Tags element after the last Cluster) requires
the whole file before any pose is readable. The v0.3 **streaming layout** makes both
live recording and progressive playback possible; it is what writers SHOULD emit.

Element order inside the Segment:

```
Info, Tracks, Tags(CHROMAPAKZ), Tags(WURLD)          <- header, before any Cluster
[ Tags(WURLD_POSES chunk), Tags(WURLD_IMU_* chunk)?, Cluster ] ...
Tags(WURLD_FRAMES [, WURLD_IMU_*]), Cues         <- finalize (absent if crashed)
```

- **Header tag**: the `WURLD` JSON document is written before the first Cluster
  with `"frames": []` — cameras, conventions, signals, world, rigs, and IMU stream
  descriptors (with `"count"` omitted or 0) are known at recording start; poses are
  not. A progressive reader has full calibration before the first video byte.
- **Pose chunks**: `WURLD_POSES` SimpleTags (TagBinary, §7 records) appear
  repeatedly, each SHOULD directly precede the Cluster containing the frames it
  describes. Repeated tags concatenate in file order; records MUST be ordered by
  frame index across the whole file. Camera index resolves against the header
  document's **sorted camera keys** (a `frames_binary.cameras` list, when present,
  overrides). IMU chunks (`WURLD_IMU_<id>`, §8.3 records) interleave the same
  way and also concatenate.
- **Finalize**: on clean close, writers SHOULD append a consolidated
  `WURLD_FRAMES` table (and consolidated IMU tags) so whole-file readers get one
  contiguous table, then Cues. A recording that dies mid-stream is still a valid,
  fully-posed file up to its last flushed chunk.
- **Precedence** for readers: `WURLD_FRAMES` table, if present, is authoritative;
  otherwise the concatenation of `WURLD_POSES` chunks; otherwise the JSON
  `frames` array.
- **Batch writers** (all poses known up front) MAY put the complete pose data in the
  header region instead of chunking — a JSON `frames` array in the header document,
  or a `WURLD_FRAMES` table directly after the header tag — and omit per-cluster
  chunks. Playback is equally streamable: all poses arrive before the first Cluster.
  The chunked form exists for live recording, where poses are not known at header
  time.
- Cues, when written, MUST reflect final Cluster offsets (writers that insert chunk
  tags between Clusters rebuild Cues).

### 9.1 SeekHead and range-request access (v0.4)

Batch writers SHOULD begin the Segment with a Matroska `SeekHead` (ID `0x114D9B74`)
whose entries cover: `Info`, `Tracks`, every `Tags` element in the header region, the
**first `Cluster`**, and `Cues`. `SeekPosition` values are byte offsets relative to
the Segment payload start (which includes the SeekHead itself) and SHOULD be encoded
as fixed-width 8-byte unsigned integers so the SeekHead's own size is independent of
the values it carries.

With the metadata-first layout (§9) plus a SeekHead, a range-request client needs:

1. one small read of the file head (EBML/Segment headers + SeekHead);
2. one read of the header region, whose extent is exactly `[0, first-Cluster
   position)` — yielding **all calibration and every pose without touching video
   bytes**;
3. optional reads of `Cues` and individual Clusters for random video access.

Live recordings omit the SeekHead (final positions are unknowable at header time);
a remuxing finalizer MAY add one.

## 10. Compatibility & versioning

- Unknown JSON fields and unknown `WURLD_*` tags MUST be ignored (forward
  compatibility).
- `version` is `major.minor`; minor bumps are additive-only. **1.0 freezes the
  semantics of every construct in this document**: any change to the meaning of
  an existing field, tag, record layout, or convention requires a major bump;
  1.x revisions may only add. Early readers see later files as valid
  (binary-frame files appear to have zero posed frames to a pre-§7 reader —
  writers targeting maximum compatibility should prefer JSON frames below
  ~100k frames).
- Repeated SimpleTag names: binary payloads concatenate in file order; for string
  payloads the last occurrence wins.
- A file with no `WURLD` tag is a plain chromapakz file; libraries SHOULD read it
  as a sequence with no poses.

## 11. File scope: one rig, one clock

A wurld file's atomic unit is **one rigidly-coupled sensor head on one
clock**. The container's job is to make that head's internal synchronization
unbreakable — shared timeline, shared Clusters, poses that cannot drift from
pixels or be separated from them in transit.

**Belongs in one file**: a stereo pair, a phone's RGB + LiDAR + IMU, a robot's
head assembly — sensors that are hardware- or tightly-synced, rigidly mounted,
and jointly calibrated (the `rigs` block, §8.1, describes exactly this
coupling). Multiple synchronized RGB streams for such a rig are stored as
display streams keyed by camera id (§4.4).

**Belongs in several files**: separate agents, clocks, or mountings — two
robots covering one scene, external mocap alongside an egocentric camera,
camera arrays with independent clocks, or derived artifacts (a reconstruction
stored beside its source capture). Their coupling is *soft* (estimated
transforms, estimated clock offsets), and a single container would force a
merge between producers that are naturally separate processes on separate
machines. Composition across files is a **scene manifest** concern — a
lightweight sidecar listing member files, each with a world-frame alignment
(SE(3)/Sim(3)) and a clock offset — still reserved for a future revision. The
layering mirrors USD composition over single assets: the container stays
simple and atomic; the scene lives one level above.

A **collection manifest** (§13) is a different sidecar and does not fill that
role: it indexes many files as a dataset and asserts no spatial or temporal
relationship between them at all.

Two rules follow:

- Writers MUST NOT span multiple clocks in one file. If timestamps had to be
  aligned by estimation, the sources belong in separate files under a manifest.
- Writers SHOULD NOT interleave multiple cameras' frames in a single video
  track. It is technically legal (frames carry per-frame `camera` ids and
  equal timestamps are permitted), but it breaks plain-video playback,
  misrepresents the track's frame rate, and muddies Cluster-level random
  access. Use display streams (§4.4) instead.

## 12. Non-goals

- Not an editing/composition format (that's USD's job).
- Not a delivery format for reconstructed 3D (that's glTF/3D Tiles/splats).
- Not a general robot log (that's MCAP); wurld is the *camera-centric interchange*
  that logs and datasets convert through.
- Not a multi-agent scene format — one file is one rig on one clock (§11);
  cross-agent composition belongs to the future scene manifest, not the
  container.

## 13. Collection manifest (v1.2)

A **collection** indexes many wurld files as one dataset. It is a sidecar JSON
document, not a container: every member remains an ordinary wurld file that
plays and parses alone, and deleting the manifest loses only the index.

This is deliberately **not** the scene manifest reserved in §11. A collection
claims no spatial or temporal relationship between its members — no alignment,
no clock offset, no shared world frame. Two members may be unrelated captures on
opposite sides of the planet. A reader MUST NOT infer that member poses are
comparable across files.

```json
{
  "format": "wurld-collection",
  "version": 1,
  "description": "",
  "totals": { "members": 2, "frames": 71, "posed_frames": 67 },
  "members": [
    { "uri": "captures/take00.wl.webm", "frames": 12, "posed_frames": 11,
      "cameras": ["0"], "rgb_streams": ["rgb"], "signals": ["depth"],
      "metric_scale": true, "t_start": 0.0, "t_end": 0.367,
      "width": 64, "height": 48, "bytes": 25462, "sha256": null }
  ]
}
```

- `format` MUST be `"wurld-collection"`. A reader MUST reject other values.
- `version` is the manifest schema version. A reader MUST reject a manifest whose
  version exceeds what it understands, rather than guessing.
- `uri` is a path relative to the manifest's directory, an absolute path, or an
  `http(s)` URL. Relative paths keep a collection movable as a directory.
- Every other member field is a **cache of what the member's own header says**.
  The file is authoritative: a reader that needs a guarantee MUST read the
  member. `sha256` is optional and absent unless requested, since computing it
  requires reading every byte.
- Member order defines the collection's frame ordering: global frame index `k`
  is resolved by walking members in order and accumulating `frames`.
- Members MAY disagree — different resolutions, cameras, signals, or
  `metric_scale`. A consumer requiring homogeneity MUST filter; mixing scaled
  and unscaled reconstructions is the hazard worth checking first.
