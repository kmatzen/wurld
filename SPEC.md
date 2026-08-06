# Wurld — posed sensor video, v0.1 (working title)

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
   and versioning, but in v0.1 its values are fixed.
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

This mirrors chromapakz's own `CHROMAPAKZ` SimpleTag. Writers append the wurld
`Tags` element at the **end of the Segment payload** (after all Clusters) and rewrite
the Segment size vint; this preserves all Cue/Cluster offsets, which are relative to
the Segment payload start. Readers scan top-level Segment children for Tags elements
and select the SimpleTag named `WURLD`.

Recommended file suffix: `.wl.webm` (plain `.webm` also valid).

## 3. Canonical conventions (fixed in v0.1)

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
  "version": "0.1",
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
resolution in v0.1 (a `scale` extension is reserved for v0.2).

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
```

When a chromapakz signal already carries a `quant` spec (e.g. `inverse-depth`), the
wurld `value_map` MUST agree with it; the chromapakz spec is authoritative for
decode, wurld's copy is for consumers that read metadata only.

## 7. Compatibility & versioning

- Unknown JSON fields MUST be ignored (forward compatibility).
- `version` is `major.minor`; minor bumps are additive-only.
- A file with no `WURLD` tag is a plain chromapakz file; libraries SHOULD read it
  as a sequence with no poses.
- Multi-camera rigs, IMU streams, per-frame intrinsics overrides, binary frame tables
  (for >10^6 frames), and interleaved streaming pose tracks are reserved for v0.2+;
  the JSON layout above is designed so all of these are additive.

## 8. Non-goals

- Not an editing/composition format (that's USD's job).
- Not a delivery format for reconstructed 3D (that's glTF/3D Tiles/splats).
- Not a general robot log (that's MCAP); wurld is the *camera-centric interchange*
  that logs and datasets convert through.
