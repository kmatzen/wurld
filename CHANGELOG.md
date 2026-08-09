# Changelog

## Unreleased

### Added

- **`float16_bits` value map (SPEC §6.1)** — scene-referred HDR. An IEEE half is
  exactly 16 bits, so a lossless signal carries EXR half-float bit-exactly, one
  signal per channel: NaN, ±Inf, −0.0 and denormals all round-trip, and there is
  no `invalid` code because every bit pattern denotes a value.
- **`Sequence.signal_values(id)`** — codes through their value map, the general
  form of `depth_meters`.
- **`examples/05_hdr_exr_render.py`**, and `wurld validate` accepts the new type.
- **Multi-camera display streams (SPEC §4.4)** — `wl.write(rgb={camera_id: array})`
  stores several cameras' pixels in one file, bound by id so a reader knows which
  intrinsics apply. `Sequence.rgb_streams` / `rgb_for(id)` read them. Poses stay
  single-camera with rig-derived siblings; a lone stream keeps the conventional
  id `rgb` and binds implicitly, so existing files are unaffected.
- **HDR display track (SPEC §4.5)** — `wl.write(..., hdr={"transfer": "pq"})`;
  `Sequence.hdr` reports the signalling and `Sequence.rgb` returns `uint16`
  10-bit codes. Display-referred, and explicitly not a substitute for a
  `float16_bits` signal. The browser viewer says it cannot draw HDR colour
  rather than showing an empty pane.

Whether this beats EXR depends on temporal coherence: measured against EXR/ZIP,
13.5x smaller for a static denoised render, 1.5x when the camera moves, and
0.8x — *larger* — for raw path-traced output with per-frame Monte Carlo noise.
Denoise before archiving.


## 1.1.1 — 2026-08-06

Documentation-only republish: the package page now carries the current README
and CHANGELOG. No code changes.

## 1.1.0 — 2026-08-06

The project takes its final name: **wurld** — *the World's Unbroken Record of
Localization & Depth*. Python package `wurld`, module `import wurld`, CLI
`wurld`; files carry `WURLD*` Matroska tags and `"format": "wurld"`. First
release published to PyPI.

## 1.0.0 — 2026-08-06

First stable release. SPEC 1.0 freezes the semantics of everything below;
future additions are minor-versioned and additive (readers ignore unknown
fields and unknown `WURLD_*` tags).

The road here, in brief:

- **0.1** — the container: chromapakz WebM + a `WURLD` Matroska tag holding
  cameras (COLMAP-style models), per-frame canonical-convention poses (RDF,
  camera-to-world, wxyz, meters, seconds) and signal value maps. Converters:
  COLMAP, nerfstudio `transforms.json`, TUM RGB-D. CLI, browser viewer
  (WebCodecs point-cloud scrubber), synthetic ground-truth scene.
- **0.2** — binary frame tables (45 B/frame), multi-camera rig calibration +
  `rig_c2w`, per-frame intrinsics overrides, IMU streams (32 B/sample), Stray
  Scanner importer.
- **0.3** — fully streamable playback and recording: metadata-first layout with
  rebuilt Cues, live chunked form (`WURLD_POSES` ahead of Clusters,
  consolidated table on finalize, crash-safe), `StreamReader`/`StreamWriter`
  (the latter over chromapakz's streaming encode, added upstream for this),
  browser live record→play demo.
- **0.4** — SeekHead + `wurld.remote.fetch_header` (all poses via ranged
  reads, <2% of file bytes), Polycam raw importer.
- **0.5** — Record3D importer, MCAP/Foxglove export, progressive ranged viewer,
  nerfstudio DataParser (verified against a live install).
- **0.6** — WurldCam iOS app records wurld on-device (libvpx +
  chromapakz cross-compiled; Swift port of the stream writer; device-free
  verification harness); ffprobe shown to surface the WURLD document with
  no ffmpeg patch.
- **0.7** — random video access: cluster-independent decode (chromapakz signal
  keyframe cadence, upstream), `remote.fetch_frames`, viewer lazy scrubbing,
  on-device confidence signal.
- **0.8** — `Sequence.fetch_frames` local partial decode, EuRoC stereo+IMU
  importer (rigs + IMU exercised together), MCAP `/tf`.
- **0.9** — `Sequence.iter_frames` bounded-memory iteration, `wurld trim`,
  viewer auto-lazy, multi-RGB design proposal (ChromaPakZ #47).
- **1.0** — chromapakz pinned to the released 0.4.0; SPEC declared 1.0; LICENSE,
  CI, packaging validation. SPEC §11 codifies file scope: one rig, one clock.

Known deliberate gaps at 1.0: multi-RGB pixel storage awaits the ChromaPakZ #47
design review (calibration for extra cameras is carried today; pixels are not);
the phone importers (Stray/Polycam/Record3D) and WurldCam are validated
against synthetic fixtures and a device-free harness — not yet against
real-device captures.
