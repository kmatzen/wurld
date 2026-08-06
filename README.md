# wurld

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

## Quickstart

```sh
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"

# write a synthetic demo sequence (analytic RGBD + orbit poses)
.venv/bin/wurld demo demo.wl.webm

# inspect
.venv/bin/wurld info demo.wl.webm

# convert real datasets (auto-detects TUM / transforms.json / COLMAP layouts)
.venv/bin/wurld convert path/to/rgbd_dataset_freiburg1_desk desk.wl.webm
.venv/bin/wurld convert path/to/transforms.json scene.wl.webm
.venv/bin/wurld convert path/to/colmap_project scene.wl.webm --images path/to/images

# extract back out
.venv/bin/wurld extract demo.wl.webm out/ --format tum        # or transforms | colmap
```

Python API:

```python
import wurld as wl

seq = wl.read("demo.wl.webm")
seq.rgb                  # (T, H, W, 4) uint8, lazily decoded
seq.depth_meters(0)      # (H, W) float, NaN = invalid
seq.c2w(0)               # 4x4 camera-to-world (RDF, meters)
seq.K("0")               # 3x3 intrinsics
seq.frames[0].t          # sensor timestamp (authoritative, may be non-uniform)
```

## Browser viewer

```sh
npm install                      # fetches the chromapakz decoder
python3 -m http.server 8000      # from the repo root
# open http://localhost:8000/viewer/index.html  and drop a .wl.webm
# (or .../viewer/index.html?src=../demo.wl.webm)
```

Zero-install, WebCodecs-decoded: seekable playback with per-frame point-cloud
reprojection, camera trajectory + frustum, RGB and metric-depth panes, and the same
file playing in a plain `<video>` element.

## Layout

- `SPEC.md` — the format (v0.1)
- `wurld/` — Python reference implementation (container, conventions, EBML tag
  layer, converters, CLI, synthetic test scene)
- `tests/` — round-trip suite (`pytest`): bit-exact depth, pose fidelity, converter
  round trips, COLMAP binary parsing, validation errors
- `viewer/` — single-file browser viewer
- `LANDSCAPE.md` — the market research that motivated this project

## Status / verified

- 17/17 tests pass: depth round-trips bit-exactly through VP9; poses, timestamps, and
  intrinsics survive every converter round trip; TUM native-unit depth is bit-exact
  end to end.
- `ffmpeg`/`ffprobe` read and fully decode wurld files with zero warnings (all
  three VP9 streams); the WURLD tag is invisible to standard demuxers.
- Viewer verified in Chrome (native WebCodecs decode, no WASM).

## Roadmap (SPEC v0.2+)

Multi-camera rigs, IMU track, per-frame intrinsics, binary frame tables for 10^6+
frames, interleaved pose track for live streaming; converters for Record3D, Polycam,
Stray Scanner, ARKit straight-from-device; LeRobot / nerfstudio / Foxglove
integrations; ffmpeg demuxer patch.
