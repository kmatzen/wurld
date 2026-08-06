# Spatial Data Formats — Landscape & Gap Analysis

*Researched August 2026 via six parallel deep-dives: 3D asset interchange, neural
representations, geospatial/point clouds, capture/sensor data, AI/world models, and
web/engine delivery. Full raw findings summarized here; links preserved.*

## The one-paragraph thesis

Spatial data lives in roughly six mutually incompatible layers — capture containers,
pose conventions, depth encodings, robot-episode formats, generated-world exports, and
eval inputs — connected only by lossy, ad-hoc, per-app bridges. The *output* side of 3D
(delivery of meshes and splats) is standardizing fast (glTF 2.1, KHR_gaussian_splatting,
3D Tiles 2.0, OpenUSD Core Spec 1.0). The *input* side — posed sensor video, the raw
material of 3D reconstruction, robot learning, and world-model training — has **no
standard at all**. Every serious gap found in this research is either a **codec**, a
**container**, or a **metric**. All three are your home turf.

---

## Landscape summary by area

### 1. 3D asset interchange (glTF, USD, FBX, materials)
- **glTF** dominates delivery; glTF 2.1 (Q4 2026 target) adds 64-bit GLB, composition,
  visibility; KHR_interactivity submitted for ratification July 2026. But glTF is
  explicitly *not* an interchange format (baked animation, hardcoded materials).
- **OpenUSD** won authoring/composition; AOUSD Core Spec 1.0 shipped Dec 2025 (animation
  deferred to 1.1). Chronic complaints: 200 MB SDK, >10 MB WASM, per-DCC subset
  roulette, lossy round-trips. Godot literally cannot ship it.
- **FBX**: zombie incumbent; **ufbx** (solo-dev, single-file MIT loader, adopted by Godot
  4.3 and Blender 4.5) is the canonical proof that one expert can fill a format gap.
- **Materials Babel**: glTF PBR ≠ OpenPBR ≠ Standard Surface ≠ MDL ≠ Substrate; .sbsar
  procedural materials have no open interchange (AWE 2025 panel devoted to exactly this).
- USD↔glTF conversion is lossy both directions; only heroic one-off tools exist
  (pablode/guc; Google's converter is dead). Khronos/AOUSD liaison produces guidance,
  not code.

### 2. Neural representations (splats, NeRF)
- Static splat "format war" is consolidating: **SPZ** (Niantic, Khronos-blessed) for
  interchange, **SOG** (PlayCanvas, image-backed) for web delivery, raw PLY as archive.
  KHR_gaussian_splatting release candidate Feb 2026, ratification ~Q2 2026.
- The base Khronos spec explicitly **excludes animation, streaming, and LOD**.
- **No streaming/random-access container**: four incompatible LOD/streaming answers
  shipped in 2026 alone (World Labs `.rad`, PlayCanvas Streamed-SOG, Cesium 3D Tiles
  splat tiling, ksplat progressive).
- **Dynamic/4D splats have zero format story** — Gracia streams at ~80 Mbit/s
  (pre-codec-era bitrates); a dozen incompatible research codecs; MPEG Gaussian Splat
  Coding CfP: proposals due **January 2027** — an open standards window.
- Color science unspecified everywhere (implied sRGB, exposure baked in, no HDR story);
  KHR reserves a `colorSpace` enum but defines almost nothing. SH silently destroyed by
  common tools (SuperSplat export). No provenance/generation-loss tracking.
- NeRF checkpoints are dead as interchange; industry answer was "convert to splats."

### 3. Geospatial & point clouds
- Cloud-native wave (COG, PMTiles, STAC, GeoParquet) fully covered 2D; in 3D only COPC
  exists. **No single-file, range-request-streamable format for tiled meshes** — the
  "3tz" issue (CesiumGS/3d-tiles #422) has been open ~7 years; PMTiles proved the
  demand pattern and adoption path in 2D.
- **E57** — the neutral pivot of the entire terrestrial-scan economy (NavVis, Leica,
  Faro, Matterport) — is frozen since 2011: XML+binary dual parse, no streaming, RAM
  blowups >500M points, vendor-folklore structured-scan semantics. No "COPC for E57"
  exists.
- Trajectories/pose-graphs: de facto standard is **SBET, a closed Applanix binary**. No
  open, indexed, streamable pose format links raw logs (MCAP) to derived products.
- MCAP won the *time* axis (ROS 2 default); nothing standardizes the *space* axis.
- Splat georeferencing (CRS metadata in SPZ/SOG) is unsolved in any open format.

### 4. Capture & sensor data ← *strongest convergence*
- **There is no standard for "video + per-frame pose + depth + semantics."** Every
  producer invents a directory layout (Record3D, Polycam, Stray, ARKitScenes,
  ScanNet++, Aria VRS+MPS CSVs, Ego-Exo4D, DL3DV, ViPE); every consumer writes N
  parsers (nerfstudio maintains per-app dataparsers). kapture (Naver) tried a unified
  pivot and stalled outside localization research.
- **COLMAP binary model** is the de-facto pose interchange but has no timestamps, no
  depth, no IMU, no metric scale, no gravity; OpenCV-vs-OpenGL convention flips are a
  recurring bug class (nerfstudio #2402, #2182, #3748).
- **Lossless uint16 depth cannot ride mainstream video codecs** — labs ship millions of
  16-bit PNGs. LeRobot PR #455 was closed with: *"no suitable codec was found to
  preserve absolute depth values without normalization"*; follow-up demand in issue
  #1144. This is literally chromapakz's function.
- Depth semantics underspecified even when shipped: mm-uint16 vs float32-m vs disparity,
  confidence conventions inconsistent, ScanNet++ ships no LiDAR intrinsics.
- MV-HEVC spatial video (mass-produced by every iPhone 15 Pro+) is a reconstruction
  dead-end: no poses, no depth; no browser outside visionOS Safari can decode it.

### 5. AI & world models
- Fei-Fei Li, Nov 2025: *"Where is the data for spatial intelligence? It's all in our
  heads."* The stated #1 bottleneck of the field is spatial training data.
- World models emit siloed or unusable representations: Genie 3 outputs only video
  frames; Marble exports splats+meshes but semantics/articulation are future work;
  Marble→Isaac Sim is a manual multi-tool conversion chain.
- Training-side de facto standards: WebDataset tar shards (Cosmos/NeMo), LeRobotDataset
  v3 (Parquet + MP4 — **no depth stream, no intrinsics/extrinsics in the video**),
  RLDS/OXE widely criticized, ARIO as academic alternative. Conversions drop
  depth/calibration modalities.
- Spatial eval is fragmented: VLMs score <50% on VSI-Bench; benchmark proliferation with
  no shared harness; E3D-Bench covers 16 geometric FMs but the documented failure regime
  (high-res, many views, large scenes) is untested; **nobody benchmarks world-model 3D
  exports geometrically**; WorldScore is static-only; LoViF 2026 is openly soliciting 4D
  quality metrics.

### 6. Web & engine delivery
- WebGPU is Baseline (Jan 2026); `<model>` element is Apple-only and USDZ-only;
  iOS remains the web-AR dead zone.
- glTF is not streamable — no ABR/DASH equivalent for 3D; 3D Tiles is the only deployed
  analog.
- **No browser can decode MV-HEVC** outside visionOS Safari; x265 and NVIDIA SDK now
  encode it; only closed/server-side converters exist (SpatialGen).
- WebCodecs is crippled >8-bit (10-bit decode returns `format: null`; w3c/webcodecs
  #427) — chromapakz's bit-packing approach is the known workaround.
- Interactivity doesn't travel (KHR_interactivity just submitted; USD has no ratified
  behavior model); no pixel-accurate rendering contract between viewers.

---

## The gap shortlist (scored for you specifically)

Your differentiated assets: lossless video-codec engineering with browser-native
WebCodecs delivery (chromapakz), color science (chromacal, framewright), 3D geometry
eval at paper-match rigor (plumbline), C++/Python/JS triple implementations, CV research
background.

### ★ 1. The posed-sensor-video container ("the missing interchange")
One layer above chromapakz: a spec + reference implementation for **video + per-frame
pose + intrinsics/distortion + timestamps + depth + confidence + optional semantics**
in one seekable, streamable, ordinary-player-compatible file (Matroska/WebM profile or
ISOBMFF tracks), with explicit coordinate conventions and metric scale.

- **Why it wins**: three markets are independently reinventing this bundle *right now* —
  robot learning (LeRobot v3 has no depth/pose-in-video), world-model training
  (ViPE/Cosmos emit ad-hoc npz+mp4), and splat/NeRF capture (per-app nerfstudio
  parsers). No incumbent. kapture stalled; COLMAP can't grow into it; MCAP is a log,
  not a video.
- **Concrete adoption wedges, cheap to ship**: LeRobot depth backend (issue #1144 is an
  open door), nerfstudio dataparser, Foxglove/MCAP schema, ffmpeg demuxer, converters
  to/from COLMAP / transforms.json / TUM / EuRoC / Record3D / Polycam / Stray / VRS.
- **Demo that sells it**: a zero-install browser viewer — seekable posed-RGBD playback
  with per-frame point-cloud reprojection via WebCodecs + WebGPU. Nothing comparable
  exists.
- chromapakz becomes the payload layer instead of a codec looking for a market.

### ★ 2. Video-codec-backed splat delivery + the 4D splat codec
Nobody ships a browser-native path: Morton/atlas-packed splat attribute planes →
AV1/HEVC → WebCodecs hardware decode → GPU textures, with per-attribute rate-distortion
control. Extends naturally to 4D: attribute planes become video tracks → fMP4/DASH/CMAF
streaming of dynamic splats. MPEG CfP (proposals due Jan 2027) is both a distribution
channel and a deadline; SOG proved image-backed packing works but insists on lossless
WebP; the V-PCC amendment targets ISO pipelines, not browsers.
- Bigger upside, more competitive risk (PlayCanvas, Niantic, Cesium, World Labs, MPEG
  all orbiting). Best entered *after* #1 establishes the container credibility, or as a
  CfP response.

### 3. "PMTiles for 3D Tiles" / cloud-optimized single-file 3D
Seven-year-open issue, proven adoption pattern, pure systems work, no GIS lore needed.
Pragmatic, well-scoped, but geospatial-niche and Cesium/Bentley may eventually claim it.
Same pattern applies to a "COPC for E57" (structured scans: poses + panoramas + depth
panoramas + octree, range-request streamable) — every TLS vendor funnels through frozen
2011-era E57.

### 4. MV-HEVC in the browser
WASM/WebCodecs-assisted MV-HEVC decode + WebGPU stereo renderer — the only way to play
iPhone spatial video on Chrome/Quest/desktop web. Narrow, unclaimed, perfect skill fit;
a strong wedge project but a feature, not a platform. (Pairs well with #1: stereo →
disparity → depth into the container.)

### Anti-recommendations
- A new grand-unified 3D format (STF shows the fate); competing with ufbx; 3MF core
  work (ISO done); GeoZarr-style committee plays (gridlocked); V3C/MPEG volumetric
  tooling (licensor-owned moat, no indie demand); event-camera unification
  (vendor-driven, crowded); metrics/eval work (ruled out by preference).

---

## Recommendation

**Build #1 — the posed-sensor-video container — and keep #2 (4D/WebCodecs splat codec)
as the follow-on once the container has traction.**

The reasoning: the biggest names in AI say spatial *data* is the bottleneck; the data
exists but is trapped in incompatible per-app layouts; the missing piece is exactly a
codec+container problem; and you have already built the hardest sub-component (lossless
16-bit-in-video with browser decode). The window is 2026-shaped: LeRobot v3 just
shipped without depth, the KHR splat extension shows the standards moment for the
*output* side, and nobody has claimed the *input* side.

Distribution without an eval play: the adoption wedges are all builder-facing —
ecosystem PRs where practitioners already are (LeRobot #1144, nerfstudio dataparser,
Foxglove/MCAP schema, ffmpeg demuxer) plus the zero-install browser scrubber as the
shareable demo. Converters do the marketing: anyone who uses the CLI to escape a
per-app format has already adopted the container.

First four shippable milestones, each independently valuable:
1. Spec v0 + C++/Python/JS reference implementation (chromapakz tracks + pose/intrinsics/
   timestamp metadata track), with a coordinate-convention validator built in.
2. Converters: COLMAP ⇄, transforms.json ⇄, TUM/EuRoC ⇄, Record3D/Polycam/Stray →.
3. The browser demo: seekable posed-RGBD scrubber with live point-cloud reprojection
   (WebCodecs + WebGPU).
4. Ecosystem PRs: LeRobot depth backend (#1144), nerfstudio dataparser, Foxglove schema.
