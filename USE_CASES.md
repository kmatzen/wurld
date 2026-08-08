# Pipelines, scenarios, and where wurld fits

[LANDSCAPE.md](LANDSCAPE.md) surveyed the *format* space and argued for a posed
sensor-video container. This document looks at the other axis: what people
actually do with posed RGBD, which scenarios fall out of that, and — for each —
whether wurld helps, and how. It ends with the cases where it does not.

**On the evidence here.** The pipeline survey cites sources for each area
surveyed; claims about this repository's own behaviour are verifiable by reading
the code or running the examples. Where something rests on working knowledge
rather than a citation it says so. Four scenarios ship as runnable examples under
`examples/`, marked ▶; the rest are analysis, not demonstrations.

---

## Part 1 — What the pipelines actually look like

### Producers

**Consumer phone capture.** LiDAR phones emit RGB, a low-resolution depth map, a
per-pixel confidence class and a 6-DoF pose per frame. Every app invents its own
container: Record3D writes a zip of JPEGs plus LZFSE float32 depth, Polycam and
Stray Scanner each differ again. This repository has importers for all three,
which is itself the evidence: four formats carrying the same five quantities.

**SLAM and VIO.** Systems output a trajectory, conventionally a text file — TUM
RGB-D's `timestamp tx ty tz qx qy qz qw`, scalar-last — evaluated by associating
on timestamp against ground truth. The images and depth live in adjacent
directories, related only by filename. Research rigs go further: Project Aria
records to VRS and runs cloud Machine Perception Services that return SLAM
trajectories, eye gaze and hand tracking as separate derived artefacts, with
"millimeter-accurate 6 DoF poses for every captured frame and high-frequency
(1 kHz) motion in-between frames" ([Aria docs][aria], [AEA dataset][aea]).

**Feed-forward reconstruction.** The direction that has moved most since
LANDSCAPE was written. DUSt3R, MASt3R and VGGT predict dense geometry *and*
camera parameters in a single network pass, with no per-scene SfM — VGGT
"jointly predicting camera parameters, depth, point maps, and tracks from one to
hundreds of views in a single forward pass, without any pose optimization"
([review][ff-review], [ScienceDirect][ff-sd]). This changes the producer
population: posed RGBD is now something a *model* emits, in bulk, with
per-pixel confidence and occasional total failure, usually only up to scale.

**Synthetic renderers and world models.** Blender, Isaac Sim and Habitat produce
exact poses and exact depth. More consequentially, generative world models are
now themselves bulk producers of training data: NVIDIA's Cosmos was trained on
"20 million hours of real-world data" and exists to emit physics-aware synthetic
video for robotics and AV training, and Cosmos 3 adds joint video-action
prediction ([Cosmos 3 technical report][cosmos3], [NVIDIA][nv-wam]). Genie 3
generates persistent interactive 3D environments in real time ([Genie][genie]).
The field's own framing is that progress is judged by "closed-loop usefulness,
controllability, and policy relevance" rather than visual realism — which makes
the *pose and depth* accompanying generated video the part that matters, and
that is exactly what tends to get dropped in an MP4.

### Consumers

**NeRF and 3D Gaussian splatting.** The de-facto input is a COLMAP workspace or
a nerfstudio `transforms.json`. Trainers want SfM points to initialise from —
"Gaussian splatting works much better if you initialize it from pre-existing
geometry, such as SfM points from COLMAP" ([nerfstudio][splat],
[learnopencv][locv]) — and depth-supervised variants such as DN-Splatter add a
depth prior per image ([DN-Splatter][dn]). Camera-model handling is a live
source of error: fisheye captures must be rectified to pinhole because the
rasteriser assumes it ([FIORD][fiord]).

**Robot learning.** LeRobot's v3 format moved from one Parquet and one MP4 per
episode to chunked multi-episode files with unified Parquet metadata, explicitly
for "OXE-level" scale and streaming ([LeRobotDataset v3][lerobot-v3],
[announcement][lerobot-blog]). Vision is MP4; state and action are columnar.
Depth, where it exists, is an extra modality bolted alongside.

**SLAM benchmarking.** Ground truth and estimate are both trajectories; tooling
associates on timestamp and reports ATE/RPE. The friction is conventions —
quaternion order and pose direction — not the maths.

**Robot logging and replay.** MCAP is the default rosbag2 storage format since
ROS 2 Iron (May 2023) and is "used in production by a wide range of companies,
from autonomous vehicles to drones" ([Foxglove][mcap-ros2], [mcap.dev][mcap]).
It is serialization-agnostic, row-oriented and append-only, and carries schemas
alongside data. It is the right tool for heterogeneous robot time-series — and
the reason wurld exports to it rather than competing with it.

**Dataset distribution.** The large-scale ML convention is sharded sequential
I/O: WebDataset stores samples in POSIX tar shards, which is fast to stream but
"does not support efficient random access to individual samples" without a
side index; Zarr v3 offers chunked cloud-native storage with sharding for
random-access workloads ([WebDataset on HF][webds], [Zarr][zarr]). Neither is
camera-aware — they are containers for arrays, so calibration and poses ride
along as whatever the author invented.

**Reconstruction, meshing and inspection.** TSDF fusion and similar want posed
depth in metres with invalid pixels marked. Inspection wants to open a file and
look at it. *(not independently surveyed)*

---

## Part 2 — Scenarios

Runnable examples are marked ▶. The rest are analysis, not demonstrations.

### ▶ 1. Feed-forward model output — `examples/01_feedforward_reconstruction.py`

A VGGT-style pass emits poses, depth and per-pixel confidence, fails on some
frames, and recovers geometry only up to scale. The container has to carry that
uncertainty instead of flattening it:

- `pose_valid=False` on frames the model could not localise, so a consumer skips
  them rather than training on a wrong pose;
- `confidence` as a full-resolution signal beside depth, with a `linear`
  `value_map` — these models report per-pixel reliability, not a scalar;
- `world.metric_scale=False`, which is the claim a monocular model cannot make.

The example writes 24 frames, three of them unlocalised, and reads back the
scale flag and confidence range.

### ▶ 2. Training a splat or NeRF — `examples/02_gaussian_splatting.py`

wurld does not replace `transforms.json`; trainers read that, so wurld emits it.
What one file buys is that the depth handed to a depth-supervised trainer is the
same depth the capture recorded — bit-exact, in metres, invalid pixels still
marked NaN rather than 0, which is the classic 16-bit-PNG failure where 0 means
both "no return" and "at the sensor".

The example exports `transforms.json` plus per-frame metric depth, and asserts
the RDF→OpenGL axis conversion actually happened. A silent axis flip trains a
model that renders mirrored, and nothing downstream catches it.

### ▶ 3. SLAM benchmarking — `examples/03_slam_evaluation.py`

Ground truth and estimate become the same kind of artefact, and `wurld validate`
checks both. Conventions convert once in the exporter rather than in every
consumer. Verified on the real TUM `freiburg1_desk` sequence: 572/573 poses
associated within 10 ms, **0.000 mm** translation error against the original
`groundtruth.txt`.

### ▶ 4. Robot rig with IMU — `examples/04_robot_rig_imu.py`

Extrinsics live in `rigs` once, not per frame; poses are stored for one camera
and derived for the rest, so calibration cannot drift out of sync with the
trajectory. IMU is a stream at its own rate on the same clock — resampling
200 Hz IMU to the 30 Hz video rate destroys exactly the signal a VIO consumer
wants. The example builds a 12 cm stereo rig with 200 Hz IMU and derives the
second camera's pose from the first.

### 5. Phone capture to a training set

The path this repository already implements end to end: WurldCam records
on-device, or an existing app's output is imported (`wurld convert` handles
Record3D, Polycam, Stray, TUM, COLMAP, nerfstudio, EuRoC), and the result feeds
scenario 2 or 4. The value is the count: seven input layouts, one output.

### 6. Robot-learning episodes

LeRobot v3 stores vision as MP4 and state/action as Parquet. wurld is
complementary rather than competing: it is the *camera-centric* part — posed
RGBD with calibration — while actions and rewards stay columnar. A wurld file
per episode alongside the Parquet keeps depth and calibration together instead
of scattering them, and the depth stays lossless. Not demonstrated here; the
integration exists as a staged PR against LeRobot's depth backend.

### 7. Licensing is a first-class constraint on corpus building

Not a format feature, but the thing that actually decides what a corpus can
contain, and easy to get wrong. The indoor RGBD datasets differ sharply:
**TUM RGB-D is CC BY 4.0** and redistributable with attribution; **Hypersim is
CC BY-SA 3.0**; **ARKitScenes** is under Apple's licence, non-commercial;
**Replica** and **ScanNet** carry their own terms of use ([ARKitScenes][arkitscenes]).
**EuRoC** is *In Copyright — Non-Commercial Use Permitted*, which despite
appearances grants no redistribution right at all: non-commercial *use* needs no
permission, "for other uses you need to obtain permission from the
rights-holder(s)" ([rightsstatements.org][inc-nc]).

The practical consequence for a converted corpus: TUM and Hypersim can be
republished with attribution; EuRoC, ScanNet and Replica cannot, and want a
conversion recipe plus a checksum instead of a mirror. This repository's TUM
conversion is verified bit-exact against the source PNGs, which is what makes
republishing it defensible.

### 8. Dataset distribution and streaming

The header carries every pose in one contiguous region, so a client can read
calibration and the full trajectory in two range requests without touching
video. That is what makes "index 10,000 files' trajectories" cheap. Verified
against GitHub Pages and Hugging Face, both of which return HTTP 206.

### 9. Inspection and triage

Open the file in a browser and look at it: [kmatzen.com/wurld](https://kmatzen.com/wurld/).
Or read it with tools nobody here controls — `ffprobe` prints the metadata
document, and poses come out of the WebVTT track with plain ffmpeg. See
[EXTRACTING.md](EXTRACTING.md).

### 10. Simulation-to-real comparison

Synthetic and captured sequences in one format means a pipeline consumes both
without branching. `world.metric_scale` and `gravity_in_world` carry the
distinctions that actually differ. Not demonstrated.

### 11. Long-horizon capture and archival

Bounded-memory iteration (`iter_frames`), cluster-level random access, and a
crash-safe streaming layout — a recording that dies mid-take is still valid and
fully posed up to its last flush. Demonstrated incidentally: an interrupted
device capture produced a readable 1.8 KB file.

---

## Part 3 — Where wurld is the wrong tool

A format's limits are part of its specification. These are real, and most are
consequences of deliberate choices in SPEC §11–12.

**Multi-agent scenes.** One file is one rig on one clock (§11). Two robots
observing each other need two files plus something above them to relate the
frames — a scene manifest wurld deliberately does not define. Anything requiring
a shared world frame across independently-clocked agents is out of scope today.

**LiDAR without a camera.** Depth is stored as a camera raster: a `uint16` plane
on an image grid with pinhole intrinsics. A spinning LiDAR's sweep is not that —
it is a time-ordered point stream with per-point timestamps and no image plane.
Forcing it into a range image loses the sweep structure and the per-point
timing. Use MCAP or a point-cloud format; wurld exports to MCAP for exactly this
reason.

**Geospatial survey.** No CRS, no georeferencing, no spatial index, no tiling. A
city-scale capture wants 3D Tiles — an OGC Community Standard since December
2022, using glTF 2.0 as tile content and built for streaming massive
heterogeneous 3D geospatial data ([OGC][ogc-3dt], [spec][3dt-spec]) — or COPC
for point clouds. Those are organised for spatial queries over an *area*; wurld
is organised for temporal playback of one *trajectory*. Different index, and
not a gap to be closed by adding fields.

**Non-rigid and articulated capture.** Poses are for cameras. There is nowhere
to put per-object or per-joint motion, so a deforming subject is representable
only as the pixels that saw it. Object IDs exist as a signal role; object
*poses* do not.

**Apple-native playback.** QuickTime Player, iOS Photos, Quick Look and Final
Cut cannot open the file: AVFoundation has no WebM demuxer. Desktop viewing is
VLC or IINA, browser viewing is the hosted viewer. See Part 4 for the
measurements and why the container choice stands.

**Event cameras.** Microsecond-resolution asynchronous events have no frame
raster to occupy. The container is frame-indexed at its core.

**Sparse or non-raster depth.** Depth from a sparse SfM point cloud has no dense
grid; padding it into a raster mostly stores "invalid".

**Very wide multi-camera rigs.** SPEC v1 carries one RGB track. Rig extrinsics
are supported and additional cameras' poses derive from them, but genuine
multi-camera *pixel* storage is unimplemented — the design is
[ChromaPakZ #47](https://github.com/kmatzen/ChromaPakZ/issues/47), now being
implemented. Until it lands, a stereo pair means either two files or one
camera's pixels.

**Where the frame budget actually is.** On-device capture reaches 30 fps at
256×192 with RGB, depth and confidence. Higher depth resolutions or more signals
will exceed the budget again; lossless coding of sensor noise is the cost, and
it is inherent rather than a tuning problem.

---

## Part 4 — Playback reach, HDR, and what we plan to do about them

Measured 2026-08-08 on macOS 15 (arm64), against
`docs/samples/synthetic-orbit.wl.webm`. Re-run before trusting these; players
change.

| player / stack | opens the file | renders the right track |
|---|---|---|
| `ffmpeg` / `ffprobe` | yes | all six tracks, addressable by title |
| VLC 3.x | yes | yes — RGB, verified by dumping the rendered frame |
| Chrome (WebCodecs) | yes | yes — the hosted viewer decodes natively |
| IINA | expected, **not measured** | expected — same ffmpeg core as above |
| QuickTime Player | **no** — AVFoundation `Cannot Open` | — |
| iOS Photos | **no** — same AVFoundation path | — |

The multi-track question is the one that could have quietly broken the "plays in
an ordinary player" claim: a wurld file carries three to six video tracks, and a
player defaulting to `signal-depth-hi` would show a grey gradient. VLC selects
RGB (track 1) correctly. That is checked by rendering a frame, not by reading a
track list.

### Why WebM, given that cost

The container was chosen for properties that no Apple-native combination
provides today:

- **Lossless 16-bit integer planes.** VP9 lossless is what makes bit-exact depth
  possible at all. It is the format's reason for existing.
- **Native browser decode with no WASM.** WebCodecs decodes VP9 directly, which
  is what makes the zero-install viewer work.
- **Royalty-free**, with no licensing question attached to distributing files.
- **Element-aligned streaming** with an unknown-size Segment, so a recording that
  dies mid-take is still valid — demonstrated by an interrupted device capture.
- **Matroska tags and tracks** for metadata, which is how the pose data rides
  along without a sidecar.

### What it costs, precisely

AVFoundation has no WebM demuxer, so **QuickTime Player, iOS Photos, Quick Look,
Preview and Final Cut cannot open a wurld file** — not a bit-depth or HDR issue,
and not fixable by changing the video codec inside WebM. Safari plays VP9
through WebKit's own decoder, not the system's, which is why the browser works
and the Finder does not.

This is accepted, not an oversight. Desktop viewing is VLC or IINA; browser
viewing is the hosted viewer; everything else goes through ffmpeg
([EXTRACTING.md](EXTRACTING.md)).

**Not planned:** an MP4/HEVC binding for Apple-native playback. It was measured
as feasible — HEVC Main 10 / HLG / BT.2020 in MP4 is `isPlayable`, and a
two-video-track HDR MP4 stays playable — but it is a second container binding
with the whole Matroska metadata layer to re-do against `udta`/`mebx`, and it
rests on an unresolved question: whether VideoToolbox decodes lossless HEVC at
all. Without that, the depth payload has nowhere to live, and the payload is the
point.

### HDR, in two separate senses

**Scene-referred HDR data — works today, wants a spec entry.** EXR half-float is
exactly 16 bits, so the raw bit patterns store losslessly as `uint16` signal
codes, one signal per channel. Measured on synthetic render content with
path-tracing noise (320×240 half-float RGB):

    raw half-float             450.0 KiB/frame
    zlib per frame (~EXR/ZIP)  242.7 KiB/frame    1.85x
    chromapakz lossless         38.6 KiB/frame   11.66x      17.8 ms/frame

**6.3× smaller than EXR/ZIP, bit-exact**, with NaN, ±Inf, −0.0 and denormals all
surviving. The advantage is temporal — VP9 exploits inter-frame redundancy where
EXR compresses each frame alone — so expect less on sequences with hard cuts.

*Plan:* add a `value_map` of `{"type": "float16_bits"}` so the reinterpretation
is declared rather than a private convention, with writer and reader support and
a SPEC paragraph. Small and well understood. Until then this works but no
consumer knows to do it.

**HDR10 display track — feasible, deliberately deferred.** Verified against the
libvpx we build (v1.16.0, `--enable-vp9-highbitdepth`): profile-2 10-bit encoder
init succeeds, `VP9E_SET_COLOR_SPACE` accepts BT.2020, `I42016` allocates. Three
things are missing: chromapakz pins `g_profile=0` and 8-bit `I420`; the ABI
takes `const uint8_t* rgba` and would need a 16-bit input path; and **no WebM
`Colour` element is written at all** — no transfer characteristics, primaries,
mastering metadata or MaxCLL/MaxFALL — so a 10-bit track would still be
displayed as washed-out SDR.

*Plan:* deferred. It buys HDR in browsers only, since Apple players cannot open
the container regardless, and the display track is a colour reference beside
bit-exact depth rather than the deliverable. Worth doing after the scene-referred
work, and it needs verification on real browsers, which is unavailable here.

### Multi-camera pixel storage

Genuine multi-camera *pixel* storage (stereo pairs in one file) is the remaining
structural gap; rig extrinsics and derived poses already work. The design is
[ChromaPakZ #47](https://github.com/kmatzen/ChromaPakZ/issues/47) and is being
implemented separately — not covered here to avoid duplicating that work.

---

## Sources

- [Aria Gen 1 documentation — VRS and Machine Perception Services][aria]
- [Aria Everyday Activities Dataset][aea]
- [Review of Feed-forward 3D Reconstruction: From DUSt3R to VGGT][ff-review]
- [Feed-forward 3D reconstruction with point-cloud representations][ff-sd]
- [nerfstudio — Splatfacto][splat]
- [3D Gaussian Splatting, explained (LearnOpenCV)][locv]
- [DN-Splatter: depth and normal priors for Gaussian splatting][dn]
- [FIORD: fisheye indoor-outdoor dataset with LiDAR ground truth][fiord]
- [LeRobotDataset v3.0][lerobot-v3]
- [LeRobotDataset v3.0 announcement][lerobot-blog]
- [MCAP as the ROS 2 default bag format][mcap-ros2] · [mcap.dev][mcap]
- [WebDataset on Hugging Face][webds] · [Zarr streaming concepts][zarr]
- [Cosmos 3: Omnimodal World Models for Physical AI][cosmos3] · [NVIDIA on world-action models][nv-wam] · [Genie][genie]
- [OGC adopts 3D Tiles v1.1][ogc-3dt] · [3D Tiles specification][3dt-spec]
- [ARKitScenes][arkitscenes] (Apple licence, non-commercial)
- [TUM RGB-D benchmark](https://cvg.cit.tum.de/data/datasets/rgbd-dataset) (CC BY 4.0)

[aria]: https://facebookresearch.github.io/projectaria_tools/docs/faq
[aea]: https://arxiv.org/html/2402.13349v1
[ff-review]: https://arxiv.org/pdf/2507.08448
[ff-sd]: https://www.sciencedirect.com/science/article/pii/S2096579626000203
[splat]: https://docs.nerf.studio/nerfology/methods/splat.html
[locv]: https://learnopencv.com/3d-gaussian-splatting/
[dn]: https://github.com/maturk/dn-splatter
[fiord]: https://arxiv.org/pdf/2504.01732
[lerobot-v3]: https://huggingface.co/docs/lerobot/main/en/lerobot-dataset-v3
[lerobot-blog]: https://huggingface.co/blog/lerobot-datasets-v3
[mcap-ros2]: https://foxglove.dev/blog/mcap-as-the-ros2-default-bag-format
[mcap]: https://mcap.dev/
[webds]: https://huggingface.co/docs/hub/en/datasets-webdataset
[zarr]: https://acquire-project.github.io/acquire-docs/dev/core_concepts/
[cosmos3]: https://research.nvidia.com/labs/cosmos-lab/cosmos3/technical-report.pdf
[nv-wam]: https://developer.nvidia.com/blog/pretrained-to-imagine-fine-tuned-to-act-the-rise-of-world-action-models/
[genie]: https://arxiv.org/pdf/2402.15391
[ogc-3dt]: https://www.ogc.org/announcement/ogc-adopts-3d-tiles-v1-1-as-community-standard/
[3dt-spec]: https://docs.ogc.org/cs/22-025r4/22-025r4.html
[arkitscenes]: https://github.com/apple/ARKitScenes
[inc-nc]: http://rightsstatements.org/page/InC-NC/1.0/
