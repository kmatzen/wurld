# Pipelines, scenarios, and where wurld fits

[LANDSCAPE.md](LANDSCAPE.md) surveyed the *format* space and argued for a posed
sensor-video container. This document looks at the other axis: what people
actually do with posed RGBD, which scenarios fall out of that, and — for each —
whether wurld helps, and how. It ends with the cases where it does not.

**On the evidence here.** The pipeline survey below cites sources for four areas
I researched directly; the rest draws on the format work in this repository
(importers, exporters and the specification), which is verifiable by reading the
code. This session's web-search budget ran out partway, so areas marked *(not
independently surveyed)* rest on working knowledge rather than a citation, and
should be treated accordingly. Four of the scenarios ship as runnable examples
under `examples/`; the rest are described but not demonstrated, and are marked.

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

**Synthetic renderers.** Blender, Isaac Sim, Habitat produce exact poses and
exact depth. The container question is trivial; the interesting part is that
synthetic and real data should be interchangeable downstream, which they are not
when each renderer invents an export layout. *(not independently surveyed)*

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

**Reconstruction and meshing, dataset curation, browser inspection.** TSDF fusion
and similar want posed depth in metres with invalid pixels marked. Curation wants
streaming and random access without downloading everything. Inspection wants to
open a file and look at it. *(not independently surveyed)*

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

### 7. Dataset distribution and streaming

The header carries every pose in one contiguous region, so a client can read
calibration and the full trajectory in two range requests without touching
video. That is what makes "index 10,000 files' trajectories" cheap. Verified
against GitHub Pages and Hugging Face, both of which return HTTP 206.

### 8. Inspection and triage

Open the file in a browser and look at it: [kmatzen.com/wurld](https://kmatzen.com/wurld/).
Or read it with tools nobody here controls — `ffprobe` prints the metadata
document, and poses come out of the WebVTT track with plain ffmpeg. See
[EXTRACTING.md](EXTRACTING.md).

### 9. Simulation-to-real comparison

Synthetic and captured sequences in one format means a pipeline consumes both
without branching. `world.metric_scale` and `gravity_in_world` carry the
distinctions that actually differ. Not demonstrated.

### 10. Long-horizon capture and archival

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

**Geospatial survey.** No CRS, no georeferencing, no spatial index, no tiling.
A city-scale capture wants 3D Tiles or COPC, which are built for spatial
queries over areas rather than temporal playback of one trajectory.

**Non-rigid and articulated capture.** Poses are for cameras. There is nowhere
to put per-object or per-joint motion, so a deforming subject is representable
only as the pixels that saw it. Object IDs exist as a signal role; object
*poses* do not.

**Event cameras.** Microsecond-resolution asynchronous events have no frame
raster to occupy. The container is frame-indexed at its core.

**Sparse or non-raster depth.** Depth from a sparse SfM point cloud has no dense
grid; padding it into a raster mostly stores "invalid".

**Very wide multi-camera rigs.** SPEC v1 carries one RGB track. Rig extrinsics
are supported and additional cameras' poses derive from them, but genuine
multi-camera *pixel* storage is unimplemented — the design is filed as
[ChromaPakZ #47](https://github.com/kmatzen/ChromaPakZ/issues/47) awaiting
review. Today a stereo pair means either two files or one camera's pixels.

**Where the frame budget actually is.** On-device capture reaches 30 fps at
256×192 with RGB, depth and confidence. Higher depth resolutions or more signals
will exceed the budget again; lossless coding of sensor noise is the cost, and
it is inherent rather than a tuning problem.

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
