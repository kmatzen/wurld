# Changelog

## Unreleased

### Added

- **C++ streaming writer** (`cpp/include/wurld_stream.hpp`) — the incremental
  counterpart of `attach()`, which rebuilds a whole file in memory. Metadata is
  woven between the encoder's Clusters as they arrive, so peak memory is one
  Cluster plus 45 bytes per pose, and a recording killed mid-flight still reads
  back what preceded the interruption (asserted by truncating one). Layout is
  SPEC §9's live form, byte-compatible with the Python `StreamWriter`; the two
  are checked against each other on the same captured encoder chunks.
- **`wurld.stream.write_streaming`** — writes a wurld file from an *iterator* of
  frames, holding one frame rather than the sequence. `container.write` takes
  whole arrays, which is fine for a 573-frame TUM capture and impossible for a
  real EuRoC one.
- **`StreamWriter` supports multi-camera display streams** (`rgb_streams=`,
  `add_frame(rgbs=...)`), the streaming counterpart of `write(rgb={id: array})`.
  Stream ids are validated against the declared cameras at construction, before
  any encoding.
- **The EuRoC importer picks its writer by size** (`wurld.stream.should_stream`):
  streaming when materialising would take over a quarter of memory, the exact
  batch path otherwise, and `streaming=True/False` to force either.
  `from_euroc(..., max_frames=N)` converts a prefix. The full real V1_01_easy sequence now converts in 284 s at a **68 MB**
  peak, against the 8.4 GB it would have needed materialised — it previously
  could not be converted at all on an 8 GB machine. Its poses now live in the
  binary frame table, so they are float32 (~1e-7 on metre-scale translations)
  where the batch path used float64 JSON for short sequences; timestamps stay
  float64.
- **`tests/test_real_tum.py`** — the TUM claim quoted in three documents was a
  one-off measurement that nothing re-checked. It now runs against the real
  `freiburg1_desk` download: 572/573 poses associated within 10 ms at
  **0.000000000 mm**, poses stored verbatim rather than merely close, depth
  within 2e-3 relative of the source 16-bit PNGs, and TUM's `0` (no return)
  still NaN. `scripts/fetch_tum.sh` fetches the sequence (CC BY 4.0, not
  redistributed); the tests skip without it, and a weekly CI job runs them.

### Verified

- **The EuRoC importer, against the real V1_01_easy download** — not just the
  fixture. Poses recomputed independently from the raw ground-truth csv agree to
  3.2e-13, the stereo baseline comes out at 11.01 cm, and the 21 frames before
  ground truth starts stay unposed. Real EuRoC csvs are CRLF, which the importer
  strips and an LF-only fixture never exercised.

## 1.2.1 — 2026-08-09

Memory fixes for streaming and random access. No format change, no API change;
`wurld-core` (JavaScript) is unaffected and stays at 1.2.0 — its decoder is
pull-based and never had the defect.

### Fixed

- **`Sequence.iter_frames` dropped every display stream but the primary.** A
  streaming consumer of a stereo file silently saw one eye — which looks like a
  working conversion until someone needs the other camera. It now yields
  `rgbs` (`{camera_id: plane}`) alongside `rgb`.
- **The ROS 2 exporter dropped every camera but one**, for the same reason, and
  decoded the whole file to do it. It now streams, exports a topic per display
  stream, and puts a `/tf` frame on each camera — the second derived from the
  rig, so a stereo pair keeps its calibrated baseline (measured on real EuRoC:
  11.01 cm, constant to four decimals). Found by exporting a real stereo
  sequence and noticing only one `image_raw` topic came out.
- **Collections streamed only the primary eye** of a stereo member, the same
  defect as the ROS exporter's. `iter_frames(fields=("rgb",))` now also yields
  `rgbs`; `rgb` stays the primary so single-camera consumers are unaffected.
- **Exporting a multi-camera file to TUM, COLMAP or nerfstudio warns.** Those
  formats hold one camera's images, so dropping the rest is correct — doing it
  without a word was not.
- EuRoC dropped frames whose cam1 image was missing without saying so; it now
  warns with the count. Zero on a real hardware-synced sequence, non-zero on a
  trimmed one — and an unexplained frame count is hard to chase later.
- **Streaming held a whole file, not a Cluster.** `Sequence.iter_frames`
  documented bounded memory and delivered it for packets but not for buffers: a
  spliced single-Cluster file still advertised the whole sequence's frame count,
  so chromapakz sized its output arrays for the entire file. Measured on a
  600-frame 320x240 sequence: **277 MB per Cluster, now 14.7 MB**. The spliced
  header now carries the Cluster's own frame count, and drops the wurld tags a
  decoder never reads.
- **`Collection.iter_frames` used the whole-member decode**, so peak memory
  tracked the longest member (291 MB on the fixture above) and briefly doubled
  at member boundaries. It now uses the bounded iterator and releases each
  member before opening the next: **581 MB -> 69 MB**.
- **Random access had the same defect, and worse.** `remote.fetch_frames` (and
  so `Sequence.fetch_frames`, ranged HTTP reads, and `Collection.frame`) spliced
  the original header too, so fetching *one* frame from a 600-frame 320x240 file
  allocated **279 MB**; now **16.4 MB**. Partial decode exists to make reading a
  few frames of a long file cheap, which it was not.
- **Metadata-only collection iteration read pixel bytes.** It now stops at the
  header region: **291 MB -> 1.0 MB**.
- Frames yielded from a Cluster are copied rather than aliased, so a shuffle
  buffer cannot pin every Cluster it has seen.

Nothing failed while these were wrong — frames were correct and in order, just
an order of magnitude more expensive than claimed. `tests/test_streaming_memory.py`
asserts the bounds, and `scripts/bench_collection.py` produces the numbers.

### Added

- **`scripts/bench_collection.py`** — measures a collection at scale, each case
  in its own process so peak RSS is attributable. Results for 10,000 members are
  in USE_CASES scenario 7.

## 1.2.0 — 2026-08-08

Three reader implementations that provably agree, a C++ writer, datasets built
from many files, and a ROS 2 bridge — plus HDR, stereo and scene-referred
signals in the format itself.

The document `version` field moves from `0.4` to **`1.2`**, matching the SPEC
revision it has always been meant to track (SPEC §10). Nothing reads it as a
gate, and §10's forward-compatibility rules already cover it — early readers see
later files as valid — so this is a self-description fix, not a break. Files
written by wurld 1.1.1 and by the WurldCam build currently in review still say
`0.4` and remain readable.

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
- **ROS 2 bridge** (`wurld ros2 export|import`, `pip install 'wurld[ros2]'`) —
  a real rosbag2 of CDR-encoded `sensor_msgs`/`tf2_msgs`, so `ros2 bag play`
  works and nodes can subscribe. The existing MCAP export writes Foxglove
  jsonschema channels, which Foxglove reads but no ROS node can. Handles the two
  conventions that fail silently: ROS quaternions are xyzw where wurld's are
  wxyz, and ROS optical frames are RDF (REP 145), so `world -> *_optical_frame`
  is `c2w` unconverted. Depth is `32FC1` metres so NaN survives. IMU declares
  `orientation_covariance[0] = -1` rather than shipping a believable identity.
  Import is lossy and says so: depth is requantized (<5 µm measured) and images
  re-encode through VP9.
- **C++ writer** (`cpp/include/wurld_write.hpp`) — attaches calibration, poses
  and IMU to an already-encoded WebM, keeping the zero-dependency property:
  chromapakz encodes, wurld attaches. Rebuilds Cues at the shifted offsets and
  regenerates the SeekHead, and replaces previous wurld tags on re-attach while
  leaving foreign tags alone. Verified byte-identical to the Python packers, and
  its output satisfies `wurld validate` and reads identically through all three
  readers. `wurld_attach` drives it from the command line.
- **Conformance corpus** (`conformance/`) — small vectors plus the parse a
  reader must produce, checked against the Python, JavaScript and C++ readers by
  one harness. Expectations are generated from intent rather than captured from
  a reader, so the corpus cannot enshrine a shared bug. Ships in the npm package
  so a JS consumer can verify their own build.
- **`readDocument` / `resolvePoses` / `readImuStreams` in `wurld-core`** — the
  SPEC §9 pose-precedence chain now lives in the published JS library rather
  than only in the viewer page. Anyone installing the package previously had to
  reimplement it, and getting it wrong means a phone recording reads as zero
  poses — which is exactly the bug the viewer once shipped.
- **C++ reader** (`cpp/include/wurld.hpp`) — single-header C++17, zero
  dependencies, reading calibration, poses, timestamps, signals, rigs and IMU.
  Deliberately does not decode pixels: that needs libvpx and chromapakz, and the
  value here is dropping onto a robot without a codec stack. `cluster_start`
  points at the pixels for consumers that want them. Traversal seeks over
  payloads, so file size does not drive I/O. Checked against the Python reader
  field-by-field rather than against its own expectations.
- **Collections (SPEC §14)** — many wurld files as one dataset. `wurld index`
  builds a manifest; `Collection` gives global frame addressing across members
  and sharded streaming. Indexing reads headers only: a file with 100x the
  pixels of another, at the same frame count, indexes for the same ~8 KiB.
  Members stay ordinary playable wurld files — a collection is a sidecar, not a
  container, and asserts nothing about how members relate in space or time
  (that remains the separate scene-manifest concern reserved in §11).
- **`Collection.verify()` / `wurld collection --verify`** — re-reads every
  member's header and reports drift. Global frame indexing is computed from the
  manifest's cached counts, so a member that gained or lost frames silently
  shifts every index after it while `locate()` keeps returning an answer.
  `--checksum` also compares recorded hashes, which catches content changes a
  header check cannot see.
- **`wurld.integrations.torch_data`** (`pip install 'wurld[torch]'`) —
  `WurldIterableDataset` shards across DataLoader workers *and* distributed
  ranks, decoding each member exactly once; `WurldFrameDataset` is map-style for
  evaluation. Sharding is tested with real multi-worker DataLoaders asserting
  disjointness and completeness, because a bad split does not crash — it
  silently trains on duplicated frames while the loss curve looks healthy.
- **`examples/08_collection_training.py`**, which verifies its own shard split.
- **Viewer exports** — `poses.csv`, `imu_<id>.csv`, `wurld.json` and a per-frame
  `depth_NNNNN.npy` (float32 metres, NaN where there was no return). These cover
  precisely what ffmpeg cannot reach: binary pose tables and IMU streams. Poses
  come through the full SPEC §9 precedence chain, so binary tables export like
  JSON ones, and unposed frames are omitted rather than written as identity.
- **Camera picker in the viewer** — appears only for multi-stream files, and
  drives both the RGB pane and the point-cloud colours. Showing only the primary
  without saying so hid half a stereo file.
- **HDR display track (SPEC §4.5)** — `wl.write(..., hdr={"transfer": "pq"})`;
  `Sequence.hdr` reports the signalling and `Sequence.rgb` returns `uint16`
  10-bit codes. Display-referred, and explicitly not a substitute for a
  `float16_bits` signal. The browser viewer says it cannot draw HDR colour
  rather than showing an empty pane.
- **`examples/06_stereo_rig.py`** — both eyes, shared depth, rig-derived pose.

On lossless video versus EXR, since `float16_bits` invites the comparison:
whether it wins depends on temporal coherence, not on the codec. Measured
against EXR/ZIP — 13.5x smaller for a static denoised render, 1.5x when the
camera moves, and 0.8x (*larger*) for raw path-traced output with per-frame
Monte Carlo noise. Denoise before archiving.

### Changed

- **JS `unpackFrames` validates like the other readers.** A buffer length that
  is not a whole number of records, or a camera index with no camera behind it,
  now raises instead of yielding a frame whose camera is `undefined`. Found by
  the conformance work: Python and C++ already rejected both.
- **EuRoC importer carries both eyes.** cam0 and cam1 ship as display streams
  keyed by camera id, now that multi-stream exists; `--mono` (or `stereo=False`)
  restores the previous single-track output, which is byte-identical in layout —
  one track still named `rgb`. Calibration for both cameras and the `body` rig
  are recorded either way.
- **EuRoC poses are interpolated, not snapped.** Ground truth is 200 Hz on a
  clock unrelated to the 20 Hz shutter, so the nearest sample sat up to 2.5 ms
  away — about a millimetre of systematic error on every frame. Poses now
  interpolate linearly on position and SLERP on rotation; frames the ground
  truth does not bracket stay `pose_valid: false` rather than being clamped to
  the nearest pose, which would invent a stationary camera.
- EuRoC nanosecond timestamps parse as integers before conversion. The absolute
  epoch in float64 seconds still quantises to ~238 ns — inherent to the
  representation, far below sensor accuracy — so cam0/cam1 pairing compares the
  integer nanoseconds instead.
- **Document `version` is now `1.2`** (was `0.4`), tracking the SPEC revision as
  §10 always specified. Additive-only; no reader gates on it.

### Verified

- **HDR10 playback measured** on macOS: Chrome and IINA both honour the PQ
  transfer function, established by an A/B of two files with byte-identical
  decoded pixels differing only in colour tag. Appearance on a true HDR panel
  remains unverified — the measurement ran on a 500-nit LCD.
- **Three readers agree** on the conformance corpus, and the C++ reader is
  additionally diffed field-by-field against Python on the example files.
- **Sharding is disjoint and complete** across ranks x DataLoader workers, with
  real multi-worker DataLoaders rather than simulated ones.


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
