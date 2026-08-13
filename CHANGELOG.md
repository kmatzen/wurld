# Changelog

## 1.5.0 — 2026-08-12

### Added — per-stream resolution (SPEC v1.3, chromapakz >= 0.10.0)

Streams no longer share one resolution: a 256×192 LiDAR depth map rides beside
full-resolution RGB instead of one being resampled to the other. Format v1.3 is
additive — a signal (or display stream) may carry its own `width`/`height`, the
file pair stays the primary display resolution, and uniform files are written
byte-identically to before.

- **SPEC.** New §4.6 defines what chromapakz's format v4 explicitly leaves to
  its wrapper: signal grids are FOV-aligned with their camera's image, so
  intrinsics scale linearly onto them. §4.1's calibration rule becomes
  per-stream (each camera at its own stream's resolution — the same statement
  as before for pre-1.3 files). Signal entries may state their geometry
  (§4.3); the chromapakz metadata stays authoritative.
- **Python.** `write()` accepts mixed-resolution arrays (each camera is checked
  against its own stream); the document self-describes signal geometry even
  when the caller never mentioned it. `SignalMeta` carries optional
  `width`/`height`, which `StreamWriter` declares to the streaming encoder.
  New `Sequence.signal_resolution(id)`, and `Sequence.K(..., signal_id=...)`
  returns intrinsics scaled to a signal's own grid. `validate` checks
  resolution agreement per stream and signal-geometry consistency.
- **Viewer.** Reads per-stream geometry from the codec metadata: the point
  cloud samples depth on its own grid with scaled intrinsics and maps colours
  across geometries; the 2D panes and the `.npy` export use each plane's own
  size. Both the buffered and lazy paths.
- **WurldCam (iOS).** The recorder no longer downscales RGB to the depth grid:
  RGB records at half the camera resolution (960×720) beside native 256×192
  depth/confidence, via the chromapakz `dc_stream_create3` streaming ABI.
  Native pin moves to v0.10.0.
- **Conformance.** New `v11_mixed_resolution` vector; pre-v4 readers must fail
  loudly on it rather than return misshapen data.

## 1.4.0 — 2026-08-11

The recommended file suffix becomes `.wurld.webm`, and the browser viewer stops
showing you the previous file when you open a new one.

No format change: `FORMAT_VERSION` stays at 1.2 and every existing file remains
valid and unchanged. The suffix has always been a recommendation, never a thing
readers key on.

### Fixed

- **`wurld index` skipped valid files named `.webm`.** SPEC §2 allows the plain
  suffix alongside the recommended `.wurld.webm`, but the default glob was
  `*.wurld.webm`, so a corpus named the other legal way indexed as nothing without
  a word. The default is now `*.webm`, which covers both.
- A `.webm` that is *not* a wurld file is now passed over in silence rather than
  reported as a broken member — a directory holding ordinary video beside
  captures is unremarkable. The two are told apart by whether the container
  parses and lacks a `WURLD` tag, not by the error text and not by sniffing for
  the string, since truncated rubbish can contain it.

### Changed

- **The recommended suffix is now `.wurld.webm`, was `.wl.webm`.** The `wl` stood
  for worldline, the format's name before it became wurld, and nothing had used
  that name for some time. OME-TIFF — the precedent SPEC §2 cites — spells its
  format's name in full as `.ome.tiff`, so this follows it properly.

  Nothing about reading changes and no existing file becomes invalid. The suffix
  has always been a recommendation: a reader identifies a wurld file by its
  `WURLD` tag, never by its name, and no code in this repository branches on the
  suffix. `.wl.webm` files keep working, `wurld index` already globs `*.webm` so
  it finds both, and SPEC §2 now records the old spelling as still valid. Only
  the defaults, docs and the conformance vector filenames have moved.
- SPEC §2 now says *why* the suffix is `.wurld.webm`: the trailing `.webm` is
  load-bearing, because every OS, player and CDN keys off the last suffix, and a
  bare `.wurld` would forfeit the property the design is built on. Tools must not
  assume the `.wurld` part is present.

### Fixed (viewer)

- **Opening a second file could leave the first one on screen.** A file the
  decoder rejected threw out of the drop handler into nothing: no error was
  shown and every pane kept displaying the previous capture, so the viewer
  looked like it had simply ignored the drop. Failures are now reported in the
  metadata line and the view is emptied.
- The RGB and depth panes were write-only — each was painted when data existed
  and otherwise left alone — so a file with no depth kept showing the previous
  file's depth map. Panes with nothing to show are now cleared, on file swap and
  per frame.
- `camera`, `depthMap` and the lazy-loading cursor survived a load. Stale
  intrinsics would have reprojected new depth into a plausible but wrong point
  cloud, and a stale cursor kept issuing ranged reads against the previous
  URL. All three are reset before the new file is applied.
- The previous file's blob URL is now revoked instead of being pinned for the
  life of the page, and re-picking the same path in the file dialog works.

  The symptom that started this — a dropped file appearing to be ignored — was
  mostly **chromapakz** refusing to decode RGB-only files at all, fixed in
  chromapakz 0.9.1 and required for the viewer to open a capture with no depth
  signal. The dependency floor moves accordingly.

### Changed (packaging)

- chromapakz floor raised to 0.9.1 in `pyproject.toml` and `package.json`.
- The hosted viewer's pinned chromapakz CDN build had drifted a minor behind the
  dependency — the published demo ran an older decoder than the tests did. It is
  now pinned to 0.9.1 and `scripts/build-pages.sh` says to keep the two in step.
- `docs/samples/synthetic-orbit` follows the suffix rename; the hosted page links
  it by name and would otherwise have 404'd.

### Changed (iOS)

- WurldCam writes `.wurld.webm` rather than `.wl.webm`.
- The camera purpose string now gives the reason and an example of the use, which
  is what App Store guideline 5.1.1 asks for; the old one restated the permission
  in jargon. Changed in both `project.yml` and `Sources/Info.plist`, which
  xcodegen keeps in step.
- `ios/APP_REVIEW_NOTES.md` answers the App Review 2.1 information request.

## 1.3.0 — 2026-08-09

Streaming: sequences that did not fit in memory now convert, in Python and in
C++. Plus a sweep for assumptions that newer features had quietly invalidated —
one display stream, 8-bit pixels, every frame posed — which turned up seven
defects across the exporters, none of which had failed a test.

Requires **chromapakz 0.9.0**, which carries two fixes written for this work:
partial decodes no longer allocate for the whole sequence, and batch encoding
runs its tracks concurrently — a real TUM conversion drops from 106 s to 87 s.

`wurld-core` (JavaScript) is unchanged since 1.2.0 and stays there.

### Changed

- **Requires chromapakz >= 0.9.0**, and the header-rewriting workaround is gone.
  wurld used to rewrite a spliced Cluster's frame count before decoding it,
  because chromapakz sized its output buffers from the header and so allocated
  for the whole sequence on every partial decode. That is fixed upstream
  (ChromaPakZ #58), so the workaround came out; the memory bounds it bought are
  still asserted by measurement rather than assumed. 0.9.0 also encodes batch
  tracks concurrently (ChromaPakZ #59), which makes conversion 1.3-1.5x faster
  wherever a lossless signal is present — a real TUM conversion went 106 s to
  87 s — with byte-identical output.

### Fixed

- **`wurld validate` now catches a truncated file.** A copy that lost Clusters —
  an interrupted transfer, a partial upload — keeps its header and pose table,
  so the document stays self-consistent, every pose is well formed, and a reader
  reports the full frame count. Only the video is gone. A 150-frame file cut
  after its first Cluster carried 30 frames of pixels, reported 150 poses, and
  produced no findings. The check compares the declared count against the blocks
  actually present (`chromapakz.frames_present`, new in 0.9.0) and also reports
  poses that reference frames past the end of the video.
- **EXTRACTING.md never mentioned the second camera.** A multi-camera file
  carries the other streams titled by camera id, and the recipe now shows how to
  reach them — with the warning that the id is the only thing binding those
  pixels to their calibration, so a stream taken by position and undistorted
  with the primary's intrinsics is quietly wrong.
- **EXTRACTING.md told readers to pull depth from `0:v:1`.** That is
  `signal-depth-hi` on a plain RGB+depth file, the *second camera's colour* on a
  stereo capture, and something else again when a file carries confidence.
  Nothing errors — the colour plane dequantizes into plausible nonsense. The
  recipe now resolves planes by track title, and `tests/test_extracting_doc.py`
  executes it against both a mono and a stereo file, checking the result matches
  the Python reader (1.2e-07 m, identical NaN pattern).

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
- **A `Collection` mixing metric and up-to-scale members warns.** Resolution,
  camera count and bit depth may legitimately differ across a corpus; scale may
  not — poses that do not share a unit will be trained on as if they did, and
  nothing downstream can tell. The CLI already said so; the API a training run
  actually uses did not.
- **`tests/test_export_matrix.py`** — every exporter against every shape of
  legal file (unposed frames, no display track, stereo, rig+IMU, HDR). Each side
  was covered and the combination was not, which is how four separate export
  defects survived. The invariant is deliberately weak: succeed, or refuse with
  a `ValueError` that explains itself — never an internal `TypeError` that reads
  as a wurld bug when the request was impossible.
- **The Foxglove MCAP export had the HDR defect too** and was missed in the
  first pass; the matrix caught it.
- **Feed-forward output could not be exported to nerfstudio or COLMAP at all.**
  Both raised "frame 9: pose not valid" on the unlocalised frames that
  `pose_valid=False` exists to represent — so scenario 1 could not feed scenario
  2, the two pipelines the format most wants to connect. Unposed frames are now
  omitted (a pose-indexed format has no place for them) and the count is warned
  rather than dropped in silence.
- **A signals-only file (`rgb=None`, legal for scene-referred HDR) died with
  "'NoneType' object is not subscriptable"** in all three image exporters. They
  now say the file has no display track and what that means for the format.
- **Exporting an HDR file to TUM, COLMAP or nerfstudio died in PIL** with
  "Cannot handle this data type: (1, 1, 3), <u2", which names neither the file
  nor the format nor a way out. It now refuses with all three: those formats
  store 8-bit images, an HDR track decodes to 10-bit codes, and wurld will not
  silently crush one into the other.
- **An HDR file exported to ROS was labelled `rgb8` while carrying uint16.**
  The declared layout was half the actual byte count, so a consumer read a
  corrupted double-width image and nothing raised. HDR images now go out as
  `rgb16`, with a warning that the values are display-referred PQ codes rather
  than linear colour — `sensor_msgs/Image` has no field to say so. Importing
  `rgb16` is refused with that reason instead of "no images found".
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
