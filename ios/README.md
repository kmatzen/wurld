# WurldCam — iOS posed-RGBD recorder

Records ARKit LiDAR captures (RGB + metric depth + confidence + camera poses)
into a **Record3D-compatible `.r3d`** file that wurld ingests directly:

```sh
wurld convert wurld-2026-….r3d out.wurld.webm     # needs [record3d] extra
```

## Two formats (toggle in the UI)

- **`.wurld.webm` (default, v2)** — wurld recorded directly on-device:
  chromapakz VP9-lossless depth via the `dc_stream_*` C ABI (libvpx cross-built
  for iOS), poses woven live as `WURLD_POSES` chunk tags with a consolidated
  table on stop. Crash-safe: an interrupted take is a valid, fully-posed file up
  to the last flushed cluster. RGB rides at the depth grid (256x192), near/far
  0.1–12 m. Build the native libs once: `ios/scripts/build-native.sh`.
- **`.r3d` (v1)** — Record3D-compatible zip (JPEG + LZFSE depth/confidence),
  zero native deps, converted on the desktop with `wurld convert`.

The v2 pipeline is verified without a device: `ios/scripts/verify-pipeline.sh`
compiles the app's own writer + encoder for macOS, records a synthetic take, and
validates poses/depth/layout with the Python wurld reader and ffmpeg.

Design notes:
- Frames stay in **sensor (landscape) orientation** so pixels and
  `ARCamera.transform` poses agree — the importer's ARKit RUB→RDF conversion
  assumes unrotated frames. (Record3D-the-app rotates to portrait; the format
  fields don't care.)
- RGB is halved to 960×720 by default (`rgbScale`), K scaled to match; depth
  stays at native LiDAR 256×192.
- ~30 fps target cadence; ZIP uses store-mode (payloads are already
  compressed). Keep takes under 4 GB (no ZIP64).

## Build

```sh
brew install xcodegen
cd ios/WurldCam
xcodegen                      # generates WurldCam.xcodeproj
open WurldCam.xcodeproj   # set your signing team, run on a LiDAR device
```

Requires an iPhone/iPad with LiDAR (Pro models). Not testable in the simulator
(no ARKit camera).

To put a build on a connected phone without opening Xcode:

```sh
ios/scripts/install-device.sh          # finds the paired device itself
```

Note this is **not** `archive.sh`. That produces an App Store build signed for
distribution with `get-task-allow=false`, which a device refuses to install — it
is for upload, not for running. `install-device.sh` builds Debug with an Apple
Development identity, the only kind that sideloads, and checks that before it
tries. The phone must be unlocked and either plugged in or reachable on the
network; `xcrun devicectl list devices` shows `tunnelState`, and `disconnected`
means the install fails with a connection reset however good the build is.

## Status

Compiles against the iOS SDK; the recording pipeline (encoder, pose weaving,
conventions) is validated on macOS via the harness. **ARKit on hardware is the
one untested link** — the first real capture in either format should be
eyeballed in the viewer (trajectory shape, depth alignment).
