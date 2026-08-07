# WurldCam — iOS posed-RGBD recorder

Records ARKit LiDAR captures (RGB + metric depth + confidence + camera poses)
into a **Record3D-compatible `.r3d`** file that wurld ingests directly:

```sh
wurld convert wurld-2026-….r3d out.wl.webm     # needs [record3d] extra
```

## Why .r3d and not wurld directly (v1)

The wurld container carries chromapakz VP9-lossless tracks, which needs
libvpx built for iOS. v1 sidesteps that: JPEG frames + LZFSE depth/confidence
(Apple's Compression framework — zero third-party dependencies), the exact
layout the existing importer is tested against. v2 will embed the chromapakz C
core (`dc_stream_*` streaming ABI) and weave WURLD pose tags on-device.

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

## Status

Compiles against the iOS SDK; **not yet validated on a device** — the first
real capture should be round-tripped through `wurld convert` and eyeballed
in the viewer (trajectory shape, depth alignment) to confirm conventions.
