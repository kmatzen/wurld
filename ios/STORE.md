# WurldCam — App Store submission

Everything buildable is done. What remains needs your Apple account, your
device, or your judgement. Work top to bottom.

## Done in the repo

| Item | Where |
|---|---|
| App icon, 1024², opaque sRGB, square corners | `Sources/Assets.xcassets/AppIcon.appiconset` — generated artwork (depth-gradient point surface). `ios/scripts/make-icon.py` renders an alternative programmatic icon and is kept as a fallback |
| Privacy manifest (no data collected, no required-reason APIs) | `Sources/PrivacyInfo.xcprivacy` |
| Marketing/build version wired to `$(MARKETING_VERSION)` / `$(CURRENT_PROJECT_VERSION)` | `project.yml` |
| Export-compliance exemption (`ITSAppUsesNonExemptEncryption: false`) | `project.yml` → Info.plist |
| Files-app access so users can retrieve older captures | `UIFileSharingEnabled`, `LSSupportsOpeningDocumentsInPlace` |
| libvpx BSD notice (a binary-distribution obligation) + MIT notices, in-app | `Sources/Acknowledgements.swift`, ⓘ button |
| Archive + `.ipa` export | `ios/scripts/archive.sh`, `ios/scripts/ExportOptions.plist` |
| Privacy-policy text to host | `ios/PRIVACY.md` |

## Signing and the build — done

`ios/scripts/archive.sh` ran end to end. `-allowProvisioningUpdates` created
the missing pieces automatically:

- **Apple Distribution certificate** — issued, cloud-managed by Xcode. It does
  not appear in `security find-identity` because the private key lives in
  Xcode's managed store rather than your login keychain. That is normal for
  automatic signing, but it does tie the identity to Xcode on this Mac; if you
  ever sign from another machine or CI, export it from Xcode → Settings →
  Accounts, or switch to an App Store Connect API key.
- **App ID `dev.wurld.WurldCam`** — registered (an explicit, non-wildcard
  profile was issued for it).
- **`iOS Team Store Provisioning Profile: dev.wurld.WurldCam`** — created.

Result: `ios/build/export/WurldCam.ipa`, 2.9 MB, signed
`Apple Distribution: Kevin Matzen (R5326Y7EZ4)`, `get-task-allow = false`,
`MinimumOSVersion 17.0`, carrying `Assets.car`, the app icons and
`PrivacyInfo.xcprivacy`. This is a valid App Store binary.

## Blockers only you can clear

1. **Create the app record.** <https://appstoreconnect.apple.com> → Apps → +.
   The App ID already exists; this is the store-listing record. Pick the name
   there — "WurldCam" may be taken, and the listing name need not match the
   binary's display name.

2. **Host the privacy policy.** A URL is mandatory. `ios/PRIVACY.md` is written
   and accurate; publish it (GitHub Pages on the `wurld` repo works, but that
   repo is currently **private** — the URL must be publicly reachable).

3. **Screenshots.** Required: 6.9" iPhone, 1290×2796 or 1320×2868. Take them on
   your 15 Pro (Volume-Up + Side), mid-capture. The 15 Pro shoots 1179×2556,
   which App Store Connect will not accept directly — but it is the same 0.4613
   aspect ratio, so upscaling to 1290×2796 is exact and lossless in framing.
   Neither the simulator nor any tooling here can substitute: the vendored libs
   are device-only arm64, there is no LiDAR in the simulator, and generated
   screenshots would misrepresent the app.

4. **Privacy nutrition labels** in App Store Connect. Answer **"No, we do not
   collect data from this app"** — that matches `PrivacyInfo.xcprivacy` and the
   absence of any network code. A mismatch here is a common rejection.

5. **Upload.** Transporter.app or Xcode Organizer, using the `.ipa` above. Bump
   `CURRENT_PROJECT_VERSION` in `project.yml` before each subsequent upload —
   App Store Connect rejects a repeat build number.

## Review risks worth pre-empting

**LiDAR.** The app is useless without it, and there is no
`UIRequiredDeviceCapabilities` key for LiDAR, so App Store review may run it on
a non-Pro iPhone, see "this device has no LiDAR scene depth", and reject it as
non-functional. Put this in **App Review Notes** verbatim:

> WurldCam requires a LiDAR-equipped device (iPhone 12 Pro or later Pro/Pro Max,
> or an iPad Pro with LiDAR). On devices without LiDAR the app reports that
> scene depth is unavailable. Please review on a Pro device. Tap the red button
> to record; the file is written to Documents and is retrievable via the share
> button or the Files app.

**"Apple Vision Pro support issue … [arkit]".** Expected, and not a rejection.
visionOS has no `arkit` device capability, so this warning reports that the app
will not be offered on Vision Pro as a compatible iPhone app — which is right:
Vision Pro does not give compatible iPhone apps rear-camera LiDAR scene depth,
so the app could only launch and report that depth is unavailable. Silence the
notice in App Store Connect → Pricing and Availability → turn off Apple Vision
Pro; there is no Info.plist key for it. Do **not** silence it by dropping
`arkit` from `UIRequiredDeviceCapabilities` — that makes the app eligible for a
platform it cannot work on. The key gates nothing else: at iOS 17 the oldest
installable device is an A12 iPhone, and ARKit needs only A9.

**Guideline 4.2 (minimum functionality).** This is a single-purpose developer
tool: point, record, get a file. That is a legitimate category, but the listing
should make the audience explicit — say who it is for (researchers, robotics
and 3D-reconstruction developers) and what the output is used for, rather than
pitching it as a consumer camera.

## Draft listing copy

**Subtitle** (30 char max): `Posed LiDAR capture, on device`

**Promotional text**: Record RGB, LiDAR depth, per-frame camera poses and
confidence in one streamable file. Open standard, no cloud, no account.

**Description**:

> WurldCam turns an iPhone Pro into a posed RGBD capture rig.
>
> Every recording contains the colour video, the LiDAR depth map, the camera's
> position and orientation for each frame, and per-pixel depth confidence —
> written together into a single file as you record, not stitched together
> afterwards. Depth is stored losslessly, so the millimetres your sensor
> measured are the millimetres you get back.
>
> Built for people who need the geometry, not just the picture: 3D
> reconstruction, robotics datasets, SLAM evaluation, NeRF and Gaussian-splat
> capture, photogrammetry research.
>
> • Records to the open wurld format (Matroska/WebM container) or Record3D .r3d
> • Lossless 16-bit depth, per-frame poses in metres with quaternion rotations
> • Files stay on your device — no account, no cloud, no analytics
> • Retrieve captures via the share sheet or the Files app
> • Read them with the open-source Python and JavaScript tools
>
> Requires a LiDAR-equipped iPhone.

**Keywords** (100 char max):
`lidar,depth,rgbd,3d scan,point cloud,slam,photogrammetry,nerf,capture,pose,dataset,robotics`

**Category**: Developer Tools (primary), Graphics & Design (secondary).

## Capture quality — verified on device

The backpressure fix is confirmed against real captures pulled off the phone:

| take | frames | fps | stalls (dt > 3× median) | worst dt |
|---|---|---|---|---|
| before fix | 99 | 17.6 | 10 (10%) | 0.317 s |
| before fix | 124 | 17.6 | 13 (11%) | 0.333 s |
| **after fix** | 102 | 15.2 | **0 (0%)** | **0.183 s** |

Stalls are gone. Nominal rate fell 17.6 → 15.2 fps, which is the intended
trade: skipping a frame outright costs one sample, while queuing one starves
ARKit's buffer pool and costs several. If you want the rate back, the lever is
`frameInterval` in `CaptureController.swift` — but check the skip count after
changing it.
