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

Verified: `ARCHIVE SUCCEEDED` unsigned, and the shipped `Info.plist` carries
every key above.

## Blockers only you can clear

1. **Apple Distribution certificate.** You have *Apple Development* and
   *Developer ID Application*, but no *Apple Distribution* — App Store export
   cannot sign without it. Xcode creates one on first use: Organizer →
   Distribute App → App Store Connect. Certificates are account-wide and
   limited in number, so this is yours to make, not a script's.

2. **Register the App ID and create the app record.** Bundle ID
   `dev.wurld.WurldCam` on <https://appstoreconnect.apple.com>. Pick the app
   name there — "WurldCam" may already be taken; the binary's display name and
   the store listing name do not have to match.

3. **Host the privacy policy.** A URL is mandatory. `ios/PRIVACY.md` is written
   and accurate; publish it (GitHub Pages on the `wurld` repo works, but note
   that repo is currently **private** — it must be reachable publicly).

4. **Screenshots.** Required: 6.9" iPhone (1320×2868 or 1290×2796). Take them
   on your 15 Pro with Volume-Up + Side, mid-capture with the point cloud
   visible. The simulator is not an option — the vendored libs are device-only
   arm64 and there is no LiDAR in the simulator.

5. **Privacy nutrition labels** in App Store Connect. Answer **"No, we do not
   collect data from this app"** — that matches `PrivacyInfo.xcprivacy` and the
   absence of any network code. A mismatch here is a common rejection.

6. **Upload.** `ios/scripts/archive.sh` produces the `.ipa`; upload with
   Transporter or Xcode Organizer. Bump `CURRENT_PROJECT_VERSION` in
   `project.yml` before each upload — App Store Connect rejects a repeat build
   number.

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

## Before you ship

The capture pipeline has had exactly one real-world test, which surfaced ten
dropped-frame stalls in 5.6 s. The backpressure fix is in but unproven on
hardware — record a few takes of varying length and check the reported skip
count before submitting.
