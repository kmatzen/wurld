# App Review reply — Guideline 2.1, Information Needed

Paste items 2–7 into **App Store Connect → App Review Information → Notes**, and
attach the screen recording for item 1. The same text serves future submissions;
Apple asked for it to live in Notes permanently.

Two things only you can supply are marked **[YOU]**. Everything else is drawn
from the app as built.

---

## 1. Screen recording — **[YOU]**

Must be captured on a physical LiDAR device on current iOS, starting from launch.
The app has no accounts, no purchases, no user-generated content shared with
anyone, and exactly one permission prompt, so the recording is short. Shot list:

1. Home screen → tap WurldCam (start from launch, as Apple asks).
2. On first launch a three-page walkthrough appears: what the app records, a
   live check that the device has a LiDAR scanner, and how to hold the phone.
   Page through it rather than skipping — it shows the reviewer the app's
   intent in the app's own words.
3. The camera permission prompt appears as the walkthrough closes — **pause on
   it long enough to read the purpose string**, then tap Allow. This is the
   only sensitive-data prompt in the app.
4. The live camera preview with the depth readout.
5. Choose the output format (`.wurld.webm` default, or `.r3d`).
6. Tap record, walk around an object for ~10 seconds, tap stop.
7. The finished take appears in the on-device list with its duration and size.
8. Open the ⓘ button to show the licence acknowledgements.
9. Open the **Files** app → On My iPhone → WurldCam, showing the recording is a
   normal file the user owns and can copy off the device.

Do not cut the permission prompt — a reviewer looking for step 4 of their list
("prompts requesting access to sensitive data") needs to see it.

## 2. Devices and OS versions tested — **[YOU]**

Fill in the actual devices. Format Apple expects, for example:

> Tested on iPhone 15 Pro (iOS 18.x) and iPad Pro 11-inch M4 (iPadOS 18.x).

⚠️ **Read this before answering.** `ios/README.md` records that ARKit on real
hardware is the one link never exercised — the pipeline is verified on macOS via
`ios/scripts/verify-pipeline.sh`, not on a device. Apple reviews on physical
hardware. If the app has not yet completed a real capture on a LiDAR iPhone,
do that first: a 2.1 reply claiming device testing that has not happened will
fail again, and more slowly. The app also requires LiDAR (`arkit` +
`UIRequiredDeviceCapabilities`) and cannot run in the Simulator, so a reviewer
on a non-Pro device would see nothing — say explicitly which models have the
scanner.

## 3. Purpose and target audience

WurldCam is a 3D capture recorder for iPhone and iPad models with a LiDAR
scanner. It records what the camera sees together with the depth of every pixel
and the position the camera was in for each frame, and writes them into a single
file the user can copy off the device.

**Problem it solves.** An ordinary video records colour only, so the geometry of
the scene is lost. Recovering it afterwards requires photogrammetry, which is
slow and often fails on plain or reflective surfaces. The LiDAR scanner already
measures that geometry directly, but there is no simple way to save it with the
video in one file that other tools accept. WurldCam writes the colour, the
metric depth and the camera track together, so the capture arrives complete.

**Value.** One tap produces a file that opens both as an ordinary video — the
recording plays in any standard player — and as 3D data in desktop tools.

**Audience.** Developers, researchers and technical artists working in 3D
reconstruction, robotics and computer vision. It is a professional tool rather
than a consumer camera app; there is no social feature of any kind.

## 4. Setup and access to the main features

**No login of any kind.** There are no accounts, no registration, no demo
credentials to supply, and no gated content — every feature is available on
first launch. Nothing needs to be configured, and no sample file is required to
exercise the app, since the reviewer creates the content by recording.

Complete flow:

1. Launch the app. A three-page walkthrough appears once, on first launch: what
   the app records, a live check for the LiDAR scanner, and capture technique.
   It can be skipped, and reopened any time from the ? button.
2. Allow camera access when prompted as the walkthrough closes (required — the
   app records from the camera and LiDAR scanner and cannot function without it).
3. Point at any object a metre or two away. The preview shows live depth.
4. Optionally switch the output format between `.wurld.webm` (default) and
   `.r3d`.
5. Tap record; move slowly around the subject; tap stop.
6. The take is listed in the app and written to the app's Documents folder,
   reachable from the **Files** app (Files → On My iPhone → WurldCam) so it can
   be copied to a computer via AirDrop, iCloud Drive or a cable.

**Hardware requirement:** a LiDAR scanner is mandatory (iPhone Pro / Pro Max, or
iPad Pro). On a device without one the app cannot record, and it will not run in
the Simulator at all.

## 5. External services, tools and platforms

**None.** The app makes no network requests whatsoever — it contains no
networking code, no analytics, no advertising, no crash reporting, no
authentication provider, no payment processing and no AI or cloud service. Every
step runs on the device.

It uses only Apple's own frameworks (ARKit, AVFoundation, Metal, SwiftUI) plus
two statically linked open-source libraries compiled into the binary:

- **libvpx** (BSD-3-Clause) — VP9 video encoding.
- **chromapakz** (MIT) — packs the depth data losslessly into the video file.

Both licences are reproduced in the app under the ⓘ button, which also satisfies
the binary-distribution notice requirement. No third party receives any data,
because nothing leaves the device.

Consistent with this, the app's privacy manifest declares no collected data
types and no tracking, and the app uses no required-reason APIs.

## 6. Regional differences

**None.** The app functions identically in every region. There is no
geo-restricted content, no regional pricing, no region-dependent feature, and no
server whose availability could vary — the app is entirely offline. The only
variation is the system language used for standard iOS interface elements.

## 7. Regulated industry / third-party material

**Not applicable.** The app does not operate in a regulated industry and
includes no protected third-party material. It records only content the user
creates with their own device's camera.

The sole third-party components are the two open-source libraries in item 5,
both under permissive licences (BSD-3-Clause and MIT) that allow binary
redistribution with attribution; that attribution is included in the app.

---

## Also fixed for this resubmission

Apple's "prevent common issues" note calls out purpose strings (Guideline
5.1.1): they must give the reason and, usually, an example of the use. The
camera string previously read "WurldCam records posed RGBD captures with the
camera and LiDAR", which restates the permission in jargon and gives no example.
It now reads:

> WurldCam uses the camera and LiDAR scanner to record a 3D capture. Each
> frame's colour image, per-pixel depth and camera position are written to a
> file on this device — for example, walking around a chair records a short clip
> you can open on a computer and view in 3D. Recordings stay on this device and
> are never uploaded.

Changed in both `WurldCam/project.yml` and `WurldCam/Sources/Info.plist`, which
must stay identical — xcodegen regenerates the plist from the yaml.

## Still to check before resubmitting — **[YOU]**

- **Screenshots** (Guideline 2.3.3): must show the app in use, not the launch or
  title screen. Use the live capture view with depth visible and the take list —
  not the permission prompt and not an empty start screen.
- ~~**Privacy policy URL**~~ — **done**. Live and publicly reachable over HTTPS at
  <https://kmatzen.com/wurld/privacy.html>. Paste that into App Store Connect →
  App Privacy → Privacy Policy URL. (Verified 2026-08-11: 200 over both http and
  https. GitHub Pages serves `/docs` from the public `kmatzen/wurld` repo;
  `https_enforced` is off, so prefer the https URL explicitly.)
- A real capture on hardware, per item 2.
