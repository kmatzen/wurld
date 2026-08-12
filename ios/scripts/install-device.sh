#!/usr/bin/env bash
# Build WurldCam for a connected iPhone and install it.
#
#   ios/scripts/install-device.sh [udid]
#
# Not the same thing as archive.sh. That produces an App Store build signed for
# distribution with get-task-allow=false, which a device will refuse to install —
# it is for upload, not for running. This builds a Debug configuration signed
# with an Apple Development identity, which is the only kind you can sideload.
#
# Requires the phone to be present: plugged in over USB, or unlocked and on the
# same network with an active tunnel. `xcrun devicectl list devices` reports
# tunnelState; "disconnected" means the install will fail with a connection
# reset no matter how good the build is.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PROJ="$ROOT/ios/WurldCam/WurldCam.xcodeproj"
DD="$ROOT/ios/build/dd"
APP="$DD/Build/Products/Debug-iphoneos/WurldCam.app"

udid="${1:-}"
if [ -z "$udid" ]; then
  udid="$(xcrun devicectl list devices --json-output /tmp/wurldcam-devices.json >/dev/null 2>&1 \
    && python3 -c "
import json
d = json.load(open('/tmp/wurldcam-devices.json'))
for x in d['result']['devices']:
    p = x.get('connectionProperties', {})
    if p.get('pairingState') == 'paired':
        print(x['hardwareProperties']['udid']); break
" || true)"
fi
if [ -z "$udid" ]; then
  echo "error: no paired device found. Plug the iPhone in and unlock it." >&2
  exit 1
fi
echo "device: $udid"

echo "building (Debug, development signing)…"
xcodebuild -project "$PROJ" -scheme WurldCam -configuration Debug \
  -destination 'generic/platform=iOS' -derivedDataPath "$DD" \
  -allowProvisioningUpdates build >/dev/null

# A distribution-signed bundle installs nowhere; fail loudly rather than let
# devicectl report it as an opaque transport error.
if ! codesign -d --entitlements - --xml "$APP" 2>/dev/null \
     | plutil -convert json -o - - 2>/dev/null \
     | grep -q '"get-task-allow":true'; then
  echo "error: built app is not development-signed (get-task-allow is false)." >&2
  exit 1
fi

echo "installing…"
xcrun devicectl device install app --device "$udid" "$APP"
echo "installed. Launch it from the home screen; the first run asks for camera access."
