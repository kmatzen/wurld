#!/usr/bin/env bash
# Build a distributable WurldCam archive and export an .ipa for App Store Connect.
#
#   ios/scripts/archive.sh            # signed archive + .ipa (needs a distribution cert)
#   ios/scripts/archive.sh --unsigned # mechanical check of the archive step only
#
# This never uploads. Hand the resulting .ipa to Transporter, or use
#   xcrun altool --upload-app -f <ipa> -t ios -u <apple-id> -p <app-specific-password>
# Bump the build number before every upload: App Store Connect rejects a
# CFBundleVersion it has already seen for the same CFBundleShortVersionString.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PROJ="$ROOT/WurldCam/WurldCam.xcodeproj"
OUT="${OUT:-$ROOT/build}"
ARCHIVE="$OUT/WurldCam.xcarchive"
UNSIGNED=0
[ "${1:-}" = "--unsigned" ] && UNSIGNED=1

command -v xcodegen >/dev/null || { echo "xcodegen not found (brew install xcodegen)" >&2; exit 1; }
(cd "$ROOT/WurldCam" && xcodegen generate >/dev/null)

if [ ! -f "$ROOT/WurldCam/Vendor/lib/libvpx.a" ]; then
  echo "vendored libs missing — run ios/scripts/build-native.sh first" >&2
  exit 1
fi

mkdir -p "$OUT"
rm -rf "$ARCHIVE"

echo "==> archiving (Release)"
if [ "$UNSIGNED" = 1 ]; then
  xcodebuild -project "$PROJ" -scheme WurldCam -sdk iphoneos -configuration Release \
    -archivePath "$ARCHIVE" CODE_SIGNING_ALLOWED=NO archive
  echo "unsigned archive at $ARCHIVE (cannot be exported for the App Store)"
  exit 0
fi

xcodebuild -project "$PROJ" -scheme WurldCam -sdk iphoneos -configuration Release \
  -archivePath "$ARCHIVE" -allowProvisioningUpdates archive

echo "==> exporting .ipa"
xcodebuild -exportArchive -archivePath "$ARCHIVE" \
  -exportOptionsPlist "$ROOT/scripts/ExportOptions.plist" \
  -exportPath "$OUT/export" -allowProvisioningUpdates

echo
echo "ipa: $(find "$OUT/export" -name '*.ipa' -maxdepth 1 | head -1)"
echo "Upload it with Transporter.app, or Xcode's Organizer (Window > Organizer)."
