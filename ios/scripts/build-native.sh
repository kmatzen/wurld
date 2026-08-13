#!/usr/bin/env bash
# Cross-compile libvpx + the chromapakz native core as iOS arm64 static libs.
#
#   ios/scripts/build-native.sh              # device  -> Vendor/lib
#   ios/scripts/build-native.sh --simulator  # arm64 simulator -> Vendor/lib-sim
#
# The simulator slice exists so the app can be run in Simulator for App Store
# screenshots. ARKit does not deliver frames there, so the capture path itself
# is only ever exercised on a device.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENDOR="$ROOT/WurldCam/Vendor"
BUILD="$ROOT/.native-build"
# Which ChromaPakZ the vendored libs are built from. Vendor/ is a build artifact
# (gitignored), so this pin is the only thing that makes an iOS build reproducible
# — without it the libs come from whatever state a local checkout happens to be in.
# Keep in step with the pins in pyproject.toml / package.json.
CHROMAPAKZ_REF="${CHROMAPAKZ_REF:-v0.10.0}"
# A local checkout wins when explicitly asked for (that is how you test an
# unreleased change); otherwise clone the pinned tag, so a fresh machine needs
# nothing set up.
if [ -n "${CHROMAPAKZ_SRC:-}" ]; then
  echo "==> chromapakz: local checkout $CHROMAPAKZ_SRC (ref NOT enforced)"
else
  CHROMAPAKZ_SRC="$BUILD/chromapakz-$CHROMAPAKZ_REF"
  if [ ! -d "$CHROMAPAKZ_SRC" ]; then
    mkdir -p "$BUILD"
    git clone --depth 1 --branch "$CHROMAPAKZ_REF" \
      https://github.com/kmatzen/ChromaPakZ "$CHROMAPAKZ_SRC"
  fi
  echo "==> chromapakz: $CHROMAPAKZ_REF"
fi
MIN_IOS="17.0"

if [ "${1:-}" = "--simulator" ]; then
  PLATFORM="simulator"
  SDK_NAME="iphonesimulator"
  LIBDIR="$VENDOR/lib-sim"
  # libvpx has no simulator target, and its darwin targets hardcode both
  # -miphoneos-version-min and the iPhoneOS sysroot — which conflicts with a
  # simulator triple and stamps objects as platform iOS. generic-gnu injects
  # nothing, leaving CC/CXX fully in control. Costs the hand-written asm, which
  # does not matter for a build that exists only to take screenshots.
  TRIPLE="-target arm64-apple-ios$MIN_IOS-simulator"
  VPX_TARGET="generic-gnu"
else
  PLATFORM="device"
  SDK_NAME="iphoneos"
  LIBDIR="$VENDOR/lib"
  TRIPLE="-miphoneos-version-min=$MIN_IOS"
  VPX_TARGET="arm64-darwin-gcc"
fi

SDK="$(xcrun --sdk $SDK_NAME --show-sdk-path)"
CLANG="$(xcrun --sdk $SDK_NAME -f clang)"
CLANGXX="$(xcrun --sdk $SDK_NAME -f clang++)"
FLAGS="-arch arm64 -isysroot $SDK $TRIPLE"
WORK="$BUILD/libvpx-$PLATFORM"

echo "==> building for $PLATFORM -> $LIBDIR"
mkdir -p "$BUILD" "$LIBDIR" "$VENDOR/include"

# ── libvpx ──
if [ ! -f "$LIBDIR/libvpx.a" ]; then
  if [ ! -d "$BUILD/libvpx" ]; then
    git clone --depth 1 https://github.com/webmproject/libvpx "$BUILD/libvpx"
  fi
  mkdir -p "$WORK"
  pushd "$WORK" >/dev/null
  CC="$CLANG $FLAGS" CXX="$CLANGXX $FLAGS" LD="$CLANG $FLAGS" \
    ../libvpx/configure \
      --target=$VPX_TARGET \
      --disable-examples --disable-tools --disable-docs --disable-unit-tests \
      --disable-vp8 --enable-vp9 --enable-vp9-highbitdepth \
      --disable-shared --enable-static
  make -j"$(sysctl -n hw.ncpu)"
  cp libvpx.a "$LIBDIR/libvpx.a"
  mkdir -p "$VENDOR/include/vpx"
  cp ../libvpx/vpx/*.h "$VENDOR/include/vpx/"
  cp vpx_config.h "$VENDOR/include/vpx/" 2>/dev/null || true
  popd >/dev/null
fi

# ── chromapakz core ──
if [ ! -d "$CHROMAPAKZ_SRC/native" ]; then
  echo "chromapakz checkout not found at $CHROMAPAKZ_SRC (set CHROMAPAKZ_SRC)" >&2
  exit 1
fi
"$CLANGXX" $FLAGS -std=c++17 -O2 -c \
  -I "$VENDOR/include" -I "$CHROMAPAKZ_SRC/native" \
  "$CHROMAPAKZ_SRC/native/chromapakz.cpp" -o "$BUILD/chromapakz-$PLATFORM.o"
libtool -static -o "$LIBDIR/libchromapakz.a" "$BUILD/chromapakz-$PLATFORM.o"
cp "$CHROMAPAKZ_SRC/native/chromapakz.h" "$VENDOR/include/"

echo "done (chromapakz $CHROMAPAKZ_REF):"
ls -la "$LIBDIR"
lipo -info "$LIBDIR/libvpx.a" "$LIBDIR/libchromapakz.a"
