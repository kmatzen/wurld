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
CHROMAPAKZ_SRC="${CHROMAPAKZ_SRC:-$HOME/git/chromapakz}"
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

echo "done:"
ls -la "$LIBDIR"
lipo -info "$LIBDIR/libvpx.a" "$LIBDIR/libchromapakz.a"
