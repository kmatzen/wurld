#!/usr/bin/env bash
# Cross-compile libvpx + the chromapakz native core as iOS arm64 static libs.
# Output: ios/WurldCam/Vendor/{lib/libvpx.a, lib/libchromapakz.a, include/...}
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENDOR="$ROOT/WurldCam/Vendor"
BUILD="$ROOT/.native-build"
CHROMAPAKZ_SRC="${CHROMAPAKZ_SRC:-$HOME/git/chromapakz}"
MIN_IOS="17.0"

SDK="$(xcrun --sdk iphoneos --show-sdk-path)"
CLANG="$(xcrun --sdk iphoneos -f clang)"
CLANGXX="$(xcrun --sdk iphoneos -f clang++)"
FLAGS="-arch arm64 -isysroot $SDK -miphoneos-version-min=$MIN_IOS"

mkdir -p "$BUILD" "$VENDOR/lib" "$VENDOR/include"

# ── libvpx ──
if [ ! -f "$VENDOR/lib/libvpx.a" ]; then
  if [ ! -d "$BUILD/libvpx" ]; then
    git clone --depth 1 https://github.com/webmproject/libvpx "$BUILD/libvpx"
  fi
  mkdir -p "$BUILD/libvpx-ios"
  pushd "$BUILD/libvpx-ios" >/dev/null
  CC="$CLANG $FLAGS" CXX="$CLANGXX $FLAGS" LD="$CLANG $FLAGS" \
    ../libvpx/configure \
      --target=arm64-darwin-gcc \
      --disable-examples --disable-tools --disable-docs --disable-unit-tests \
      --disable-vp8 --enable-vp9 --enable-vp9-highbitdepth \
      --disable-shared --enable-static
  make -j"$(sysctl -n hw.ncpu)"
  cp libvpx.a "$VENDOR/lib/libvpx.a"
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
  "$CHROMAPAKZ_SRC/native/chromapakz.cpp" -o "$BUILD/chromapakz.o"
libtool -static -o "$VENDOR/lib/libchromapakz.a" "$BUILD/chromapakz.o"
cp "$CHROMAPAKZ_SRC/native/chromapakz.h" "$VENDOR/include/"

echo "done:"
ls -la "$VENDOR/lib"
lipo -info "$VENDOR/lib/libvpx.a" "$VENDOR/lib/libchromapakz.a"
