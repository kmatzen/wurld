#!/usr/bin/env bash
# Verify the WurldCam recording pipeline WITHOUT a device: compile the
# app's Swift writer + encoder for macOS, record a synthetic take, and
# validate the file with the Python wurld reader + ffmpeg.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO="$(cd "$ROOT/.." && pwd)"
CHROMAPAKZ_SRC="${CHROMAPAKZ_SRC:-$HOME/git/chromapakz}"
OUT="$(mktemp -d)"
clang++ -std=c++17 -O2 -dynamiclib -arch arm64 \
  -I /opt/homebrew/opt/libvpx/include -I "$CHROMAPAKZ_SRC/native" \
  "$CHROMAPAKZ_SRC/native/chromapakz.cpp" \
  -L /opt/homebrew/opt/libvpx/lib -lvpx -o "$OUT/libchromapakz_mac.dylib"
swiftc -O "$ROOT/WurldCam/Harness/main.swift" \
  "$ROOT/WurldCam/Sources/WurldStreamWriter.swift" \
  "$ROOT/WurldCam/Sources/ChromapakzEncoder.swift" \
  "$ROOT/WurldCam/Sources/ZipWriter.swift" \
  -import-objc-header "$ROOT/WurldCam/Sources/WurldCam-Bridging-Header.h" \
  -I "$CHROMAPAKZ_SRC/native" -L "$OUT" -lchromapakz_mac -o "$OUT/harness"
"$OUT/harness" "$OUT/take.wurld.webm"
"$REPO/.venv/bin/python" - "$OUT/take.wurld.webm" <<'EOF'
import sys, numpy as np
sys.path.insert(0, ".")
import wurld as wl
from wurld.stream import StreamReader
seq = wl.read(sys.argv[1])
assert len(seq.frames) == 20 and seq.frames[3].t == 0.3
# RGB at 64x48, depth/confidence on their own 32x24 grid (SPEC 4.6).
assert seq.rgb.shape == (20, 48, 64, 4)
assert seq.signal_resolution("depth") == (32, 24)
dm = seq.depth_meters(5)
assert dm.shape == (24, 32)
assert np.isnan(dm[0, 0]) and abs(dm[7, 10] - (0.5 + 17 / 56 * 8.0)) < 0.01
r = StreamReader()
data = open(sys.argv[1], "rb").read()
for off in range(0, len(data), 512):
    r.feed(data[off : off + 512])
assert len(r.frames) == 20
conf = seq.signal("confidence")
assert conf.shape == (20, 24, 32) and conf[0, 0, 0] == 0 and (conf[:, 1:, :] == 2).all()
assert seq.signal_meta("confidence").value_map["labels"]["2"] == "high"
print("pipeline verification OK:", len(seq.frames), "mixed-res frames + confidence, bit-checked")
EOF
