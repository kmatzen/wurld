// Standalone round-trip check for PixelConvert.swift. Builds with plain swiftc
// on macOS (Accelerate + CoreVideo + CoreGraphics, no ARKit, no chromapakz), so
// CI gets a fast signal on the vImage colour path that the app build can't give
// cheaply.
//
//   swiftc -O ../../Sources/PixelConvert.swift main.swift -o roundtrip && ./roundtrip
//
// sRGB RGBA -> tagged 4:2:0 (YCbCr420.make) -> colour-managed sRGB RGBA
// (PixelConverter). Solid colour blocks, interiors compared, so the only
// expected loss is chroma subsampling + rounding — a channel swap, wrong matrix,
// or wrong range would blow past the tolerance.

import CoreVideo
import Foundation

func fail(_ msg: String) -> Never {
    FileHandle.standardError.write(Data(("FAIL: " + msg + "\n").utf8))
    exit(1)
}

let W = 64, H = 48
let cols = 3, rows = 3
let bw = W / cols, bh = H / rows
let colors: [(UInt8, UInt8, UInt8)] = [
    (255, 255, 255), (0, 0, 0), (128, 128, 128),
    (230, 40, 40), (40, 200, 80), (50, 90, 230),
    (240, 220, 30), (200, 60, 200), (30, 200, 220),
]

var rgba = [UInt8](repeating: 0, count: W * H * 4)
for y in 0..<H {
    for x in 0..<W {
        let block = min((y / bh) * cols + (x / bw), colors.count - 1)
        let c = colors[block]
        let o = (y * W + x) * 4
        rgba[o] = c.0; rgba[o + 1] = c.1; rgba[o + 2] = c.2; rgba[o + 3] = 255
    }
}

guard let pb = YCbCr420.make(fromSRGBA: rgba, width: W, height: H) else {
    fail("YCbCr420.make returned nil")
}

let out: [UInt8]
do {
    out = try PixelConverter().sRGBA(from: pb, width: W, height: H)
} catch {
    fail("PixelConverter.sRGBA threw: \(error)")
}
guard out.count == W * H * 4 else { fail("unexpected output size \(out.count)") }

// Compare block interiors (>=2px from block edges) to skip chroma bleed at seams.
var maxErr = 0
for y in 0..<H where (y % bh) >= 2 && (y % bh) < bh - 2 {
    for x in 0..<W where (x % bw) >= 2 && (x % bw) < bw - 2 {
        let o = (y * W + x) * 4
        for k in 0..<3 { maxErr = max(maxErr, abs(Int(out[o + k]) - Int(rgba[o + k]))) }
    }
}

let tolerance = 12
print("PixelConvert round-trip: \(W)x\(H), \(colors.count) blocks, max channel error \(maxErr)")
if maxErr > tolerance { fail("round-trip error \(maxErr) exceeds tolerance \(tolerance)") }
print("OK")

// A converter is reused across frames, and the conversion it caches is specific
// to the buffer's range and matrix. Feed it a BT.601-tagged buffer after a
// BT.709 one: if the cache is keyed on "generated once" rather than on the
// tagging, the second conversion silently uses 709 coefficients on 601 data.
let reused = PixelConverter()
guard let pb709 = YCbCr420.make(fromSRGBA: rgba, width: W, height: H) else {
    fail("could not build the 709 buffer")
}
guard let pb601 = YCbCr420.make(fromSRGBA: rgba, width: W, height: H) else {
    fail("could not build the 601 buffer")
}
CVBufferSetAttachment(pb601, kCVImageBufferYCbCrMatrixKey,
                      kCVImageBufferYCbCrMatrix_ITU_R_601_4, .shouldPropagate)

let first: [UInt8], second: [UInt8], fresh: [UInt8]
do {
    first = try reused.sRGBA(from: pb709, width: W, height: H)
    second = try reused.sRGBA(from: pb601, width: W, height: H)
    fresh = try PixelConverter().sRGBA(from: pb601, width: W, height: H)
} catch {
    fail("re-tagged conversion threw: \(error)")
}
guard first.count == second.count, second.count == fresh.count else {
    fail("size mismatch across conversions")
}

// The 601 result must match a converter that only ever saw 601 — i.e. the cache
// regenerated. It must also differ from the 709 result, or the tags did nothing.
var reuseErr = 0, matrixDelta = 0
for i in 0..<second.count where i % 4 != 3 {
    reuseErr = max(reuseErr, abs(Int(second[i]) - Int(fresh[i])))
    matrixDelta = max(matrixDelta, abs(Int(second[i]) - Int(first[i])))
}
print("cached conversion vs fresh converter (601 after 709): max delta \(reuseErr)")
if reuseErr != 0 { fail("stale conversion reused across a matrix change (delta \(reuseErr))") }
print("601 vs 709 output differs by \(matrixDelta) — the tag is being honoured")
if matrixDelta == 0 { fail("601 and 709 produced identical output; the matrix tag is ignored") }
print("OK")
