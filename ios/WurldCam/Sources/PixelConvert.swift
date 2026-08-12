import Accelerate
import CoreGraphics
import CoreVideo
import Foundation

/// CPU color conversions shared by the capture path and the simulator, kept
/// free of ARKit so they compile and unit-test on macOS without a device.
///
/// The device path converts the camera's YpCbCr (4:2:0 biplanar) buffer into
/// RGBA at the depth grid with vImage: the YpCbCr->RGB matrix (from the buffer's
/// tagged matrix + range) and chroma upsample in one SIMD pass, then a downscale.
/// No Core Image filter graph and no synchronous GPU->CPU readback — the step
/// that made the write chain miss frames.
///
/// This applies the color *matrix* but not a primaries/transfer gamut remap, so
/// the result is the camera's own R'G'B' treated as sRGB. That matches ARKit's
/// capture (BT.709 / sRGB-compatible) and the simulator's tagging; a wide-gamut
/// (Display P3) source would need the fuller color-managed path.
final class PixelConverter {
    enum ConvertError: Error {
        case unsupportedFormat(OSType)
        case generate(vImage_Error)
        case convert(vImage_Error)
        case scale(vImage_Error)
        case scratch(vImage_Error)
    }

    /// YpCbCr->RGB conversion, built once. `full` is the RGBA image at the source
    /// resolution, reused across frames. All touched only on the caller's serial
    /// write queue, one frame at a time (`maxInFlight == 1`), so no locking.
    private var info = vImage_YpCbCrToARGB()
    /// The (range, matrix) the cached conversion was generated for. A conversion
    /// built for one tagging silently produces wrong color for another, so key
    /// the cache on both rather than on "have we generated once".
    private var generatedFor: (fullRange: Bool, matrix709: Bool)?
    private var full = vImage_Buffer()
    private var fullW = 0, fullH = 0

    deinit { if full.data != nil { free(full.data) } }

    func sRGBA(from buffer: CVPixelBuffer, width: Int, height: Int) throws -> [UInt8] {
        let fmt = CVPixelBufferGetPixelFormatType(buffer)
        let fullRange = fmt == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange
        guard fullRange || fmt == kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange
        else { throw ConvertError.unsupportedFormat(fmt) }

        let matrix709 = isBT709(buffer)
        if generatedFor?.fullRange != fullRange || generatedFor?.matrix709 != matrix709 {
            try generate(fullRange: fullRange, matrix709: matrix709)
        }

        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }

        let sw = CVPixelBufferGetWidthOfPlane(buffer, 0)
        let sh = CVPixelBufferGetHeightOfPlane(buffer, 0)
        try ensureFull(width: sw, height: sh)

        var yp = vImage_Buffer(data: CVPixelBufferGetBaseAddressOfPlane(buffer, 0),
                               height: vImagePixelCount(sh), width: vImagePixelCount(sw),
                               rowBytes: CVPixelBufferGetBytesPerRowOfPlane(buffer, 0))
        var cbcr = vImage_Buffer(data: CVPixelBufferGetBaseAddressOfPlane(buffer, 1),
                                 height: vImagePixelCount(CVPixelBufferGetHeightOfPlane(buffer, 1)),
                                 width: vImagePixelCount(CVPixelBufferGetWidthOfPlane(buffer, 1)),
                                 rowBytes: CVPixelBufferGetBytesPerRowOfPlane(buffer, 1))

        // The convert emits canonical ARGB; permuteMap [1,2,3,0] rewrites it to
        // R,G,B,A in memory — the order the chromapakz encoder reads.
        let permute: [UInt8] = [1, 2, 3, 0]
        let convRC = permute.withUnsafeBufferPointer { map in
            vImageConvert_420Yp8_CbCr8ToARGB8888(&yp, &cbcr, &full, &info, map.baseAddress, 255,
                                                 vImage_Flags(kvImageNoFlags))
        }
        guard convRC == kvImageNoError else { throw ConvertError.convert(convRC) }

        var out = [UInt8](repeating: 0, count: width * height * 4)
        let scaleRC: vImage_Error = out.withUnsafeMutableBytes { dstBytes in
            var dst = vImage_Buffer(data: dstBytes.baseAddress,
                                    height: vImagePixelCount(height), width: vImagePixelCount(width),
                                    rowBytes: width * 4)
            return vImageScale_ARGB8888(&full, &dst, nil, vImage_Flags(kvImageNoFlags))
        }
        guard scaleRC == kvImageNoError else { throw ConvertError.scale(scaleRC) }
        return out
    }

    /// BT.709 unless the buffer explicitly tags BT.601. ARKit and the simulator
    /// both use 709.
    private func isBT709(_ buffer: CVPixelBuffer) -> Bool {
        guard let m = CVBufferGetAttachment(buffer, kCVImageBufferYCbCrMatrixKey, nil)?
            .takeUnretainedValue() as? String
        else { return true }
        return m != (kCVImageBufferYCbCrMatrix_ITU_R_601_4 as String)
    }

    private func generate(fullRange: Bool, matrix709: Bool) throws {
        var range = fullRange
            ? vImage_YpCbCrPixelRange(Yp_bias: 0, CbCr_bias: 128,
                                      YpRangeMax: 255, CbCrRangeMax: 255,
                                      YpMax: 255, YpMin: 0, CbCrMax: 255, CbCrMin: 0)
            : vImage_YpCbCrPixelRange(Yp_bias: 16, CbCr_bias: 128,
                                      YpRangeMax: 235, CbCrRangeMax: 240,
                                      YpMax: 255, YpMin: 0, CbCrMax: 255, CbCrMin: 0)
        var matrix = (matrix709 ? kvImage_YpCbCrToARGBMatrix_ITU_R_709_2
                                : kvImage_YpCbCrToARGBMatrix_ITU_R_601_4).pointee
        let rc = vImageConvert_YpCbCrToARGB_GenerateConversion(
            &matrix, &range, &info, kvImage420Yp8_CbCr8, kvImageARGB8888,
            vImage_Flags(kvImageNoFlags))
        guard rc == kvImageNoError else { throw ConvertError.generate(rc) }
        generatedFor = (fullRange, matrix709)
    }

    private func ensureFull(width: Int, height: Int) throws {
        if full.data != nil, fullW == width, fullH == height { return }
        if full.data != nil { free(full.data); full.data = nil }
        let rc = vImageBuffer_Init(&full, vImagePixelCount(height), vImagePixelCount(width),
                                   32, vImage_Flags(kvImageNoFlags))
        guard rc == kvImageNoError else { full.data = nil; throw ConvertError.scratch(rc) }
        fullW = width; fullH = height
    }
}

// MARK: - Simulator source

/// Pack an sRGB RGBA image into a 4:2:0 full-range biplanar `CVPixelBuffer` that
/// looks, to the capture path, exactly like an ARKit camera frame — including the
/// color attachments, so `PixelConverter` reads real tags and round-trips the
/// color back to sRGB. Used only by the simulator; the math is a plain BT.709
/// full-range encode (the grid is tiny, so a straightforward loop is fine).
enum YCbCr420 {
    static func make(fromSRGBA rgba: [UInt8], width: Int, height: Int) -> CVPixelBuffer? {
        precondition(width % 2 == 0 && height % 2 == 0, "4:2:0 needs even dimensions")
        var pb: CVPixelBuffer?
        let attrs = [kCVPixelBufferIOSurfacePropertiesKey: [:] as CFDictionary] as CFDictionary
        guard CVPixelBufferCreate(kCFAllocatorDefault, width, height,
                                  kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,
                                  attrs, &pb) == kCVReturnSuccess,
              let pb else { return nil }

        // Tag the color so vImage's CV format picks it up on the way back: the
        // BT.709 matrix for the YpCbCr<->RGB step, and an explicit sRGB color
        // space for the RGB gamut/transfer. The samples were encoded straight
        // from sRGB values, so sRGB source -> sRGB dest makes the transfer step
        // an identity and the round-trip loses only chroma subsampling.
        CVBufferSetAttachment(pb, kCVImageBufferYCbCrMatrixKey,
                              kCVImageBufferYCbCrMatrix_ITU_R_709_2, .shouldPropagate)
        CVBufferSetAttachment(pb, kCVImageBufferCGColorSpaceKey,
                              CGColorSpace(name: CGColorSpace.sRGB)!, .shouldPropagate)
        // Chroma siting so vImage knows how to upsample the 4:2:0 planes.
        CVBufferSetAttachment(pb, kCVImageBufferChromaLocationTopFieldKey,
                              kCVImageBufferChromaLocation_Center, .shouldPropagate)

        CVPixelBufferLockBaseAddress(pb, [])
        defer { CVPixelBufferUnlockBaseAddress(pb, []) }
        guard let yBase = CVPixelBufferGetBaseAddressOfPlane(pb, 0)?.assumingMemoryBound(to: UInt8.self),
              let cBase = CVPixelBufferGetBaseAddressOfPlane(pb, 1)?.assumingMemoryBound(to: UInt8.self)
        else { return nil }
        let yStride = CVPixelBufferGetBytesPerRowOfPlane(pb, 0)
        let cStride = CVPixelBufferGetBytesPerRowOfPlane(pb, 1)

        @inline(__always) func clamp8(_ v: Float) -> UInt8 {
            UInt8(max(0, min(255, v.rounded())))
        }
        // BT.709 full-range on the gamma-encoded (R'G'B') samples.
        for y in 0..<height {
            for x in 0..<width {
                let o = (y * width + x) * 4
                let r = Float(rgba[o]), g = Float(rgba[o + 1]), b = Float(rgba[o + 2])
                yBase[y * yStride + x] = clamp8(0.2126 * r + 0.7152 * g + 0.0722 * b)
            }
        }
        // CbCr subsampled 2x2, interleaved (Cb, Cr) per chroma sample.
        for cy in 0..<(height / 2) {
            for cx in 0..<(width / 2) {
                var rs: Float = 0, gs: Float = 0, bs: Float = 0
                for dy in 0..<2 {
                    for dx in 0..<2 {
                        let o = ((cy * 2 + dy) * width + (cx * 2 + dx)) * 4
                        rs += Float(rgba[o]); gs += Float(rgba[o + 1]); bs += Float(rgba[o + 2])
                    }
                }
                let r = rs / 4, g = gs / 4, b = bs / 4
                let yp = 0.2126 * r + 0.7152 * g + 0.0722 * b
                let base = cy * cStride + cx * 2
                cBase[base] = clamp8((b - yp) / 1.8556 + 128)      // Cb
                cBase[base + 1] = clamp8((r - yp) / 1.5748 + 128)  // Cr
            }
        }
        return pb
    }
}
