import Accelerate
import CoreGraphics
import CoreVideo
import Foundation

/// CPU colour conversions shared by the capture path and the simulator, kept
/// free of ARKit so they compile and unit-test on macOS without a device.
///
/// The device path converts the camera's YpCbCr (4:2:0 biplanar) buffer into
/// colour-managed **sRGB** RGBA at the depth grid. vImage reads the buffer's own
/// colour tags — YpCbCr matrix, chroma siting, primaries, transfer function,
/// range — and does the matrix, chroma upsample, gamut map and transfer
/// re-encode in one SIMD pass. That is the same colour science Core Image
/// applied when it rendered into an sRGB colour space, but without building a
/// filter graph or forcing a synchronous GPU->CPU readback every frame.
///
/// Contrast with a bare `vImageConvert_420Yp8_CbCr8ToARGB8888`, which does only
/// the YpCbCr matrix and leaves the result in the camera's native space — wrong
/// whenever the camera is not already sRGB (wide-gamut capture is often P3).
final class PixelConverter {
    enum ConvertError: Error {
        case unsupportedFormat(OSType)
        case cvFormat
        case converter(vImage_Error)
        case convert(vImage_Error)
        case scale(vImage_Error)
        case scratch(vImage_Error)
    }

    private let srgb = CGColorSpace(name: CGColorSpace.sRGB)!
    private var converter: vImageConverter?
    /// Colour-managed RGBA at the source resolution, reused across frames. Only
    /// ever touched on the caller's serial write queue, one frame at a time.
    private var srgbFull = vImage_Buffer()
    private var srgbW = 0, srgbH = 0

    deinit {
        if srgbFull.data != nil { free(srgbFull.data) }
    }

    /// Camera YpCbCr -> colour-managed sRGB RGBA, downscaled to `width`x`height`.
    /// Throws on any failure so the caller surfaces it rather than silently
    /// writing wrong colour.
    func sRGBA(from buffer: CVPixelBuffer, width: Int, height: Int) throws -> [UInt8] {
        let fmt = CVPixelBufferGetPixelFormatType(buffer)
        guard fmt == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange
                || fmt == kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange
        else { throw ConvertError.unsupportedFormat(fmt) }

        if converter == nil { try buildConverter(for: buffer) }

        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }

        let srcW = CVPixelBufferGetWidthOfPlane(buffer, 0)
        let srcH = CVPixelBufferGetHeightOfPlane(buffer, 0)
        try ensureScratch(width: srcW, height: srcH)

        // AnyToAny takes the source as one vImage_Buffer per plane (Y, then CbCr).
        var srcs = [
            vImage_Buffer(data: CVPixelBufferGetBaseAddressOfPlane(buffer, 0),
                          height: vImagePixelCount(srcH), width: vImagePixelCount(srcW),
                          rowBytes: CVPixelBufferGetBytesPerRowOfPlane(buffer, 0)),
            vImage_Buffer(data: CVPixelBufferGetBaseAddressOfPlane(buffer, 1),
                          height: vImagePixelCount(CVPixelBufferGetHeightOfPlane(buffer, 1)),
                          width: vImagePixelCount(CVPixelBufferGetWidthOfPlane(buffer, 1)),
                          rowBytes: CVPixelBufferGetBytesPerRowOfPlane(buffer, 1)),
        ]
        let diag = "DIAG AnyToAny src0=\(srcW)x\(srcH) "
            + "src1=\(CVPixelBufferGetWidthOfPlane(buffer, 1))x\(CVPixelBufferGetHeightOfPlane(buffer, 1)) "
            + "dst=\(srgbW)x\(srgbH) "
            + "nsrc=\(vImageConverter_GetNumberOfSourceBuffers(converter!)) "
            + "ndst=\(vImageConverter_GetNumberOfDestinationBuffers(converter!))\n"
        FileHandle.standardError.write(Data(diag.utf8))
        let convRC = srcs.withUnsafeMutableBufferPointer { sp -> vImage_Error in
            vImageConvert_AnyToAny(converter!, sp.baseAddress!, &srgbFull, nil,
                                   vImage_Flags(kvImageNoFlags))
        }
        guard convRC == kvImageNoError else { throw ConvertError.convert(convRC) }

        var out = [UInt8](repeating: 0, count: width * height * 4)
        let scaleRC: vImage_Error = out.withUnsafeMutableBytes { dstBytes in
            var dst = vImage_Buffer(data: dstBytes.baseAddress,
                                    height: vImagePixelCount(height), width: vImagePixelCount(width),
                                    rowBytes: width * 4)
            return vImageScale_ARGB8888(&srgbFull, &dst, nil, vImage_Flags(kvImageNoFlags))
        }
        guard scaleRC == kvImageNoError else { throw ConvertError.scale(scaleRC) }
        return out
    }

    private func buildConverter(for buffer: CVPixelBuffer) throws {
        // Both sources are fully colour-tagged — ARKit tags the camera frame, and
        // `YCbCr420.make` tags the simulator buffer with matrix, colour space and
        // chroma siting — so the CV format builds straight from the buffer with no
        // accessor/mutation dance (and no fragile const-handle cast).
        let cvFmt = vImageCVImageFormat_CreateWithCVPixelBuffer(buffer).takeRetainedValue()
        var cgFmt = vImage_CGImageFormat(
            bitsPerComponent: 8, bitsPerPixel: 32,
            colorSpace: Unmanaged.passRetained(srgb),
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
            version: 0, decode: nil, renderingIntent: .defaultIntent)
        var err = kvImageNoError
        guard let conv = vImageConverter_CreateForCVToCGImageFormat(
            cvFmt, &cgFmt, nil, vImage_Flags(kvImagePrintDiagnosticsToConsole), &err),
            err == kvImageNoError
        else { throw ConvertError.converter(err) }
        converter = conv.takeRetainedValue()
    }

    private func ensureScratch(width: Int, height: Int) throws {
        if srgbFull.data != nil, srgbW == width, srgbH == height { return }
        if srgbFull.data != nil { free(srgbFull.data); srgbFull.data = nil }
        let rc = vImageBuffer_Init(&srgbFull, vImagePixelCount(height), vImagePixelCount(width),
                                   32, vImage_Flags(kvImageNoFlags))
        guard rc == kvImageNoError else { srgbFull.data = nil; throw ConvertError.scratch(rc) }
        srgbW = width; srgbH = height
    }
}

// MARK: - Simulator source

/// Pack an sRGB RGBA image into a 4:2:0 full-range biplanar `CVPixelBuffer` that
/// looks, to the capture path, exactly like an ARKit camera frame — including the
/// colour attachments, so `PixelConverter` reads real tags and round-trips the
/// colour back to sRGB. Used only by the simulator; the maths is a plain BT.709
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

        // Tag the colour so vImage's CV format picks it up on the way back: the
        // BT.709 matrix for the YpCbCr<->RGB step, and an explicit sRGB colour
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
