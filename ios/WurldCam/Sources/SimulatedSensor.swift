#if targetEnvironment(simulator)
import ARKit
import CoreImage
import UIKit
import simd

/// A stand-in for the camera and LiDAR when running in Simulator.
///
/// ARKit delivers no frames there — no camera, no depth — so the app would
/// otherwise sit on "this device has no LiDAR scene depth" with a blank
/// preview. This feeds the real capture path a bundled scene image plus a
/// synthesised depth map and a slow dolly-in pose track, which makes the UI
/// exercisable and lets the encoder run end to end without hardware.
///
/// Compiled only for the simulator, and the bundled asset is excluded from
/// device builds (see EXCLUDED_SOURCE_FILE_NAMES in project.yml). Nothing here
/// ships to a real device.
///
/// The imagery is synthetic. Screenshots taken this way show the app's real
/// interface over a scene the sensors never saw.
final class SimulatedSensor {
    let depthWidth = 256
    let depthHeight = 192
    /// Matches the app's own downscaled RGB grid, so encoding behaves as on device.
    private(set) var rgb: [UInt8] = []
    private(set) var depth: [Float] = []
    private(set) var confidence: [UInt8] = []
    /// Full-resolution scene for the on-screen preview.
    let previewImage: UIImage?

    /// Roughly an iPhone wide lens at this grid: ~70° horizontal field of view.
    var intrinsics: simd_float3x3 {
        var k = matrix_identity_float3x3
        let f = Float(depthWidth) * 0.70
        k[0][0] = f; k[1][1] = f
        k[2][0] = Float(depthWidth) / 2; k[2][1] = Float(depthHeight) / 2
        return k
    }

    init() {
        let image = UIImage(named: "SimulatedScene")
        previewImage = image
        guard let cg = image?.cgImage else { return }
        rgb = Self.resample(cg, width: depthWidth, height: depthHeight)
        (depth, confidence) = Self.synthesiseDepth(rgb: rgb, width: depthWidth,
                                                  height: depthHeight)
    }

    /// Camera-to-world for frame `i`: a slow dolly forward with a gentle arc, so
    /// consecutive poses differ the way a handheld take would.
    func transform(forFrame i: Int) -> simd_float4x4 {
        let t = Float(i) / 30.0
        let yaw = 0.10 * sin(t * 0.6)
        let translation = SIMD3<Float>(0.16 * sin(t * 0.5), 0.02 * sin(t * 1.7), -0.12 * t)
        let rotation = simd_quatf(angle: yaw, axis: SIMD3<Float>(0, 1, 0))
        var m = simd_float4x4(rotation)
        m.columns.3 = SIMD4<Float>(translation.x, translation.y, translation.z, 1)
        return m
    }

    // MARK: - Fabricating plausible sensor data

    private static func resample(_ cg: CGImage, width: Int, height: Int) -> [UInt8] {
        var out = [UInt8](repeating: 0, count: width * height * 4)
        let space = CGColorSpaceCreateDeviceRGB()
        let info = CGImageAlphaInfo.premultipliedLast.rawValue
        out.withUnsafeMutableBytes { buf in
            guard let ctx = CGContext(data: buf.baseAddress, width: width, height: height,
                                      bitsPerComponent: 8, bytesPerRow: width * 4,
                                      space: space, bitmapInfo: info) else { return }
            // Aspect-fill, matching how the preview is displayed.
            let scale = max(CGFloat(width) / CGFloat(cg.width),
                            CGFloat(height) / CGFloat(cg.height))
            let w = CGFloat(cg.width) * scale, h = CGFloat(cg.height) * scale
            ctx.draw(cg, in: CGRect(x: (CGFloat(width) - w) / 2,
                                    y: (CGFloat(height) - h) / 2, width: w, height: h))
        }
        return out
    }

    /// Depth is invented, not measured. A scene shot from standing height mostly
    /// recedes with image height, so the base is a vertical ramp; luminance adds
    /// local relief so the surface is not perfectly flat, and the result is
    /// smoothed to avoid the texture-shaped ridges a raw image would produce.
    private static func synthesiseDepth(rgb: [UInt8], width: Int, height: Int)
        -> ([Float], [UInt8]) {
        let near: Float = 0.55, far: Float = 4.8
        var d = [Float](repeating: 0, count: width * height)
        for y in 0..<height {
            let v = Float(y) / Float(height - 1)
            let ramp = near + (far - near) * powf(1 - v, 1.35)
            for x in 0..<width {
                let o = (y * width + x) * 4
                let luma = (0.299 * Float(rgb[o]) + 0.587 * Float(rgb[o + 1])
                            + 0.114 * Float(rgb[o + 2])) / 255
                let u = Float(x) / Float(width - 1)
                let bulge = 0.22 * sinf(u * .pi)          // centre of frame nearer
                d[y * width + x] = max(0.3, ramp - bulge - 0.45 * (luma - 0.5))
            }
        }
        // Separable box blur, twice — cheap approximation of a smooth surface.
        var tmp = d
        for _ in 0..<2 {
            let r = 3
            for y in 0..<height {
                for x in 0..<width {
                    var s: Float = 0, n: Float = 0
                    for k in -r...r where x + k >= 0 && x + k < width {
                        s += d[y * width + x + k]; n += 1
                    }
                    tmp[y * width + x] = s / n
                }
            }
            for x in 0..<width {
                for y in 0..<height {
                    var s: Float = 0, n: Float = 0
                    for k in -r...r where y + k >= 0 && y + k < height {
                        s += tmp[(y + k) * width + x]; n += 1
                    }
                    d[y * width + x] = s / n
                }
            }
        }
        return (d, [UInt8](repeating: 2, count: width * height))  // "high" confidence
    }
}

/// Wrap the synthetic planes as CVPixelBuffers so they flow through exactly the
/// same encode path as ARKit's, rather than a parallel one that could drift.
extension SimulatedSensor {
    static func makeBGRA(fromRGBA rgba: [UInt8], width: Int, height: Int) -> CVPixelBuffer? {
        guard let pb = allocate(width: width, height: height,
                                format: kCVPixelFormatType_32BGRA) else { return nil }
        CVPixelBufferLockBaseAddress(pb, [])
        defer { CVPixelBufferUnlockBaseAddress(pb, []) }
        guard let base = CVPixelBufferGetBaseAddress(pb)?.assumingMemoryBound(to: UInt8.self)
        else { return nil }
        let stride = CVPixelBufferGetBytesPerRow(pb)
        for y in 0..<height {
            for x in 0..<width {
                let s = (y * width + x) * 4, d = y * stride + x * 4
                base[d] = rgba[s + 2]; base[d + 1] = rgba[s + 1]   // RGBA -> BGRA
                base[d + 2] = rgba[s]; base[d + 3] = 255
            }
        }
        return pb
    }

    static func makeDepth(from depth: [Float], width: Int, height: Int) -> CVPixelBuffer? {
        guard let pb = allocate(width: width, height: height,
                                format: kCVPixelFormatType_DepthFloat32) else { return nil }
        CVPixelBufferLockBaseAddress(pb, [])
        defer { CVPixelBufferUnlockBaseAddress(pb, []) }
        guard let base = CVPixelBufferGetBaseAddress(pb) else { return nil }
        let stride = CVPixelBufferGetBytesPerRow(pb)
        for y in 0..<height {
            let row = base.advanced(by: y * stride).assumingMemoryBound(to: Float.self)
            for x in 0..<width { row[x] = depth[y * width + x] }
        }
        return pb
    }

    static func makeConfidence(from conf: [UInt8], width: Int, height: Int) -> CVPixelBuffer? {
        guard let pb = allocate(width: width, height: height,
                                format: kCVPixelFormatType_OneComponent8) else { return nil }
        CVPixelBufferLockBaseAddress(pb, [])
        defer { CVPixelBufferUnlockBaseAddress(pb, []) }
        guard let base = CVPixelBufferGetBaseAddress(pb)?.assumingMemoryBound(to: UInt8.self)
        else { return nil }
        let stride = CVPixelBufferGetBytesPerRow(pb)
        for y in 0..<height {
            for x in 0..<width { base[y * stride + x] = conf[y * width + x] }
        }
        return pb
    }

    private static func allocate(width: Int, height: Int, format: OSType) -> CVPixelBuffer? {
        var pb: CVPixelBuffer?
        let attrs = [kCVPixelBufferIOSurfacePropertiesKey: [:] as CFDictionary] as CFDictionary
        guard CVPixelBufferCreate(kCFAllocatorDefault, width, height, format, attrs, &pb)
                == kCVReturnSuccess else { return nil }
        return pb
    }
}
#endif
