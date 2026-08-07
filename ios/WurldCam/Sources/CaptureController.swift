import ARKit
import Combine
import Compression
import CoreImage
import Foundation
import simd

/// Records ARKit LiDAR frames into a Record3D-compatible .r3d zip:
///
///     metadata            JSON: w h dw dh K(flat, column-major, RGB res) fps
///                         poses [[qx,qy,qz,qw,tx,ty,tz]] (ARKit c2w, scalar-last)
///                         initPose frameTimestamps cameraType
///     rgbd/<N>.jpg        RGB frame (sensor orientation, optionally downscaled)
///     rgbd/<N>.depth      LZFSE float32 meters, shape (dh, dw)
///     rgbd/<N>.conf       LZFSE uint8 ARKit confidence 0/1/2
///
/// Import on the desktop with `wurld convert capture.r3d out.wl.webm`.
/// Frames stay in sensor (landscape) orientation so poses and pixels agree;
/// the importer's ARKit RUB->RDF conversion assumes exactly that.
final class CaptureController: NSObject, ObservableObject, ARSessionDelegate {
    @Published var isRecording = false
    @Published var frameCount = 0
    @Published var lastCaptureURL: URL?
    @Published var statusText = "ready"

    let session = ARSession()
    private let ciContext = CIContext()
    private var zip: ZipWriter?
    private var poses: [[Double]] = []
    private var timestamps: [Double] = []
    private var intrinsics: simd_float3x3?
    private var rgbSize: CGSize = .zero
    private var depthSize: (w: Int, h: Int) = (0, 0)
    private var captureURL: URL?
    private let writeQueue = DispatchQueue(label: "wurld.capture.write")
    /// Target capture cadence; ARKit delivers 60fps, LiDAR depth updates slower.
    private let frameInterval: TimeInterval = 1.0 / 30.0
    private var lastFrameTime: TimeInterval = -1
    /// Halve RGB (1920x1440 -> 960x720) to keep captures small; K scales to match.
    private let rgbScale: CGFloat = 0.5

    func start() {
        let config = ARWorldTrackingConfiguration()
        guard ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth) else {
            statusText = "this device has no LiDAR scene depth"
            return
        }
        config.frameSemantics = [.sceneDepth]
        session.delegate = self
        session.run(config)
    }

    func beginRecording() {
        let name = ISO8601DateFormatter().string(from: Date())
            .replacingOccurrences(of: ":", with: "-")
        let url = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("wurld-\(name).r3d")
        do {
            zip = try ZipWriter(url: url)
        } catch {
            statusText = "cannot create \(url.lastPathComponent): \(error.localizedDescription)"
            return
        }
        captureURL = url
        poses = []; timestamps = []; frameCount = 0; lastFrameTime = -1
        isRecording = true
        statusText = "recording"
    }

    func endRecording() {
        isRecording = false
        statusText = "finalizing…"
        writeQueue.async { [self] in
            defer { zip = nil }
            do {
                try zip?.add(name: "metadata", data: metadataJSON())
                try zip?.finish()
                DispatchQueue.main.async {
                    self.lastCaptureURL = self.captureURL
                    self.statusText = "saved \(self.captureURL?.lastPathComponent ?? "") (\(self.frameCount) frames)"
                }
            } catch {
                DispatchQueue.main.async { self.statusText = "finalize failed: \(error.localizedDescription)" }
            }
        }
    }

    // MARK: ARSessionDelegate

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        guard isRecording, let sceneDepth = frame.sceneDepth else { return }
        if frame.timestamp - lastFrameTime < frameInterval { return }
        lastFrameTime = frame.timestamp

        let index = frameCount
        let camera = frame.camera
        intrinsics = camera.intrinsics
        rgbSize = camera.imageResolution

        // ARKit camera.transform is camera-to-world in the gravity-aligned world.
        let q = simd_quatf(camera.transform)
        let t = camera.transform.columns.3
        poses.append([Double(q.imag.x), Double(q.imag.y), Double(q.imag.z), Double(q.real),
                      Double(t.x), Double(t.y), Double(t.z)])
        timestamps.append(frame.timestamp)
        frameCount += 1

        let capturedImage = frame.capturedImage
        let depthMap = sceneDepth.depthMap
        let confidenceMap = sceneDepth.confidenceMap
        writeQueue.async { [self] in
            do {
                try writeFrame(index: index, image: capturedImage,
                               depth: depthMap, confidence: confidenceMap)
            } catch {
                DispatchQueue.main.async { self.statusText = "write failed: \(error.localizedDescription)" }
            }
        }
    }

    private func writeFrame(index: Int, image: CVPixelBuffer,
                            depth: CVPixelBuffer, confidence: CVPixelBuffer?) throws {
        var ci = CIImage(cvPixelBuffer: image)
        if rgbScale != 1 {
            ci = ci.transformed(by: CGAffineTransform(scaleX: rgbScale, y: rgbScale))
        }
        guard let jpeg = ciContext.jpegRepresentation(
            of: ci, colorSpace: CGColorSpace(name: CGColorSpace.sRGB)!,
            options: [kCGImageDestinationLossyCompressionQuality as CIImageRepresentationOption: 0.9]
        ) else { throw CaptureError.jpegFailed }
        try zip?.add(name: "rgbd/\(index).jpg", data: jpeg)

        depthSize = (CVPixelBufferGetWidth(depth), CVPixelBufferGetHeight(depth))
        try zip?.add(name: "rgbd/\(index).depth",
                     data: lzfse(planeData(depth, bytesPerPixel: 4)))
        if let confidence {
            try zip?.add(name: "rgbd/\(index).conf",
                         data: lzfse(planeData(confidence, bytesPerPixel: 1)))
        }
    }

    private func planeData(_ buffer: CVPixelBuffer, bytesPerPixel: Int) -> Data {
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        let w = CVPixelBufferGetWidth(buffer), h = CVPixelBufferGetHeight(buffer)
        let stride = CVPixelBufferGetBytesPerRow(buffer)
        let base = CVPixelBufferGetBaseAddress(buffer)!
        let rowBytes = w * bytesPerPixel
        var out = Data(capacity: rowBytes * h)
        for row in 0..<h {
            out.append(Data(bytes: base + row * stride, count: rowBytes))
        }
        return out
    }

    private func lzfse(_ data: Data) -> Data {
        data.withUnsafeBytes { (src: UnsafeRawBufferPointer) -> Data in
            let capacity = data.count + 4096
            let dst = UnsafeMutablePointer<UInt8>.allocate(capacity: capacity)
            defer { dst.deallocate() }
            let n = compression_encode_buffer(
                dst, capacity,
                src.bindMemory(to: UInt8.self).baseAddress!, data.count,
                nil, COMPRESSION_LZFSE)
            return Data(bytes: dst, count: n)
        }
    }

    private func metadataJSON() -> Data {
        // K flat in COLUMN-major order at the (scaled) RGB resolution, matching Record3D.
        let K = intrinsics ?? matrix_identity_float3x3
        let s = Double(rgbScale)
        let flatK: [Double] = [Double(K[0][0]) * s, 0, 0,
                               0, Double(K[1][1]) * s, 0,
                               Double(K[2][0]) * s, Double(K[2][1]) * s, 1]
        let meta: [String: Any] = [
            "w": Int(rgbSize.width * rgbScale),
            "h": Int(rgbSize.height * rgbScale),
            "dw": depthSize.w, "dh": depthSize.h,
            "K": flatK,
            "fps": Int(round(1.0 / frameInterval)),
            "poses": poses,
            "initPose": poses.first ?? [0, 0, 0, 1, 0, 0, 0],
            "frameTimestamps": timestamps,
            "cameraType": 1,
            "app": "WurldCam",
        ]
        return (try? JSONSerialization.data(withJSONObject: meta)) ?? Data("{}".utf8)
    }
}

enum CaptureError: Error { case jpegFailed }
