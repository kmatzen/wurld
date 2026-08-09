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
enum CaptureFormat: String, CaseIterable, Identifiable {
    case wurld = ".wl.webm"
    case r3d = ".r3d"
    var id: String { rawValue }
}

final class CaptureController: NSObject, ObservableObject, ARSessionDelegate {
    @Published var format: CaptureFormat = .wurld
    @Published var isRecording = false
    @Published var frameCount = 0
    @Published var lastCaptureURL: URL?
    @Published var statusText = "ready"

    let session = ARSession()
    private let ciContext = CIContext()
    /// Camera YpCbCr -> colour-managed sRGB RGBA at the depth grid, on the CPU
    /// with vImage. Built once; touched only on `writeQueue`, one frame at a
    /// time (`maxInFlight == 1`), so no locking is needed.
    private let pixelConverter = PixelConverter()
    private var zip: ZipWriter?
    private var poses: [[Double]] = []
    private var timestamps: [Double] = []
    private var intrinsics: simd_float3x3?
    private var rgbSize: CGSize = .zero
    private var depthSize: (w: Int, h: Int) = (0, 0)
    private var captureURL: URL?
    // wurld path (chromapakz on-device; RGB downscaled to the depth grid)
    private var wlEncoder: ChromapakzStreamEncoder?
    private var wlWriter: WurldStreamWriter?
    private var wlFile: FileHandle?
    private let wlNear = 0.1, wlFar = 12.0  // ARKit LiDAR effective range, metres
    private var wlFirstTimestamp: TimeInterval?
    private let writeQueue = DispatchQueue(label: "wurld.capture.write")
    /// Frames handed to writeQueue but not yet encoded. The buffers we pass it
    /// belong to the ARFrame, so holding a backlog starves ARKit's pixel-buffer
    /// pool and it stops delivering frames entirely — a burst of drops rather
    /// than an even slowdown. Skipping while busy keeps spacing uniform.
    private let inFlightLock = NSLock()
    private var inFlight = 0
    private let maxInFlight = 1
    private(set) var droppedFrames = 0
    /// Target capture cadence; ARKit delivers 60fps, LiDAR depth updates slower.
    private let frameInterval: TimeInterval = 1.0 / 30.0
    private var lastFrameTime: TimeInterval = -1
    /// Halve RGB (1920x1440 -> 960x720) to keep captures small; K scales to match.
    private let rgbScale: CGFloat = 0.5
    #if targetEnvironment(simulator)
    let simulated = SimulatedSensor()
    private var simTimer: Timer?
    #endif

    func start() {
        #if targetEnvironment(simulator)
        // No camera or LiDAR here; drive the same pipeline from a synthetic source.
        statusText = "ready"
        startSimulatedFeed()
        #else
        let config = ARWorldTrackingConfiguration()
        guard ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth) else {
            statusText = "this device has no LiDAR scene depth"
            return
        }
        config.frameSemantics = [.sceneDepth]
        session.delegate = self
        session.run(config)
        #endif
    }

    #if targetEnvironment(simulator)
    // MARK: Simulated sensor (simulator only — never compiled for a device)

    private func startSimulatedFeed() {
        simTimer = Timer.scheduledTimer(withTimeInterval: frameInterval, repeats: true) {
            [weak self] _ in self?.simulatedTick()
        }
    }

    private func simulatedTick() {
        guard isRecording else { return }
        let sim = simulated
        let index = frameCount
        let timestamp = Double(index) * frameInterval
        intrinsics = sim.intrinsics
        rgbSize = CGSize(width: sim.depthWidth, height: sim.depthHeight)

        inFlightLock.lock()
        let busy = inFlight >= maxInFlight
        if busy { droppedFrames += 1 } else { inFlight += 1 }
        inFlightLock.unlock()
        if busy { return }

        guard let image = SimulatedSensor.makeYCbCr420(fromSRGBA: sim.rgb,
                                                       width: sim.depthWidth, height: sim.depthHeight),
              let depth = SimulatedSensor.makeDepth(from: sim.depth,
                                                    width: sim.depthWidth, height: sim.depthHeight),
              let conf = SimulatedSensor.makeConfidence(from: sim.confidence,
                                                        width: sim.depthWidth, height: sim.depthHeight)
        else {
            inFlightLock.lock(); inFlight -= 1; inFlightLock.unlock()
            return
        }
        let transform = sim.transform(forFrame: index)
        if format == .r3d {
            let q = simd_quatf(transform), t = transform.columns.3
            poses.append([Double(q.imag.x), Double(q.imag.y), Double(q.imag.z), Double(q.real),
                          Double(t.x), Double(t.y), Double(t.z)])
            timestamps.append(timestamp)
        }
        frameCount += 1

        writeQueue.async { [self] in
            defer { inFlightLock.lock(); inFlight -= 1; inFlightLock.unlock() }
            do {
                switch format {
                case .r3d:
                    try writeFrame(index: index, image: image, depth: depth, confidence: conf)
                case .wurld:
                    try writeWurldFrame(index: index, timestamp: timestamp,
                                        transform: transform, image: image,
                                        depth: depth, confidence: conf)
                }
            } catch {
                DispatchQueue.main.async { self.statusText = "write failed: \(error.localizedDescription)" }
            }
        }
    }
    #endif

    func beginRecording() {
        let name = ISO8601DateFormatter().string(from: Date())
            .replacingOccurrences(of: ":", with: "-")
        let url = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("wurld-\(name)\(format.rawValue)")
        do {
            switch format {
            case .r3d:
                zip = try ZipWriter(url: url)
            case .wurld:
                FileManager.default.createFile(atPath: url.path, contents: nil)
                wlFile = try FileHandle(forWritingTo: url)
                // encoder + writer are created on the first frame, when the
                // depth grid and intrinsics are known
            }
        } catch {
            statusText = "cannot create \(url.lastPathComponent): \(error.localizedDescription)"
            return
        }
        captureURL = url
        poses = []; timestamps = []; frameCount = 0; lastFrameTime = -1; wlFirstTimestamp = nil
        inFlightLock.lock(); inFlight = 0; droppedFrames = 0; inFlightLock.unlock()
        isRecording = true
        statusText = "recording"
    }

    func endRecording() {
        isRecording = false
        statusText = "finalizing…"
        writeQueue.async { [self] in
            defer { zip = nil; wlEncoder = nil; wlWriter = nil; wlFile = nil }
            do {
                switch format {
                case .r3d:
                    try zip?.add(name: "metadata", data: metadataJSON())
                    try zip?.finish()
                case .wurld:
                    try wlEncoder?.finish()   // tail cluster(s) flow through weave
                    wlWriter?.finish()        // consolidated WURLD_FRAMES table
                    wlEncoder?.destroy()
                    try wlFile?.close()
                }
                DispatchQueue.main.async {
                    self.lastCaptureURL = self.captureURL
                    let skipped = self.droppedFrames > 0 ? ", \(self.droppedFrames) skipped" : ""
                    self.statusText = "saved \(self.captureURL?.lastPathComponent ?? "") (\(self.frameCount) frames\(skipped))"
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

        // Apply backpressure before claiming the frame: if the writer is still
        // busy, drop this one now rather than queue it. A skipped frame costs
        // one sample; a queued one stalls the session for several.
        inFlightLock.lock()
        let busy = inFlight >= maxInFlight
        if busy { droppedFrames += 1 } else { inFlight += 1 }
        inFlightLock.unlock()
        if busy { return }

        lastFrameTime = frame.timestamp

        let index = frameCount
        let camera = frame.camera
        intrinsics = camera.intrinsics
        rgbSize = camera.imageResolution

        // ARKit camera.transform is camera-to-world in the gravity-aligned world.
        let transform = camera.transform
        if format == .r3d {
            let q = simd_quatf(transform)
            let t = transform.columns.3
            poses.append([Double(q.imag.x), Double(q.imag.y), Double(q.imag.z), Double(q.real),
                          Double(t.x), Double(t.y), Double(t.z)])
            timestamps.append(frame.timestamp)
        }
        frameCount += 1

        let capturedImage = frame.capturedImage
        let depthMap = sceneDepth.depthMap
        let confidenceMap = sceneDepth.confidenceMap
        let timestamp = frame.timestamp
        writeQueue.async { [self] in
            defer {
                inFlightLock.lock(); inFlight -= 1; inFlightLock.unlock()
            }
            do {
                switch format {
                case .r3d:
                    try writeFrame(index: index, image: capturedImage,
                                   depth: depthMap, confidence: confidenceMap)
                case .wurld:
                    try writeWurldFrame(index: index, timestamp: timestamp,
                                            transform: transform, image: capturedImage,
                                            depth: depthMap, confidence: confidenceMap)
                }
            } catch {
                DispatchQueue.main.async { self.statusText = "write failed: \(error.localizedDescription)" }
            }
        }
    }

    // MARK: wurld path

    private func writeWurldFrame(index: Int, timestamp: Double,
                                     transform: simd_float4x4, image: CVPixelBuffer,
                                     depth: CVPixelBuffer, confidence: CVPixelBuffer?) throws {
        let dw = CVPixelBufferGetWidth(depth), dh = CVPixelBufferGetHeight(depth)
        depthSize = (dw, dh)

        if wlEncoder == nil {
            // First frame: intrinsics and the depth grid are now known.
            let K = intrinsics ?? matrix_identity_float3x3
            let sx = Double(dw) / Double(rgbSize.width), sy = Double(dh) / Double(rgbSize.height)
            let doc: [String: Any] = [
                "format": "wurld", "version": "1.2",   // SPEC §10; see wurld.container.FORMAT_VERSION
                "conventions": ["camera_axes": "RDF", "pose_direction": "camera_to_world",
                                "quaternion_order": "wxyz", "units": "meters",
                                "timestamp_units": "seconds"],
                "world": ["metric_scale": true, "gravity_in_world": [0, -1, 0],
                          "description": "WurldCam on-device recording (ARKit, RGB at depth grid)"],
                "cameras": ["0": ["model": "PINHOLE", "width": dw, "height": dh,
                                  "params": [Double(K[0][0]) * sx, Double(K[1][1]) * sy,
                                             Double(K[2][0]) * sx, Double(K[2][1]) * sy]]],
                "signals": [
                    ["id": "depth", "role": "depth",
                     "value_map": ["type": "inverse_depth", "near": wlNear,
                                   "far": wlFar, "levels": 65536, "invalid": 0]],
                    ["id": "confidence", "role": "confidence",
                     "value_map": ["type": "labels",
                                   "labels": ["0": "low", "1": "medium", "2": "high"]]],
                ],
                "frames": [],
            ]
            let writer = try WurldStreamWriter(doc: doc) { [weak self] data in
                self?.wlFile?.write(data)
            }
            wlWriter = writer
            // Creating the encoder emits the mux header through weave immediately.
            wlEncoder = try ChromapakzStreamEncoder(
                width: dw, height: dh, fps: Int(round(1.0 / frameInterval)),
                rgbKbps: 2000, near: wlNear, far: wlFar, includeConfidence: true,
                // Poses also go out as WebVTT cues so a capture that never touches a
                // desktop is still readable by plain ffmpeg. The binary table written
                // by WurldStreamWriter stays authoritative.
                textTrack: "wurld-poses"
            ) { chunk in writer.weave(chunk) }
        }

        // Pose first: pending poses flush ahead of the cluster holding this frame.
        let pose = canonicalPose(index: UInt32(index), time: timestamp, arkitTransform: transform)
        wlWriter?.addPose(pose)
        // Cue times rebase to the media timeline: ARKit timestamps are device uptime,
        // so an absolute cue would land far past the end of the video. The true value
        // stays in the payload.
        if wlFirstTimestamp == nil { wlFirstTimestamp = timestamp }
        let q = pose.qWXYZ, tr = pose.translation
        try? wlEncoder?.addText(
            "i=\(index) t=\(timestamp) camera=0 "
            + "q_wxyz=\(q.x),\(q.y),\(q.z),\(q.w) tr=\(tr.x),\(tr.y),\(tr.z)",
            timestamp: max(0, timestamp - (wlFirstTimestamp ?? timestamp)),
            duration: frameInterval)

        let rgba = try pixelConverter.sRGBA(from: image, width: dw, height: dh)
        let z = floatPlane(depth)
        var conf = [UInt16](repeating: 2, count: dw * dh)  // "high" when ARKit omits the map
        if let confidence {
            CVPixelBufferLockBaseAddress(confidence, .readOnly)
            let stride = CVPixelBufferGetBytesPerRow(confidence)
            let base = CVPixelBufferGetBaseAddress(confidence)!.assumingMemoryBound(to: UInt8.self)
            for row in 0..<dh {
                for col in 0..<dw { conf[row * dw + col] = UInt16(base[row * stride + col]) }
            }
            CVPixelBufferUnlockBaseAddress(confidence, .readOnly)
        }
        try wlEncoder?.addFrame(rgba: rgba,
                                depth: ChromapakzStreamEncoder.quantize(z, near: wlNear, far: wlFar),
                                confidence: conf)
    }

    private func floatPlane(_ buffer: CVPixelBuffer) -> [Float] {
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        let w = CVPixelBufferGetWidth(buffer), h = CVPixelBufferGetHeight(buffer)
        let stride = CVPixelBufferGetBytesPerRow(buffer) / MemoryLayout<Float>.size
        let base = CVPixelBufferGetBaseAddress(buffer)!.assumingMemoryBound(to: Float.self)
        var out = [Float](repeating: 0, count: w * h)
        for row in 0..<h {
            for col in 0..<w { out[row * w + col] = base[row * stride + col] }
        }
        return out
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
