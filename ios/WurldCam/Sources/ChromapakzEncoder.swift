import Foundation

/// Swift wrapper over the chromapakz streaming C ABI (`dc_stream_*`).
///
/// Chunks are element-aligned (the header is the whole file prefix; later
/// chunks are whole Cluster elements), so `WurldStreamWriter` can weave
/// pose tags between them without parsing byte boundaries. `emit_cues` is
/// off: injected tag bytes would invalidate cue offsets (wurld SPEC §9).
final class ChromapakzStreamEncoder {
    enum EncoderError: Error { case create(Int32), addFrame(Int32), addText(Int32), finish(Int32) }

    private var handle: OpaquePointer?
    private let onChunk: (Data) -> Void
    private let withConfidence: Bool
    private var hasTextTrack = false

    /// `textTrack` declares a WebVTT metadata track (chromapakz >= 0.5.0). It exists so
    /// tools that will not install anything can read per-frame data: ffmpeg's Matroska
    /// demuxer drops TagBinary, so the binary pose table is invisible to it.
    ///
    /// `depthWidth`/`depthHeight` give the depth and confidence planes their own
    /// geometry (chromapakz format v4, >= 0.10.0) — LiDAR depth stays 256x192 while
    /// RGB records at the camera's resolution. 0,0 keeps them at `width`x`height`,
    /// and such a file is byte-identical to what the pre-v4 entry points wrote.
    init(width: Int, height: Int, fps: Int, rgbKbps: Int,
         near: Double, far: Double, includeConfidence: Bool = false,
         depthWidth: Int = 0, depthHeight: Int = 0,
         textTrack: String? = nil,
         onChunk: @escaping (Data) -> Void) throws {
        self.onChunk = onChunk
        self.withConfidence = includeConfidence
        self.hasTextTrack = textTrack != nil
        var h: OpaquePointer?
        let rc = "depth".withCString { depthPtr -> Int32 in
            "confidence".withCString { confPtr -> Int32 in
                var specs = [dc_signal_spec3_t(id: depthPtr, data: nil, inverse_depth: 1,
                                               near_: near, far_: far, levels: 65536,
                                               view: nil, width: Int32(depthWidth),
                                               height: Int32(depthHeight))]
                if includeConfidence {
                    // The ARKit confidence map shares the depth grid.
                    specs.append(dc_signal_spec3_t(id: confPtr, data: nil, inverse_depth: 0,
                                                   near_: 0, far_: 0, levels: 65536,
                                                   view: nil, width: Int32(depthWidth),
                                                   height: Int32(depthHeight)))
                }
                // A single NULL id takes the conventional stream name "rgb".
                let rgbs = [dc_rgb_spec2_t(id: nil, kbps: Int32(rgbKbps), width: 0, height: 0)]
                func create(_ namePtr: UnsafePointer<CChar>?) -> Int32 {
                    specs.withUnsafeBufferPointer { specBuf in
                        rgbs.withUnsafeBufferPointer { rgbBuf in
                            dc_stream_create3(Int32(width), Int32(height), Int32(fps),
                                              rgbBuf.baseAddress, 1, 0 /* emit_cues */,
                                              specBuf.baseAddress, Int32(specs.count),
                                              namePtr, &h)
                        }
                    }
                }
                guard let name = textTrack else { return create(nil) }
                return name.withCString(create)
            }
        }
        guard rc == 0, h != nil else { throw EncoderError.create(rc) }
        handle = h
        try takeChunk { dc_stream_header(h, $0, $1) }
    }

    /// rgba: width*height*4 bytes; depth/confidence: uint16 planes at their own
    /// geometry (depthWidth*depthHeight when given, else width*height);
    /// confidence carries raw codes (0/1/2), required iff includeConfidence.
    func addFrame(rgba: [UInt8], depth: [UInt16], confidence: [UInt16]? = nil) throws {
        guard let h = handle else { return }
        precondition((confidence != nil) == withConfidence,
                     "confidence plane must match includeConfidence")
        let conf = confidence ?? []
        let rc = rgba.withUnsafeBufferPointer { rgbaBuf in
            depth.withUnsafeBufferPointer { depthBuf -> Int32 in
                conf.withUnsafeBufferPointer { confBuf -> Int32 in
                    var planes: [UnsafePointer<UInt16>?] = [depthBuf.baseAddress]
                    if withConfidence { planes.append(confBuf.baseAddress) }
                    return planes.withUnsafeMutableBufferPointer { planesBuf in
                        var out: UnsafeMutablePointer<UInt8>?
                        var outLen: Int = 0
                        let rc = dc_stream_add_frame(h, rgbaBuf.baseAddress,
                                                     planesBuf.baseAddress, &out, &outLen)
                        if rc == 0 { emit(out, outLen) }
                        return rc
                    }
                }
            }
        }
        guard rc == 0 else { throw EncoderError.addFrame(rc) }
    }

    /// Append one timed-text cue. Times are seconds on the media timeline.
    func addText(_ text: String, timestamp: Double, duration: Double) throws {
        guard let h = handle, hasTextTrack else { return }
        var utf8 = Array(text.utf8)
        try utf8.withUnsafeMutableBufferPointer { buf in
            try takeChunk { out, len in
                dc_stream_add_text(h, Int32((timestamp * 1000).rounded()),
                                   Int32(max(0, (duration * 1000).rounded())),
                                   buf.baseAddress, buf.count, out, len)
            }
        }
    }

    func finish() throws {
        guard let h = handle else { return }
        try takeChunk { dc_stream_finish(h, $0, $1) }
    }

    func destroy() {
        if let h = handle { dc_stream_destroy(h); handle = nil }
    }

    deinit { destroy() }

    private func takeChunk(
        _ fn: (UnsafeMutablePointer<UnsafeMutablePointer<UInt8>?>, UnsafeMutablePointer<Int>) -> Int32
    ) throws {
        var out: UnsafeMutablePointer<UInt8>?
        var outLen: Int = 0
        let rc = withUnsafeMutablePointer(to: &out) { o in
            withUnsafeMutablePointer(to: &outLen) { l in fn(o, l) }
        }
        guard rc == 0 else { throw EncoderError.finish(rc) }
        emit(out, outLen)
    }

    private func emit(_ out: UnsafeMutablePointer<UInt8>?, _ len: Int) {
        guard let out, len > 0 else { return }
        onChunk(Data(bytes: out, count: len))
        dc_free(out)
    }

    /// Float meters -> uint16 inverse-depth codes (NaN/out-of-range -> 0 invalid).
    static func quantize(_ z: [Float], near: Double, far: Double) -> [UInt16] {
        var out = [UInt16](repeating: 0, count: z.count)
        z.withUnsafeBufferPointer { zp in
            out.withUnsafeMutableBufferPointer { op in
                dc_quantize_inverse(zp.baseAddress, Int32(z.count), near, far, 65536, op.baseAddress)
            }
        }
        return out
    }
}
