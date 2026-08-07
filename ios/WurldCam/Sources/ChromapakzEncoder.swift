import Foundation

/// Swift wrapper over the chromapakz streaming C ABI (`dc_stream_*`).
///
/// Chunks are element-aligned (the header is the whole file prefix; later
/// chunks are whole Cluster elements), so `WurldStreamWriter` can weave
/// pose tags between them without parsing byte boundaries. `emit_cues` is
/// off: injected tag bytes would invalidate cue offsets (wurld SPEC §9).
final class ChromapakzStreamEncoder {
    enum EncoderError: Error { case create(Int32), addFrame(Int32), finish(Int32) }

    private var handle: OpaquePointer?
    private let onChunk: (Data) -> Void
    private let signalID = "depth"

    init(width: Int, height: Int, fps: Int, rgbKbps: Int,
         near: Double, far: Double, onChunk: @escaping (Data) -> Void) throws {
        self.onChunk = onChunk
        var h: OpaquePointer?
        let rc = signalID.withCString { idPtr -> Int32 in
            var spec = dc_signal_spec_t(id: idPtr, data: nil, inverse_depth: 1,
                                        near_: near, far_: far, levels: 65536)
            return withUnsafePointer(to: &spec) { specPtr in
                dc_stream_create(Int32(width), Int32(height), Int32(fps), Int32(rgbKbps),
                                 1 /* has_rgb */, 0 /* emit_cues */, specPtr, 1, &h)
            }
        }
        guard rc == 0, h != nil else { throw EncoderError.create(rc) }
        handle = h
        try takeChunk { dc_stream_header(h, $0, $1) }
    }

    /// rgba: W*H*4 bytes; depth: W*H uint16 inverse-depth codes.
    func addFrame(rgba: [UInt8], depth: [UInt16]) throws {
        guard let h = handle else { return }
        let rc = try rgba.withUnsafeBufferPointer { rgbaBuf in
            try depth.withUnsafeBufferPointer { depthBuf -> Int32 in
                var planes: [UnsafePointer<UInt16>?] = [depthBuf.baseAddress]
                return try planes.withUnsafeMutableBufferPointer { planesBuf in
                    var out: UnsafeMutablePointer<UInt8>?
                    var outLen: Int = 0
                    let rc = dc_stream_add_frame(h, rgbaBuf.baseAddress,
                                                 planesBuf.baseAddress, &out, &outLen)
                    if rc == 0 { emit(out, outLen) }
                    return rc
                }
            }
        }
        guard rc == 0 else { throw EncoderError.addFrame(rc) }
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

    /// Float metres -> uint16 inverse-depth codes (NaN/out-of-range -> 0 invalid).
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
