import Foundation
import simd

/// Live wurld recording (SPEC §9 chunked form) — the Swift port of
/// Python's `wurld.StreamWriter` / JS `WurldRecorder`.
///
/// Wraps the element-aligned chromapakz chunk stream: the WURLD header
/// document goes out right after the mux prefix, WURLD_POSES chunk tags
/// flush ahead of each cluster chunk, and a consolidated WURLD_FRAMES
/// table lands on `finish()`. Every emitted prefix is a valid, fully-posed
/// file up to the last flushed chunk — a crash loses the tail, not the take.
final class WurldStreamWriter {
    struct Pose {
        let index: UInt32
        let time: Double
        let qWXYZ: simd_double4  // unit quaternion, w first
        let translation: simd_double3
    }

    private let emit: (Data) -> Void
    private let headerTag: Data
    private var headerEmitted = false
    private var pending: [Pose] = []
    private var all: [Pose] = []

    /// `doc` is the WURLD header document (frames: []) — cameras, signals,
    /// conventions, world — serialized as-is.
    init(doc: [String: Any], onChunk: @escaping (Data) -> Void) throws {
        emit = onChunk
        let json = try JSONSerialization.data(withJSONObject: doc)
        headerTag = Self.buildTags([("WURLD", .string(json))])
    }

    /// Feed every chromapakz chunk through here, in order.
    func weave(_ chunk: Data) {
        if !headerEmitted {
            // First chunk is the whole file prefix; our header doc follows it.
            emit(chunk)
            emit(headerTag)
            headerEmitted = true
            return
        }
        flushPending()
        emit(chunk)
    }

    func addPose(_ pose: Pose) {
        pending.append(pose)
        all.append(pose)
    }

    /// Call after the encoder's finish() chunks have gone through `weave`.
    func finish() {
        flushPending()
        if !all.isEmpty {
            emit(Self.buildTags([("WURLD_FRAMES", .binary(Self.packFrames(all)))]))
        }
    }

    private func flushPending() {
        guard !pending.isEmpty else { return }
        emit(Self.buildTags([("WURLD_POSES", .binary(Self.packFrames(pending)))]))
        pending = []
    }

    // MARK: EBML (wurld SPEC §2: Tags / Tag / SimpleTag / TagName / TagString|TagBinary)

    enum TagValue { case string(Data), binary(Data) }

    private static let idTags: [UInt8] = [0x12, 0x54, 0xC3, 0x67]
    private static let idTag: [UInt8] = [0x73, 0x73]
    private static let idSimpleTag: [UInt8] = [0x67, 0xC8]
    private static let idTagName: [UInt8] = [0x45, 0xA3]
    private static let idTagString: [UInt8] = [0x44, 0x87]
    private static let idTagBinary: [UInt8] = [0x44, 0x85]

    private static func vintSize(_ size: Int) -> Data {
        var length = 1
        while size >= (1 << (7 * length)) - 1 { length += 1 }
        var v = UInt64(size) | (UInt64(1) << UInt64(7 * length))
        var bytes = [UInt8](repeating: 0, count: length)
        for i in stride(from: length - 1, through: 0, by: -1) {
            bytes[i] = UInt8(v & 0xFF)
            v >>= 8
        }
        return Data(bytes)
    }

    private static func element(_ id: [UInt8], _ payload: Data) -> Data {
        var out = Data(id)
        out.append(vintSize(payload.count))
        out.append(payload)
        return out
    }

    static func buildTags(_ entries: [(String, TagValue)]) -> Data {
        var body = Data()
        for (name, value) in entries {
            var simple = element(idTagName, Data(name.utf8))
            switch value {
            case .string(let d): simple.append(element(idTagString, d))
            case .binary(let d): simple.append(element(idTagBinary, d))
            }
            body.append(element(idTag, element(idSimpleTag, simple)))
        }
        return element(idTags, body)
    }

    // MARK: frame records (wurld SPEC §7: 45 bytes little-endian)

    static func packFrames(_ poses: [Pose]) -> Data {
        var out = Data(capacity: poses.count * 45)
        for p in poses {
            out.appendLE(p.index)
            out.appendLE(UInt32(0))  // camera index (single camera "0")
            out.appendLE(p.time.bitPattern)
            for k in 0..<4 { out.appendLE(Float(p.qWXYZ[k]).bitPattern) }
            for k in 0..<3 { out.appendLE(Float(p.translation[k]).bitPattern) }
            out.append(1)  // flags: pose_valid
        }
        return out
    }
}

/// ARKit camera-to-world (RUB axes) -> wurld canonical RDF pose.
/// Right-multiplies by diag(1, -1, -1): flips the camera's Y and Z columns.
func canonicalPose(index: UInt32, time: Double, arkitTransform m: simd_float4x4)
    -> WurldStreamWriter.Pose {
    var c2w = simd_double4x4(
        simd_double4(Double(m.columns.0.x), Double(m.columns.0.y), Double(m.columns.0.z), 0),
        simd_double4(Double(m.columns.1.x), Double(m.columns.1.y), Double(m.columns.1.z), 0),
        simd_double4(Double(m.columns.2.x), Double(m.columns.2.y), Double(m.columns.2.z), 0),
        simd_double4(Double(m.columns.3.x), Double(m.columns.3.y), Double(m.columns.3.z), 1)
    )
    c2w.columns.1 = -c2w.columns.1
    c2w.columns.2 = -c2w.columns.2
    let rotation = simd_double3x3(
        simd_double3(c2w.columns.0.x, c2w.columns.0.y, c2w.columns.0.z),
        simd_double3(c2w.columns.1.x, c2w.columns.1.y, c2w.columns.1.z),
        simd_double3(c2w.columns.2.x, c2w.columns.2.y, c2w.columns.2.z)
    )
    var q = simd_quatd(rotation)
    if q.real < 0 { q = simd_quatd(real: -q.real, imag: -q.imag) }  // w >= 0 canonical
    return WurldStreamWriter.Pose(
        index: index,
        time: time,
        qWXYZ: simd_double4(q.real, q.imag.x, q.imag.y, q.imag.z),
        translation: simd_double3(c2w.columns.3.x, c2w.columns.3.y, c2w.columns.3.z)
    )
}
