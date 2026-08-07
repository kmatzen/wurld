import Foundation
import simd

// Synthetic recording through the exact Swift pipeline WurldCam uses:
// ChromapakzStreamEncoder + WurldStreamWriter. Output validated by Python.

let W = 64, H = 48, N = 20, FPS = 10
let NEAR = 0.1, FAR = 12.0

let outURL = URL(fileURLWithPath: CommandLine.arguments.count > 1
                 ? CommandLine.arguments[1] : "swift_take.wl.webm")
FileManager.default.createFile(atPath: outURL.path, contents: nil)
let file = try FileHandle(forWritingTo: outURL)

let doc: [String: Any] = [
    "format": "wurld", "version": "0.4",
    "conventions": ["camera_axes": "RDF", "pose_direction": "camera_to_world",
                    "quaternion_order": "wxyz", "units": "meters",
                    "timestamp_units": "seconds"],
    "world": ["metric_scale": true, "gravity_in_world": [0, -1, 0],
              "description": "swift harness take"],
    "cameras": ["0": ["model": "PINHOLE", "width": W, "height": H,
                      "params": [48.0, 48.0, 31.5, 23.5]]],
    "signals": [["id": "depth", "role": "depth",
                 "value_map": ["type": "inverse_depth", "near": NEAR, "far": FAR,
                               "levels": 65536, "invalid": 0]]],
    "frames": [],
]

let writer = try WurldStreamWriter(doc: doc) { data in file.write(data) }
let enc = try ChromapakzStreamEncoder(width: W, height: H, fps: FPS, rgbKbps: 500,
                                      near: NEAR, far: FAR) { writer.weave($0) }

for i in 0..<N {
    // ARKit-style RUB c2w: orbit in the XZ plane, y-up world
    let ang = Double(i) * 0.1
    let eye = simd_double3(2 * cos(ang), 0.5, 2 * sin(ang))
    // camera looks at origin: forward = normalize(origin - eye); RUB: -Z = forward
    let fwd = simd_normalize(-eye)
    let right = simd_normalize(simd_cross(fwd, simd_double3(0, 1, 0)))
    let up = simd_cross(right, fwd)
    var m = simd_float4x4(1)
    m.columns.0 = simd_float4(Float(right.x), Float(right.y), Float(right.z), 0)
    m.columns.1 = simd_float4(Float(up.x), Float(up.y), Float(up.z), 0)
    m.columns.2 = simd_float4(Float(-fwd.x), Float(-fwd.y), Float(-fwd.z), 0)  // RUB: +Z back
    m.columns.3 = simd_float4(Float(eye.x), Float(eye.y), Float(eye.z), 1)

    writer.addPose(canonicalPose(index: UInt32(i), time: Double(i) / Double(FPS),
                                 arkitTransform: m))

    var rgba = [UInt8](repeating: 0, count: W * H * 4)
    var z = [Float](repeating: 0, count: W * H)
    for v in 0..<H {
        for u in 0..<W {
            let k = v * W + u
            rgba[k * 4] = UInt8((u * 4 + i * 7) % 256)
            rgba[k * 4 + 1] = UInt8((v * 5) % 256)
            rgba[k * 4 + 2] = UInt8((i * 12) % 256)
            rgba[k * 4 + 3] = 255
            // depth ramp with a NaN hole
            z[k] = (u == 0 && v == 0) ? Float.nan
                 : Float(0.5) + Float(u + v) / Float(W + H) * 8.0
        }
    }
    try enc.addFrame(rgba: rgba, depth: ChromapakzStreamEncoder.quantize(z, near: NEAR, far: FAR))
}

try enc.finish()
writer.finish()
enc.destroy()
try file.close()
print("wrote \(outURL.path)")
