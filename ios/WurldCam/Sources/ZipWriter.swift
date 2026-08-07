import Foundation
import zlib

/// Minimal streaming ZIP writer (store method — payloads are already JPEG/LZFSE
/// compressed). Entries are written as they arrive; the central directory lands
/// on `finish()`. No ZIP64: keep captures under 4 GB (a few minutes of LiDAR).
final class ZipWriter {
    private let handle: FileHandle
    private var central = Data()
    private var entryCount: UInt16 = 0
    private var offset: UInt32 = 0

    init(url: URL) throws {
        FileManager.default.createFile(atPath: url.path, contents: nil)
        handle = try FileHandle(forWritingTo: url)
    }

    func add(name: String, data: Data) throws {
        let nameBytes = Data(name.utf8)
        let crc = data.crc32()
        let size = UInt32(data.count)

        var local = Data()
        local.appendLE(UInt32(0x04034b50))
        local.appendLE(UInt16(20))      // version needed
        local.appendLE(UInt16(0))       // flags
        local.appendLE(UInt16(0))       // method: store
        local.appendLE(UInt32(0))       // dos time/date
        local.appendLE(crc)
        local.appendLE(size)            // compressed == uncompressed (store)
        local.appendLE(size)
        local.appendLE(UInt16(nameBytes.count))
        local.appendLE(UInt16(0))       // extra len
        local.append(nameBytes)

        try handle.write(contentsOf: local)
        try handle.write(contentsOf: data)

        var entry = Data()
        entry.appendLE(UInt32(0x02014b50))
        entry.appendLE(UInt16(20)); entry.appendLE(UInt16(20))
        entry.appendLE(UInt16(0)); entry.appendLE(UInt16(0)); entry.appendLE(UInt32(0))
        entry.appendLE(crc); entry.appendLE(size); entry.appendLE(size)
        entry.appendLE(UInt16(nameBytes.count))
        entry.appendLE(UInt16(0)); entry.appendLE(UInt16(0))    // extra, comment
        entry.appendLE(UInt16(0)); entry.appendLE(UInt16(0))    // disk, internal attrs
        entry.appendLE(UInt32(0))                               // external attrs
        entry.appendLE(offset)
        entry.append(nameBytes)
        central.append(entry)

        offset += UInt32(local.count) + size
        entryCount += 1
    }

    func finish() throws {
        try handle.write(contentsOf: central)
        var end = Data()
        end.appendLE(UInt32(0x06054b50))
        end.appendLE(UInt16(0)); end.appendLE(UInt16(0))
        end.appendLE(entryCount); end.appendLE(entryCount)
        end.appendLE(UInt32(central.count))
        end.appendLE(offset)
        end.appendLE(UInt16(0))
        try handle.write(contentsOf: end)
        try handle.close()
    }
}

extension Data {
    mutating func appendLE<T: FixedWidthInteger>(_ value: T) {
        Swift.withUnsafeBytes(of: value.littleEndian) { append(contentsOf: $0) }
    }

    func crc32() -> UInt32 {
        withUnsafeBytes { buf in
            UInt32(zlib.crc32(0, buf.bindMemory(to: UInt8.self).baseAddress, uInt(count)))
        }
    }
}
