// wurld writer — attach poses, calibration and IMU to an encoded WebM.
//
// Scope, and why it is this shape. Producing a *playable* wurld file needs a
// video encoder, and pulling libvpx into this header would destroy the property
// that makes the reader useful on a robot: no dependencies. But a robot already
// has an encoder — chromapakz is C++ — so the split that matters is the one
// this file implements. chromapakz writes the WebM; wurld attaches the
// metadata layer to it. Neither half needs the other's dependencies.
//
//   #include "wurld_write.hpp"
//   wurld::WriteDoc doc;
//   doc.cameras["0"] = {"PINHOLE", 640, 480, {525, 525, 320, 240}};
//   doc.frames.push_back({0, 0.0, "0", true, {1,0,0,0}, {0,0,0}});
//   doc.world_json = R"({"metric_scale":true})";
//   wurld::write_file("in.webm", "out.wl.webm", doc);
//
// The bytes it emits are checked against the Python writer for equality, not
// merely for readability: `pack_frames` and `pack_imu` must produce identical
// buffers, and a file written here must satisfy `wurld validate` and read back
// through all three readers (tests/test_cpp_writer.py).
//
// Layout follows SPEC §9.1: SeekHead, the source's header elements, the wurld
// Tags, Clusters, rebuilt Cues. Inserting the Tags shifts every Cluster, so
// Cues are rebuilt at the new offsets rather than carried over — a stale Cues
// element seeks to the wrong place, which looks like corrupt video rather than
// a muxing bug.

#ifndef WURLD_WRITE_HPP
#define WURLD_WRITE_HPP

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "wurld.hpp"

namespace wurld {

// ------------------------------------------------------------------ document

struct WriteCamera {
    std::string model = "PINHOLE";
    int width = 0;
    int height = 0;
    std::vector<double> params;
};

/// What to write. `*_json` fields take raw JSON so a caller can pass through
/// blocks this struct does not model without the writer having to understand
/// them.
struct WriteDoc {
    std::map<std::string, WriteCamera> cameras;
    std::vector<Frame> frames;
    std::vector<SignalMeta> signals;   ///< value_map is used as written
    std::map<std::string, std::vector<ImuSample>> imu;
    std::string world_json = "{}";
    std::string rigs_json;             ///< empty means omit
    /// Write poses as the binary table (SPEC §7) rather than the JSON array.
    /// Matches the Python writer's "auto": binary above 10k frames.
    enum class FramesFormat { Auto, Json, Binary } frames_format = FramesFormat::Auto;
};

constexpr size_t BINARY_FRAMES_THRESHOLD = 10000;

// ------------------------------------------------------------------- packing

namespace detail {

inline void put_le32(std::string& out, uint32_t v) {
    for (int k = 0; k < 4; ++k) out += static_cast<char>((v >> (8 * k)) & 0xFF);
}
inline void put_lef32(std::string& out, float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, 4);
    put_le32(out, bits);
}
inline void put_lef64(std::string& out, double d) {
    uint64_t bits;
    std::memcpy(&bits, &d, 8);
    for (int k = 0; k < 8; ++k) out += static_cast<char>((bits >> (8 * k)) & 0xFF);
}

}  // namespace detail

/// SPEC §7 binary frame table: "<IId4f3fB", 45 bytes per record.
inline std::string pack_frames(const std::vector<Frame>& frames,
                               const std::vector<std::string>& camera_keys) {
    std::string out;
    out.reserve(frames.size() * FRAME_RECORD_SIZE);
    for (const auto& f : frames) {
        auto it = std::find(camera_keys.begin(), camera_keys.end(), f.camera);
        // An unposed frame legitimately has no camera; anything else naming a
        // camera that is not declared would write a record no reader can resolve.
        uint32_t idx = 0;
        if (it != camera_keys.end()) {
            idx = static_cast<uint32_t>(it - camera_keys.begin());
        } else if (f.pose_valid && !f.camera.empty()) {
            throw Error("frame " + std::to_string(f.i) + " names camera '" + f.camera +
                        "' which is not declared");
        }
        detail::put_le32(out, f.i);
        detail::put_le32(out, idx);
        detail::put_lef64(out, f.t);
        for (int k = 0; k < 4; ++k) detail::put_lef32(out, static_cast<float>(f.q_wxyz[static_cast<size_t>(k)]));
        for (int k = 0; k < 3; ++k) detail::put_lef32(out, static_cast<float>(f.tr[static_cast<size_t>(k)]));
        out += static_cast<char>(f.pose_valid ? 1 : 0);
    }
    return out;
}

/// SPEC §8.3 IMU record: "<d3f3f", 32 bytes per sample.
inline std::string pack_imu(const std::vector<ImuSample>& samples) {
    std::string out;
    out.reserve(samples.size() * IMU_RECORD_SIZE);
    for (const auto& s : samples) {
        detail::put_lef64(out, s.t);
        for (float g : s.gyro) detail::put_lef32(out, g);
        for (float a : s.accel) detail::put_lef32(out, a);
    }
    return out;
}

// -------------------------------------------------------------- ebml writing

namespace ebml {

inline std::string encode_id(uint32_t id) {
    std::string out;
    int len = id > 0x00FFFFFF ? 4 : id > 0x0000FFFF ? 3 : id > 0x000000FF ? 2 : 1;
    for (int k = len - 1; k >= 0; --k) out += static_cast<char>((id >> (8 * k)) & 0xFF);
    return out;
}

/// Data size vint. `length` forces a width, which callers use to reserve room.
inline std::string encode_size(uint64_t value, int length = 0) {
    if (length == 0) {
        for (int n = 1; n <= 8; ++n) {
            if (value < (1ULL << (7 * n)) - 1) { length = n; break; }
        }
        if (length == 0) throw Error("value too large for an EBML size");
    }
    std::string out(static_cast<size_t>(length), '\0');
    uint64_t v = value | (1ULL << (7 * length));
    for (int k = 0; k < length; ++k)
        out[static_cast<size_t>(length - 1 - k)] = static_cast<char>((v >> (8 * k)) & 0xFF);
    return out;
}

inline std::string element(uint32_t id, const std::string& payload) {
    return encode_id(id) + encode_size(payload.size()) + payload;
}

inline std::string encode_uint(uint32_t id, uint64_t value) {
    std::string bytes;
    if (value == 0) {
        bytes = std::string(1, '\0');
    } else {
        for (int k = 7; k >= 0; --k) {
            uint8_t b = static_cast<uint8_t>((value >> (8 * k)) & 0xFF);
            if (!bytes.empty() || b) bytes += static_cast<char>(b);
        }
    }
    return element(id, bytes);
}

constexpr uint32_t CUES = 0x1C53BB6B;
constexpr uint32_t CUE_POINT = 0xBB;
constexpr uint32_t CUE_TIME = 0xB3;
constexpr uint32_t CUE_TRACK_POSITIONS = 0xB7;
constexpr uint32_t CUE_TRACK = 0xF7;
constexpr uint32_t CUE_CLUSTER_POSITION = 0xF1;
constexpr uint32_t SEEK_HEAD = 0x114D9B74;
constexpr uint32_t SEEK = 0x4DBB;
constexpr uint32_t SEEK_ID = 0x53AB;
constexpr uint32_t SEEK_POSITION = 0x53AC;
constexpr uint32_t CLUSTER_TIMESTAMP = 0xE7;
constexpr uint32_t VOID = 0xEC;

/// One Tags element, one SimpleTag per entry. `binary` picks TagBinary.
inline std::string build_tags(
    const std::vector<std::pair<std::string, std::pair<std::string, bool>>>& tags) {
    std::string body;
    for (const auto& kv : tags) {
        const std::string& name = kv.first;
        const std::string& value = kv.second.first;
        bool binary = kv.second.second;
        std::string v = element(binary ? TAG_BINARY : TAG_STRING, value);
        body += element(TAG, element(SIMPLE_TAG, element(TAG_NAME, name) + v));
    }
    return element(TAGS, body);
}

inline std::string build_cues(const std::vector<std::pair<uint64_t, uint64_t>>& clusters,
                              uint64_t track = 1) {
    std::string body;
    for (const auto& c : clusters) {
        std::string positions = encode_uint(CUE_TRACK, track) +
                                encode_uint(CUE_CLUSTER_POSITION, c.second);
        body += element(CUE_POINT,
                        encode_uint(CUE_TIME, c.first) +
                            element(CUE_TRACK_POSITIONS, positions));
    }
    return element(CUES, body);
}

/// Positions are fixed 8-byte uints so the SeekHead's size does not depend on
/// its values — that is what lets offsets be computed in one pass (SPEC §9.1).
inline std::string build_seek_head(const std::vector<std::pair<uint32_t, uint64_t>>& entries) {
    std::string body;
    for (const auto& e : entries) {
        std::string pos(8, '\0');
        for (int k = 0; k < 8; ++k)
            pos[static_cast<size_t>(7 - k)] = static_cast<char>((e.second >> (8 * k)) & 0xFF);
        body += element(SEEK, element(SEEK_ID, encode_id(e.first)) +
                                  element(SEEK_POSITION, pos));
    }
    return element(SEEK_HEAD, body);
}

inline size_t seek_head_size(size_t n_entries) {
    std::vector<std::pair<uint32_t, uint64_t>> dummy(n_entries, {SEGMENT, 0});
    return build_seek_head(dummy).size();
}

}  // namespace ebml

// ------------------------------------------------------------- json emission

namespace detail {

inline std::string json_str(const std::string& s) {
    std::string out;
    dump_string(out, s);
    return out;
}

inline std::string json_num(double d) { return dump_number(d); }

/// Validate that a caller-supplied JSON blob really is JSON before embedding it.
/// Splicing an unparsed string in would produce a file no reader can open, and
/// the failure would surface far from here.
inline std::string checked_json(const std::string& blob, const std::string& what) {
    if (blob.empty()) return "";
    try {
        Json v = parse_json(blob);
        if (!v.is_object()) throw Error(what + " must be a json object");
        return v.dump();
    } catch (const Error&) {
        throw;
    } catch (const std::exception& e) {
        throw Error(what + " is not valid json: " + e.what());
    }
}

}  // namespace detail

/// The WURLD document as JSON. Exposed because it is worth being able to
/// inspect what the writer will emit without writing a file.
inline std::string build_document(const WriteDoc& doc, bool binary_frames,
                                  const std::vector<std::string>& camera_keys) {
    // Keep in step with wurld.container.FORMAT_VERSION (SPEC §10).
    std::string out = "{\"format\":\"wurld\",\"version\":\"1.2\"";
    out += ",\"conventions\":{\"camera_axes\":\"RDF\","
           "\"pose_direction\":\"camera_to_world\","
           "\"quaternion_order\":\"wxyz\",\"units\":\"meters\","
           "\"timestamp_units\":\"seconds\"}";

    std::string world = detail::checked_json(doc.world_json, "world");
    out += ",\"world\":" + (world.empty() ? "{}" : world);

    out += ",\"cameras\":{";
    bool first = true;
    for (const auto& kv : doc.cameras) {
        if (!first) out += ',';
        first = false;
        out += detail::json_str(kv.first) + ":{\"model\":" + detail::json_str(kv.second.model) +
               ",\"width\":" + detail::json_num(kv.second.width) +
               ",\"height\":" + detail::json_num(kv.second.height) + ",\"params\":[";
        for (size_t k = 0; k < kv.second.params.size(); ++k) {
            if (k) out += ',';
            out += detail::json_num(kv.second.params[k]);
        }
        out += "]}";
    }
    out += "}";

    out += ",\"signals\":[";
    for (size_t k = 0; k < doc.signals.size(); ++k) {
        if (k) out += ',';
        out += "{\"id\":" + detail::json_str(doc.signals[k].id) +
               ",\"role\":" + detail::json_str(doc.signals[k].role) +
               ",\"value_map\":" + doc.signals[k].value_map.dump() + "}";
    }
    out += "]";

    if (binary_frames) {
        // The descriptor: readers use it to resolve camera indices and to check
        // the table is complete.
        out += ",\"frames\":[],\"frames_binary\":{\"version\":1,\"count\":" +
               std::to_string(doc.frames.size()) + ",\"cameras\":[";
        for (size_t k = 0; k < camera_keys.size(); ++k) {
            if (k) out += ',';
            out += detail::json_str(camera_keys[k]);
        }
        out += "]}";
    } else {
        out += ",\"frames\":[";
        for (size_t k = 0; k < doc.frames.size(); ++k) {
            const Frame& f = doc.frames[k];
            if (k) out += ',';
            out += "{\"i\":" + std::to_string(f.i) + ",\"t\":" + detail::json_num(f.t);
            if (f.pose_valid) {
                out += ",\"camera\":" + detail::json_str(f.camera) + ",\"q_wxyz\":[";
                for (size_t j = 0; j < 4; ++j) {
                    if (j) out += ',';
                    out += detail::json_num(f.q_wxyz[j]);
                }
                out += "],\"tr\":[";
                for (size_t j = 0; j < 3; ++j) {
                    if (j) out += ',';
                    out += detail::json_num(f.tr[j]);
                }
                out += "]";
            } else {
                out += ",\"pose_valid\":false";
            }
            out += "}";
        }
        out += "]";
    }

    if (!doc.imu.empty()) {
        out += ",\"imu\":{";
        first = true;
        for (const auto& kv : doc.imu) {
            if (!first) out += ',';
            first = false;
            double rate = 0.0;
            if (kv.second.size() > 1) {
                double span = kv.second.back().t - kv.second.front().t;
                if (span > 0)
                    rate = static_cast<double>(kv.second.size() - 1) / span;
            }
            out += detail::json_str(kv.first) + ":{\"count\":" +
                   std::to_string(kv.second.size());
            if (rate > 0) out += ",\"rate_hz\":" + detail::json_num(std::round(rate * 10) / 10);
            out += "}";
        }
        out += "}";
    }

    std::string rigs = detail::checked_json(doc.rigs_json, "rigs");
    if (!rigs.empty()) out += ",\"rigs\":" + rigs;

    out += "}";
    return out;
}

// ---------------------------------------------------------------- the writer

/// Rebuild `webm` into the batch layout with wurld metadata attached.
///
/// Takes and returns whole files in memory. A robot recording for hours should
/// stream instead; this is for the finalise-a-clip case, and saying so is
/// better than pretending an in-memory rebuild scales.
inline std::string attach(const std::string& webm, const WriteDoc& doc) {
    std::vector<std::string> camera_keys;
    for (const auto& kv : doc.cameras) camera_keys.push_back(kv.first);  // sorted: std::map

    bool binary = doc.frames_format == WriteDoc::FramesFormat::Binary ||
                  (doc.frames_format == WriteDoc::FramesFormat::Auto &&
                   doc.frames.size() > BINARY_FRAMES_THRESHOLD);

    std::vector<std::pair<std::string, std::pair<std::string, bool>>> tags;
    tags.emplace_back("WURLD", std::make_pair(build_document(doc, binary, camera_keys), false));
    if (binary)
        tags.emplace_back("WURLD_FRAMES",
                          std::make_pair(pack_frames(doc.frames, camera_keys), true));
    for (const auto& kv : doc.imu)
        tags.emplace_back("WURLD_IMU_" + kv.first, std::make_pair(pack_imu(kv.second), true));

    const std::string tags_bytes = ebml::build_tags(tags);

    // Locate the Segment.
    const auto* p = reinterpret_cast<const uint8_t*>(webm.data());
    size_t pos = 0, idu = 0, szu = 0;
    uint32_t id = ebml::read_id(p, webm.size(), idu);
    if (id != 0x1A45DFA3) throw Error("not an EBML/Matroska file");
    uint64_t size = ebml::read_size(p + idu, webm.size() - idu, szu);
    pos = idu + szu + static_cast<size_t>(size);

    size_t seg_start = pos;
    id = ebml::read_id(p + pos, webm.size() - pos, idu);
    if (id != ebml::SEGMENT) throw Error("expected a Segment element");
    size = ebml::read_size(p + pos + idu, webm.size() - pos - idu, szu);
    size_t payload_start = pos + idu + szu;
    size_t payload_end = (size == UINT64_MAX) ? webm.size()
                                              : payload_start + static_cast<size_t>(size);
    if (payload_end > webm.size()) payload_end = webm.size();

    // Split the Segment: header elements, Clusters, everything after.
    std::vector<std::string> head_parts, body_parts;
    std::vector<std::pair<uint64_t, uint64_t>> cluster_offsets;  // (timestamp, body offset)
    size_t body_len = 0;
    bool seen_cluster = false;

    size_t cur = payload_start;
    while (cur < payload_end) {
        size_t eidu = 0, eszu = 0;
        uint32_t eid = ebml::read_id(p + cur, payload_end - cur, eidu);
        uint64_t esz = ebml::read_size(p + cur + eidu, payload_end - cur - eidu, eszu);
        size_t pstart = cur + eidu + eszu;
        size_t pend = (esz == UINT64_MAX) ? payload_end : pstart + static_cast<size_t>(esz);
        if (pend > payload_end) throw Error("truncated element in Segment");
        std::string raw = webm.substr(cur, pend - cur);

        if (eid == ebml::CUES || eid == ebml::SEEK_HEAD || eid == ebml::VOID) {
            // Rebuilt below. Void is reserved padding some muxers leave in the
            // header; carrying it forward would break the fixed-size SeekHead.
        } else if (eid == ebml::TAGS && !seen_cluster) {
            // Attaching to a file that already has wurld metadata replaces it.
            // Carrying the old tags forward would leave two WURLD documents in
            // one file, and which one a reader picks is a coin toss. Foreign
            // tags — chromapakz's own, most of all — must survive untouched.
            detail::Tags existing;
            detail::collect_simple_tags(webm.substr(pstart, pend - pstart), existing);
            std::vector<std::pair<std::string, std::pair<std::string, bool>>> keep;
            for (const auto& kv : existing) {
                if (kv.first.rfind("WURLD", 0) == 0) continue;
                keep.push_back(kv);
            }
            if (!keep.empty()) head_parts.push_back(ebml::build_tags(keep));
        } else if (eid == ebml::CLUSTER) {
            seen_cluster = true;
            uint64_t ts = 0;
            size_t c = pstart;
            while (c < pend) {
                size_t ciu = 0, csu = 0;
                uint32_t cid = ebml::read_id(p + c, pend - c, ciu);
                uint64_t csz = ebml::read_size(p + c + ciu, pend - c - ciu, csu);
                size_t cps = c + ciu + csu;
                size_t cpe = (csz == UINT64_MAX) ? pend : cps + static_cast<size_t>(csz);
                if (cid == ebml::CLUSTER_TIMESTAMP) {
                    ts = 0;
                    for (size_t k = cps; k < cpe && k < webm.size(); ++k)
                        ts = (ts << 8) | static_cast<uint8_t>(webm[k]);
                    break;
                }
                if (cpe <= c) break;
                c = cpe;
            }
            cluster_offsets.emplace_back(ts, body_len);
            body_len += raw.size();
            body_parts.push_back(std::move(raw));
        } else if (!seen_cluster) {
            head_parts.push_back(std::move(raw));
        } else {
            body_len += raw.size();
            body_parts.push_back(std::move(raw));
        }
        cur = pend;
    }

    // SeekHead entries: every header element, the Tags, the first Cluster, Cues.
    size_t n_entries = head_parts.size() + 1 + (cluster_offsets.empty() ? 1 : 2);
    size_t sh_size = ebml::seek_head_size(n_entries);

    std::vector<std::pair<uint32_t, uint64_t>> entries;
    uint64_t at = sh_size;
    std::string head;
    {
        // Re-read each header element's id for its SeekHead entry.
        for (const auto& raw : head_parts) {
            size_t hu = 0;
            uint32_t hid = ebml::read_id(reinterpret_cast<const uint8_t*>(raw.data()),
                                         raw.size(), hu);
            entries.emplace_back(hid, at);
            head += raw;
            at += raw.size();
        }
    }
    entries.emplace_back(ebml::TAGS, at);
    head += tags_bytes;
    at += tags_bytes.size();
    uint64_t first_cluster_pos = at;
    if (!cluster_offsets.empty()) entries.emplace_back(ebml::CLUSTER, first_cluster_pos);
    entries.emplace_back(ebml::CUES, first_cluster_pos + body_len);

    std::string seek_head = ebml::build_seek_head(entries);
    if (seek_head.size() != sh_size) throw Error("SeekHead size must be value-independent");

    std::vector<std::pair<uint64_t, uint64_t>> cues;
    cues.reserve(cluster_offsets.size());
    for (const auto& c : cluster_offsets)
        cues.emplace_back(c.first, first_cluster_pos + c.second);

    std::string payload = seek_head + head;
    for (const auto& b : body_parts) payload += b;
    payload += ebml::build_cues(cues);

    return webm.substr(0, seg_start) + ebml::encode_id(ebml::SEGMENT) +
           ebml::encode_size(payload.size(), 8) + payload + webm.substr(payload_end);
}

/// Read `in_path`, attach `doc`, write `out_path`.
inline void write_file(const std::string& in_path, const std::string& out_path,
                       const WriteDoc& doc) {
    std::ifstream in(in_path, std::ios::binary);
    if (!in) throw Error("cannot open " + in_path);
    std::ostringstream buf;
    buf << in.rdbuf();
    std::string result = attach(buf.str(), doc);

    std::ofstream out(out_path, std::ios::binary);
    if (!out) throw Error("cannot write " + out_path);
    out.write(result.data(), static_cast<std::streamsize>(result.size()));
    if (!out) throw Error("failed writing " + out_path);
}

}  // namespace wurld

#endif  // WURLD_WRITE_HPP
