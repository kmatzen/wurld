// wurld — single-header C++17 reader for posed sensor video.
//
// Scope, stated up front so nothing is implied: this reads a wurld file's
// *metadata* — calibration, per-frame poses, timestamps, signal descriptors,
// rigs and IMU. It does not decode pixels or depth. Decoding needs libvpx and
// chromapakz; this header deliberately has **no dependencies at all** so it can
// drop into a robot without dragging a codec stack behind it. For consumers
// that do want pixels, `Document::cluster_start` reports where the Clusters
// begin, which is what a decoder needs.
//
// This is the C++ counterpart of `wurld.remote.fetch_header`, and it is checked
// against the Python implementation field-by-field on generated files
// (tests/test_cpp_reader.py) rather than merely against its own expectations.
//
// Conventions are fixed by the spec and are not negotiable per file: RDF camera
// axes, camera-to-world poses, wxyz quaternions, metres, seconds. `c2w()`
// returns a row-major 4x4.
//
//   #include "wurld.hpp"
//   auto doc = wurld::read("scene.wurld.webm");
//   for (const auto& f : doc.frames)
//       if (f.pose_valid) use(f.c2w());
//
// Traversal seeks over payloads and reads only Tags elements, so opening a
// 10 GB file costs a handful of seeks rather than 10 GB of I/O.

#ifndef WURLD_HPP
#define WURLD_HPP

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <map>

#include <stdexcept>
#include <string>
#include <vector>

namespace wurld {

// ---------------------------------------------------------------- exceptions

struct Error : std::runtime_error {
    explicit Error(const std::string& what) : std::runtime_error("wurld: " + what) {}
};

// --------------------------------------------------------------------- json
//
// A small JSON value. The document this parses is machine-written by the wurld
// writers, but "machine-written" is not a licence to be sloppy: unterminated
// strings, bad escapes and trailing garbage are rejected rather than
// half-parsed, because a truncated file is exactly when a reader must not
// invent plausible values.

class Json {
  public:
    enum class Type { Null, Bool, Number, String, Array, Object };

    Json() = default;
    Type type = Type::Null;
    bool boolean = false;
    double number = 0.0;
    std::string string;
    std::vector<Json> array;
    std::vector<std::pair<std::string, Json>> object;  // insertion order preserved

    bool is_null() const { return type == Type::Null; }
    bool is_object() const { return type == Type::Object; }
    bool is_array() const { return type == Type::Array; }
    bool is_number() const { return type == Type::Number; }
    bool is_string() const { return type == Type::String; }
    bool is_bool() const { return type == Type::Bool; }

    const Json* find(const std::string& key) const {
        if (type != Type::Object) return nullptr;
        for (const auto& kv : object)
            if (kv.first == key) return &kv.second;
        return nullptr;
    }
    bool has(const std::string& key) const { return find(key) != nullptr; }

    const Json& at(const std::string& key) const {
        const Json* v = find(key);
        if (!v) throw Error("missing json key '" + key + "'");
        return *v;
    }

    double num(const std::string& key, double dflt = 0.0) const {
        const Json* v = find(key);
        return v && v->is_number() ? v->number : dflt;
    }
    std::string str(const std::string& key, const std::string& dflt = "") const {
        const Json* v = find(key);
        return v && v->is_string() ? v->string : dflt;
    }
    // Tri-state: a missing `metric_scale` is not the same claim as `false`.
    int tribool(const std::string& key) const {
        const Json* v = find(key);
        if (!v || !v->is_bool()) return -1;
        return v->boolean ? 1 : 0;
    }

    std::string dump() const {
        std::string out;
        dump_into(out);
        return out;
    }

  private:
    void dump_into(std::string& out) const;
};

namespace detail {

class JsonParser {
  public:
    explicit JsonParser(const std::string& text) : s_(text) {}

    Json parse() {
        Json v = value();
        skip_ws();
        if (p_ != s_.size()) fail("trailing content after json value");
        return v;
    }

  private:
    const std::string& s_;
    size_t p_ = 0;

    [[noreturn]] void fail(const std::string& why) const {
        throw Error("json at byte " + std::to_string(p_) + ": " + why);
    }
    void skip_ws() {
        while (p_ < s_.size() && (s_[p_] == ' ' || s_[p_] == '\t' || s_[p_] == '\n' ||
                                  s_[p_] == '\r'))
            ++p_;
    }
    char peek() {
        skip_ws();
        if (p_ >= s_.size()) fail("unexpected end of input");
        return s_[p_];
    }
    void expect(char c) {
        if (peek() != c) fail(std::string("expected '") + c + "'");
        ++p_;
    }
    bool literal(const char* lit) {
        size_t n = std::strlen(lit);
        if (s_.compare(p_, n, lit) != 0) return false;
        p_ += n;
        return true;
    }

    Json value() {
        switch (peek()) {
            case '{': return object();
            case '[': return array();
            case '"': {
                Json v;
                v.type = Json::Type::String;
                v.string = string();
                return v;
            }
            case 't': {
                if (!literal("true")) fail("bad literal");
                Json v; v.type = Json::Type::Bool; v.boolean = true; return v;
            }
            case 'f': {
                if (!literal("false")) fail("bad literal");
                Json v; v.type = Json::Type::Bool; v.boolean = false; return v;
            }
            case 'n': {
                if (!literal("null")) fail("bad literal");
                return Json{};
            }
            default: return number();
        }
    }

    Json object() {
        expect('{');
        Json v;
        v.type = Json::Type::Object;
        if (peek() == '}') { ++p_; return v; }
        for (;;) {
            std::string key = (peek() == '"') ? string() : (fail("object key must be a string"), "");
            expect(':');
            v.object.emplace_back(std::move(key), value());
            char c = peek();
            ++p_;
            if (c == '}') break;
            if (c != ',') fail("expected ',' or '}'");
        }
        return v;
    }

    Json array() {
        expect('[');
        Json v;
        v.type = Json::Type::Array;
        if (peek() == ']') { ++p_; return v; }
        for (;;) {
            v.array.push_back(value());
            char c = peek();
            ++p_;
            if (c == ']') break;
            if (c != ',') fail("expected ',' or ']'");
        }
        return v;
    }

    static void utf8(std::string& out, uint32_t cp) {
        if (cp < 0x80) {
            out += static_cast<char>(cp);
        } else if (cp < 0x800) {
            out += static_cast<char>(0xC0 | (cp >> 6));
            out += static_cast<char>(0x80 | (cp & 0x3F));
        } else if (cp < 0x10000) {
            out += static_cast<char>(0xE0 | (cp >> 12));
            out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
            out += static_cast<char>(0x80 | (cp & 0x3F));
        } else {
            out += static_cast<char>(0xF0 | (cp >> 18));
            out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
            out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
            out += static_cast<char>(0x80 | (cp & 0x3F));
        }
    }

    uint32_t hex4() {
        if (p_ + 4 > s_.size()) fail("truncated \\u escape");
        uint32_t v = 0;
        for (int k = 0; k < 4; ++k) {
            char c = s_[p_++];
            v <<= 4;
            if (c >= '0' && c <= '9') v |= static_cast<uint32_t>(c - '0');
            else if (c >= 'a' && c <= 'f') v |= static_cast<uint32_t>(c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') v |= static_cast<uint32_t>(c - 'A' + 10);
            else fail("bad hex digit in \\u escape");
        }
        return v;
    }

    std::string string() {
        expect('"');
        std::string out;
        for (;;) {
            if (p_ >= s_.size()) fail("unterminated string");
            char c = s_[p_++];
            if (c == '"') break;
            if (c != '\\') { out += c; continue; }
            if (p_ >= s_.size()) fail("unterminated escape");
            char e = s_[p_++];
            switch (e) {
                case '"': out += '"'; break;
                case '\\': out += '\\'; break;
                case '/': out += '/'; break;
                case 'b': out += '\b'; break;
                case 'f': out += '\f'; break;
                case 'n': out += '\n'; break;
                case 'r': out += '\r'; break;
                case 't': out += '\t'; break;
                case 'u': {
                    uint32_t cp = hex4();
                    // Surrogate pair: Python's json emits these for astral chars.
                    if (cp >= 0xD800 && cp <= 0xDBFF && p_ + 1 < s_.size() &&
                        s_[p_] == '\\' && s_[p_ + 1] == 'u') {
                        size_t save = p_;
                        p_ += 2;
                        uint32_t lo = hex4();
                        if (lo >= 0xDC00 && lo <= 0xDFFF)
                            cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                        else
                            p_ = save;  // lone high surrogate; emit as-is
                    }
                    utf8(out, cp);
                    break;
                }
                default: fail("unknown escape");
            }
        }
        return out;
    }

    Json number() {
        size_t start = p_;
        bool negative = false;
        if (p_ < s_.size() && (s_[p_] == '-' || s_[p_] == '+')) {
            negative = s_[p_] == '-';
            ++p_;
        }
        bool any = false;
        while (p_ < s_.size() && ((s_[p_] >= '0' && s_[p_] <= '9') || s_[p_] == '.' ||
                                  s_[p_] == 'e' || s_[p_] == 'E' || s_[p_] == '-' ||
                                  s_[p_] == '+')) {
            any = true;
            ++p_;
        }
        // NaN / Infinity: Python's json emits these bare, and depth value maps
        // legitimately contain them.
        if (!any) {
            // The sign was already consumed above, so apply it here — losing it
            // would turn -Infinity into +Infinity, which reads as valid data.
            if (literal("NaN")) {
                Json v; v.type = Json::Type::Number; v.number = NAN; return v;
            }
            if (literal("Infinity")) {
                Json v; v.type = Json::Type::Number;
                v.number = negative ? -INFINITY : INFINITY;
                return v;
            }
            fail("expected a value");
        }
        std::string tok = s_.substr(start, p_ - start);
        if (tok == "-" || tok == "+") fail("bad number");
        try {
            size_t used = 0;
            double d = std::stod(tok, &used);
            if (used != tok.size()) fail("bad number '" + tok + "'");
            Json v; v.type = Json::Type::Number; v.number = d; return v;
        } catch (const Error&) {
            throw;
        } catch (const std::exception&) {
            fail("bad number '" + tok + "'");
        }
    }
};

inline void dump_string(std::string& out, const std::string& s) {
    out += '"';
    for (char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof buf, "\\u%04x", c);
                    out += buf;
                } else {
                    out += c;
                }
        }
    }
    out += '"';
}

inline std::string dump_number(double d) {
    if (std::isnan(d)) return "NaN";
    if (std::isinf(d)) return d > 0 ? "Infinity" : "-Infinity";
    char buf[40];
    if (d == static_cast<double>(static_cast<long long>(d)) && std::fabs(d) < 1e15) {
        std::snprintf(buf, sizeof buf, "%lld", static_cast<long long>(d));
    } else {
        std::snprintf(buf, sizeof buf, "%.17g", d);
    }
    return buf;
}

}  // namespace detail

inline void Json::dump_into(std::string& out) const {
    switch (type) {
        case Type::Null: out += "null"; break;
        case Type::Bool: out += boolean ? "true" : "false"; break;
        case Type::Number: out += detail::dump_number(number); break;
        case Type::String: detail::dump_string(out, string); break;
        case Type::Array: {
            out += '[';
            for (size_t k = 0; k < array.size(); ++k) {
                if (k) out += ',';
                array[k].dump_into(out);
            }
            out += ']';
            break;
        }
        case Type::Object: {
            out += '{';
            for (size_t k = 0; k < object.size(); ++k) {
                if (k) out += ',';
                detail::dump_string(out, object[k].first);
                out += ':';
                object[k].second.dump_into(out);
            }
            out += '}';
            break;
        }
    }
}

inline Json parse_json(const std::string& text) { return detail::JsonParser(text).parse(); }

// --------------------------------------------------------------------- model

struct Camera {
    std::string model;
    int width = 0;
    int height = 0;
    std::vector<double> params;
};

struct Frame {
    uint32_t i = 0;
    double t = 0.0;
    std::string camera;
    bool pose_valid = true;
    std::array<double, 4> q_wxyz{{1, 0, 0, 0}};
    std::array<double, 3> tr{{0, 0, 0}};

    /// Row-major 4x4 camera-to-world. Undefined when !pose_valid, so check first.
    std::array<double, 16> c2w() const {
        const double w = q_wxyz[0], x = q_wxyz[1], y = q_wxyz[2], z = q_wxyz[3];
        const double n = std::sqrt(w * w + x * x + y * y + z * z);
        const double s = (n > 0) ? 1.0 / n : 1.0;
        const double W = w * s, X = x * s, Y = y * s, Z = z * s;
        return {{1 - 2 * (Y * Y + Z * Z), 2 * (X * Y - W * Z), 2 * (X * Z + W * Y), tr[0],
                 2 * (X * Y + W * Z), 1 - 2 * (X * X + Z * Z), 2 * (Y * Z - W * X), tr[1],
                 2 * (X * Z - W * Y), 2 * (Y * Z + W * X), 1 - 2 * (X * X + Y * Y), tr[2],
                 0, 0, 0, 1}};
    }
};

struct SignalMeta {
    std::string id;
    std::string role;
    Json value_map;
};

struct ImuSample {
    double t = 0.0;
    std::array<float, 3> gyro{{0, 0, 0}};
    std::array<float, 3> accel{{0, 0, 0}};
};

struct ImuStream {
    std::string id;
    std::vector<ImuSample> samples;
};

struct Document {
    std::string version;
    std::map<std::string, Camera> cameras;
    std::vector<Frame> frames;
    std::vector<SignalMeta> signals;
    Json world;
    Json rigs;
    std::map<std::string, ImuStream> imu;
    Json raw;  ///< the whole WURLD document, for fields this struct does not model

    /// Byte range [first, last) covering Clusters — what to hand a decoder.
    uint64_t cluster_start = 0;
    uint64_t file_size = 0;

    const Camera* camera_for(const Frame& f) const {
        auto it = cameras.find(f.camera);
        return it == cameras.end() ? nullptr : &it->second;
    }
    size_t posed_frames() const {
        size_t n = 0;
        for (const auto& f : frames) n += f.pose_valid ? 1 : 0;
        return n;
    }
};

// ---------------------------------------------------------------------- ebml

namespace ebml {

constexpr uint32_t SEGMENT = 0x18538067;
constexpr uint32_t TAGS = 0x1254C367;
constexpr uint32_t TAG = 0x7373;
constexpr uint32_t SIMPLE_TAG = 0x67C8;
constexpr uint32_t TAG_NAME = 0x45A3;
constexpr uint32_t TAG_STRING = 0x4487;
constexpr uint32_t TAG_BINARY = 0x4485;
constexpr uint32_t CLUSTER = 0x1F43B675;

/// Element id: the leading byte's high bits give the width, and the marker bit
/// is *kept* (ids are conventionally written with it).
inline uint32_t read_id(const uint8_t* p, size_t avail, size_t& used) {
    if (avail < 1) throw Error("truncated element id");
    uint8_t b = p[0];
    int len = b & 0x80 ? 1 : b & 0x40 ? 2 : b & 0x20 ? 3 : b & 0x10 ? 4 : 0;
    if (!len) throw Error("invalid element id marker byte");
    if (avail < static_cast<size_t>(len)) throw Error("truncated element id");
    uint32_t v = 0;
    for (int k = 0; k < len; ++k) v = (v << 8) | p[k];
    used = static_cast<size_t>(len);
    return v;
}

/// Data size vint: the marker bit is *stripped*. Returns UINT64_MAX for the
/// all-ones "unknown size" form, which live recordings use for the Segment.
inline uint64_t read_size(const uint8_t* p, size_t avail, size_t& used) {
    if (avail < 1) throw Error("truncated element size");
    uint8_t b = p[0];
    int len = 0;
    for (int k = 0; k < 8; ++k)
        if (b & (0x80 >> k)) { len = k + 1; break; }
    if (!len) throw Error("invalid element size marker byte");
    if (avail < static_cast<size_t>(len)) throw Error("truncated element size");
    uint64_t v = b & static_cast<uint8_t>(0xFF >> len);
    bool all_ones = v == static_cast<uint64_t>(0xFF >> len);
    for (int k = 1; k < len; ++k) {
        v = (v << 8) | p[k];
        all_ones = all_ones && p[k] == 0xFF;
    }
    used = static_cast<size_t>(len);
    return all_ones ? UINT64_MAX : v;
}

}  // namespace ebml

// -------------------------------------------------------------- binary tables

// SPEC §7: "<IId4f3fB" — 45 bytes, little-endian, no padding.
constexpr size_t FRAME_RECORD_SIZE = 45;
// SPEC §8.3: "<d3f3f" — 32 bytes.
constexpr size_t IMU_RECORD_SIZE = 32;

namespace detail {

inline uint32_t le32(const uint8_t* p) {
    return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
           (static_cast<uint32_t>(p[2]) << 16) | (static_cast<uint32_t>(p[3]) << 24);
}
inline float lef32(const uint8_t* p) {
    uint32_t bits = le32(p);
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}
inline double lef64(const uint8_t* p) {
    uint64_t bits = 0;
    for (int k = 7; k >= 0; --k) bits = (bits << 8) | p[k];
    double d;
    std::memcpy(&d, &bits, 8);
    return d;
}

}  // namespace detail

inline std::vector<Frame> unpack_frames(const std::string& buf,
                                        const std::vector<std::string>& camera_keys) {
    if (buf.size() % FRAME_RECORD_SIZE)
        throw Error("WURLD_FRAMES length " + std::to_string(buf.size()) +
                    " is not a multiple of " + std::to_string(FRAME_RECORD_SIZE));
    const auto* p = reinterpret_cast<const uint8_t*>(buf.data());
    std::vector<Frame> out;
    out.reserve(buf.size() / FRAME_RECORD_SIZE);
    for (size_t off = 0; off + FRAME_RECORD_SIZE <= buf.size(); off += FRAME_RECORD_SIZE) {
        const uint8_t* r = p + off;
        Frame f;
        f.i = detail::le32(r);
        uint32_t cam = detail::le32(r + 4);
        // A camera index the table cannot resolve is corruption, not a default.
        if (cam >= camera_keys.size())
            throw Error("frame record camera index " + std::to_string(cam) +
                        " but only " + std::to_string(camera_keys.size()) +
                        " cameras are declared");
        f.camera = camera_keys[cam];
        f.t = detail::lef64(r + 8);
        for (int k = 0; k < 4; ++k) f.q_wxyz[static_cast<size_t>(k)] = detail::lef32(r + 16 + 4 * k);
        for (int k = 0; k < 3; ++k) f.tr[static_cast<size_t>(k)] = detail::lef32(r + 32 + 4 * k);
        f.pose_valid = (r[44] & 1) != 0;
        out.push_back(f);
    }
    return out;
}

inline std::vector<ImuSample> unpack_imu(const std::string& buf) {
    if (buf.size() % IMU_RECORD_SIZE)
        throw Error("IMU buffer " + std::to_string(buf.size()) +
                    " is not a multiple of " + std::to_string(IMU_RECORD_SIZE));
    const auto* p = reinterpret_cast<const uint8_t*>(buf.data());
    std::vector<ImuSample> out;
    out.reserve(buf.size() / IMU_RECORD_SIZE);
    for (size_t off = 0; off + IMU_RECORD_SIZE <= buf.size(); off += IMU_RECORD_SIZE) {
        const uint8_t* r = p + off;
        ImuSample s;
        s.t = detail::lef64(r);
        for (int k = 0; k < 3; ++k) s.gyro[static_cast<size_t>(k)] = detail::lef32(r + 8 + 4 * k);
        for (int k = 0; k < 3; ++k) s.accel[static_cast<size_t>(k)] = detail::lef32(r + 20 + 4 * k);
        out.push_back(s);
    }
    return out;
}

// ---------------------------------------------------------------------- read

namespace detail {

/// Tag values in file order. Repeated binary tags concatenate — WURLD_POSES is
/// written as a chunk per flush by the streaming writer, and dropping any of
/// them would silently lose poses rather than fail.
using Tags = std::vector<std::pair<std::string, std::pair<std::string, bool>>>;  // name -> (value, is_binary)

inline void collect_simple_tags(const std::string& buf, Tags& out) {
    const auto* p = reinterpret_cast<const uint8_t*>(buf.data());
    const size_t n = buf.size();

    // Walk a nested element range, gathering SimpleTag name/value pairs.
    struct Walker {
        const uint8_t* p;
        size_t n;
        Tags& out;

        void range(size_t start, size_t end) {
            size_t pos = start;
            while (pos < end) {
                size_t idu = 0, szu = 0;
                uint32_t id = ebml::read_id(p + pos, end - pos, idu);
                uint64_t size = ebml::read_size(p + pos + idu, end - pos - idu, szu);
                size_t pstart = pos + idu + szu;
                if (size == UINT64_MAX || pstart + size > end)
                    return;  // truncated or unknown-size child: stop, do not guess
                size_t pend = pstart + static_cast<size_t>(size);
                if (id == ebml::TAG || id == ebml::TAGS) {
                    range(pstart, pend);
                } else if (id == ebml::SIMPLE_TAG) {
                    simple_tag(pstart, pend);
                }
                pos = pend;
            }
        }

        void simple_tag(size_t start, size_t end) {
            std::string name, value;
            bool have_value = false, binary = false;
            size_t pos = start;
            while (pos < end) {
                size_t idu = 0, szu = 0;
                uint32_t id = ebml::read_id(p + pos, end - pos, idu);
                uint64_t size = ebml::read_size(p + pos + idu, end - pos - idu, szu);
                size_t pstart = pos + idu + szu;
                if (size == UINT64_MAX || pstart + size > end) return;
                size_t pend = pstart + static_cast<size_t>(size);
                if (id == ebml::TAG_NAME) {
                    name.assign(reinterpret_cast<const char*>(p + pstart), pend - pstart);
                } else if (id == ebml::TAG_STRING) {
                    value.assign(reinterpret_cast<const char*>(p + pstart), pend - pstart);
                    have_value = true;
                    binary = false;
                } else if (id == ebml::TAG_BINARY) {
                    value.assign(reinterpret_cast<const char*>(p + pstart), pend - pstart);
                    have_value = true;
                    binary = true;
                } else if (id == ebml::SIMPLE_TAG) {
                    simple_tag(pstart, pend);  // nested
                }
                pos = pend;
            }
            if (!name.empty() && have_value)
                out.emplace_back(name, std::make_pair(std::move(value), binary));
        }
    } walker{p, n, out};

    walker.range(0, n);
}

inline std::string tag_string(const Tags& tags, const std::string& name, bool* found = nullptr) {
    if (found) *found = false;
    std::string joined;
    for (const auto& kv : tags) {
        if (kv.first != name) continue;
        if (found) *found = true;
        // Repeated binary tags concatenate in file order; a repeated string tag
        // is last-wins, matching the Python reader.
        if (kv.second.second) joined += kv.second.first;
        else joined = kv.second.first;
    }
    return joined;
}

inline bool tag_is_binary(const Tags& tags, const std::string& name) {
    for (const auto& kv : tags)
        if (kv.first == name) return kv.second.second;
    return false;
}

}  // namespace detail

/// Parse a wurld document from a whole in-memory file.
inline Document parse(const std::string& bytes);

/// Read a wurld file's metadata, seeking over payloads rather than reading them.
inline Document read(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw Error("cannot open " + path);
    in.seekg(0, std::ios::end);
    const auto file_size = static_cast<uint64_t>(in.tellg());
    in.seekg(0);

    auto read_at = [&](uint64_t off, size_t n) {
        std::string b(n, '\0');
        in.seekg(static_cast<std::streamoff>(off));
        in.read(&b[0], static_cast<std::streamsize>(n));
        b.resize(static_cast<size_t>(in.gcount()));
        in.clear();
        return b;
    };

    // Find the Segment, then walk its top-level children. Only Tags payloads are
    // read; Clusters are skipped by seeking, so file size does not drive I/O.
    uint64_t pos = 0;
    uint64_t segment_end = file_size;
    uint64_t cluster_start = 0;
    detail::Tags tags;

    auto header = read_at(0, 64);
    if (header.size() < 4) throw Error("file is too short to be Matroska");
    {
        size_t idu = 0, szu = 0;
        const auto* hp = reinterpret_cast<const uint8_t*>(header.data());
        uint32_t id = ebml::read_id(hp, header.size(), idu);
        if (id != 0x1A45DFA3) throw Error("not an EBML/Matroska file");
        uint64_t size = ebml::read_size(hp + idu, header.size() - idu, szu);
        if (size == UINT64_MAX) throw Error("EBML header has unknown size");
        pos = idu + szu + size;
    }
    {
        auto seg = read_at(pos, 16);
        size_t idu = 0, szu = 0;
        const auto* sp = reinterpret_cast<const uint8_t*>(seg.data());
        uint32_t id = ebml::read_id(sp, seg.size(), idu);
        if (id != ebml::SEGMENT) throw Error("expected a Segment element");
        uint64_t size = ebml::read_size(sp + idu, seg.size() - idu, szu);
        pos += idu + szu;
        segment_end = (size == UINT64_MAX) ? file_size : pos + size;
        if (segment_end > file_size) segment_end = file_size;
    }

    while (pos < segment_end) {
        auto head = read_at(pos, 16);
        if (head.size() < 2) break;
        size_t idu = 0, szu = 0;
        uint32_t id = 0;
        uint64_t size = 0;
        try {
            const auto* hp = reinterpret_cast<const uint8_t*>(head.data());
            id = ebml::read_id(hp, head.size(), idu);
            size = ebml::read_size(hp + idu, head.size() - idu, szu);
        } catch (const Error&) {
            break;  // trailing garbage: keep what was parsed rather than failing
        }
        uint64_t pstart = pos + idu + szu;
        if (size == UINT64_MAX) break;  // unknown-size cluster: nothing after it to index
        uint64_t pend = pstart + size;
        if (pend > segment_end) break;

        if (id == ebml::TAGS) {
            detail::collect_simple_tags(read_at(pstart, static_cast<size_t>(size)), tags);
        } else if (id == ebml::CLUSTER && cluster_start == 0) {
            cluster_start = pos;
        }
        pos = pend;
    }

    bool have_doc = false;
    std::string wurld = detail::tag_string(tags, "WURLD", &have_doc);
    if (!have_doc) throw Error(path + ": no WURLD tag (not a wurld file)");

    Document doc = parse(wurld);
    doc.cluster_start = cluster_start;
    doc.file_size = file_size;

    // Pose precedence (SPEC §9): consolidated table > streamed chunks > JSON.
    std::vector<std::string> camera_keys;
    const Json* fb = doc.raw.find("frames_binary");
    if (fb && fb->is_object() && fb->has("cameras")) {
        for (const auto& c : fb->at("cameras").array)
            if (c.is_string()) camera_keys.push_back(c.string);
    } else {
        for (const auto& kv : doc.cameras) camera_keys.push_back(kv.first);  // std::map: sorted
    }

    bool has_table = false, has_chunks = false;
    std::string table = detail::tag_string(tags, "WURLD_FRAMES", &has_table);
    std::string chunks = detail::tag_string(tags, "WURLD_POSES", &has_chunks);
    has_table = has_table && detail::tag_is_binary(tags, "WURLD_FRAMES");
    has_chunks = has_chunks && detail::tag_is_binary(tags, "WURLD_POSES");

    if (has_table) {
        if (fb && fb->is_object() && fb->has("version") &&
            static_cast<int>(fb->num("version", 1)) != 1)
            throw Error("unsupported frames_binary version");
        doc.frames = unpack_frames(table, camera_keys);
        if (fb && fb->is_object() && fb->has("count")) {
            auto want = static_cast<size_t>(fb->num("count", 0));
            if (doc.frames.size() != want)
                throw Error("WURLD_FRAMES has " + std::to_string(doc.frames.size()) +
                            " records, expected " + std::to_string(want));
        }
    } else if (has_chunks) {
        doc.frames = unpack_frames(chunks, camera_keys);
    } else if (fb && fb->is_object()) {
        throw Error("frames_binary declared but WURLD_FRAMES tag missing");
    }

    // IMU streams are declared in the document and carried as binary tags.
    const Json* imu = doc.raw.find("imu");
    if (imu && imu->is_object()) {
        for (const auto& kv : imu->object) {
            bool found = false;
            std::string buf = detail::tag_string(tags, "WURLD_IMU_" + kv.first, &found);
            if (!found || !detail::tag_is_binary(tags, "WURLD_IMU_" + kv.first)) continue;
            ImuStream s;
            s.id = kv.first;
            s.samples = unpack_imu(buf);
            doc.imu.emplace(kv.first, std::move(s));
        }
    }
    return doc;
}

inline Document parse(const std::string& text) {
    Document doc;
    doc.raw = parse_json(text);
    if (doc.raw.str("format") != "wurld")
        throw Error("WURLD tag present but format=" + doc.raw.str("format", "(absent)"));
    doc.version = doc.raw.str("version");

    if (const Json* cams = doc.raw.find("cameras"); cams && cams->is_object()) {
        for (const auto& kv : cams->object) {
            Camera c;
            c.model = kv.second.str("model");
            c.width = static_cast<int>(kv.second.num("width"));
            c.height = static_cast<int>(kv.second.num("height"));
            if (const Json* ps = kv.second.find("params"); ps && ps->is_array())
                for (const auto& v : ps->array) c.params.push_back(v.number);
            doc.cameras.emplace(kv.first, std::move(c));
        }
    }

    if (const Json* sigs = doc.raw.find("signals"); sigs && sigs->is_array()) {
        for (const auto& s : sigs->array) {
            SignalMeta m;
            m.id = s.str("id");
            m.role = s.str("role");
            if (const Json* vm = s.find("value_map")) m.value_map = *vm;
            doc.signals.push_back(std::move(m));
        }
    }

    // Absent means empty, not null — matching the Python reader's doc.get(k, {}).
    // A consumer iterating rigs should find nothing, not dereference a null.
    doc.world.type = Json::Type::Object;
    doc.rigs.type = Json::Type::Object;
    if (const Json* w = doc.raw.find("world"); w && !w->is_null()) doc.world = *w;
    if (const Json* r = doc.raw.find("rigs"); r && !r->is_null()) doc.rigs = *r;

    // JSON frames are the fallback; read() overwrites these from a binary table.
    if (const Json* fs = doc.raw.find("frames"); fs && fs->is_array()) {
        for (const auto& fj : fs->array) {
            Frame f;
            f.i = static_cast<uint32_t>(fj.num("i"));
            f.t = fj.num("t");
            f.camera = fj.str("camera");
            int pv = fj.tribool("pose_valid");
            f.pose_valid = pv != 0;  // absent means valid
            if (const Json* q = fj.find("q_wxyz"); q && q->array.size() == 4)
                for (size_t k = 0; k < 4; ++k) f.q_wxyz[k] = q->array[k].number;
            if (const Json* t = fj.find("tr"); t && t->array.size() == 3)
                for (size_t k = 0; k < 3; ++k) f.tr[k] = t->array[k].number;
            doc.frames.push_back(std::move(f));
        }
    }
    return doc;
}

}  // namespace wurld

#endif  // WURLD_HPP
