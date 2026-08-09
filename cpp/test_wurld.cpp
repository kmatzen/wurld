// Unit tests for the parts of the C++ reader that do not need a file.
//
// File-level correctness is checked against the Python reader in
// tests/test_cpp_reader.py — agreeing with itself proves nothing. What lives
// here is the JSON parser and the binary record layouts, where the failure mode
// is a plausible-but-wrong value rather than a crash.

#include <cmath>
#include <cstdio>
#include <string>

#include "wurld.hpp"

static int failures = 0;

#define CHECK(cond)                                                             \
    do {                                                                        \
        if (!(cond)) {                                                          \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++failures;                                                         \
        }                                                                       \
    } while (0)

#define CHECK_THROWS(expr)                                                      \
    do {                                                                        \
        bool threw = false;                                                     \
        try { (void)(expr); } catch (const std::exception&) { threw = true; }   \
        if (!threw) {                                                           \
            std::fprintf(stderr, "FAIL %s:%d: expected throw from %s\n",        \
                         __FILE__, __LINE__, #expr);                            \
            ++failures;                                                         \
        }                                                                       \
    } while (0)

static void test_json_basics() {
    auto v = wurld::parse_json(R"({"a":1,"b":[1,2,3],"c":"x","d":true,"e":null})");
    CHECK(v.is_object());
    CHECK(v.num("a") == 1.0);
    CHECK(v.at("b").array.size() == 3);
    CHECK(v.str("c") == "x");
    CHECK(v.tribool("d") == 1);
    CHECK(v.at("e").is_null());
    CHECK(v.find("missing") == nullptr);
}

static void test_json_tribool_distinguishes_absent_from_false() {
    // metric_scale=false is a claim; metric_scale absent is not.
    auto f = wurld::parse_json(R"({"metric_scale":false})");
    auto a = wurld::parse_json(R"({})");
    CHECK(f.tribool("metric_scale") == 0);
    CHECK(a.tribool("metric_scale") == -1);
}

static void test_json_numbers() {
    auto v = wurld::parse_json(R"({"i":-42,"f":1.5e-3,"big":1e300,"neg":-0.0})");
    CHECK(v.num("i") == -42.0);
    CHECK(std::fabs(v.num("f") - 0.0015) < 1e-18);
    CHECK(v.num("big") == 1e300);
    CHECK(std::signbit(v.num("neg")));
}

static void test_json_non_finite() {
    // Python's json writes these bare, and depth value maps really contain them.
    auto v = wurld::parse_json(R"({"a":NaN,"b":Infinity,"c":-Infinity})");
    CHECK(std::isnan(v.num("a")));
    CHECK(std::isinf(v.num("b")) && v.num("b") > 0);
    CHECK(std::isinf(v.num("c")) && v.num("c") < 0);
}

static void test_json_escapes() {
    auto v = wurld::parse_json(R"({"s":"a\"b\\c\nd\teéA"})");
    CHECK(v.str("s") == "a\"b\\c\nd\te\xc3\xa9" "A");
    // Astral plane via surrogate pair (U+1F600).
    auto e = wurld::parse_json(R"({"s":"😀"})");
    CHECK(e.str("s") == "\xf0\x9f\x98\x80");
}

static void test_json_rejects_malformed() {
    CHECK_THROWS(wurld::parse_json(R"({"a":1)"));          // unterminated object
    CHECK_THROWS(wurld::parse_json(R"({"a":"x)"));         // unterminated string
    CHECK_THROWS(wurld::parse_json(R"({"a":1} trailing)"));
    CHECK_THROWS(wurld::parse_json(R"({a:1})"));           // unquoted key
    CHECK_THROWS(wurld::parse_json(R"({"a":tru})"));
    CHECK_THROWS(wurld::parse_json(R"({"a":"\q"})"));      // unknown escape
    CHECK_THROWS(wurld::parse_json(R"({"a":"\u00"})"));    // truncated escape
    CHECK_THROWS(wurld::parse_json(""));
    CHECK_THROWS(wurld::parse_json(R"([1,2)"));
}

static void test_json_round_trip_of_dump() {
    const std::string src = R"({"a":[1,2],"b":{"c":"x"},"d":true})";
    auto v = wurld::parse_json(src);
    auto again = wurld::parse_json(v.dump());
    CHECK(again.at("a").array.size() == 2);
    CHECK(again.at("b").str("c") == "x");
    CHECK(again.tribool("d") == 1);
}

static void test_vint_widths() {
    // Data sizes strip the marker bit; ids keep it.
    struct { const char* bytes; size_t n; uint64_t want; } cases[] = {
        {"\x81", 1, 1}, {"\x40\x01", 2, 1}, {"\x20\x00\x01", 3, 1},
        {"\xfe", 1, 126},
    };
    for (auto& c : cases) {
        size_t used = 0;
        uint64_t got = wurld::ebml::read_size(reinterpret_cast<const uint8_t*>(c.bytes),
                                              c.n, used);
        CHECK(got == c.want);
        CHECK(used == c.n);
    }
    // The all-ones form means "unknown size", not a huge number.
    size_t used = 0;
    CHECK(wurld::ebml::read_size(reinterpret_cast<const uint8_t*>("\xff"), 1, used) ==
          UINT64_MAX);

    // Truncation must raise rather than read past the buffer.
    size_t u = 0;
    CHECK_THROWS(wurld::ebml::read_size(reinterpret_cast<const uint8_t*>("\x40"), 1, u));
    CHECK_THROWS(wurld::ebml::read_id(reinterpret_cast<const uint8_t*>("\x10\x00"), 2, u));
    CHECK_THROWS(wurld::ebml::read_id(reinterpret_cast<const uint8_t*>("\x00"), 1, u));
}

static std::string frame_record(uint32_t i, uint32_t cam, double t, const float q[4],
                                const float tr[3], uint8_t flags) {
    std::string b(wurld::FRAME_RECORD_SIZE, '\0');
    auto* p = reinterpret_cast<uint8_t*>(&b[0]);
    auto put32 = [&](size_t off, uint32_t v) {
        for (int k = 0; k < 4; ++k) p[off + static_cast<size_t>(k)] =
            static_cast<uint8_t>((v >> (8 * k)) & 0xFF);
    };
    auto putf = [&](size_t off, float f) {
        uint32_t bits;
        std::memcpy(&bits, &f, 4);
        put32(off, bits);
    };
    put32(0, i);
    put32(4, cam);
    uint64_t tb;
    std::memcpy(&tb, &t, 8);
    for (int k = 0; k < 8; ++k) p[8 + static_cast<size_t>(k)] =
        static_cast<uint8_t>((tb >> (8 * k)) & 0xFF);
    for (int k = 0; k < 4; ++k) putf(16 + 4 * static_cast<size_t>(k), q[k]);
    for (int k = 0; k < 3; ++k) putf(32 + 4 * static_cast<size_t>(k), tr[k]);
    p[44] = flags;
    return b;
}

static void test_unpack_frames() {
    const float q[4] = {1.0f, 0.0f, 0.0f, 0.0f};
    const float tr[3] = {1.5f, -2.0f, 3.25f};
    std::string buf = frame_record(7, 1, 0.5, q, tr, 1) +
                      frame_record(8, 0, 0.75, q, tr, 0);
    auto frames = wurld::unpack_frames(buf, {"cam0", "cam1"});
    CHECK(frames.size() == 2);
    CHECK(frames[0].i == 7);
    CHECK(frames[0].camera == "cam1");
    CHECK(frames[0].t == 0.5);
    CHECK(frames[0].pose_valid);
    CHECK(frames[0].tr[2] == 3.25);
    CHECK(frames[1].camera == "cam0");
    CHECK(!frames[1].pose_valid);

    // A length that is not a whole number of records is corruption.
    CHECK_THROWS(wurld::unpack_frames(buf.substr(0, buf.size() - 3), {"cam0", "cam1"}));
    // A camera index with no camera behind it must not silently become cam0.
    CHECK_THROWS(wurld::unpack_frames(buf, {"cam0"}));
}

static void test_unpack_imu() {
    std::string b(wurld::IMU_RECORD_SIZE, '\0');
    auto* p = reinterpret_cast<uint8_t*>(&b[0]);
    double t = 1.5;
    uint64_t tb;
    std::memcpy(&tb, &t, 8);
    for (int k = 0; k < 8; ++k) p[k] = static_cast<uint8_t>((tb >> (8 * k)) & 0xFF);
    float vals[6] = {0.1f, 0.2f, 0.3f, 0.0f, 0.0f, 9.81f};
    for (int k = 0; k < 6; ++k) {
        uint32_t bits;
        std::memcpy(&bits, &vals[k], 4);
        for (int j = 0; j < 4; ++j)
            p[8 + 4 * static_cast<size_t>(k) + static_cast<size_t>(j)] =
                static_cast<uint8_t>((bits >> (8 * j)) & 0xFF);
    }
    auto s = wurld::unpack_imu(b);
    CHECK(s.size() == 1);
    CHECK(s[0].t == 1.5);
    CHECK(std::fabs(s[0].gyro[1] - 0.2f) < 1e-6);
    CHECK(std::fabs(s[0].accel[2] - 9.81f) < 1e-5);
    CHECK_THROWS(wurld::unpack_imu(b + std::string(7, '\0')));
}

static void test_c2w_identity_and_rotation() {
    wurld::Frame f;
    f.q_wxyz = {{1, 0, 0, 0}};
    f.tr = {{1, 2, 3}};
    auto m = f.c2w();
    CHECK(m[0] == 1 && m[5] == 1 && m[10] == 1);
    CHECK(m[3] == 1 && m[7] == 2 && m[11] == 3);
    CHECK(m[12] == 0 && m[13] == 0 && m[14] == 0 && m[15] == 1);

    // 90 degrees about z: x -> y.
    const double s = std::sqrt(0.5);
    f.q_wxyz = {{s, 0, 0, s}};
    f.tr = {{0, 0, 0}};
    m = f.c2w();
    CHECK(std::fabs(m[0] - 0.0) < 1e-12);
    CHECK(std::fabs(m[1] + 1.0) < 1e-12);
    CHECK(std::fabs(m[4] - 1.0) < 1e-12);

    // An unnormalised quaternion must still give a rotation matrix.
    f.q_wxyz = {{2, 0, 0, 0}};
    m = f.c2w();
    CHECK(std::fabs(m[0] - 1.0) < 1e-12);
}

int main() {
    test_json_basics();
    test_json_tribool_distinguishes_absent_from_false();
    test_json_numbers();
    test_json_non_finite();
    test_json_escapes();
    test_json_rejects_malformed();
    test_json_round_trip_of_dump();
    test_vint_widths();
    test_unpack_frames();
    test_unpack_imu();
    test_c2w_identity_and_rotation();
    if (failures) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("all C++ unit checks passed\n");
    return 0;
}
