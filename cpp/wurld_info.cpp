// wurld_info — dump a wurld file's metadata as JSON.
//
// Exists for two reasons: it is the smallest useful demonstration of the C++
// reader, and it gives the cross-implementation test something to diff. The
// output is deliberately machine-readable so `tests/test_cpp_reader.py` can
// compare it against the Python reader field-by-field rather than eyeballing it.
//
//   wurld_info scene.wurld.webm            summary
//   wurld_info scene.wurld.webm --json     everything the reader parsed

#include <cstdio>
#include <cstring>
#include <string>

#include "wurld.hpp"

namespace {

std::string quote(const std::string& s) {
    std::string out;
    wurld::detail::dump_string(out, s);
    return out;
}

std::string num(double d) { return wurld::detail::dump_number(d); }

void dump_json(const wurld::Document& doc) {
    std::string out = "{";
    out += "\"version\":" + quote(doc.version);
    out += ",\"cluster_start\":" + num(static_cast<double>(doc.cluster_start));

    out += ",\"cameras\":{";
    bool first = true;
    for (const auto& kv : doc.cameras) {
        if (!first) out += ',';
        first = false;
        out += quote(kv.first) + ":{\"model\":" + quote(kv.second.model) +
               ",\"width\":" + num(kv.second.width) +
               ",\"height\":" + num(kv.second.height) + ",\"params\":[";
        for (size_t k = 0; k < kv.second.params.size(); ++k) {
            if (k) out += ',';
            out += num(kv.second.params[k]);
        }
        out += "]}";
    }
    out += "}";

    out += ",\"signals\":[";
    for (size_t k = 0; k < doc.signals.size(); ++k) {
        if (k) out += ',';
        out += "{\"id\":" + quote(doc.signals[k].id) +
               ",\"role\":" + quote(doc.signals[k].role) +
               ",\"value_map\":" + doc.signals[k].value_map.dump() + "}";
    }
    out += "]";

    out += ",\"frames\":[";
    for (size_t k = 0; k < doc.frames.size(); ++k) {
        const auto& f = doc.frames[k];
        if (k) out += ',';
        out += "{\"i\":" + num(f.i) + ",\"t\":" + num(f.t) +
               ",\"camera\":" + quote(f.camera) +
               ",\"pose_valid\":" + (f.pose_valid ? "true" : "false");
        if (f.pose_valid) {
            out += ",\"q_wxyz\":[";
            for (size_t j = 0; j < 4; ++j) { if (j) out += ','; out += num(f.q_wxyz[j]); }
            out += "],\"tr\":[";
            for (size_t j = 0; j < 3; ++j) { if (j) out += ','; out += num(f.tr[j]); }
            out += "],\"c2w\":[";
            auto m = f.c2w();
            for (size_t j = 0; j < 16; ++j) { if (j) out += ','; out += num(m[j]); }
            out += "]";
        }
        out += "}";
    }
    out += "]";

    out += ",\"imu\":{";
    first = true;
    for (const auto& kv : doc.imu) {
        if (!first) out += ',';
        first = false;
        out += quote(kv.first) + ":[";
        for (size_t k = 0; k < kv.second.samples.size(); ++k) {
            const auto& s = kv.second.samples[k];
            if (k) out += ',';
            out += "[" + num(s.t);
            for (float g : s.gyro) out += "," + num(g);
            for (float a : s.accel) out += "," + num(a);
            out += "]";
        }
        out += "]";
    }
    out += "}";

    out += ",\"world\":" + doc.world.dump();
    out += ",\"rigs\":" + doc.rigs.dump();
    out += "}";
    std::printf("%s\n", out.c_str());
}

void dump_summary(const wurld::Document& doc, const char* path) {
    std::printf("%s\n", path);
    std::printf("  wurld v%s; %zu frames, %zu posed\n", doc.version.c_str(),
                doc.frames.size(), doc.posed_frames());
    for (const auto& kv : doc.cameras) {
        std::printf("  camera %s: %s %dx%d [", kv.first.c_str(), kv.second.model.c_str(),
                    kv.second.width, kv.second.height);
        for (size_t k = 0; k < kv.second.params.size(); ++k)
            std::printf("%s%g", k ? ", " : "", kv.second.params[k]);
        std::printf("]\n");
    }
    for (const auto& s : doc.signals)
        std::printf("  signal %s (%s)\n", s.id.c_str(), s.role.c_str());
    for (const auto& kv : doc.imu)
        std::printf("  imu %s: %zu samples\n", kv.first.c_str(), kv.second.samples.size());
    if (!doc.frames.empty()) {
        const auto& f = doc.frames.front();
        std::printf("  t: %.6f .. %.6f\n", f.t, doc.frames.back().t);
    }
    // Pixels are not this reader's job; say where they start rather than imply.
    std::printf("  clusters begin at byte %llu of %llu (pixels need chromapakz)\n",
                static_cast<unsigned long long>(doc.cluster_start),
                static_cast<unsigned long long>(doc.file_size));
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: wurld_info <file.wurld.webm> [--json]\n");
        return 2;
    }
    bool as_json = argc > 2 && std::strcmp(argv[2], "--json") == 0;
    try {
        auto doc = wurld::read(argv[1]);
        if (as_json) dump_json(doc);
        else dump_summary(doc, argv[1]);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
    return 0;
}
