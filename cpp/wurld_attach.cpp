// wurld_attach — attach a wurld metadata layer to an encoded WebM.
//
// The producer half of the C++ side: chromapakz (or any VP9 muxer) writes the
// video, this attaches calibration, poses and IMU. It reads the document to
// attach as JSON so the tests can drive it without a C++ test harness having to
// construct every case in code.
//
//   wurld_attach in.webm out.wurld.webm spec.json
//
// spec.json:
//   { "cameras": {...}, "frames": [...], "signals": [...],
//     "imu": {"imu0": [[t, gx,gy,gz, ax,ay,az], ...]},
//     "world": {...}, "rigs": {...}, "frames_format": "auto"|"json"|"binary" }

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>

#include "wurld_write.hpp"

namespace {

std::string slurp(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw wurld::Error("cannot open " + path);
    std::ostringstream buf;
    buf << in.rdbuf();
    return buf.str();
}

wurld::WriteDoc parse_spec(const std::string& text) {
    const wurld::Json spec = wurld::parse_json(text);
    wurld::WriteDoc doc;

    if (const wurld::Json* cams = spec.find("cameras")) {
        for (const auto& kv : cams->object) {
            wurld::WriteCamera c;
            c.model = kv.second.str("model", "PINHOLE");
            c.width = static_cast<int>(kv.second.num("width"));
            c.height = static_cast<int>(kv.second.num("height"));
            if (const wurld::Json* ps = kv.second.find("params"))
                for (const auto& v : ps->array) c.params.push_back(v.number);
            doc.cameras[kv.first] = std::move(c);
        }
    }

    if (const wurld::Json* fs = spec.find("frames")) {
        for (const auto& fj : fs->array) {
            wurld::Frame f;
            f.i = static_cast<uint32_t>(fj.num("i"));
            f.t = fj.num("t");
            f.camera = fj.str("camera");
            f.pose_valid = fj.tribool("pose_valid") != 0;   // absent means valid
            if (const wurld::Json* q = fj.find("q_wxyz"))
                for (size_t k = 0; k < 4 && k < q->array.size(); ++k)
                    f.q_wxyz[k] = q->array[k].number;
            if (const wurld::Json* t = fj.find("tr"))
                for (size_t k = 0; k < 3 && k < t->array.size(); ++k)
                    f.tr[k] = t->array[k].number;
            doc.frames.push_back(std::move(f));
        }
    }

    if (const wurld::Json* sigs = spec.find("signals")) {
        for (const auto& s : sigs->array) {
            wurld::SignalMeta m;
            m.id = s.str("id");
            m.role = s.str("role");
            if (const wurld::Json* vm = s.find("value_map")) m.value_map = *vm;
            doc.signals.push_back(std::move(m));
        }
    }

    if (const wurld::Json* imu = spec.find("imu")) {
        for (const auto& kv : imu->object) {
            std::vector<wurld::ImuSample> out;
            for (const auto& row : kv.second.array) {
                if (row.array.size() < 7) throw wurld::Error("imu row needs 7 numbers");
                wurld::ImuSample s;
                s.t = row.array[0].number;
                for (size_t k = 0; k < 3; ++k)
                    s.gyro[k] = static_cast<float>(row.array[1 + k].number);
                for (size_t k = 0; k < 3; ++k)
                    s.accel[k] = static_cast<float>(row.array[4 + k].number);
                out.push_back(s);
            }
            doc.imu[kv.first] = std::move(out);
        }
    }

    if (const wurld::Json* w = spec.find("world")) doc.world_json = w->dump();
    if (const wurld::Json* r = spec.find("rigs")) doc.rigs_json = r->dump();

    const std::string ff = spec.str("frames_format", "auto");
    if (ff == "json") doc.frames_format = wurld::WriteDoc::FramesFormat::Json;
    else if (ff == "binary") doc.frames_format = wurld::WriteDoc::FramesFormat::Binary;
    else if (ff != "auto") throw wurld::Error("frames_format must be auto, json or binary");

    return doc;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: wurld_attach <in.webm> <out.wurld.webm> <spec.json>\n");
        return 2;
    }
    try {
        wurld::WriteDoc doc = parse_spec(slurp(argv[3]));
        wurld::write_file(argv[1], argv[2], doc);
        // Read it straight back, so the tool cannot report success on a file
        // its own reader rejects.
        wurld::Document check = wurld::read(argv[2]);
        std::printf("%s: %zu frames (%zu posed), %zu cameras, %zu imu streams\n",
                    argv[2], check.frames.size(), check.posed_frames(),
                    check.cameras.size(), check.imu.size());
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
    return 0;
}
