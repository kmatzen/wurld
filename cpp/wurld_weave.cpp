// wurld_weave — drive the C++ StreamWriter from encoder chunks on disk.
//
// The C++ side does not encode video (deliberately — see wurld_stream.hpp), so
// exercising the streaming writer needs chunks from a real encoder. This takes
// them as files and interleaves the metadata, which is exactly what a robot's
// encoder callback would do, with the callback replaced by a directory listing.
//
//   wurld_weave <chunkdir> <out.wl.webm> <spec.json>
//
// chunkdir holds chunk_00000.bin, chunk_00001.bin, ... in encoder order.
// spec.json is the same shape wurld_attach takes: cameras, frames, imu, world,
// rigs. Frames are recorded in order, one per encoded frame.
//
// Frames are added before the Cluster carrying them arrives, which is the
// ordering that matters: `frames_per_chunk` says how many frames each chunk
// after the first contains, mirroring a live writer adding a pose, encoding an
// image, and receiving a Cluster once the encoder has enough.

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "wurld_stream.hpp"

namespace {

std::string slurp(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw wurld::Error("cannot open " + path);
    std::ostringstream buf;
    buf << in.rdbuf();
    return buf.str();
}

std::vector<std::string> chunk_paths(const std::string& dir) {
    std::vector<std::string> out;
    for (int i = 0;; ++i) {
        char name[64];
        std::snprintf(name, sizeof name, "/chunk_%05d.bin", i);
        std::string p = dir + name;
        std::ifstream probe(p, std::ios::binary);
        if (!probe) break;
        out.push_back(p);
    }
    return out;
}

wurld::WriteDoc parse_spec(const std::string& text, std::vector<wurld::Frame>& frames,
                           std::map<std::string, std::vector<wurld::ImuSample>>& imu) {
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
            f.pose_valid = fj.tribool("pose_valid") != 0;
            if (const wurld::Json* q = fj.find("q_wxyz"))
                for (size_t k = 0; k < 4 && k < q->array.size(); ++k)
                    f.q_wxyz[k] = q->array[k].number;
            if (const wurld::Json* t = fj.find("tr"))
                for (size_t k = 0; k < 3 && k < t->array.size(); ++k)
                    f.tr[k] = t->array[k].number;
            frames.push_back(std::move(f));
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
    if (const wurld::Json* im = spec.find("imu")) {
        for (const auto& kv : im->object) {
            std::vector<wurld::ImuSample> rows;
            for (const auto& row : kv.second.array) {
                if (row.array.size() < 7) throw wurld::Error("imu row needs 7 numbers");
                wurld::ImuSample s;
                s.t = row.array[0].number;
                for (size_t k = 0; k < 3; ++k)
                    s.gyro[k] = static_cast<float>(row.array[1 + k].number);
                for (size_t k = 0; k < 3; ++k)
                    s.accel[k] = static_cast<float>(row.array[4 + k].number);
                rows.push_back(s);
            }
            imu[kv.first] = std::move(rows);
        }
    }
    if (const wurld::Json* w = spec.find("world")) doc.world_json = w->dump();
    if (const wurld::Json* r = spec.find("rigs")) doc.rigs_json = r->dump();
    return doc;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: wurld_weave <chunkdir> <out.wl.webm> <spec.json>\n");
        return 2;
    }
    try {
        std::vector<wurld::Frame> frames;
        std::map<std::string, std::vector<wurld::ImuSample>> imu;
        wurld::WriteDoc doc = parse_spec(slurp(argv[3]), frames, imu);
        // Declared in the document as well as streamed: a reader unpacks
        // WURLD_IMU_<id> only for streams the document names.
        doc.imu = imu;

        const wurld::Json spec = wurld::parse_json(slurp(argv[3]));
        size_t per_chunk = static_cast<size_t>(spec.num("frames_per_chunk", 0));

        auto chunks = chunk_paths(argv[1]);
        if (chunks.empty()) throw wurld::Error(std::string("no chunks in ") + argv[1]);

        std::ofstream out(argv[2], std::ios::binary);
        if (!out) throw wurld::Error(std::string("cannot write ") + argv[2]);

        wurld::StreamWriter w([&](const std::string& b) {
            out.write(b.data(), static_cast<std::streamsize>(b.size()));
        }, doc);

        size_t next = 0;
        for (size_t c = 0; c < chunks.size(); ++c) {
            if (c > 0) {
                // Poses for the frames this Cluster carries are added before it,
                // as a live writer would: add pose, encode image, receive Cluster.
                size_t take = per_chunk ? per_chunk : frames.size();
                for (size_t k = 0; k < take && next < frames.size(); ++k, ++next)
                    w.add_frame(frames[next]);
            }
            w.on_encoder_chunk(slurp(chunks[c]));
        }
        for (; next < frames.size(); ++next) w.add_frame(frames[next]);
        for (const auto& kv : imu) w.add_imu(kv.first, kv.second);

        size_t n = w.finish();
        out.close();
        std::printf("%s: wove %zu frames across %zu encoder chunks\n",
                    argv[2], n, chunks.size());
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
    return 0;
}
