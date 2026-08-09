// The C++ snippets printed in README.md, compiled so they cannot rot.
//
// Nothing here runs — the point is that the code a reader copies out of the
// README still type-checks against the current headers. A snippet that no
// longer compiles is worse than no snippet: it is the first thing someone
// tries, and it fails on their machine rather than in CI.
//
// Keep these in step with README.md by hand; if one changes, change both.

#include "wurld.hpp"
#include "wurld_write.hpp"
#include "wurld_stream.hpp"
#include <fstream>
static void use(const std::array<double,16>&) {}

void reader_snippet() {                       // README: C++ reader
    auto doc = wurld::read("scene.wl.webm");
    for (const auto& f : doc.frames)
        if (f.pose_valid) use(f.c2w());
}

void writer_snippet() {                       // README: Writing from C++
    wurld::WriteDoc doc;
    doc.cameras["0"] = {"PINHOLE", 640, 480, {525, 525, 320, 240}};
    doc.frames.push_back({0, 0.0, "0", true, {1,0,0,0}, {0,0,0}});
    doc.world_json = R"({"metric_scale":true})";
    wurld::write_file("encoded.webm", "out.wl.webm", doc);
}

void stream_snippet(std::ofstream& out) {     // README: Recording from C++
    wurld::WriteDoc doc;
    doc.cameras["0"] = {"PINHOLE", 640, 480, {525, 525, 320, 240}};
    wurld::StreamWriter w([&](const std::string& b){ out.write(b.data(), b.size()); }, doc);
    (void)w;
}
