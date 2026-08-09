// wurld streaming writer — record a wurld file without holding it in memory.
//
// `attach()` (wurld_write.hpp) rebuilds a whole file at once, which is fine for
// finalising a clip and useless to a robot recording for hours. This is the
// incremental form: metadata is woven between the encoder's Clusters as they
// arrive, so peak memory is one Cluster plus the poses seen so far (45 bytes
// each), and a file that is killed mid-recording still holds everything written
// before the interruption.
//
// The dependency split is the same as the reader's, and is the point. wurld does
// not encode video; chromapakz does, and it already emits WebM in chunks. Hand
// those chunks here and this interleaves the pose and IMU tags:
//
//   wurld::StreamWriter w(
//       [&](const std::string& bytes) { out.write(bytes.data(), bytes.size()); },
//       doc);
//   // chromapakz encoder callback:
//   encoder.on_chunk = [&](const std::string& chunk) { w.on_encoder_chunk(chunk); };
//   for (...) {
//       w.add_frame(pose);            // before encoding the matching image
//       encoder.add_frame(pixels);
//   }
//   encoder.finish();                 // tail Clusters arrive via on_encoder_chunk
//   w.finish();
//
// Layout is SPEC §9's live form, byte-compatible with the Python StreamWriter:
// the encoder's file prefix, the WURLD document, then for each Cluster the
// poses it contains as a `WURLD_POSES` chunk *ahead* of it, and finally one
// consolidated `WURLD_FRAMES` table. A reader resolves the table first (SPEC
// §9 precedence), so an interrupted recording still reads through the chunks.
//
// No Cues are written. Interleaving tags between Clusters moves them, and a
// stale Cue seeks into the middle of one — SPEC §9 forbids it. Run the file
// through `wurld::attach()` afterwards to get the indexed batch layout.

#ifndef WURLD_STREAM_HPP
#define WURLD_STREAM_HPP

#include <functional>
#include <map>
#include <string>
#include <vector>

#include "wurld_write.hpp"

namespace wurld {

class StreamWriter {
  public:
    using Sink = std::function<void(const std::string&)>;

    /// `doc.frames` is ignored — poses arrive through `add_frame`.
    StreamWriter(Sink sink, const WriteDoc& doc) : sink_(std::move(sink)) {
        if (!sink_) throw Error("StreamWriter needs a sink");
        for (const auto& kv : doc.cameras) camera_keys_.push_back(kv.first);  // sorted
        if (camera_keys_.empty()) throw Error("StreamWriter needs at least one camera");
        header_ = ebml::build_tags({{"WURLD",
            {build_document(doc, FramesMode::Streaming, camera_keys_), false}}});
    }

    StreamWriter(const StreamWriter&) = delete;
    StreamWriter& operator=(const StreamWriter&) = delete;

    /// Record the pose of the frame about to be encoded.
    ///
    /// Called before handing the image to the encoder, so that when the Cluster
    /// carrying it arrives the pose is already pending and goes out ahead of it.
    void add_frame(const Frame& f) {
        if (finished_) throw Error("StreamWriter is finished");
        if (f.pose_valid) {
            const auto& q = f.q_wxyz;
            double n = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
            if (std::fabs(n - 1.0) > 1e-3)
                throw Error("frame " + std::to_string(f.i) + ": quaternion not unit");
        }
        pending_.push_back(f);
        all_.push_back(f);
    }

    /// IMU samples at whatever cadence they arrive; flushed with the next Cluster.
    void add_imu(const std::string& stream_id, const std::vector<ImuSample>& samples) {
        if (finished_) throw Error("StreamWriter is finished");
        auto& dst = pending_imu_[stream_id];
        dst.insert(dst.end(), samples.begin(), samples.end());
    }

    /// Feed one chunk from the video encoder, in the order it produced them.
    void on_encoder_chunk(const std::string& chunk) {
        if (finished_) throw Error("StreamWriter is finished");
        if (!header_emitted_) {
            // The encoder's first chunk is the file prefix — EBML header,
            // Segment header, Tracks. The document follows it so a reader that
            // stops early still has calibration.
            emit(chunk);
            emit(header_);
            header_emitted_ = true;
            return;
        }
        // Later chunks are whole Clusters holding frames already added, so their
        // poses are pending now and belong in front (SPEC §9).
        flush_tags();
        emit(chunk);
    }

    /// Finalise: flush the tail, then write the consolidated pose table.
    ///
    /// The encoder must be finished first, so its tail Clusters have already
    /// arrived through `on_encoder_chunk`.
    size_t finish() {
        if (finished_) throw Error("StreamWriter is finished");
        if (!header_emitted_)
            throw Error("finish() before any encoder chunk — nothing was recorded");
        flush_tags();
        if (!all_.empty()) {
            emit(ebml::build_tags({{"WURLD_FRAMES",
                {pack_frames(all_, camera_keys_), true}}}));
        }
        finished_ = true;
        return all_.size();
    }

    size_t frames_written() const { return all_.size(); }

  private:
    void emit(const std::string& bytes) {
        if (!bytes.empty()) sink_(bytes);
    }

    void flush_tags() {
        std::vector<std::pair<std::string, std::pair<std::string, bool>>> tags;
        if (!pending_.empty()) {
            tags.emplace_back("WURLD_POSES",
                              std::make_pair(pack_frames(pending_, camera_keys_), true));
            pending_.clear();
        }
        for (auto& kv : pending_imu_) {
            if (kv.second.empty()) continue;
            tags.emplace_back("WURLD_IMU_" + kv.first,
                              std::make_pair(pack_imu(kv.second), true));
        }
        pending_imu_.clear();
        if (!tags.empty()) emit(ebml::build_tags(tags));
    }

    Sink sink_;
    std::vector<std::string> camera_keys_;
    std::string header_;
    bool header_emitted_ = false;
    bool finished_ = false;
    std::vector<Frame> pending_, all_;
    std::map<std::string, std::vector<ImuSample>> pending_imu_;
};

}  // namespace wurld

#endif  // WURLD_STREAM_HPP
