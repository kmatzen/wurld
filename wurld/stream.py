"""Incremental wurld streaming (SPEC §9): live writing and live parsing.

``StreamWriter`` records live: it wraps chromapakz's streaming encoder
(``cz.create_encoder``, chromapakz >= PR #43) and weaves WURLD_POSES /
WURLD_IMU chunk tags into the element-aligned chunk stream, so every emitted
byte range is a valid, fully-posed file up to the last flushed chunk. On
``finish()`` it appends the consolidated WURLD_FRAMES table.

``StreamReader`` consumes a wurld byte stream as it arrives — from a live
recorder, a socket, or a growing file — and emits metadata and pose events the
moment their elements are complete, without waiting for the end of the stream.

Video decode is intentionally out of scope in the reader: hand the same bytes to
a chromapakz decoder (the JS network decoder for progressive video; the Python
batch decoder once the stream ends).

Events yielded by ``feed()``:

    ("wurld", doc)          -- the header JSON document (dict)
    ("poses", [Frame, ...])     -- a WURLD_POSES chunk (streamed form)
    ("frames_table", [Frame, ...]) -- consolidated WURLD_FRAMES (authoritative)
    ("imu", stream_id, samples) -- an IMU chunk, samples (N, 7) float64
    ("cluster", byte_length)    -- a video Cluster passed by
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import ebml
from .container import (
    CONVENTIONS,
    FORMAT_VERSION,
    Camera,
    Frame,
    ImuStream,
    SignalMeta,
    pack_frames,
    unpack_frames,
)


class StreamWriter:
    """Live wurld recording (requires ``chromapakz.create_encoder``).

    ::

        w = StreamWriter(out.write, cameras={"0": cam}, has_rgb=True,
                         signal_meta=[SignalMeta("depth", "depth",
                             {"type": "inverse_depth", "near": 0.4, "far": 12.0})])
        w.add_frame(pose_frame, rgb=rgba, signals={"depth": {"float": z}})
        w.add_imu("imu0", samples_chunk)     # optional, any cadence
        w.finish()

    Every declared signal (and rgb when ``has_rgb``) must appear on every frame,
    matching the chromapakz streaming contract.
    """

    def __init__(
        self,
        on_chunk,
        *,
        cameras: dict[str, Camera],
        signal_meta: list[SignalMeta] | None = None,
        world: dict | None = None,
        rigs: dict | None = None,
        imu: dict[str, dict] | None = None,  # stream id -> descriptor (rate_hz, extrinsics, ...)
        fps: float = 30,
        has_rgb: bool = True,
        rgb_kbps: int = 2000,
        pose_track: bool = False,
        rgb_streams: list[str] | None = None,
    ):
        import chromapakz as cz

        if not hasattr(cz, "create_encoder"):
            raise RuntimeError(
                "live recording needs chromapakz with streaming encode "
                "(cz.create_encoder; chromapakz PR #43)"
            )
        cam0 = next(iter(cameras.values()))
        # Multi-camera display streams (SPEC §4.4), the streaming counterpart of
        # write(rgb={id: array}). Stream ids are camera ids, checked here rather
        # than at the first add_frame so a mistake surfaces before any encoding.
        self._rgb_streams = list(rgb_streams) if rgb_streams else []
        if self._rgb_streams:
            unknown = [s for s in self._rgb_streams if s not in cameras]
            if unknown:
                raise ValueError(
                    f"rgb_streams {unknown} are not declared cameras "
                    f"(SPEC 4.4: stream ids are camera ids); cameras are {sorted(cameras)}")
            if not has_rgb:
                raise ValueError("rgb_streams given but has_rgb=False")
            cam0 = cameras[self._rgb_streams[0]]
        self._on_chunk = on_chunk
        self._camera_keys = sorted(cameras)
        self._signal_meta = list(signal_meta or [])
        self._pending: list[Frame] = []
        self._all: list[Frame] = []
        self._pending_imu: dict[str, list[np.ndarray]] = {}
        self._finished = False

        doc = {
            "format": "wurld",
            "version": FORMAT_VERSION,
            "conventions": dict(CONVENTIONS),
            "world": dict(world or {"metric_scale": True, "gravity_in_world": None, "description": ""}),
            "cameras": {k: c.to_json() for k, c in cameras.items()},
            "signals": [s.to_json() for s in self._signal_meta],
            "frames": [],
        }
        if rigs:
            doc["rigs"] = rigs
        if imu:
            doc["imu"] = {k: dict(v) for k, v in imu.items()}
        self._header_tag = ebml.build_tags({"WURLD": json.dumps(doc, separators=(",", ":"))})
        self._header_emitted = False

        cz_signals = []
        for s in self._signal_meta:
            spec = {"id": s.id}
            if s.value_map.get("type") == "inverse_depth":
                spec.update(
                    near=s.value_map["near"],
                    far=s.value_map["far"],
                    levels=s.value_map.get("levels", 65536),
                )
            # A signal at its own resolution (SPEC §4.6, chromapakz format v4,
            # >= 0.10.0): declared up front because a streaming encoder plans
            # its tracks before the first frame.
            if s.width is not None:
                spec.update(width=s.width, height=s.height)
            cz_signals.append(spec)
        # cues=False: we interleave tag elements between clusters, which would
        # invalidate cue byte offsets (SPEC §9 forbids stale Cues).
        # A WebVTT pose track alongside the binary table, so ffmpeg can read poses
        # out of a live recording (its Matroska demuxer never surfaces TagBinary).
        # The table stays authoritative and contiguous — see SPEC §9; this is an
        # interop copy, not a replacement.
        self._pose_track = bool(pose_track)
        if self._pose_track and not hasattr(cz.StreamEncoder, "add_text"):
            raise RuntimeError(
                "pose_track needs chromapakz >= 0.5.0 (cz.create_encoder(text_track=...))"
            )
        kwargs = {"text_track": "wurld-poses"} if self._pose_track else {}
        if self._rgb_streams:
            # A list of stream ids; order fixes track numbering, first is primary.
            # A camera calibrated at its own resolution (SPEC §4.1/§4.6) gives
            # its stream that geometry via the dict spec form.
            kwargs["rgbs"] = [
                sid if (cameras[sid].width, cameras[sid].height)
                       == (cam0.width, cam0.height)
                else {"id": sid, "width": cameras[sid].width,
                      "height": cameras[sid].height}
                for sid in self._rgb_streams
            ]
            kwargs["has_rgb"] = False       # rgbs and has_rgb are exclusive
        else:
            kwargs["has_rgb"] = has_rgb
        self._enc = cz.create_encoder(
            cam0.width, cam0.height, signals=cz_signals, fps=max(1, round(fps)),
            rgb_kbps=rgb_kbps, on_chunk=self._weave, cues=False,
            **kwargs,
        )
        self._t0 = None

    def _emit(self, data: bytes) -> None:
        if data:
            self._on_chunk(data)

    def _flush_tags(self) -> None:
        tags: dict[str, str | bytes] = {}
        if self._pending:
            tags["WURLD_POSES"] = pack_frames(self._pending, self._camera_keys)
            self._pending = []
        for stream_id, chunks in self._pending_imu.items():
            samples = np.concatenate(chunks)
            tags[f"WURLD_IMU_{stream_id}"] = ImuStream(stream_id, samples).pack()
        self._pending_imu = {}
        if tags:
            self._emit(ebml.build_tags(tags))

    def _weave(self, chunk: bytes) -> None:
        if not self._header_emitted:
            # First chunk is the whole file prefix; our header doc follows it.
            self._emit(chunk)
            self._emit(self._header_tag)
            self._header_emitted = True
            return
        # Later chunks are whole Clusters holding already-added frames: their
        # poses are pending right now, so flush ahead of the cluster (SPEC §9).
        self._flush_tags()
        self._emit(chunk)

    def add_frame(self, pose: Frame | None, *, rgb=None, signals=None, rgbs=None) -> None:
        """One frame. ``rgbs`` is ``{camera_id: plane}`` for a multi-stream writer.

        Every declared stream must appear on every frame — the encoder plans its
        tracks from the first frame, and a stream that vanishes later would
        desynchronise the timeline rather than simply be absent.
        """
        if self._finished:
            raise RuntimeError("StreamWriter is finished")
        if self._rgb_streams:
            if rgb is not None:
                raise ValueError("this writer has rgb_streams; pass rgbs={id: plane}")
            missing = [s for s in self._rgb_streams if s not in (rgbs or {})]
            if missing:
                raise ValueError(f"frame is missing rgb streams {missing}")
        elif rgbs is not None:
            raise ValueError("rgbs given but the writer declared no rgb_streams")
        if pose is not None:
            if pose.pose_valid:
                q = np.asarray(pose.q_wxyz, dtype=np.float64)
                if abs(np.linalg.norm(q) - 1.0) > 1e-3:
                    raise ValueError(f"frame {pose.i}: quaternion not unit")
            self._pending.append(pose)
            self._all.append(pose)
        if self._rgb_streams:
            self._enc.add_frame(rgbs=rgbs, signals=signals or {})
        else:
            self._enc.add_frame(rgb=rgb, signals=signals or {})
        if self._pose_track and pose is not None and pose.pose_valid:
            # Cue times are rebased to the media timeline: sensor clocks are absolute
            # (ARKit reports device uptime) and a cue at t=71877s would sit far past
            # the end of the video. The absolute value stays in the cue text.
            if self._t0 is None:
                self._t0 = pose.t
            q, tr = pose.q_wxyz, pose.tr
            self._enc.add_text(
                f"i={pose.i} t={pose.t!r} camera={pose.camera} "
                f"q_wxyz={q[0]!r},{q[1]!r},{q[2]!r},{q[3]!r} "
                f"tr={tr[0]!r},{tr[1]!r},{tr[2]!r}",
                timestamp=max(0.0, pose.t - self._t0),
            )

    def add_imu(self, stream_id: str, samples: np.ndarray) -> None:
        """samples: (N, 7) [t, gyro xyz, accel xyz]; flushed with the next cluster."""
        samples = np.asarray(samples, dtype=np.float64).reshape(-1, 7)
        self._pending_imu.setdefault(stream_id, []).append(samples)

    def finish(self) -> dict:
        if self._finished:
            raise RuntimeError("StreamWriter is finished")
        self._enc.finish()  # tail cluster(s) arrive via _weave
        self._flush_tags()  # poses for frames in the tail cluster
        if self._all:
            self._emit(
                ebml.build_tags(
                    {"WURLD_FRAMES": pack_frames(self._all, self._camera_keys)}
                )
            )
        self._finished = True
        return {"frames": len(self._all)}


class StreamReader:
    def __init__(self):
        self._buf = bytearray()
        self._pos = 0  # parse offset into _buf
        self._in_segment = False
        self._camera_keys: list[str] = []
        self.doc: dict | None = None
        self.frames: list[Frame] = []  # accumulated streamed poses
        self.finished = False

    @staticmethod
    def _try_vint(buf, pos: int, keep_marker: bool):
        """Like ebml._read_vint but None when the vint isn't fully buffered."""
        if pos >= len(buf):
            return None
        if buf[pos] == 0:
            raise ValueError(f"invalid EBML vint at stream offset {pos}")
        length = 8 - buf[pos].bit_length() + 1
        if pos + length > len(buf):
            return None
        return ebml._read_vint(buf, pos, keep_marker)

    def _try_element(self):
        """Parse one complete element at _pos, or None if more bytes are needed."""
        buf, pos = self._buf, self._pos
        got = self._try_vint(buf, pos, keep_marker=True)
        if got is None:
            return None
        eid, p = got
        size_start = p
        got = self._try_vint(buf, p, keep_marker=False)
        if got is None:
            return None
        size, p = got
        size_len = p - size_start
        if ebml._unknown_size(size, size_len):
            # Only the Segment may be unknown-size in a live stream: descend into it.
            if eid != ebml.SEGMENT:
                raise ValueError(f"unknown-size element {eid:#x} in stream")
            return (eid, p, None)
        if p + size > len(buf):
            return None  # element not fully buffered yet
        return (eid, p, p + size)

    def feed(self, chunk: bytes) -> list[tuple]:
        """Consume bytes; return the events completed by this chunk."""
        self._buf += chunk
        events: list[tuple] = []
        while True:
            parsed = self._try_element()
            if parsed is None:
                break
            eid, payload_start, payload_end = parsed
            if payload_end is None:  # unknown-size Segment: parse children in place
                self._in_segment = True
                self._pos = payload_start
                continue
            if eid == ebml.SEGMENT:
                # Known-size segment (batch file): descend rather than skip.
                self._in_segment = True
                self._pos = payload_start
                continue
            if eid == ebml.TAGS:
                events.extend(self._handle_tags(payload_start, payload_end))
            elif eid == ebml.CLUSTER:
                events.append(("cluster", payload_end - self._pos))
            self._pos = payload_end
            self._compact()
        return events

    def _handle_tags(self, start: int, end: int) -> list[tuple]:
        events = []
        buf = self._buf
        for tid, ts, te in ebml.iter_children(buf, start, end):
            if tid != ebml.TAG:
                continue
            for sid, ss, se in ebml.iter_children(buf, ts, te):
                if sid != ebml.SIMPLE_TAG:
                    continue
                name, string, binary = None, None, None
                for fid, fs, fe in ebml.iter_children(buf, ss, se):
                    if fid == ebml.TAG_NAME:
                        name = bytes(buf[fs:fe]).decode()
                    elif fid == ebml.TAG_STRING:
                        string = bytes(buf[fs:fe]).decode()
                    elif fid == ebml.TAG_BINARY:
                        binary = bytes(buf[fs:fe])
                if name == "WURLD" and string is not None:
                    self.doc = json.loads(string)
                    fb = self.doc.get("frames_binary")
                    self._camera_keys = (
                        list(fb["cameras"]) if fb and "cameras" in fb
                        else sorted(self.doc.get("cameras", {}))
                    )
                    if self.doc.get("frames"):
                        self.frames = [Frame.from_json(f) for f in self.doc["frames"]]
                        events.append(("frames_table", list(self.frames)))
                    events.append(("wurld", self.doc))
                elif name == "WURLD_POSES" and binary is not None:
                    chunk_frames = unpack_frames(binary, self._camera_keys)
                    self.frames.extend(chunk_frames)
                    events.append(("poses", chunk_frames))
                elif name == "WURLD_FRAMES" and binary is not None:
                    self.frames = unpack_frames(binary, self._camera_keys)
                    events.append(("frames_table", list(self.frames)))
                elif name and name.startswith("WURLD_IMU_") and binary is not None:
                    stream_id = name[len("WURLD_IMU_"):]
                    meta = (self.doc or {}).get("imu", {}).get(stream_id, {})
                    events.append(
                        ("imu", stream_id, ImuStream.unpack(stream_id, binary, meta).samples)
                    )
        return events

    def finish(self) -> None:
        self.finished = True

    def _compact(self) -> None:
        # Drop consumed bytes so a long stream doesn't grow the buffer forever.
        if self._pos > 1 << 20:
            del self._buf[: self._pos]
            self._pos = 0


def should_stream(n_frames, width, height, streams=1, budget_fraction=0.25):
    """Would materialising this sequence be reckless?

    `container.write` takes whole arrays and gives float64 poses in the JSON
    frames array; `write_streaming` holds one frame and stores poses in the
    binary table, which SPEC §7 defines as float32. Neither is strictly better:
    the batch path keeps ~60 nanometres of precision that streaming rounds away,
    and the streaming path converts sequences the batch path cannot open at all.

    So pick by size rather than by preference. A real EuRoC run is 2912 stereo
    frames at 752x480 — 8.4 GB materialised — while a 573-frame TUM capture is
    0.7 GB and better served by the exact path.

    Returns False when the machine's memory cannot be determined, because
    guessing "stream" would silently degrade precision on every platform that
    does not report it.
    """
    need = int(n_frames) * int(width) * int(height) * 4 * max(1, int(streams))
    try:
        import os

        budget = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):     # not POSIX, or not reported
        return False
    return need > budget_fraction * budget


def write_streaming(
    path,
    *,
    cameras,
    frames,
    rgb_streams=None,
    signal_meta=None,
    world=None,
    rigs=None,
    imu=None,
    fps=30,
    rgb_kbps=2000,
):
    """Write a wurld file from an *iterator* of frames, one frame in memory at a time.

    `container.write` takes whole arrays, so a converter must materialise the
    sequence before encoding it. That is fine for a 573-frame TUM capture and
    impossible for a real EuRoC one: 2912 stereo frames at 752x480 is 8.4 GB of
    RGBA held at once. This is the same file, produced without ever holding more
    than one frame.

    ``frames`` yields ``(Frame, rgb, rgbs, signals)``:

    * ``rgb`` — an (H, W, 4) array for a single-stream file, else None;
    * ``rgbs`` — ``{camera_id: (H, W, 4)}`` when ``rgb_streams`` is given;
    * ``signals`` — ``{signal_id: {"u16": codes} | {"float": values}}``, matching
      the chromapakz streaming contract.

    The result uses the streaming layout (SPEC §9): poses arrive as
    ``WURLD_POSES`` chunks ahead of their Clusters and are consolidated into a
    single table on finish, so a reader sees the same poses either way. It is a
    normal wurld file — `read` and `validate` do not care which path wrote it.

    `imu` takes `ImuStream` objects, as `write` does; they are emitted whole
    because an IMU stream is small next to the video it accompanies.
    """
    path = Path(path)
    imu = list(imu or [])
    descriptors = {s.id: s.to_json() if hasattr(s, "to_json") else {} for s in imu}

    with open(path, "wb") as fh:
        writer = StreamWriter(
            fh.write,
            cameras=cameras,
            signal_meta=list(signal_meta or []),
            world=world,
            rigs=rigs,
            imu=descriptors or None,
            fps=fps,
            has_rgb=True,
            rgb_kbps=rgb_kbps,
            rgb_streams=rgb_streams,
        )
        n = 0
        for frame, rgb, rgbs, signals in frames:
            writer.add_frame(frame, rgb=rgb, rgbs=rgbs, signals=signals or {})
            n += 1
        if n == 0:
            raise ValueError("write_streaming: the frame iterator yielded nothing")
        for s in imu:
            writer.add_imu(s.id, s.samples)
        writer.finish()
    return path
