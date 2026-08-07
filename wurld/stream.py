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
    ):
        import chromapakz as cz

        if not hasattr(cz, "create_encoder"):
            raise RuntimeError(
                "live recording needs chromapakz with streaming encode "
                "(cz.create_encoder; chromapakz PR #43)"
            )
        cam0 = next(iter(cameras.values()))
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
        self._enc = cz.create_encoder(
            cam0.width, cam0.height, signals=cz_signals, fps=max(1, round(fps)),
            has_rgb=has_rgb, rgb_kbps=rgb_kbps, on_chunk=self._weave, cues=False,
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

    def add_frame(self, pose: Frame | None, *, rgb=None, signals=None) -> None:
        if self._finished:
            raise RuntimeError("StreamWriter is finished")
        if pose is not None:
            if pose.pose_valid:
                q = np.asarray(pose.q_wxyz, dtype=np.float64)
                if abs(np.linalg.norm(q) - 1.0) > 1e-3:
                    raise ValueError(f"frame {pose.i}: quaternion not unit")
            self._pending.append(pose)
            self._all.append(pose)
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
