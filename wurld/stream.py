"""Incremental wurld stream parsing (SPEC §9).

``StreamReader`` consumes a wurld byte stream as it arrives — from a live
recorder, a socket, or a growing file — and emits metadata and pose events the
moment their elements are complete, without waiting for the end of the stream.

Video decode is intentionally out of scope here: hand the same bytes to a
chromapakz decoder (the JS network decoder for progressive video; the Python
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
from .container import Frame, ImuStream, unpack_frames


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
