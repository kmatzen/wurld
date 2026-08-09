"""Range-request access to wurld files (SPEC §9.1).

A batch wurld file is metadata-first with a SeekHead, so a client on S3/CDN
can pull **all calibration and every pose without downloading video**:

    hdr = fetch_header(http_fetcher("https://cdn/scene.wl.webm"))
    hdr.frames[0].c2w, hdr.cameras["0"].K, hdr.video["frames"]
    hdr.bytes_fetched      # typically <1% of the file

``fetch(offset, size) -> bytes`` is the only transport contract; helpers for
local files and HTTP are provided.
"""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from . import ebml
from .container import (Camera, Frame, ImuStream, SignalMeta, _decode_head,
                        unpack_frames)

_PROBE_SIZE = 8192  # first read: EBML header + Segment header + SeekHead live here


def file_fetcher(path: str | Path):
    """fetch() over a local file (for tests and local range-access parity)."""
    f = open(path, "rb")

    def fetch(offset: int, size: int) -> bytes:
        f.seek(offset)
        return f.read(size)

    return fetch


def http_fetcher(url: str, timeout: float = 30.0):
    """fetch() over HTTP Range requests (stdlib only)."""

    def fetch(offset: int, size: int) -> bytes:
        req = urllib.request.Request(url, headers={"Range": f"bytes={offset}-{offset + size - 1}"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()

    return fetch


@dataclass
class RemoteHeader:
    """Everything but pixels, plus the byte offsets needed to go get them."""

    cameras: dict[str, Camera]
    frames: list[Frame]
    signals: list[SignalMeta]
    world: dict
    rigs: dict
    imu: dict[str, ImuStream]
    video: dict  # chromapakz metadata: width/height/fps/frames/rgb/signals
    header_extent: int  # [0, first Cluster) — the bytes holding all metadata
    cues_offset: int | None
    bytes_fetched: int
    doc: dict = field(repr=False)
    head: bytes = field(default=b"", repr=False)  # the fetched [0, header_extent) bytes
    segment_start: int = 0  # offset of the Segment element id
    payload_start: int = 0  # offset of the Segment payload


def fetch_header(fetch) -> RemoteHeader:
    """Read calibration + poses via ranged reads; never touches Cluster bytes."""
    head = fetch(0, _PROBE_SIZE)
    fetched = len(head)
    # _segment_bounds only reads the prefix elements; the (out-of-range) payload
    # end it reports for a truncated buffer is never dereferenced here.
    segment_start, payload_start, _ = ebml._segment_bounds(head)
    seeks = ebml.read_seek_head(head, payload_start, len(head))
    first_cluster = seeks.get(ebml.CLUSTER)
    if first_cluster is None:
        raise ValueError(
            "no SeekHead with a Cluster entry — not a v0.4 batch wurld file "
            "(live recordings need the whole stream; use StreamReader)"
        )
    if first_cluster > len(head):
        head += fetch(len(head), first_cluster - len(head))
        fetched = len(head)

    # Parse the complete header region: every element in [payload_start, first_cluster).
    tags: dict[str, str | bytes] = {}
    chroma_meta: dict = {}
    for eid, _, pstart, pend in ebml._top_level(head, payload_start, first_cluster):
        if eid != ebml.TAGS:
            continue
        for name, value in ebml.collect_simple_tags(head, pstart, pend):
            if name == "CHROMAPAKZ" and isinstance(value, str):
                chroma_meta = json.loads(value)
            elif isinstance(value, bytes) and isinstance(tags.get(name), bytes):
                tags[name] += value
            else:
                tags[name] = value

    raw = tags.get("WURLD")
    if not isinstance(raw, str):
        raise ValueError("no WURLD tag in the header region")
    doc = json.loads(raw)

    fb = doc.get("frames_binary")
    camera_keys = list(fb["cameras"]) if fb and "cameras" in fb else sorted(doc.get("cameras", {}))
    table = tags.get("WURLD_FRAMES")
    chunks = tags.get("WURLD_POSES")
    if isinstance(table, bytes):
        frames = unpack_frames(table, camera_keys)
    elif isinstance(chunks, bytes):
        frames = unpack_frames(chunks, camera_keys)
    else:
        frames = [Frame.from_json(f) for f in doc.get("frames", [])]

    imu = {}
    for stream_id, meta in doc.get("imu", {}).items():
        buf = tags.get(f"WURLD_IMU_{stream_id}")
        if isinstance(buf, bytes):
            imu[stream_id] = ImuStream.unpack(stream_id, buf, meta)

    return RemoteHeader(
        cameras={k: Camera.from_json(v) for k, v in doc.get("cameras", {}).items()},
        frames=frames,
        signals=[SignalMeta.from_json(s) for s in doc.get("signals", [])],
        world=doc.get("world", {}),
        rigs=doc.get("rigs", {}),
        imu=imu,
        video=chroma_meta,
        header_extent=first_cluster,
        cues_offset=seeks.get(ebml.CUES),
        bytes_fetched=fetched,
        doc=doc,
        head=bytes(head[:first_cluster]),
        segment_start=segment_start,
        payload_start=payload_start,
    )


_CUES_FETCH = 1 << 16  # Cues for hour-scale files fit comfortably in 64 KiB


def fetch_frames(fetch, indices: list[int], header: RemoteHeader | None = None) -> dict:
    """Randomly access video frames via ranged reads (SPEC §9.1 + Cues).

    Fetches only the Clusters containing ``indices`` (plus the Cues, once) and
    decodes each in isolation — possible because chromapakz keyframes every
    track at cluster starts (ChromaPakZ PR #45). Returns::

        {
          "frames": {index: {"rgb": (H,W,4) uint8 | None,
                             "signals": {id: (H,W) uint16}}},
          "bytes_fetched": int,   # cues + clusters (header counted by fetch_header)
          "clusters_fetched": int,
        }

    Raises:
        ValueError: If the file has no Cues, or a fetched Cluster does not start
            with keyframes on every track (written by chromapakz without the
            signal-keyframe cadence — re-encode to enable random access).
    """
    import chromapakz as cz
    import numpy as np

    hdr = header or fetch_header(fetch)
    if hdr.cues_offset is None:
        raise ValueError("no Cues entry in the SeekHead — cannot random-access video")
    meta = hdr.video
    fps, n_frames = meta["fps"], meta["frames"]
    if not fps or not n_frames:
        raise ValueError("chromapakz metadata lacks fps/frames (live stream? use StreamReader)")

    fetched = 0
    cues_buf = fetch(hdr.cues_offset, _CUES_FETCH)
    fetched += len(cues_buf)
    cues = ebml.read_cues(cues_buf)
    if not cues:
        raise ValueError("empty Cues element")
    # cue positions are relative to the Segment payload start
    starts = [(round(t * fps / 1000), hdr.payload_start + pos) for t, pos in cues]

    def cluster_of(frame: int) -> int:
        k = 0
        for j, (f0, _) in enumerate(starts):
            if f0 <= frame:
                k = j
        return k

    wanted: dict[int, list[int]] = {}
    for i in indices:
        if not 0 <= i < n_frames:
            raise IndexError(f"frame {i} out of range [0, {n_frames})")
        wanted.setdefault(cluster_of(i), []).append(i)

    out: dict[int, dict] = {}
    for k, frame_indices in sorted(wanted.items()):
        first_frame, offset = starts[k]
        end = starts[k + 1][1] if k + 1 < len(starts) else hdr.cues_offset
        cluster = fetch(offset, end - offset)
        fetched += len(cluster)

        flags = ebml.cluster_first_block_keyframes(cluster)
        if not all(flags.values()):
            raise ValueError(
                "cluster does not start with keyframes on every track "
                f"(per-track: {flags}) — this file predates chromapakz's "
                "signal-keyframe cadence (ChromaPakZ PR #45); re-encode it to "
                "enable random access"
            )

        expected = (starts[k + 1][0] if k + 1 < len(starts) else n_frames) - first_frame
        # The wurld tags are dropped from the spliced header: a decoder never
        # reads them, and a pose table would be copied for every ranged fetch.
        spliced = ebml.splice_file(
            hdr.head, hdr.segment_start,
            [_decode_head(hdr.head[hdr.payload_start:]), cluster],
        )
        decoded = cz.decode(spliced)
        for i in frame_indices:
            local = i - first_frame
            if local >= expected:
                raise ValueError(f"frame {i} beyond cluster extent ({expected} frames)")
            out[i] = {
                "rgb": np.asarray(decoded["rgb"][local]) if decoded.get("rgb") is not None else None,
                "signals": {sid: np.asarray(arr[local]) for sid, arr in decoded["signals"].items()},
            }
    return {"frames": out, "bytes_fetched": fetched, "clusters_fetched": len(wanted)}
