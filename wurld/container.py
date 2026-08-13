"""Wurld container read/write (SPEC.md).

A wurld file is a chromapakz WebM (RGB + lossless uint16 signal tracks)
plus a Matroska SimpleTag named WURLD holding the JSON document with
cameras, per-frame poses/timestamps, signal semantics, and world metadata.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from pathlib import Path

import chromapakz as cz
import numpy as np

from . import conventions, ebml

# The document `version` field (SPEC §10), not the package version.
# Tracks the SPEC revision: minor bumps are additive-only.
FORMAT_VERSION = "1.3"

# Binary frame table (SPEC §7): u32 i, u32 camera idx, f64 t, 4×f32 q, 3×f32 tr, u8 flags
_FRAME_RECORD = struct.Struct("<IId4f3fB")
# IMU sample (SPEC §8.3): f64 t, 3×f32 gyro, 3×f32 accel
_IMU_RECORD = struct.Struct("<d3f3f")
# Above this many frames, write() with frames_format="auto" switches to binary.
_BINARY_FRAMES_THRESHOLD = 10_000

CONVENTIONS = {
    "camera_axes": "RDF",
    "pose_direction": "camera_to_world",
    "quaternion_order": "wxyz",
    "units": "meters",
    "timestamp_units": "seconds",
}

# COLMAP-style camera models: name -> parameter names (SPEC.md §4.1).
CAMERA_MODELS = {
    "SIMPLE_PINHOLE": ["f", "cx", "cy"],
    "PINHOLE": ["fx", "fy", "cx", "cy"],
    "SIMPLE_RADIAL": ["f", "cx", "cy", "k"],
    "RADIAL": ["f", "cx", "cy", "k1", "k2"],
    "OPENCV": ["fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2"],
    "OPENCV_FISHEYE": ["fx", "fy", "cx", "cy", "k1", "k2", "k3", "k4"],
}


@dataclass
class Camera:
    model: str
    width: int
    height: int
    params: list[float]

    def __post_init__(self):
        if self.model not in CAMERA_MODELS:
            raise ValueError(f"unknown camera model {self.model!r}")
        expected = len(CAMERA_MODELS[self.model])
        if len(self.params) != expected:
            raise ValueError(
                f"{self.model} expects {expected} params, got {len(self.params)}"
            )

    @property
    def K(self) -> np.ndarray:
        p = self.params
        if self.model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
            fx = fy = p[0]
            cx, cy = p[1], p[2]
        else:
            fx, fy, cx, cy = p[0], p[1], p[2], p[3]
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

    def to_json(self) -> dict:
        return {
            "model": self.model,
            "width": self.width,
            "height": self.height,
            "params": [float(v) for v in self.params],
        }

    @classmethod
    def from_json(cls, d: dict) -> "Camera":
        return cls(d["model"], d["width"], d["height"], list(d["params"]))


@dataclass
class Frame:
    i: int
    t: float
    camera: str = "0"
    q_wxyz: tuple[float, float, float, float] | None = None
    tr: tuple[float, float, float] | None = None
    pose_valid: bool = True
    params: list[float] | None = None  # per-frame intrinsics override (SPEC §8.2)

    def __post_init__(self):
        if self.pose_valid and (self.q_wxyz is None or self.tr is None):
            raise ValueError(f"frame {self.i}: pose_valid but q_wxyz/tr missing")

    @property
    def c2w(self) -> np.ndarray:
        if not self.pose_valid:
            raise ValueError(f"frame {self.i}: pose not valid")
        return conventions.pose_to_matrix(self.q_wxyz, self.tr)

    def to_json(self) -> dict:
        d = {"i": self.i, "t": self.t, "camera": self.camera}
        if self.pose_valid:
            d["q_wxyz"] = [float(v) for v in self.q_wxyz]
            d["tr"] = [float(v) for v in self.tr]
        else:
            d["pose_valid"] = False
        if self.params is not None:
            d["params"] = [float(v) for v in self.params]
        return d

    @classmethod
    def from_json(cls, d: dict) -> "Frame":
        return cls(
            i=d["i"],
            t=d["t"],
            camera=d.get("camera", "0"),
            q_wxyz=tuple(d["q_wxyz"]) if "q_wxyz" in d else None,
            tr=tuple(d["tr"]) if "tr" in d else None,
            pose_valid=d.get("pose_valid", True),
            params=list(d["params"]) if "params" in d else None,
        )


def pack_frames(frames: list[Frame], camera_keys: list[str]) -> bytes:
    """Frames -> binary table (SPEC §7). Requires no per-frame intrinsics."""
    cam_index = {k: i for i, k in enumerate(camera_keys)}
    out = bytearray()
    for f in frames:
        if f.params is not None:
            raise ValueError(
                f"frame {f.i}: per-frame intrinsics require the JSON frame form (SPEC §7)"
            )
        q = f.q_wxyz if f.pose_valid else (1.0, 0.0, 0.0, 0.0)
        tr = f.tr if f.pose_valid else (0.0, 0.0, 0.0)
        out += _FRAME_RECORD.pack(
            f.i, cam_index[f.camera], f.t, *q, *tr, 1 if f.pose_valid else 0
        )
    return bytes(out)


def unpack_frames(buf: bytes, camera_keys: list[str]) -> list[Frame]:
    if len(buf) % _FRAME_RECORD.size:
        raise ValueError(
            f"WURLD_FRAMES length {len(buf)} is not a multiple of {_FRAME_RECORD.size}"
        )
    frames = []
    for n, rec in enumerate(_FRAME_RECORD.iter_unpack(buf)):
        i, cam, t, qw, qx, qy, qz, tx, ty, tz, flags = rec
        valid = bool(flags & 1)
        # The camera index comes out of the file; a table that disagrees with the
        # document's camera list must be rejected, not indexed with.
        if cam >= len(camera_keys):
            raise ValueError(
                f"record {n}: camera index {cam} but only {len(camera_keys)} "
                f"camera(s) declared ({', '.join(camera_keys) or 'none'})"
            )
        frames.append(
            Frame(
                i=i,
                t=t,
                camera=camera_keys[cam],
                q_wxyz=(qw, qx, qy, qz) if valid else None,
                tr=(tx, ty, tz) if valid else None,
                pose_valid=valid,
            )
        )
    return frames


@dataclass
class ImuStream:
    """samples: (N, 7) float64 array — [t, gyro xyz (rad/s), accel xyz (m/s^2)]."""

    id: str
    samples: np.ndarray
    rate_hz: float | None = None
    extrinsics: dict | None = None  # {"q_wxyz": [...], "tr": [...]} imu-to-camera
    description: str = ""

    def __post_init__(self):
        self.samples = np.asarray(self.samples, dtype=np.float64)
        if self.samples.ndim != 2 or self.samples.shape[1] != 7:
            raise ValueError(f"imu {self.id}: samples must be (N, 7) [t, gyro, accel]")

    def pack(self) -> bytes:
        out = bytearray()
        for row in self.samples:
            out += _IMU_RECORD.pack(row[0], *row[1:4].astype(np.float32), *row[4:7].astype(np.float32))
        return bytes(out)

    @classmethod
    def unpack(cls, stream_id: str, buf: bytes, meta: dict) -> "ImuStream":
        if len(buf) % _IMU_RECORD.size:
            raise ValueError(f"imu {stream_id}: payload not a multiple of {_IMU_RECORD.size}")
        rows = [list(rec) for rec in _IMU_RECORD.iter_unpack(buf)]
        return cls(
            id=stream_id,
            samples=np.array(rows, dtype=np.float64).reshape(-1, 7),
            rate_hz=meta.get("rate_hz"),
            extrinsics=meta.get("extrinsics"),
            description=meta.get("description", ""),
        )

    def to_json(self) -> dict:
        d = {"count": int(self.samples.shape[0]), "description": self.description}
        if self.rate_hz is not None:
            d["rate_hz"] = float(self.rate_hz)
        if self.extrinsics is not None:
            d["extrinsics"] = self.extrinsics
        return d


@dataclass
class SignalMeta:
    id: str
    role: str  # depth | confidence | object_id | semantic_id | normal_packed | custom
    value_map: dict = field(default_factory=lambda: {"type": "identity"})
    # The signal's own stored resolution (SPEC §4.6), when it differs from the
    # file's. The chromapakz metadata is authoritative; this pair keeps the
    # WURLD document self-describing and tells a streaming writer what geometry
    # to declare. Always both or neither.
    width: int | None = None
    height: int | None = None

    def __post_init__(self):
        if (self.width is None) != (self.height is None):
            raise ValueError(
                f"signal {self.id!r}: give width and height together, or neither"
            )

    def to_json(self) -> dict:
        d = {"id": self.id, "role": self.role, "value_map": self.value_map}
        if self.width is not None:
            d["width"] = self.width
            d["height"] = self.height
        return d

    @classmethod
    def from_json(cls, d: dict) -> "SignalMeta":
        return cls(d["id"], d["role"], d.get("value_map", {"type": "identity"}),
                   d.get("width"), d.get("height"))

    def apply(self, raw: np.ndarray) -> np.ndarray:
        """Map raw uint16 values to physical values (NaN where invalid)."""
        vm = self.value_map
        kind = vm.get("type", "identity")
        if kind == "identity" or kind == "labels":
            return raw
        if kind == "linear":
            out = raw.astype(np.float64) * vm.get("scale", 1.0) + vm.get("offset", 0.0)
            invalid = vm.get("invalid")
            if invalid is not None:
                out = np.where(raw == invalid, np.nan, out)
            return out
        if kind == "inverse_depth":
            # chromapakz inverse-depth quantization is authoritative; code 0
            # is invalid and dequantizes to NaN.
            return cz.dequantize_inverse(
                raw, near=vm["near"], far=vm["far"], levels=vm.get("levels", 65536)
            )
        if kind == "float16_bits":
            # Not a quantization: the codes *are* the float. IEEE half is exactly
            # 16 bits, so a lossless uint16 signal carries scene-referred HDR
            # (EXR half) bit-exactly — NaN, +/-Inf, -0.0 and denormals included.
            # There is no `invalid` code, because every bit pattern is a value;
            # absence is expressed as NaN, which is itself a bit pattern.
            return np.ascontiguousarray(raw, dtype=np.uint16).view(np.float16)
        raise ValueError(f"unknown value_map type {kind!r}")


@dataclass
class Sequence:
    """A read wurld file. Pixel data decodes lazily on first access."""

    path: Path
    cameras: dict[str, Camera]
    frames: list[Frame]
    signals: list[SignalMeta]
    world: dict
    probe: dict
    rigs: dict = field(default_factory=dict)
    imu: dict[str, ImuStream] = field(default_factory=dict)
    _bytes: bytes = field(default=b"", repr=False)
    _decoded: dict | None = field(default=None, repr=False)

    @property
    def n_frames(self) -> int:
        return self.probe["frames"]

    def _decode(self) -> dict:
        if self._decoded is None:
            self._decoded = cz.decode(self._bytes)
        return self._decoded

    @property
    def rgb(self) -> np.ndarray | None:
        """The primary display stream. uint8 codes, or uint16 when it is HDR."""
        return self._decode().get("rgb")

    @property
    def rgb_streams(self) -> list[str]:
        """Camera ids that have stored pixels, primary first.

        A rig may declare cameras with no stream of their own — their poses come
        from `rigs` (see `rig_c2w`) but their pixels were never recorded.
        """
        probe = self.probe or {}
        return [r.get("id") for r in (probe.get("rgbs") or []) if r.get("id")]

    def rgb_for(self, camera_id: str) -> np.ndarray:
        """One camera's pixels. Stream ids are camera ids (SPEC §4.4)."""
        streams = self.rgb_streams
        if streams and camera_id not in streams:
            raise KeyError(
                f"camera {camera_id!r} has no stored pixels; streams: {streams}"
            )
        decoded = self._decode()
        per_stream = decoded.get("rgbs")
        if isinstance(per_stream, dict) and camera_id in per_stream:
            return per_stream[camera_id]
        return decoded.get("rgb")

    @property
    def hdr(self) -> dict | None:
        """The display track's HDR signalling, or None when it is SDR.

        Display-referred (PQ or HLG, absolute nits) — a different thing from a
        `float16_bits` signal, which is scene-referred radiance. A file may
        carry both.
        """
        for r in ((self.probe or {}).get("rgbs") or []):
            if r.get("hdr"):
                return r["hdr"]
        return None

    def signal(self, signal_id: str) -> np.ndarray:
        """Raw uint16 codes, exactly as stored."""
        return self._decode()["signals"][signal_id]

    def signal_values(self, signal_id: str, frame_index: int | None = None) -> np.ndarray:
        """Codes mapped through the signal's value_map to physical values.

        The general form of `depth_meters`, which is the depth-role shorthand.
        Signals with no declared value_map come back as raw codes.
        """
        raw = self.signal(signal_id)
        if frame_index is not None:
            raw = raw[frame_index]
        for s in self.signals:
            if s.id == signal_id:
                return s.apply(raw)
        return raw

    def signal_meta(self, role: str) -> SignalMeta | None:
        for s in self.signals:
            if s.role == role:
                return s
        return None

    def iter_frames(self, start: int = 0, stop: int | None = None):
        """Yield ``(index, {"rgb", "rgbs", "signals"})`` with bounded memory.

        ``rgbs`` is ``{camera_id: plane}`` for a multi-camera file (SPEC §4.4)
        and None otherwise. Without it a streaming consumer of a stereo file
        silently sees one eye — the primary — which looks like a working
        conversion right up until someone needs the other camera.

        Decodes one Cluster at a time (splice decode), so an hour-long file
        never holds more than ~1 s of decoded frames. Requires the chromapakz
        signal-keyframe cadence; files written before it fall back to a single
        full decode (correct, but memory-unbounded — a warning is logged).
        """
        import logging

        data = self._bytes
        n = self.probe["frames"]
        fps = self.probe["fps"]
        stop = n if stop is None else min(stop, n)
        if start >= stop:
            return

        seg_start, ps, pe = ebml._segment_bounds(data)
        clusters = []  # (first_frame, elem_start, elem_end)
        head_end = None
        for eid, es, pstart, pend in ebml._top_level(data, ps, pe):
            if eid == ebml.CLUSTER:
                if head_end is None:
                    head_end = es
                ts = 0
                for cid, cs, ce in ebml.iter_children(data, pstart, pend):
                    if cid == ebml.CLUSTER_TIMESTAMP:
                        ts = ebml._read_uint(data, cs, ce)
                        break
                clusters.append((round(ts * fps / 1000), es, pend))
        if head_end is None:
            return

        independent = all(
            all(ebml.cluster_first_block_keyframes(data[es:pend]).values())
            for _, es, pend in clusters[1:]
        )
        if not independent:
            logging.getLogger(__name__).warning(
                "%s predates the chromapakz signal-keyframe cadence; "
                "iter_frames falls back to one full decode", self.path
            )
            decoded = self._decode()
            for i in range(start, stop):
                yield i, {
                    "rgb": np.asarray(decoded["rgb"][i]) if decoded.get("rgb") is not None else None,
                    "rgbs": _stream_planes(decoded, i),
                    "signals": {sid: np.asarray(a[i]) for sid, a in decoded["signals"].items()},
                }
            return

        head = data[ps:head_end]
        # Built once: the wurld tags are dropped for the per-Cluster decodes, so
        # a long capture's pose table is not copied alongside every Cluster.
        decode_head = _decode_head(head)

        for k, (first, es, pend) in enumerate(clusters):
            last = (clusters[k + 1][0] if k + 1 < len(clusters) else n) - 1
            if last < start or first >= stop:
                continue
            spliced = ebml.splice_file(data[:ps], seg_start,
                                       [decode_head, data[es:pend]])
            decoded = cz.decode(spliced)
            count = last - first + 1
            for local in range(count):
                i = first + local
                if not start <= i < stop:
                    continue
                yield i, {
                    "rgb": np.asarray(decoded["rgb"][local]) if decoded.get("rgb") is not None else None,
                    "rgbs": _stream_planes(decoded, local),
                    "signals": {sid: np.asarray(a[local]) for sid, a in decoded["signals"].items()},
                }

    def fetch_frames(self, indices: list[int]) -> dict[int, dict]:
        """Decode only the Clusters containing ``indices`` (partial decode).

        For long sequences this skips decoding untouched clusters entirely —
        the local counterpart of :func:`wurld.remote.fetch_frames`, and it
        needs the same cluster-independence (chromapakz signal-keyframe
        cadence). Returns ``{index: {"rgb": ..., "signals": {id: ...}}}``.
        """
        from . import remote

        data = self._bytes

        def fetch(offset: int, size: int) -> bytes:
            return data[offset : offset + size]

        return remote.fetch_frames(fetch, indices)["frames"]

    def depth_meters(self, frame_index: int | None = None) -> np.ndarray:
        """Metric depth (meters, NaN=invalid) for one frame or all frames."""
        meta = self.signal_meta("depth")
        if meta is None:
            raise ValueError("no signal with role 'depth'")
        raw = self.signal(meta.id)
        if frame_index is not None:
            raw = raw[frame_index]
        return meta.apply(raw)

    def signal_resolution(self, signal_id: str) -> tuple[int, int]:
        """(width, height) a signal is stored at (SPEC §4.6).

        Per-stream geometry (chromapakz format v4) lets a 256x192 LiDAR depth
        map ride beside full-resolution RGB; a signal with no geometry of its
        own is at the file resolution, as every signal was before v1.3. Read
        from the codec metadata, which is authoritative.
        """
        probe = self.probe or {}
        for s in probe.get("signals") or []:
            if s.get("id") == signal_id:
                return (s.get("width") or probe["width"],
                        s.get("height") or probe["height"])
        raise KeyError(f"no signal {signal_id!r}; file has "
                       f"{[s.get('id') for s in probe.get('signals') or []]}")

    def K(self, camera_id: str = "0", frame_index: int | None = None,
          signal_id: str | None = None) -> np.ndarray:
        """Intrinsics for a camera, honoring a per-frame override when given.

        With ``signal_id``, the intrinsics are scaled to that signal's stored
        resolution (SPEC §4.6: signal grids are FOV-aligned with the camera
        image, so calibration scales linearly) — the K to unproject a depth
        plane that is stored smaller than the video.
        """
        cam = self.cameras[camera_id]
        if frame_index is not None and self.frames[frame_index].params is not None:
            cam = Camera(cam.model, cam.width, cam.height, self.frames[frame_index].params)
        K = cam.K
        if signal_id is not None:
            sw, sh = self.signal_resolution(signal_id)
            if (sw, sh) != (cam.width, cam.height):
                K = K.copy()
                K[0] *= sw / cam.width
                K[1] *= sh / cam.height
        return K

    def c2w(self, frame_index: int) -> np.ndarray:
        return self.frames[frame_index].c2w

    def rig_c2w(self, frame_index: int, camera_id: str, rig_id: str | None = None) -> np.ndarray:
        """Derive another rig camera's c2w from this frame's pose (SPEC §8.1)."""
        rig_id = rig_id or next(iter(self.rigs))
        rig = self.rigs[rig_id]["cameras"]
        f = self.frames[frame_index]

        def cam2rig(key):
            e = rig[key]
            return conventions.pose_to_matrix(e["q_wxyz"], e["tr"])

        return f.c2w @ conventions.invert_pose(cam2rig(f.camera)) @ cam2rig(camera_id)

    def to_document(self) -> dict:
        return _document(self.cameras, self.frames, self.signals, self.world, self.rigs, self.imu)


def _stream_planes(decoded: dict, index: int) -> dict | None:
    """One frame of each display stream, or None when the file has just the one."""
    rgbs = decoded.get("rgbs")
    if not rgbs or len(rgbs) < 2:
        return None
    return {sid: np.asarray(arr[index]) for sid, arr in rgbs.items()}


def _decode_head(head: bytes) -> bytes:
    """A header for decoding one spliced Cluster, with the wurld tags dropped.

    `cz.decode` never reads them, and a binary pose table for a long capture is
    hundreds of kilobytes copied per Cluster for nothing.

    This used to also rewrite CHROMAPAKZ's frame count, because chromapakz sized
    its output buffers from the header and so allocated for the whole sequence
    on every partial decode — 277 MB per Cluster against 14.7 MB. That is fixed
    upstream in chromapakz 0.9.0 (ChromaPakZ #58), which counts the blocks
    actually present, so the rewrite is gone and the pin moved to match.
    """
    out = b""
    for eid, es, pstart, pend in ebml._top_level(head, 0, len(head)):
        if eid != ebml.TAGS:
            out += head[es:pend]
            continue
        chroma = None
        for name, value in ebml.collect_simple_tags(head, pstart, pend):
            if name == "CHROMAPAKZ" and isinstance(value, str):
                chroma = value
        if chroma is None:
            out += head[es:pend]
            continue
        out += ebml.build_tags({"CHROMAPAKZ": chroma})
    return out


def _document(cameras, frames, signals, world, rigs=None, imu=None) -> dict:
    doc = {
        "format": "wurld",
        "version": FORMAT_VERSION,
        "conventions": dict(CONVENTIONS),
        "world": world,
        "cameras": {k: c.to_json() for k, c in cameras.items()},
        "signals": [s.to_json() for s in signals],
        "frames": [f.to_json() for f in frames],
    }
    if rigs:
        doc["rigs"] = rigs
    if imu:
        doc["imu"] = {k: s.to_json() for k, s in imu.items()}
    return doc


def _validate_rigs(rigs: dict, cameras: dict) -> list[str]:
    problems = []
    for rig_id, rig in rigs.items():
        for cam_key, e in rig.get("cameras", {}).items():
            if cam_key not in cameras:
                problems.append(f"rig {rig_id}: unknown camera {cam_key!r}")
            q = np.asarray(e.get("q_wxyz", []), dtype=np.float64)
            if q.shape != (4,) or abs(np.linalg.norm(q) - 1.0) > 1e-3:
                problems.append(f"rig {rig_id}/{cam_key}: q_wxyz missing or not unit")
            if len(e.get("tr", [])) != 3:
                problems.append(f"rig {rig_id}/{cam_key}: tr must be length 3")
    return problems


def write(
    path: str | Path,
    *,
    cameras: dict[str, Camera],
    frames: list[Frame],
    rgb: np.ndarray | dict[str, np.ndarray] | None = None,
    signals: dict[str, np.ndarray] | None = None,
    specs: dict[str, dict] | None = None,
    signal_meta: list[SignalMeta] | None = None,
    world: dict | None = None,
    rigs: dict | None = None,
    imu: list[ImuStream] | None = None,
    frames_format: str = "auto",  # "auto" | "json" | "binary"
    fps: float = 30,
    rgb_kbps: int = 2000,
    hdr: dict | None = None,
    validate: bool = True,
) -> Path:
    """Encode and write a wurld file.

    ``signals``/``specs`` pass straight to chromapakz (bit-exact uint16 planes);
    ``signal_meta`` attaches roles/value maps; ``frames`` carry canonical-convention
    poses (RDF, camera-to-world, wxyz) and sensor timestamps. ``frames_format``
    "binary" packs frames into a WURLD_FRAMES tag (SPEC §7); "auto" does so
    beyond 10k frames when no frame carries a per-frame intrinsics override.
    """
    path = Path(path)
    world = dict(world or {"metric_scale": True, "gravity_in_world": None, "description": ""})
    signal_meta = list(signal_meta or [])
    rigs = dict(rigs or {})
    imu_streams = {s.id: s for s in (imu or [])}
    frames = sorted(frames, key=lambda f: f.i)

    has_overrides = any(f.params is not None for f in frames)
    if frames_format == "auto":
        use_binary = len(frames) > _BINARY_FRAMES_THRESHOLD and not has_overrides
    elif frames_format == "binary":
        use_binary = True
    elif frames_format == "json":
        use_binary = False
    else:
        raise ValueError(f"frames_format must be auto|json|binary, got {frames_format!r}")

    n_video = None
    for arr in (signals or {}).values():
        n_video = arr.shape[0]
    if isinstance(rgb, dict):
        # Every declared stream shares one frame grid, so they must agree on
        # length — a short stream would silently misalign poses for that camera.
        lengths = {cid: a.shape[0] for cid, a in rgb.items()}
        if len(set(lengths.values())) > 1:
            raise ValueError(f"rgb streams disagree on frame count: {lengths}")
        n_video = next(iter(lengths.values()), None)
    elif rgb is not None:
        n_video = rgb.shape[0]
    if n_video is None:
        raise ValueError("need rgb and/or signals")

    if validate:
        problems = conventions.validate_frames(frames)
        if frames and (frames[0].i < 0 or frames[-1].i >= n_video):
            problems.append(
                f"frame indices [{frames[0].i}, {frames[-1].i}] out of range for {n_video} video frames"
            )
        known = {c for c in cameras}
        for f in frames:
            if f.camera not in known:
                problems.append(f"frame {f.i}: unknown camera {f.camera!r}")
                break
        # Each camera is calibrated at its own stream's resolution (SPEC §4.1);
        # streams need not agree with each other since v1.3 / chromapakz v4.
        if isinstance(rgb, dict):
            for cid, arr in rgb.items():
                cam = cameras.get(cid)
                if cam is not None and (cam.width != arr.shape[2]
                                        or cam.height != arr.shape[1]):
                    problems.append(
                        f"camera {cid!r} calibrated at {cam.width}x{cam.height} but its "
                        f"stream is {arr.shape[2]}x{arr.shape[1]} (SPEC 4.1 requires equality)"
                    )
        elif rgb is not None:
            # One stream shared by every posed camera: all of them see its pixels.
            for cam in cameras.values():
                if cam.width != rgb.shape[2] or cam.height != rgb.shape[1]:
                    problems.append(
                        f"camera calibrated at {cam.width}x{cam.height} but video is "
                        f"{rgb.shape[2]}x{rgb.shape[1]} (SPEC requires equality)"
                    )
        for s in signal_meta:
            arr = (signals or {}).get(s.id)
            if s.width is not None and arr is not None and (
                    s.width != arr.shape[2] or s.height != arr.shape[1]):
                problems.append(
                    f"signal {s.id!r} declares {s.width}x{s.height} but its plane is "
                    f"{arr.shape[2]}x{arr.shape[1]} (SPEC 4.3: the pair must agree)"
                )
        for f in frames:
            if f.params is not None and len(f.params) != len(cameras[f.camera].params):
                problems.append(
                    f"frame {f.i}: params override length {len(f.params)} != "
                    f"camera model {cameras[f.camera].model}"
                )
                break
        problems += _validate_rigs(rigs, cameras)
        if problems:
            raise ValueError("invalid wurld data:\n  " + "\n  ".join(problems))

    # Per-stream geometry in the document (SPEC §4.6): a signal stored at its
    # own resolution states it, so the document stays self-describing even for
    # a caller that never mentioned geometry (chromapakz infers it from the
    # arrays; the codec metadata stays authoritative either way).
    if signals:
        primary = next(iter(rgb.values())) if isinstance(rgb, dict) else rgb
        if primary is None:
            primary = next(iter(signals.values()))
        ph, pw = primary.shape[1], primary.shape[2]
        signal_meta = [
            SignalMeta(s.id, s.role, s.value_map,
                       signals[s.id].shape[2], signals[s.id].shape[1])
            if s.width is None and s.id in signals
               and signals[s.id].shape[1:3] != (ph, pw)
            else s
            for s in signal_meta
        ]

    # Video-track fps is presentation timing only (frame `t` values are the
    # authoritative timestamps); chromapakz requires an integer rate.
    # rgb may be one array (single display stream) or {camera_id: array} for a
    # rig that stores several cameras' pixels. Stream ids are camera ids, which
    # is what lets a reader tell which intrinsics apply to which pixels; the
    # first entry is the primary and keeps chromapakz's track 1 and name "rgb",
    # so pre-0.7.0 readers still see exactly one RGB track.
    rgb_arg, rgbs_arg = rgb, None
    if isinstance(rgb, dict):
        unknown = [k for k in rgb if k not in cameras]
        if unknown:
            raise ValueError(
                f"rgb streams {unknown} have no matching camera; stream ids are "
                f"camera ids (declared: {sorted(cameras)})"
            )
        rgb_arg, rgbs_arg = None, rgb
    if hdr is not None and rgb_arg is None and rgbs_arg is None:
        raise ValueError("hdr= was given but there is no RGB stream to apply it to")

    data = cz.encode(signals or {}, specs=specs, rgb=rgb_arg, rgbs=rgbs_arg,
                     fps=max(1, round(fps)), rgb_kbps=rgb_kbps, hdr=hdr)

    tags: dict[str, str | bytes] = {}
    if use_binary:
        camera_keys = sorted(cameras)
        doc = _document(cameras, [], signal_meta, world, rigs, imu_streams)
        doc["frames_binary"] = {"version": 1, "count": len(frames), "cameras": camera_keys}
        tags["WURLD"] = json.dumps(doc, separators=(",", ":"))
        tags["WURLD_FRAMES"] = pack_frames(frames, camera_keys)
    else:
        doc = _document(cameras, frames, signal_meta, world, rigs, imu_streams)
        tags["WURLD"] = json.dumps(doc, separators=(",", ":"))
    for stream_id, stream in imu_streams.items():
        tags[f"WURLD_IMU_{stream_id}"] = stream.pack()

    # Streaming layout (SPEC §9): all metadata lands before the first Cluster,
    # so a progressive reader has calibration and every pose ahead of the video.
    path.write_bytes(ebml.insert_header_tags(data, tags))
    return path


def read(path: str | Path) -> Sequence:
    """Read a wurld (or plain chromapakz) file. Pixels decode lazily."""
    path = Path(path)
    data = path.read_bytes()
    probe = cz.probe(data)
    all_tags = ebml.read_all_tags(data)
    raw = all_tags.get("WURLD")
    if not isinstance(raw, str):
        doc = {"cameras": {}, "frames": [], "signals": [], "world": {}}
    else:
        doc = json.loads(raw)
        if doc.get("format") != "wurld":
            raise ValueError(f"{path}: WURLD tag present but format={doc.get('format')!r}")

    # Pose precedence (SPEC §9): consolidated table > streamed chunks > JSON array.
    fb = doc.get("frames_binary")
    camera_keys = list(fb["cameras"]) if fb and "cameras" in fb else sorted(doc.get("cameras", {}))
    table = all_tags.get("WURLD_FRAMES")
    chunks = all_tags.get("WURLD_POSES")
    if isinstance(table, bytes):
        if fb is not None and fb.get("version", 1) != 1:
            raise ValueError(f"{path}: unsupported frames_binary version {fb.get('version')}")
        frames = unpack_frames(table, camera_keys)
        if fb is not None and len(frames) != fb.get("count", len(frames)):
            raise ValueError(
                f"{path}: WURLD_FRAMES has {len(frames)} records, expected {fb.get('count')}"
            )
    elif isinstance(chunks, bytes):
        frames = unpack_frames(chunks, camera_keys)
    else:
        if fb is not None:
            raise ValueError(f"{path}: frames_binary declared but WURLD_FRAMES tag missing")
        frames = [Frame.from_json(f) for f in doc.get("frames", [])]

    imu = {}
    for stream_id, meta in doc.get("imu", {}).items():
        buf = all_tags.get(f"WURLD_IMU_{stream_id}")
        if isinstance(buf, bytes):
            imu[stream_id] = ImuStream.unpack(stream_id, buf, meta)

    return Sequence(
        path=path,
        cameras={k: Camera.from_json(v) for k, v in doc.get("cameras", {}).items()},
        frames=frames,
        signals=[SignalMeta.from_json(s) for s in doc.get("signals", [])],
        world=doc.get("world", {}),
        rigs=doc.get("rigs", {}),
        imu=imu,
        probe=probe,
        _bytes=data,
    )


def info(path: str | Path) -> dict:
    """Cheap summary without decoding pixels."""
    seq = read(path)
    return {
        "path": str(seq.path),
        "video": {
            "width": seq.probe["width"],
            "height": seq.probe["height"],
            "frames": seq.probe["frames"],
            "fps": seq.probe["fps"],
            "has_rgb": seq.probe["has_rgb"],
        },
        "chromapakz_signals": [s["id"] for s in seq.probe.get("signals", [])],
        "wurld": {
            "posed_frames": len(seq.frames),
            "cameras": {k: c.to_json() for k, c in seq.cameras.items()},
            "signals": [s.to_json() for s in seq.signals],
            "world": seq.world,
            "rigs": seq.rigs,
            "imu": {k: s.to_json() for k, s in seq.imu.items()},
            "t_start": seq.frames[0].t if seq.frames else None,
            "t_end": seq.frames[-1].t if seq.frames else None,
        },
    }
