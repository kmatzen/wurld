"""Wurld container read/write (SPEC.md).

A wurld file is a chromapakz WebM (RGB + lossless uint16 signal tracks)
plus a Matroska SimpleTag named WURLD holding the JSON document with
cameras, per-frame poses/timestamps, signal semantics, and world metadata.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import chromapakz as cz
import numpy as np

from . import conventions, ebml

FORMAT_VERSION = "0.1"

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
        )


@dataclass
class SignalMeta:
    id: str
    role: str  # depth | confidence | object_id | semantic_id | normal_packed | custom
    value_map: dict = field(default_factory=lambda: {"type": "identity"})

    def to_json(self) -> dict:
        return {"id": self.id, "role": self.role, "value_map": self.value_map}

    @classmethod
    def from_json(cls, d: dict) -> "SignalMeta":
        return cls(d["id"], d["role"], d.get("value_map", {"type": "identity"}))

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
    _bytes: bytes = field(repr=False)
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
        return self._decode().get("rgb")

    def signal(self, signal_id: str) -> np.ndarray:
        return self._decode()["signals"][signal_id]

    def signal_meta(self, role: str) -> SignalMeta | None:
        for s in self.signals:
            if s.role == role:
                return s
        return None

    def depth_meters(self, frame_index: int | None = None) -> np.ndarray:
        """Metric depth (meters, NaN=invalid) for one frame or all frames."""
        meta = self.signal_meta("depth")
        if meta is None:
            raise ValueError("no signal with role 'depth'")
        raw = self.signal(meta.id)
        if frame_index is not None:
            raw = raw[frame_index]
        return meta.apply(raw)

    def K(self, camera_id: str = "0") -> np.ndarray:
        return self.cameras[camera_id].K

    def c2w(self, frame_index: int) -> np.ndarray:
        return self.frames[frame_index].c2w

    def to_document(self) -> dict:
        return _document(self.cameras, self.frames, self.signals, self.world)


def _document(cameras, frames, signals, world) -> dict:
    return {
        "format": "wurld",
        "version": FORMAT_VERSION,
        "conventions": dict(CONVENTIONS),
        "world": world,
        "cameras": {k: c.to_json() for k, c in cameras.items()},
        "signals": [s.to_json() for s in signals],
        "frames": [f.to_json() for f in frames],
    }


def write(
    path: str | Path,
    *,
    cameras: dict[str, Camera],
    frames: list[Frame],
    rgb: np.ndarray | None = None,
    signals: dict[str, np.ndarray] | None = None,
    specs: dict[str, dict] | None = None,
    signal_meta: list[SignalMeta] | None = None,
    world: dict | None = None,
    fps: float = 30,
    rgb_kbps: int = 2000,
    validate: bool = True,
) -> Path:
    """Encode and write a wurld file.

    ``signals``/``specs`` pass straight to chromapakz (bit-exact uint16 planes);
    ``signal_meta`` attaches roles/value maps; ``frames`` carry canonical-convention
    poses (RDF, camera-to-world, wxyz) and sensor timestamps.
    """
    path = Path(path)
    world = dict(world or {"metric_scale": True, "gravity_in_world": None, "description": ""})
    signal_meta = list(signal_meta or [])
    frames = sorted(frames, key=lambda f: f.i)

    n_video = None
    for arr in (signals or {}).values():
        n_video = arr.shape[0]
    if rgb is not None:
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
        for cam in cameras.values():
            if rgb is not None and (cam.width != rgb.shape[2] or cam.height != rgb.shape[1]):
                problems.append(
                    f"camera calibrated at {cam.width}x{cam.height} but video is "
                    f"{rgb.shape[2]}x{rgb.shape[1]} (SPEC v0.1 requires equality)"
                )
        if problems:
            raise ValueError("invalid wurld data:\n  " + "\n  ".join(problems))

    # Video-track fps is presentation timing only (frame `t` values are the
    # authoritative timestamps); chromapakz requires an integer rate.
    data = cz.encode(signals or {}, specs=specs, rgb=rgb, fps=max(1, round(fps)), rgb_kbps=rgb_kbps)
    doc = _document(cameras, frames, signal_meta, world)
    tagged = ebml.append_tag(data, "WURLD", json.dumps(doc, separators=(",", ":")))
    path.write_bytes(tagged)
    return path


def read(path: str | Path) -> Sequence:
    """Read a wurld (or plain chromapakz) file. Pixels decode lazily."""
    path = Path(path)
    data = path.read_bytes()
    probe = cz.probe(data)
    raw = ebml.read_tag(data, "WURLD")
    if raw is None:
        doc = {"cameras": {}, "frames": [], "signals": [], "world": {}}
    else:
        doc = json.loads(raw)
        if doc.get("format") != "wurld":
            raise ValueError(f"{path}: WURLD tag present but format={doc.get('format')!r}")
    return Sequence(
        path=path,
        cameras={k: Camera.from_json(v) for k, v in doc.get("cameras", {}).items()},
        frames=[Frame.from_json(f) for f in doc.get("frames", [])],
        signals=[SignalMeta.from_json(s) for s in doc.get("signals", [])],
        world=doc.get("world", {}),
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
            "t_start": seq.frames[0].t if seq.frames else None,
            "t_end": seq.frames[-1].t if seq.frames else None,
        },
    }
