"""Polycam raw capture -> wurld.

Layout parsed by Polycam's own `polyform` tooling:

    capture/
      mesh_info.json, raw.glb, ...
      keyframes/
        images/<ts_us>.jpg            original (distorted) keyframes
        corrected_images/<ts_us>.jpg  undistorted, from pose optimization
        cameras/<ts_us>.json          fx fy cx cy, t_00..t_23 (row-major 3x4 c2w),
                                      timestamp, width, height, blur_score
        corrected_cameras/<ts_us>.json  optimized versions of the above
        depth/<ts_us>.png             16-bit millimeters (LiDAR resolution)
        confidence/<ts_us>.png        8-bit ARKit confidence: 0 low / 127 medium / 255 high

Poses are ARKit convention (gravity-aligned world, +Y up; RUB camera axes) and
convert to canonical RDF on import. ``corrected_cameras``/``corrected_images``
are preferred when present (``corrected=False`` opts out). RGB and depth live on
different grids; the same ``at="depth"`` (default) / ``at="rgb"`` policy as the
Stray importer applies, with intrinsics rescaled to match.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from .. import container, conventions

DEPTH_SCALE = 1.0e-3  # mm -> m
CONFIDENCE_LABELS = {"0": "low", "127": "medium", "255": "high"}


def _keyframes_dir(capture: Path) -> Path:
    kf = capture / "keyframes"
    if not kf.is_dir():
        raise FileNotFoundError(f"{capture}: no keyframes/ directory (not a Polycam raw capture?)")
    return kf


def _camera_from_json(d: dict, sx: float = 1.0, sy: float = 1.0) -> container.Camera:
    return container.Camera(
        "PINHOLE",
        int(round(d["width"] * sx)),
        int(round(d["height"] * sy)),
        [d["fx"] * sx, d["fy"] * sy, d["cx"] * sx, d["cy"] * sy],
    )


def _c2w_from_json(d: dict) -> np.ndarray:
    """t_00..t_23 row-major 3x4 (ARKit RUB c2w) -> canonical RDF c2w."""
    m = np.eye(4)
    for r in range(3):
        for c in range(4):
            m[r, c] = d[f"t_{r}{c}"]
    return conventions.c2w_gl_to_cv(m)


def from_polycam(
    capture: str | Path,
    out_path: str | Path,
    at: str = "depth",  # "depth" | "rgb"
    corrected: bool = True,
    rgb_kbps: int = 4000,
) -> Path:
    capture = Path(capture)
    kf = _keyframes_dir(capture)

    cam_dir = kf / "corrected_cameras" if corrected and (kf / "corrected_cameras").is_dir() else kf / "cameras"
    img_dir = kf / "corrected_images" if corrected and (kf / "corrected_images").is_dir() else kf / "images"
    used_corrected = cam_dir.name == "corrected_cameras"

    def _stem_key(p: Path):
        # Stems are microsecond timestamps: sort numerically, not lexicographically
        # (unequal digit counts would otherwise scramble the frame order).
        try:
            return (0, int(p.stem))
        except ValueError:
            return (1, p.stem)

    cam_files = sorted(cam_dir.glob("*.json"), key=_stem_key)
    if not cam_files:
        raise FileNotFoundError(f"no camera json files under {cam_dir}")

    rgb_list, depth_list, conf_list, frames = [], [], [], []
    cam_json0 = None
    for i, cf in enumerate(cam_files):
        stem = cf.stem
        d = json.loads(cf.read_text())
        cam_json0 = cam_json0 or d

        img_path = img_dir / f"{stem}.jpg"
        if not img_path.exists():
            img_path = img_dir / f"{stem}.png"
        if not img_path.exists():
            raise FileNotFoundError(f"no keyframe image for camera {stem} under {img_dir}")
        rgb_list.append(np.asarray(Image.open(img_path).convert("RGBA")))

        dp = kf / "depth" / f"{stem}.png"
        if dp.exists():
            arr = np.asarray(Image.open(dp))
            if arr.dtype != np.uint16:
                raise ValueError(f"{dp}: Polycam depth must be 16-bit PNG")
            depth_list.append(arr)
        cp = kf / "confidence" / f"{stem}.png"
        if cp.exists():
            conf_list.append(np.asarray(Image.open(cp)).astype(np.uint16))

        # filenames are timestamps in microseconds; the json repeats it
        t_us = float(d.get("timestamp", float(stem)))
        q, tr = conventions.matrix_to_pose(_c2w_from_json(d))
        frames.append(container.Frame(i=i, t=t_us * 1e-6, camera="0", q_wxyz=tuple(q), tr=tuple(tr)))

    if depth_list and len(depth_list) != len(cam_files):
        raise ValueError("some keyframes have depth maps and some do not")
    if conf_list and len(conf_list) != len(cam_files):
        conf_list = []  # partial confidence: drop rather than misalign

    rgb = np.stack(rgb_list)
    rgb_h, rgb_w = rgb.shape[1:3]
    depth = np.stack(depth_list) if depth_list else None
    conf = np.stack(conf_list) if conf_list else None

    if depth is not None and (depth.shape[1], depth.shape[2]) != (rgb_h, rgb_w):
        d_h, d_w = depth.shape[1:3]
        if at == "depth":
            W, H = d_w, d_h
            resized = np.stack(
                [np.asarray(Image.fromarray(f[..., :4]).resize((W, H), Image.BOX)) for f in rgb]
            )
            rgb = resized
            camera = _camera_from_json(cam_json0, sx=d_w / cam_json0["width"], sy=d_h / cam_json0["height"])
        elif at == "rgb":
            W, H = rgb_w, rgb_h
            idx_v = (np.arange(H) * d_h // H).clip(0, d_h - 1)
            idx_u = (np.arange(W) * d_w // W).clip(0, d_w - 1)
            depth = depth[:, idx_v][:, :, idx_u]
            if conf is not None:
                conf = conf[:, idx_v][:, :, idx_u]
            camera = _camera_from_json(cam_json0, sx=rgb_w / cam_json0["width"], sy=rgb_h / cam_json0["height"])
        else:
            raise ValueError("at must be 'depth' or 'rgb'")
    else:
        camera = _camera_from_json(
            cam_json0, sx=rgb_w / cam_json0["width"], sy=rgb_h / cam_json0["height"]
        )

    signals, meta = None, []
    if depth is not None:
        signals = {"depth": depth}
        meta.append(container.SignalMeta(
            "depth", "depth", {"type": "linear", "scale": DEPTH_SCALE, "offset": 0.0, "invalid": 0}))
        if conf is not None:
            signals["confidence"] = conf
            meta.append(container.SignalMeta(
                "confidence", "confidence", {"type": "labels", "labels": CONFIDENCE_LABELS}))

    fps = 5.0  # keyframes, not video; refined from timestamps below
    if len(frames) > 1 and frames[-1].t > frames[0].t:
        fps = max(1.0, round((len(frames) - 1) / (frames[-1].t - frames[0].t), 3))

    return container.write(
        out_path,
        cameras={"0": camera},
        frames=frames,
        rgb=rgb,
        signals=signals,
        signal_meta=meta,
        fps=fps,
        rgb_kbps=rgb_kbps,
        world={
            "metric_scale": True,
            "gravity_in_world": [0.0, -1.0, 0.0],  # ARKit gravity-aligned world, +Y up
            "description": (
                f"Polycam raw import from {capture} "
                f"({'corrected' if used_corrected else 'original'} cameras/images); "
                "ARKit RUB poses converted to RDF; keyframes (not uniform video); "
                "depth is ARKit LiDAR (OS-upsampled, treat as approximate)"
            ),
        },
    )
