"""Record3D (.r3d) -> wurld.

A .r3d export is an ordinary ZIP (confirmed by the app author):

    metadata          JSON, no extension: w, h, K (flat 9, COLUMN-major, RGB res),
                      fps, dw, dh (depth grid), poses [[qx,qy,qz,qw,tx,ty,tz]] --
                      ARKit camera-to-world, scalar-LAST quaternions, meters --
                      initPose, frameTimestamps (seconds), cameraType
    rgbd/<N>.jpg      RGB frames, 0-based contiguous indices
    rgbd/<N>.depth    LZFSE-compressed float32 meters, shape (dh, dw), NaN = invalid
    rgbd/<N>.conf     LZFSE-compressed uint8 ARKit confidence 0/1/2 (optional)
    sound.m4a, icon   ignored

ARKit camera axes are RUB; poses convert to canonical RDF on import. Metric float
depth is quantized to lossless-u16 inverse-depth codes with a data-driven near/far
(recorded in the value map). ``at="depth"`` (default) resamples RGB down to the
depth grid; ``at="rgb"`` nearest-upsamples depth/confidence.

Requires ``pyliblzfse`` (``pip install wurld-video[record3d]``).
"""

from __future__ import annotations

import io
import json
import re
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from .. import container, conventions

CONFIDENCE_LABELS = {"0": "low", "1": "medium", "2": "high"}
_FRAME_RE = re.compile(r"^rgbd/(\d+)\.jpg$")


def _lzfse():
    try:
        import liblzfse
    except ImportError as e:
        raise RuntimeError(
            "the Record3D importer needs pyliblzfse (pip install pyliblzfse)"
        ) from e
    return liblzfse


def _read_plane(z: zipfile.ZipFile, name: str, dh: int, dw: int, dtype) -> np.ndarray | None:
    try:
        raw = _lzfse().decompress(z.read(name))
    except KeyError:
        return None
    n = dh * dw
    if dtype is np.float32 and len(raw) == n * 2:
        return np.frombuffer(raw, dtype=np.float16).astype(np.float32).reshape(dh, dw)
    arr = np.frombuffer(raw, dtype=dtype)
    if arr.size != n:
        raise ValueError(f"{name}: {arr.size} samples, expected {dh}x{dw}={n}")
    return arr.reshape(dh, dw)


def from_record3d(
    r3d_path: str | Path,
    out_path: str | Path,
    at: str = "depth",  # "depth" | "rgb"
    rgb_kbps: int = 4000,
) -> Path:
    r3d_path = Path(r3d_path)
    with zipfile.ZipFile(r3d_path) as z:
        names = set(z.namelist())
        meta_name = "metadata" if "metadata" in names else "metadata.json"
        if meta_name not in names:
            raise ValueError(f"{r3d_path}: no metadata file — not a Record3D export?")
        meta = json.loads(z.read(meta_name))

        # K is a flat 9-array in COLUMN-major order at RGB resolution.
        K_rgb = np.array(meta["K"], dtype=np.float64).reshape(3, 3).T
        rgb_w, rgb_h = int(meta["w"]), int(meta["h"])
        dw, dh = int(meta.get("dw", 0)), int(meta.get("dh", 0))
        has_depth = dw > 0 and dh > 0 and any(n.endswith(".depth") for n in names)

        indices = sorted(int(m.group(1)) for n in names if (m := _FRAME_RE.match(n)))
        if not indices or indices != list(range(len(indices))):
            raise ValueError(f"{r3d_path}: rgbd/<N>.jpg frames not contiguous from 0")
        n_frames = len(indices)

        poses = meta.get("poses", [])
        timestamps = meta.get("frameTimestamps", [])
        fps = float(meta.get("fps", 30))
        # Record3D stores ARKit's device uptime, so a phone awake for days starts a
        # take at t≈330000 s. SPEC §3 permits any epoch, but nothing downstream wants
        # one: it prints as nonsense and anything reading t as an offset into the
        # media has to work the origin out for itself. Rebasing to the first frame
        # leaves every interval identical. The .r3d itself is untouched — that file
        # is Record3D's format and keeps Record3D's convention.
        t0 = float(timestamps[0]) if timestamps else 0.0

        if at == "depth" and has_depth:
            W, H = dw, dh
            scale = (dw / rgb_w, dh / rgb_h)
        else:
            W, H = rgb_w, rgb_h
            scale = (1.0, 1.0)
            if at not in ("depth", "rgb"):
                raise ValueError("at must be 'depth' or 'rgb'")

        rgb = np.empty((n_frames, H, W, 4), dtype=np.uint8)
        depth_m = np.empty((n_frames, H, W), dtype=np.float32) if has_depth else None
        conf = None
        frames = []
        for i in indices:
            img = Image.open(io.BytesIO(z.read(f"rgbd/{i}.jpg"))).convert("RGBA")
            if img.size != (W, H):
                img = img.resize((W, H), Image.BOX if at == "depth" else Image.BILINEAR)
            rgb[i] = np.asarray(img)

            if has_depth:
                d = _read_plane(z, f"rgbd/{i}.depth", dh, dw, np.float32)
                c = _read_plane(z, f"rgbd/{i}.conf", dh, dw, np.uint8)
                if at == "rgb":
                    idx_v = (np.arange(H) * dh // H).clip(0, dh - 1)
                    idx_u = (np.arange(W) * dw // W).clip(0, dw - 1)
                    d = d[idx_v][:, idx_u]
                    c = c[idx_v][:, idx_u] if c is not None else None
                depth_m[i] = d
                if c is not None:
                    if conf is None:
                        conf = np.zeros((n_frames, H, W), dtype=np.uint16)
                    conf[i] = c

            t = float(timestamps[i]) - t0 if i < len(timestamps) else i / fps
            if i < len(poses):
                qx, qy, qz, qw, tx, ty, tz = (float(v) for v in poses[i])
                c2w_gl = conventions.pose_to_matrix((qw, qx, qy, qz), (tx, ty, tz))
                q, tr = conventions.matrix_to_pose(conventions.c2w_gl_to_cv(c2w_gl))
                frames.append(container.Frame(i=i, t=t, q_wxyz=tuple(q), tr=tuple(tr)))
            else:
                frames.append(container.Frame(i=i, t=t, pose_valid=False))

    sx, sy = (scale if at == "depth" and has_depth else (W / rgb_w, H / rgb_h))
    camera = container.Camera(
        "PINHOLE", W, H,
        [K_rgb[0, 0] * sx, K_rgb[1, 1] * sy, K_rgb[0, 2] * sx, K_rgb[1, 2] * sy],
    )

    signals, meta_out = None, []
    if has_depth:
        import chromapakz as cz

        finite = depth_m[np.isfinite(depth_m) & (depth_m > 0)]
        near = float(max(0.05, np.min(finite) * 0.95)) if finite.size else 0.1
        far = float(max(near * 2, np.max(finite) * 1.05)) if finite.size else 10.0
        z_clip = np.where(np.isfinite(depth_m) & (depth_m > 0), np.clip(depth_m, near, far), np.nan)
        d16 = cz.quantize_inverse(z_clip, near=near, far=far)
        signals = {"depth": d16}
        meta_out.append(container.SignalMeta(
            "depth", "depth",
            {"type": "inverse_depth", "near": near, "far": far, "levels": 65536, "invalid": 0}))
        if conf is not None:
            signals["confidence"] = conf
            meta_out.append(container.SignalMeta(
                "confidence", "confidence", {"type": "labels", "labels": CONFIDENCE_LABELS}))

    specs = {"depth": cz.inverse_depth_spec(near, far)} if has_depth else None
    return container.write(
        out_path,
        cameras={"0": camera},
        frames=frames,
        rgb=rgb,
        signals=signals,
        specs=specs,
        signal_meta=meta_out,
        fps=fps,
        rgb_kbps=rgb_kbps,
        world={
            "metric_scale": True,
            "gravity_in_world": [0.0, -1.0, 0.0],  # ARKit gravity-aligned world, +Y up
            "description": (
                f"Record3D import from {r3d_path.name} (cameraType={meta.get('cameraType')}); "
                "ARKit RUB poses converted to RDF; float depth (meters) quantized to "
                f"inverse-depth codes near={float(near) if has_depth else 'n/a'}"
                f" far={float(far) if has_depth else 'n/a'}"
            ),
        },
    )
