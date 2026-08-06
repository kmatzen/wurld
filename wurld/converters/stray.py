"""Stray Scanner (iOS) -> wurld.

Stray Scanner layout (docs.strayrobots.io):

    camera_matrix.csv   3x3 RGB intrinsics (comma-separated rows)
    odometry.csv        timestamp, frame, x, y, z, qx, qy, qz, qw  (ARKit odometry)
    rgb.mp4             H.264 video at sensor resolution
    depth/NNNNNN.png    16-bit millimeters at LiDAR resolution (256x192)
    confidence/NNNNNN.png  optional uint8 ARKit confidence (0/1/2)

ARKit camera axes are RUB (OpenGL); poses convert to canonical RDF on import.
RGB and depth resolutions differ; ``at="depth"`` (default) resamples RGB down to
the depth grid (honest sampling for 3D use), ``at="rgb"`` nearest-upsamples
depth/confidence to the RGB grid. Intrinsics are rescaled to match.

RGB decode shells out to ffmpeg (required on PATH for this importer).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

from .. import container, conventions

DEPTH_SCALE = 1.0e-3  # mm -> m
CONFIDENCE_LABELS = {"0": "low", "1": "medium", "2": "high"}


def _require_ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if exe is None:
        raise RuntimeError("the Stray importer needs ffmpeg on PATH to decode rgb.mp4")
    return exe


def _video_size(path: Path) -> tuple[int, int]:
    probe = shutil.which("ffprobe")
    if probe is None:
        raise RuntimeError("the Stray importer needs ffprobe on PATH")
    out = subprocess.run(
        [probe, "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    w, h = (int(v) for v in out.split(","))
    return w, h


def _decode_rgb(path: Path, out_w: int, out_h: int) -> np.ndarray:
    """Decode rgb.mp4 to (T, out_h, out_w, 4) RGBA via ffmpeg rawvideo pipe."""
    proc = subprocess.run(
        [_require_ffmpeg(), "-v", "error", "-i", str(path),
         "-vf", f"scale={out_w}:{out_h}:flags=area",
         "-f", "rawvideo", "-pix_fmt", "rgba", "pipe:1"],
        capture_output=True, check=True,
    )
    frame_bytes = out_w * out_h * 4
    n = len(proc.stdout) // frame_bytes
    return np.frombuffer(proc.stdout[: n * frame_bytes], dtype=np.uint8).reshape(
        n, out_h, out_w, 4
    ).copy()


def _read_odometry(path: Path) -> dict[int, tuple[float, np.ndarray]]:
    """frame index -> (timestamp, canonical RDF c2w)."""
    poses = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.lower().startswith("timestamp") or line.startswith("#"):
            continue
        vals = [v.strip() for v in line.split(",")]
        t, frame = float(vals[0]), int(float(vals[1]))
        x, y, z, qx, qy, qz, qw = (float(v) for v in vals[2:9])
        c2w_gl = conventions.pose_to_matrix(
            conventions.quat_xyzw_to_wxyz([qx, qy, qz, qw]), [x, y, z]
        )
        poses[frame] = (t, conventions.c2w_gl_to_cv(c2w_gl))
    return poses


def from_stray(
    seq_dir: str | Path,
    out_path: str | Path,
    at: str = "depth",  # "depth" | "rgb"
    rgb_kbps: int = 4000,
) -> Path:
    seq_dir = Path(seq_dir)
    K_rgb = np.loadtxt(seq_dir / "camera_matrix.csv", delimiter=",").reshape(3, 3)
    odometry = _read_odometry(seq_dir / "odometry.csv")

    depth_files = sorted((seq_dir / "depth").glob("*.png"))
    if not depth_files:
        raise FileNotFoundError(f"no depth PNGs under {seq_dir/'depth'}")
    depth = np.stack([np.asarray(Image.open(f)) for f in depth_files])
    if depth.dtype != np.uint16:
        raise ValueError("Stray depth PNGs must be 16-bit")
    conf_dir = seq_dir / "confidence"
    conf = None
    if conf_dir.is_dir() and sorted(conf_dir.glob("*.png")):
        conf = np.stack(
            [np.asarray(Image.open(f)).astype(np.uint16) for f in sorted(conf_dir.glob("*.png"))]
        )

    rgb_w, rgb_h = _video_size(seq_dir / "rgb.mp4")
    d_h, d_w = depth.shape[1:3]
    if at == "depth":
        W, H = d_w, d_h
        scale = (d_w / rgb_w, d_h / rgb_h)
    elif at == "rgb":
        W, H = rgb_w, rgb_h
        scale = (1.0, 1.0)
        idx_v = (np.arange(H) * d_h // H).clip(0, d_h - 1)
        idx_u = (np.arange(W) * d_w // W).clip(0, d_w - 1)
        depth = depth[:, idx_v][:, :, idx_u]
        if conf is not None:
            conf = conf[:, idx_v][:, :, idx_u]
    else:
        raise ValueError("at must be 'depth' or 'rgb'")

    rgb = _decode_rgb(seq_dir / "rgb.mp4", W, H)
    n = min(len(rgb), len(depth), max(odometry) + 1 if odometry else len(rgb))
    rgb, depth = rgb[:n], depth[:n]
    if conf is not None:
        conf = conf[:n]

    fx, fy = K_rgb[0, 0] * scale[0], K_rgb[1, 1] * scale[1]
    cx, cy = K_rgb[0, 2] * scale[0], K_rgb[1, 2] * scale[1]
    camera = container.Camera("PINHOLE", W, H, [fx, fy, cx, cy])

    frames = []
    for i in range(n):
        if i in odometry:
            t, c2w = odometry[i]
            q, tr = conventions.matrix_to_pose(c2w)
            frames.append(container.Frame(i=i, t=t, q_wxyz=tuple(q), tr=tuple(tr)))
        else:
            prev_t = frames[-1].t if frames else 0.0
            frames.append(container.Frame(i=i, t=prev_t, pose_valid=False))

    signals = {"depth": depth}
    meta = [
        container.SignalMeta(
            "depth", "depth", {"type": "linear", "scale": DEPTH_SCALE, "offset": 0.0, "invalid": 0}
        )
    ]
    if conf is not None:
        signals["confidence"] = conf
        meta.append(
            container.SignalMeta("confidence", "confidence", {"type": "labels", "labels": CONFIDENCE_LABELS})
        )

    fps = 30.0
    valid_t = [f.t for f in frames if f.pose_valid]
    if len(valid_t) > 1 and valid_t[-1] > valid_t[0]:
        fps = round((len(valid_t) - 1) / (valid_t[-1] - valid_t[0]), 3)

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
            "gravity_in_world": [0.0, -1.0, 0.0],  # ARKit world frame is +Y up
            "description": (
                f"Stray Scanner import from {seq_dir}; ARKit RUB poses converted to RDF; "
                f"resampled at {at} grid ({W}x{H}); depth is ARKit LiDAR (upsampled/ML-filtered "
                "by the OS, treat as approximate)"
            ),
        },
    )
