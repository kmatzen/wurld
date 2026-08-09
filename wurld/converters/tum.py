"""TUM RGB-D <-> wurld.

TUM sequences: ``rgb.txt``/``depth.txt`` (timestamp -> file), ``groundtruth.txt``
(timestamp tx ty tz qx qy qz qw, camera-to-world, meters), depth as 16-bit PNG
with 5000 units per meter. RGB/depth/pose streams tick at different rates and
are associated by nearest timestamp (the associate.py ritual, built in here).

TUM distributes no per-sequence intrinsics files; the standard published values
are selected by freiburg number in the path, or pass ``camera=`` explicitly.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PIL import Image

from . import require_8bit_pixels
from .. import container, conventions

DEPTH_SCALE = 1.0 / 5000.0  # meters per raw unit

# Published camera parameters for the TUM RGB-D benchmark.
INTRINSICS = {
    "freiburg1": container.Camera("PINHOLE", 640, 480, [517.3, 516.5, 318.6, 255.3]),
    "freiburg2": container.Camera("PINHOLE", 640, 480, [520.9, 521.0, 325.1, 249.7]),
    "freiburg3": container.Camera("PINHOLE", 640, 480, [535.4, 539.2, 320.1, 247.6]),
    "default": container.Camera("PINHOLE", 640, 480, [525.0, 525.0, 319.5, 239.5]),
}


def _read_list(path: Path) -> list[tuple[float, list[str]]]:
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        rows.append((float(parts[0]), parts[1:]))
    return rows


def _associate(a: list, b: list, max_dt: float) -> list[tuple[int, int]]:
    """Greedy nearest-timestamp matching (same policy as TUM associate.py)."""
    candidates = sorted(
        ((abs(ta - tb), ia, ib) for ia, (ta, _) in enumerate(a) for ib, (tb, _) in enumerate(b) if abs(ta - tb) < max_dt)
    )
    used_a, used_b, pairs = set(), set(), []
    for _, ia, ib in candidates:
        if ia not in used_a and ib not in used_b:
            used_a.add(ia)
            used_b.add(ib)
            pairs.append((ia, ib))
    return sorted(pairs)


def _guess_camera(path: Path) -> container.Camera:
    name = str(path).lower()
    for key, cam in INTRINSICS.items():
        if key in name:
            return cam
    return INTRINSICS["default"]


def from_tum(
    seq_dir: str | Path,
    out_path: str | Path,
    camera: container.Camera | None = None,
    max_dt: float = 0.02,
    rgb_kbps: int = 4000,
) -> Path:
    seq_dir = Path(seq_dir)
    rgb_list = _read_list(seq_dir / "rgb.txt")
    depth_list = _read_list(seq_dir / "depth.txt") if (seq_dir / "depth.txt").exists() else []
    gt_list = _read_list(seq_dir / "groundtruth.txt")

    rgb_depth = _associate(rgb_list, depth_list, max_dt) if depth_list else [(i, None) for i in range(len(rgb_list))]
    gt_times = np.array([t for t, _ in gt_list])

    rgb, depth, frames = [], [], []
    for out_i, (ri, di) in enumerate(rgb_depth):
        t_rgb, (rgb_file,) = rgb_list[ri]
        rgb.append(np.asarray(Image.open(seq_dir / rgb_file).convert("RGBA")))
        if di is not None:
            _, (depth_file,) = depth_list[di]
            d = np.asarray(Image.open(seq_dir / depth_file))
            if d.dtype != np.uint16:
                raise ValueError(f"{depth_file}: expected 16-bit PNG")
            depth.append(d)

        gi = int(np.argmin(np.abs(gt_times - t_rgb)))
        pose_valid = abs(gt_times[gi] - t_rgb) < max_dt
        if pose_valid:
            vals = [float(v) for v in gt_list[gi][1]]
            tx, ty, tz, qx, qy, qz, qw = vals
            q = conventions.quat_xyzw_to_wxyz([qx, qy, qz, qw])
            frames.append(container.Frame(i=out_i, t=t_rgb, q_wxyz=tuple(q), tr=(tx, ty, tz)))
        else:
            frames.append(container.Frame(i=out_i, t=t_rgb, pose_valid=False))

    cam = camera or _guess_camera(seq_dir)
    h, w = rgb[0].shape[:2]
    if (cam.width, cam.height) != (w, h):
        # TUM layouts carry no intrinsics; scale the published/default values
        # to the actual image size rather than failing.
        sx, sy = w / cam.width, h / cam.height
        fx, fy, cx, cy = cam.params
        cam = container.Camera("PINHOLE", w, h, [fx * sx, fy * sy, cx * sx, cy * sy])
    signals = {"depth": np.stack(depth)} if depth else None
    meta = (
        [container.SignalMeta("depth", "depth", {"type": "linear", "scale": DEPTH_SCALE, "offset": 0.0, "invalid": 0})]
        if depth
        else []
    )
    fps = 30.0
    if len(frames) > 1:
        dt = (frames[-1].t - frames[0].t) / (len(frames) - 1)
        if dt > 0:
            fps = round(1.0 / dt, 3)

    return container.write(
        out_path,
        cameras={"0": cam},
        frames=frames,
        rgb=np.stack(rgb),
        signals=signals,
        signal_meta=meta,
        fps=fps,
        rgb_kbps=rgb_kbps,
        world={
            "metric_scale": True,
            "gravity_in_world": None,
            "description": f"TUM RGB-D import from {seq_dir}",
        },
    )


def to_tum(wl_path: str | Path, out_dir: str | Path) -> Path:
    """Write rgb/, depth/, rgb.txt, depth.txt, groundtruth.txt."""
    seq = container.read(wl_path)
    out = Path(out_dir)
    (out / "rgb").mkdir(parents=True, exist_ok=True)
    if len(seq.rgb_streams) > 1:
        logging.getLogger(__name__).warning(
            "%s carries %d display streams (%s); this format holds one camera's "
            "images, so only the primary (%s) is exported",
            wl_path, len(seq.rgb_streams), ", ".join(seq.rgb_streams),
            seq.rgb_streams[0])
    require_8bit_pixels(seq, 'TUM RGB-D')
    rgb = seq.rgb
    depth_meta = seq.signal_meta("depth")
    depth_raw = seq.signal(depth_meta.id) if depth_meta else None
    if depth_raw is not None:
        (out / "depth").mkdir(exist_ok=True)

    rgb_lines, depth_lines, gt_lines = ["# timestamp filename"], ["# timestamp filename"], [
        "# timestamp tx ty tz qx qy qz qw"
    ]
    for f in seq.frames:
        ts = f"{f.t:.6f}"
        name = f"rgb/{ts}.png"
        Image.fromarray(np.asarray(rgb[f.i])[..., :3]).save(out / name)
        rgb_lines.append(f"{ts} {name}")
        if depth_raw is not None:
            vm = depth_meta.value_map
            if vm.get("type") == "linear" and abs(vm.get("scale", 0) - DEPTH_SCALE) < 1e-12:
                d16 = depth_raw[f.i]  # native TUM units: bit-exact round trip
            else:
                meters = depth_meta.apply(depth_raw[f.i])
                codes = np.round(np.nan_to_num(meters, nan=0.0) / DEPTH_SCALE)
                # out of uint16 range -> 0 (TUM "no data"), never wrap
                d16 = np.where((codes < 1) | (codes > 65535), 0, codes).astype(np.uint16)
            dname = f"depth/{ts}.png"
            Image.fromarray(d16.astype(np.uint16)).save(out / dname)
            depth_lines.append(f"{ts} {dname}")
        if f.pose_valid:
            q = conventions.quat_wxyz_to_xyzw(f.q_wxyz)
            tr = f.tr
            gt_lines.append(
                f"{ts} " + " ".join(f"{v:.6f}" for v in (*tr, *q))
            )

    (out / "rgb.txt").write_text("\n".join(rgb_lines) + "\n")
    if depth_raw is not None:
        (out / "depth.txt").write_text("\n".join(depth_lines) + "\n")
    (out / "groundtruth.txt").write_text("\n".join(gt_lines) + "\n")
    return out
