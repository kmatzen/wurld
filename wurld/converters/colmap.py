"""COLMAP sparse model <-> wurld.

COLMAP stores world-to-camera poses (qvec wxyz, tvec) in RDF camera axes and
has no timestamps, no depth, and arbitrary scale. Import synthesizes uniform
timestamps at the requested fps (image name order) and marks the world
non-metric; export writes a text model readable by COLMAP and downstream tools.
"""

from __future__ import annotations

import struct
import logging
from pathlib import Path

import numpy as np
from PIL import Image

from .. import container, conventions

# COLMAP model_id -> (name, n_params); subset wurld supports.
_MODELS = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
}
_MODEL_IDS = {name: mid for mid, (name, _) in _MODELS.items()}


def _model_dir(path: Path) -> Path:
    for base in (path, path / "sparse" / "0", path / "sparse"):
        if any((base / f"cameras.{ext}").exists() for ext in ("bin", "txt")):
            return base
    raise FileNotFoundError(f"no COLMAP model under {path}")


def read_cameras(base: Path) -> dict[str, container.Camera]:
    if (base / "cameras.bin").exists():
        return _read_cameras_bin(base / "cameras.bin")
    return _read_cameras_txt(base / "cameras.txt")


def read_images(base: Path) -> list[dict]:
    if (base / "images.bin").exists():
        return _read_images_bin(base / "images.bin")
    return _read_images_txt(base / "images.txt")


def _read_cameras_bin(path: Path) -> dict[str, container.Camera]:
    cameras = {}
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        for _ in range(n):
            cam_id, model_id, width, height = struct.unpack("<iiQQ", f.read(24))
            if model_id not in _MODELS:
                raise ValueError(f"unsupported COLMAP camera model id {model_id}")
            name, n_params = _MODELS[model_id]
            params = list(struct.unpack(f"<{n_params}d", f.read(8 * n_params)))
            cameras[str(cam_id)] = container.Camera(name, int(width), int(height), params)
    return cameras


def _read_cameras_txt(path: Path) -> dict[str, container.Camera]:
    cameras = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        cam_id, model, width, height = parts[0], parts[1], int(parts[2]), int(parts[3])
        cameras[cam_id] = container.Camera(model, width, height, [float(v) for v in parts[4:]])
    return cameras


def _read_images_bin(path: Path) -> list[dict]:
    images = []
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        for _ in range(n):
            image_id = struct.unpack("<i", f.read(4))[0]
            qvec = struct.unpack("<4d", f.read(32))  # w2c rotation, wxyz
            tvec = struct.unpack("<3d", f.read(24))  # w2c translation
            cam_id = struct.unpack("<i", f.read(4))[0]
            name = b""
            while (c := f.read(1)) != b"\x00":
                name += c
            (n_pts,) = struct.unpack("<Q", f.read(8))
            f.seek(24 * n_pts, 1)  # skip 2D points (x, y, point3D_id)
            images.append(
                {"id": image_id, "qvec": qvec, "tvec": tvec, "camera": str(cam_id), "name": name.decode()}
            )
    return images


def _read_images_txt(path: Path) -> list[dict]:
    images = []
    # Each image occupies two lines (pose header, then the 2D point list, which
    # may be blank) — keep blank lines so the pairing stays aligned.
    lines = [l.strip() for l in path.read_text().splitlines() if not l.strip().startswith("#")]
    for header in lines[0::2]:
        if not header:
            continue
        p = header.split()
        images.append(
            {
                "id": int(p[0]),
                "qvec": tuple(float(v) for v in p[1:5]),
                "tvec": tuple(float(v) for v in p[5:8]),
                "camera": p[8],
                "name": p[9],
            }
        )
    return images


def w2c_to_frame(i: int, t: float, camera: str, qvec, tvec) -> container.Frame:
    w2c = conventions.pose_to_matrix(qvec, tvec)
    q, tr = conventions.matrix_to_pose(conventions.invert_pose(w2c))
    return container.Frame(i=i, t=t, camera=camera, q_wxyz=tuple(q), tr=tuple(tr))


def from_colmap(
    model_path: str | Path,
    images_dir: str | Path,
    out_path: str | Path,
    fps: float = 30.0,
    rgb_kbps: int = 4000,
) -> Path:
    base = _model_dir(Path(model_path))
    cameras = read_cameras(base)
    images = sorted(read_images(base), key=lambda im: im["name"])
    if not images:
        raise ValueError(f"no registered images in {base}")

    frames, rgb = [], []
    for i, im in enumerate(images):
        arr = np.asarray(Image.open(Path(images_dir) / im["name"]).convert("RGBA"))
        rgb.append(arr)
        frames.append(w2c_to_frame(i, i / fps, im["camera"], im["qvec"], im["tvec"]))
    shapes = {a.shape for a in rgb}
    if len(shapes) != 1:
        raise ValueError(f"images have mixed resolutions {shapes}; v0.1 requires uniform size")

    return container.write(
        out_path,
        cameras=cameras,
        frames=frames,
        rgb=np.stack(rgb),
        fps=fps,
        rgb_kbps=rgb_kbps,
        world={
            "metric_scale": False,
            "gravity_in_world": None,
            "description": f"COLMAP model {base}; timestamps synthesized at {fps} fps in image-name order",
        },
    )


def to_colmap(wl_path: str | Path, out_dir: str | Path, write_images: bool = True) -> Path:
    """Write a COLMAP text model (cameras.txt, images.txt, points3D.txt)."""
    seq = container.read(wl_path)
    out = Path(out_dir)
    (out / "sparse" / "0").mkdir(parents=True, exist_ok=True)
    base = out / "sparse" / "0"

    cam_ids = {}
    lines = ["# Camera list: CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]"]
    for idx, (key, cam) in enumerate(seq.cameras.items(), start=1):
        cam_ids[key] = idx
        if cam.model not in _MODEL_IDS:
            raise ValueError(f"camera model {cam.model} has no COLMAP equivalent")
        params = " ".join(repr(float(v)) for v in cam.params)
        lines.append(f"{idx} {cam.model} {cam.width} {cam.height} {params}")
    (base / "cameras.txt").write_text("\n".join(lines) + "\n")

    lines = [
        "# Image list: IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME",
        "#   (wurld export: poses converted camera-to-world -> world-to-camera)",
    ]
    img_dir = out / "images"
    if write_images:
        img_dir.mkdir(exist_ok=True)
        if len(seq.rgb_streams) > 1:
            logging.getLogger(__name__).warning(
                "%s carries %d display streams (%s); this format holds one camera's "
                "images, so only the primary (%s) is exported",
                wl_path, len(seq.rgb_streams), ", ".join(seq.rgb_streams),
                seq.rgb_streams[0])
        rgb = seq.rgb
    for n, f in enumerate(seq.frames, start=1):
        name = f"frame_{f.i:06d}.png"
        w2c = conventions.invert_pose(f.c2w)
        q, t = conventions.matrix_to_pose(w2c)
        qs = " ".join(repr(float(v)) for v in q)
        ts = " ".join(repr(float(v)) for v in t)
        lines.append(f"{n} {qs} {ts} {cam_ids[f.camera]} {name}")
        lines.append("")  # empty 2D-point list
        if write_images:
            Image.fromarray(np.asarray(rgb[f.i])[..., :3]).save(img_dir / name)
    (base / "images.txt").write_text("\n".join(lines) + "\n")
    (base / "points3D.txt").write_text("# empty\n")
    return out
