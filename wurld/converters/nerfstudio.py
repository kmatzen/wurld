"""nerfstudio / instant-ngp ``transforms.json`` <-> wurld.

transforms.json uses OpenGL/Blender camera axes (RUB) with camera-to-world
matrices; wurld is canonical RDF. Depth sidecars (``depth_file_path``,
16-bit PNG millimeters per the nerfstudio convention) become a lossless
wurld depth signal with a linear value map.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from PIL import Image

from . import require_8bit_pixels
from .. import container, conventions

DEPTH_UNIT = 1.0e-3  # nerfstudio depth_unit_scale_factor default: mm -> m


def from_transforms(json_path: str | Path, out_path: str | Path, fps: float = 30.0, rgb_kbps: int = 4000) -> Path:
    json_path = Path(json_path)
    if json_path.is_dir():
        json_path = json_path / "transforms.json"
    root = json_path.parent
    doc = json.loads(json_path.read_text())

    def intrinsic(frame, key):
        return frame.get(key, doc.get(key))

    frames_json = sorted(doc["frames"], key=lambda f: f["file_path"])
    rgb, depth, frames, cameras = [], [], [], {}
    for i, fj in enumerate(frames_json):
        img_path = root / fj["file_path"]
        if not img_path.suffix:
            img_path = img_path.with_suffix(".png")  # instant-ngp style paths
        rgb.append(np.asarray(Image.open(img_path).convert("RGBA")))

        w = int(intrinsic(fj, "w") or rgb[-1].shape[1])
        h = int(intrinsic(fj, "h") or rgb[-1].shape[0])
        fl_x = intrinsic(fj, "fl_x")
        fl_y = intrinsic(fj, "fl_y") or fl_x
        cx = intrinsic(fj, "cx") or w / 2
        cy = intrinsic(fj, "cy") or h / 2
        k1, k2 = intrinsic(fj, "k1") or 0.0, intrinsic(fj, "k2") or 0.0
        p1, p2 = intrinsic(fj, "p1") or 0.0, intrinsic(fj, "p2") or 0.0
        if any((k1, k2, p1, p2)):
            cam = container.Camera("OPENCV", w, h, [fl_x, fl_y, cx, cy, k1, k2, p1, p2])
        else:
            cam = container.Camera("PINHOLE", w, h, [fl_x, fl_y, cx, cy])
        cam_key = None
        for key, existing in cameras.items():
            if existing.to_json() == cam.to_json():
                cam_key = key
                break
        if cam_key is None:
            cam_key = str(len(cameras))
            cameras[cam_key] = cam

        c2w_gl = np.asarray(fj["transform_matrix"], dtype=np.float64)
        q, tr = conventions.matrix_to_pose(conventions.c2w_gl_to_cv(c2w_gl))
        frames.append(container.Frame(i=i, t=i / fps, camera=cam_key, q_wxyz=tuple(q), tr=tuple(tr)))

        if "depth_file_path" in fj:
            d = np.asarray(Image.open(root / fj["depth_file_path"]))
            if d.dtype != np.uint16:
                raise ValueError(f"{fj['depth_file_path']}: expected 16-bit PNG depth")
            depth.append(d)

    signals = specs = None
    meta = []
    if depth:
        if len(depth) != len(rgb):
            raise ValueError("some frames have depth_file_path and some do not")
        signals = {"depth": np.stack(depth)}
        meta = [container.SignalMeta("depth", "depth", {"type": "linear", "scale": DEPTH_UNIT, "offset": 0.0, "invalid": 0})]

    return container.write(
        out_path,
        cameras=cameras,
        frames=frames,
        rgb=np.stack(rgb),
        signals=signals,
        specs=specs,
        signal_meta=meta,
        fps=fps,
        rgb_kbps=rgb_kbps,
        world={
            "metric_scale": bool(depth),
            "gravity_in_world": None,
            "description": f"imported from {json_path}; timestamps synthesized at {fps} fps",
        },
    )


def to_transforms(wl_path: str | Path, out_dir: str | Path) -> Path:
    """Write images/ (+ depth/) and a transforms.json in nerfstudio convention."""
    seq = container.read(wl_path)
    out = Path(out_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)
    if len(seq.rgb_streams) > 1:
        logging.getLogger(__name__).warning(
            "%s carries %d display streams (%s); this format holds one camera's "
            "images, so only the primary (%s) is exported",
            wl_path, len(seq.rgb_streams), ", ".join(seq.rgb_streams),
            seq.rgb_streams[0])
    require_8bit_pixels(seq, 'a nerfstudio transforms.json export')
    rgb = seq.rgb
    if rgb is None:
        raise ValueError(
            f"{wl_path} has no display track, and transforms.json indexes images. "
            "A signals-only file (depth or scene-referred HDR with rgb=None) has "
            "nothing to export here.")
    depth_meta = seq.signal_meta("depth")
    depth_raw = seq.signal(depth_meta.id) if depth_meta else None
    if depth_raw is not None:
        (out / "depth").mkdir(exist_ok=True)

    if len(seq.cameras) != 1:
        raise ValueError("to_transforms supports single-camera sequences (v0.1)")
    cam = next(iter(seq.cameras.values()))
    K = cam.K
    doc = {
        "camera_model": "OPENCV" if cam.model == "OPENCV" else "PINHOLE",
        "fl_x": K[0, 0], "fl_y": K[1, 1], "cx": K[0, 2], "cy": K[1, 2],
        "w": cam.width, "h": cam.height,
        "frames": [],
    }
    if cam.model == "OPENCV":
        doc["k1"], doc["k2"], doc["p1"], doc["p2"] = cam.params[4:8]

    skipped = 0
    for f in seq.frames:
        if not f.pose_valid:
            # transforms.json is a list of posed views: a frame the producer
            # could not localise is not a member of it. Dropping it is right;
            # crashing on it stopped feed-forward output from reaching a trainer
            # at all, which is the one pipeline this export exists for.
            skipped += 1
            continue
        name = f"images/frame_{f.i:06d}.png"
        Image.fromarray(np.asarray(rgb[f.i])[..., :3]).save(out / name)
        entry = {
            "file_path": name,
            "transform_matrix": conventions.c2w_cv_to_gl(f.c2w).tolist(),
        }
        if depth_raw is not None:
            vm = depth_meta.value_map
            if vm.get("type") == "linear" and abs(vm.get("scale", 0) - DEPTH_UNIT) < 1e-12:
                d16 = depth_raw[f.i]  # already millimeters: keep bit-exact
            else:
                meters = depth_meta.apply(depth_raw[f.i])
                codes = np.round(np.nan_to_num(meters, nan=0.0) / DEPTH_UNIT)
                # out of uint16 range -> 0 (invalid), never wrap
                d16 = np.where((codes < 1) | (codes > 65535), 0, codes).astype(np.uint16)
            dname = f"depth/frame_{f.i:06d}.png"
            Image.fromarray(d16.astype(np.uint16)).save(out / dname)
            entry["depth_file_path"] = dname
        doc["frames"].append(entry)

    (out / "transforms.json").write_text(json.dumps(doc, indent=2))
    if skipped:
        logging.getLogger(__name__).warning(
            "%s: %d frame(s) had no pose and are absent from transforms.json",
            wl_path, skipped)
    return out
