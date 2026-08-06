"""Converters between wurld and common posed-capture layouts.

Auto-detection (``detect(path)``):
- directory with ``rgb.txt`` + ``groundtruth.txt``            -> TUM RGB-D
- ``transforms.json`` file (or directory containing one)      -> nerfstudio/instant-ngp
- directory with ``odometry.csv`` + ``camera_matrix.csv``     -> Stray Scanner
- directory with ``cameras.bin|txt`` + ``images.bin|txt``
  (directly, or under ``sparse/0``)                           -> COLMAP model
"""

from __future__ import annotations

from pathlib import Path


def detect(path: str | Path) -> str | None:
    p = Path(path)
    if p.is_file() and p.name == "transforms.json":
        return "nerfstudio"
    if p.is_dir():
        if (p / "rgb.txt").exists() and (p / "groundtruth.txt").exists():
            return "tum"
        if (p / "transforms.json").exists():
            return "nerfstudio"
        if (p / "odometry.csv").exists() and (p / "camera_matrix.csv").exists():
            return "stray"
        for base in (p, p / "sparse" / "0", p / "sparse"):
            if any((base / f"cameras.{ext}").exists() for ext in ("bin", "txt")):
                return "colmap"
    return None
