"""Converters between wurld and common posed-capture layouts.

Auto-detection (``detect(path)``):
- directory with ``rgb.txt`` + ``groundtruth.txt``            -> TUM RGB-D
- ``transforms.json`` file (or directory containing one)      -> nerfstudio/instant-ngp
- ``*.r3d`` file                                              -> Record3D
- directory with ``odometry.csv`` + ``camera_matrix.csv``     -> Stray Scanner
- directory with ``[mav0/]cam0/sensor.yaml``                  -> EuRoC MAV
- directory with ``keyframes/cameras`` (or ``corrected_cameras``) -> Polycam raw
- directory with ``cameras.bin|txt`` + ``images.bin|txt``
  (directly, or under ``sparse/0``)                           -> COLMAP model
"""

from __future__ import annotations

from pathlib import Path


def detect(path: str | Path) -> str | None:
    p = Path(path)
    if p.is_file() and p.name == "transforms.json":
        return "nerfstudio"
    if p.is_file() and p.suffix.lower() == ".r3d":
        return "record3d"
    if p.is_dir():
        if (p / "rgb.txt").exists() and (p / "groundtruth.txt").exists():
            return "tum"
        if (p / "transforms.json").exists():
            return "nerfstudio"
        if (p / "odometry.csv").exists() and (p / "camera_matrix.csv").exists():
            return "stray"
        if (p / "keyframes" / "cameras").is_dir() or (p / "keyframes" / "corrected_cameras").is_dir():
            return "polycam"
        if (p / "cam0" / "sensor.yaml").exists() or (p / "mav0" / "cam0" / "sensor.yaml").exists():
            return "euroc"
        for base in (p, p / "sparse" / "0", p / "sparse"):
            if any((base / f"cameras.{ext}").exists() for ext in ("bin", "txt")):
                return "colmap"
    return None


def require_8bit_pixels(seq, fmt: str) -> None:
    """Refuse an HDR display track for a format that stores 8-bit images.

    PIL's own complaint — "Cannot handle this data type: (1, 1, 3), <u2" — names
    neither the file, the format, nor the way out. An HDR track decodes to
    uint16 (10-bit PQ codes), and PNG/JPEG here hold 8 bits, so the export
    cannot proceed; saying which of those facts collided is the least this can
    do.
    """
    if seq.hdr is None:
        return
    raise ValueError(
        f"{seq.path} carries an HDR display track ({seq.hdr.get('transfer', '?')}, "
        f"{seq.hdr.get('bits', 10)}-bit), and {fmt} stores 8-bit images. Tone-map "
        "the display track first, or export without images if the format allows "
        "it — wurld will not silently crush 10-bit display-referred codes to 8 bits."
    )
