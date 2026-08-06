"""A nerfstudio DataParser that reads a wurld file directly.

Usage (nerfstudio installed):

    from wurld.integrations.nerfstudio_parser import WurldDataParserConfig
    config = WurldDataParserConfig(data=Path("scene.wl.webm"))
    outputs = config.setup()._generate_dataparser_outputs()

or register it as a plugin via the ``nerfstudio.dataparser_configs`` entry point.

On first parse the RGB (and 16-bit depth, when present) frames are extracted to a
``<file>.cache/`` directory beside the input — nerfstudio's pipeline wants image
files on disk. Poses convert from wurld's canonical RDF to nerfstudio's
OpenGL (RUB) convention; metric depth uses ``depth_unit_scale_factor``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Type

import numpy as np

try:
    import torch
    from nerfstudio.cameras.cameras import Cameras, CameraType
    from nerfstudio.data.dataparsers.base_dataparser import (
        DataParser,
        DataParserConfig,
        DataparserOutputs,
    )
    from nerfstudio.data.scene_box import SceneBox
except ImportError as e:  # pragma: no cover - exercised only without nerfstudio
    raise ImportError(
        "wurld.integrations.nerfstudio_parser needs nerfstudio installed"
    ) from e

from PIL import Image

from .. import container, conventions

DEPTH_UNIT = 1.0e-3  # cached depth PNGs are millimeters


def _extract_cache(seq: container.Sequence, cache: Path) -> tuple[list[Path], list[Path]]:
    """Write frames to disk once; reuse on subsequent parses."""
    images_dir, depth_dir = cache / "images", cache / "depth"
    depth_meta = seq.signal_meta("depth")
    have = sorted(images_dir.glob("*.png")) if images_dir.is_dir() else []
    if len(have) == len(seq.frames) and (depth_meta is None or (depth_dir / have[0].name).exists()):
        depths = sorted(depth_dir.glob("*.png")) if depth_meta else []
        return have, depths

    images_dir.mkdir(parents=True, exist_ok=True)
    rgb = seq.rgb
    depth_raw = seq.signal(depth_meta.id) if depth_meta else None
    if depth_raw is not None:
        depth_dir.mkdir(exist_ok=True)
    image_files, depth_files = [], []
    for f in seq.frames:
        name = f"frame_{f.i:06d}.png"
        Image.fromarray(np.asarray(rgb[f.i])[..., :3]).save(images_dir / name)
        image_files.append(images_dir / name)
        if depth_raw is not None:
            meters = depth_meta.apply(depth_raw[f.i])
            codes = np.round(np.nan_to_num(meters, nan=0.0) / DEPTH_UNIT)
            d16 = np.where((codes < 1) | (codes > 65535), 0, codes).astype(np.uint16)
            Image.fromarray(d16).save(depth_dir / name)
            depth_files.append(depth_dir / name)
    return image_files, depth_files


@dataclass
class WurldDataParserConfig(DataParserConfig):
    """Parse a wurld .wl.webm as a nerfstudio dataset."""

    _target: Type = field(default_factory=lambda: WurldDataParser)
    data: Path = Path("scene.wl.webm")
    scale_factor: float = 1.0
    """Scene scale applied to camera translations."""


class WurldDataParser(DataParser):
    config: WurldDataParserConfig

    def _generate_dataparser_outputs(self, split: str = "train") -> DataparserOutputs:
        seq = container.read(self.config.data)
        frames = [f for f in seq.frames if f.pose_valid]
        if not frames:
            raise ValueError(f"{self.config.data}: no posed frames")

        cache = self.config.data.with_suffix(".cache")
        image_files, depth_files = _extract_cache(seq, cache)
        image_files = [image_files[f.i] for f in frames]
        if depth_files:
            depth_files = [depth_files[f.i] for f in frames]

        c2w = np.stack([conventions.c2w_cv_to_gl(f.c2w) for f in frames])
        c2w[:, :3, 3] *= self.config.scale_factor

        cams, fx, fy, cx, cy = seq.cameras, [], [], [], []
        for f in frames:
            K = seq.K(f.camera, frame_index=f.i)
            fx.append(K[0, 0]); fy.append(K[1, 1]); cx.append(K[0, 2]); cy.append(K[1, 2])
        cam0 = cams[frames[0].camera]

        cameras = Cameras(
            camera_to_worlds=torch.from_numpy(c2w[:, :3, :4]).float(),
            fx=torch.tensor(fx).float(), fy=torch.tensor(fy).float(),
            cx=torch.tensor(cx).float(), cy=torch.tensor(cy).float(),
            width=cam0.width, height=cam0.height,
            camera_type=CameraType.PERSPECTIVE,
        )

        centers = c2w[:, :3, 3]
        span = float(np.abs(centers).max() * 1.2) or 1.0
        scene_box = SceneBox(aabb=torch.tensor(
            [[-span, -span, -span], [span, span, span]], dtype=torch.float32))

        metadata = {}
        if depth_files:
            metadata = {"depth_filenames": depth_files,
                        "depth_unit_scale_factor": DEPTH_UNIT * self.config.scale_factor}

        return DataparserOutputs(
            image_filenames=image_files,
            cameras=cameras,
            scene_box=scene_box,
            metadata=metadata,
        )
