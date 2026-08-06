import sys
from pathlib import Path

import chromapakz as cz
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import wurld as wl
from wurld.synthetic import make_sequence

NEAR, FAR = 0.5, 40.0


@pytest.fixture(scope="session")
def scene():
    rgb, depth_m, cameras, frames = make_sequence(n_frames=10, width=128, height=96)
    z = np.where(depth_m > 0, np.clip(depth_m, NEAR, FAR), np.nan)
    d16 = cz.quantize_inverse(z, near=NEAR, far=FAR)
    rgba = np.concatenate([rgb, np.full(rgb.shape[:3] + (1,), 255, np.uint8)], -1)
    return {"rgb": rgb, "rgba": rgba, "depth_m": z, "d16": d16, "cameras": cameras, "frames": frames}


@pytest.fixture(scope="session")
def wl_file(scene, tmp_path_factory):
    path = tmp_path_factory.mktemp("wl") / "scene.wl.webm"
    wl.write(
        path,
        cameras=scene["cameras"],
        frames=scene["frames"],
        rgb=scene["rgba"],
        signals={"depth": scene["d16"]},
        specs={"depth": cz.inverse_depth_spec(NEAR, FAR)},
        signal_meta=[
            wl.SignalMeta("depth", "depth", {"type": "inverse_depth", "near": NEAR, "far": FAR, "levels": 65536, "invalid": 0})
        ],
        world={"metric_scale": True, "gravity_in_world": [0, 0, -1], "description": "test scene"},
    )
    return path
