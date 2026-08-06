import json

import numpy as np
import pytest
from PIL import Image

import wurld as wl
from wurld import conventions
from wurld.converters import detect, polycam


@pytest.fixture()
def polycam_fixture(scene, tmp_path):
    """Synthetic Polycam raw capture from the shared scene.

    RGB keyframes at 2x the depth grid, ARKit RUB c2w matrices as t_00..t_23,
    microsecond-timestamp filenames, corrected_* preferred over originals.
    """
    root = tmp_path / "capture"
    kf = root / "keyframes"
    for d in ("cameras", "corrected_cameras", "images", "corrected_images", "depth", "confidence"):
        (kf / d).mkdir(parents=True)
    (root / "mesh_info.json").write_text("{}")

    H, W = scene["rgb"].shape[1:3]
    RW, RH = W * 2, H * 2
    K = scene["cameras"]["0"].K

    for f in scene["frames"]:
        ts_us = int(round(f.t * 1e6))
        stem = f"{ts_us}"
        c2w_gl = conventions.c2w_cv_to_gl(f.c2w)
        cam = {
            "fx": K[0, 0] * 2, "fy": K[1, 1] * 2, "cx": K[0, 2] * 2, "cy": K[1, 2] * 2,
            "width": RW, "height": RH, "timestamp": ts_us, "blur_score": 1.0,
        }
        for r in range(3):
            for c in range(4):
                cam[f"t_{r}{c}"] = c2w_gl[r, c]
        # originals get poisoned poses so the test proves corrected_* is preferred
        bad = dict(cam)
        bad["t_03"] = cam["t_03"] + 100.0
        (kf / "corrected_cameras" / f"{stem}.json").write_text(json.dumps(cam))
        (kf / "cameras" / f"{stem}.json").write_text(json.dumps(bad))

        img = Image.fromarray(scene["rgb"][f.i]).resize((RW, RH), Image.NEAREST)
        img.save(kf / "corrected_images" / f"{stem}.jpg", quality=95)
        img.save(kf / "images" / f"{stem}.jpg", quality=30)

        depth_mm = np.where(np.isnan(scene["depth_m"][f.i]), 0,
                            np.round(scene["depth_m"][f.i] / 1e-3)).astype(np.uint16)
        Image.fromarray(depth_mm).save(kf / "depth" / f"{stem}.png")
        Image.fromarray(np.full((H, W), 255, np.uint8)).save(kf / "confidence" / f"{stem}.png")
    return root


def test_polycam_import(polycam_fixture, scene, tmp_path):
    assert detect(polycam_fixture) == "polycam"
    out = tmp_path / "pc.wl.webm"
    polycam.from_polycam(polycam_fixture, out)
    seq = wl.read(out)

    H, W = scene["rgb"].shape[1:3]
    assert (seq.probe["width"], seq.probe["height"]) == (W, H)  # depth-grid policy
    assert len(seq.frames) == 10
    # corrected cameras preferred: poses match, not the poisoned originals
    for i in (0, 5, 9):
        assert np.abs(seq.c2w(i) - scene["frames"][i].c2w).max() < 1e-6
        assert seq.frames[i].t == pytest.approx(scene["frames"][i].t, abs=1e-6)
    # intrinsics rescaled from the 2x keyframe resolution to the depth grid
    assert np.abs(seq.K("0") - scene["cameras"]["0"].K).max() < 1e-6
    # depth mm survive bit-for-bit (native linear units)
    dm = seq.depth_meters(0)
    ref = scene["depth_m"][0]
    both = ~np.isnan(dm) & ~np.isnan(ref)
    assert np.abs(dm[both] - ref[both]).max() < 1e-3 + 1e-9
    conf = seq.signal_meta("confidence")
    assert conf is not None and conf.value_map["labels"]["255"] == "high"
    assert np.all(seq.signal("confidence") == 255)
    assert seq.world["gravity_in_world"] == [0.0, -1.0, 0.0]
    assert "corrected" in seq.world["description"]


def test_polycam_original_cameras_opt_out(polycam_fixture, scene, tmp_path):
    out = tmp_path / "orig.wl.webm"
    polycam.from_polycam(polycam_fixture, out, corrected=False)
    seq = wl.read(out)
    # poisoned original poses: +100 on the x translation, in ARKit axes
    assert abs(seq.frames[0].tr[0] - scene["frames"][0].tr[0]) > 50


def test_polycam_at_rgb(polycam_fixture, scene, tmp_path):
    out = tmp_path / "rgb.wl.webm"
    polycam.from_polycam(polycam_fixture, out, at="rgb")
    seq = wl.read(out)
    H, W = scene["rgb"].shape[1:3]
    assert (seq.probe["width"], seq.probe["height"]) == (W * 2, H * 2)
    d = seq.signal("depth")[0]
    src = np.where(np.isnan(scene["depth_m"][0]), 0,
                   np.round(scene["depth_m"][0] / 1e-3)).astype(np.uint16)
    assert np.array_equal(d[::2, ::2], src)
