import io
import json
import zipfile

import numpy as np
import pytest
from PIL import Image

import wurld as wl
from wurld import conventions
from wurld.converters import detect

liblzfse = pytest.importorskip("liblzfse")
from wurld.converters import record3d  # noqa: E402


@pytest.fixture()
def r3d_fixture(scene, tmp_path):
    """Synthetic .r3d zip: RGB at 2x depth grid, LZFSE float32 depth (NaN invalid),
    confidence=2, column-major K at RGB resolution, scalar-last ARKit poses."""
    H, W = scene["rgb"].shape[1:3]     # depth grid
    RW, RH = W * 2, H * 2              # rgb grid
    K = scene["cameras"]["0"].K
    fx, fy, cx, cy = K[0, 0] * 2, K[1, 1] * 2, K[0, 2] * 2, K[1, 2] * 2

    poses, timestamps = [], []
    for f in scene["frames"]:
        c2w_gl = conventions.c2w_cv_to_gl(f.c2w)
        q = conventions.matrix_to_quat_wxyz(c2w_gl[:3, :3])
        qx, qy, qz, qw = conventions.quat_wxyz_to_xyzw(q)
        t = c2w_gl[:3, 3]
        poses.append([qx, qy, qz, qw, t[0], t[1], t[2]])
        timestamps.append(f.t)

    meta = {
        "w": RW, "h": RH, "dw": W, "dh": H, "fps": 30, "cameraType": 1,
        # flat column-major: [fx, 0, 0, 0, fy, 0, cx, cy, 1]
        "K": [fx, 0, 0, 0, fy, 0, cx, cy, 1],
        "poses": poses, "initPose": poses[0], "frameTimestamps": timestamps,
    }

    p = tmp_path / "capture.r3d"
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("metadata", json.dumps(meta))
        z.writestr("sound.m4a", b"")
        for f in scene["frames"]:
            buf = io.BytesIO()
            Image.fromarray(scene["rgb"][f.i]).resize((RW, RH), Image.NEAREST).save(
                buf, format="JPEG", quality=95)
            z.writestr(f"rgbd/{f.i}.jpg", buf.getvalue())
            depth = scene["depth_m"][f.i].astype(np.float32)  # NaN where invalid
            z.writestr(f"rgbd/{f.i}.depth", liblzfse.compress(depth.tobytes()))
            z.writestr(f"rgbd/{f.i}.conf", liblzfse.compress(np.full((H, W), 2, np.uint8).tobytes()))
    return p


def test_record3d_import(r3d_fixture, scene, tmp_path):
    assert detect(r3d_fixture) == "record3d"
    out = tmp_path / "r3d.wurld.webm"
    record3d.from_record3d(r3d_fixture, out)
    seq = wl.read(out)

    H, W = scene["rgb"].shape[1:3]
    assert (seq.probe["width"], seq.probe["height"]) == (W, H)  # depth-grid policy
    assert len(seq.frames) == 10
    for i in (0, 5, 9):
        assert np.abs(seq.c2w(i) - scene["frames"][i].c2w).max() < 1e-6
        assert seq.frames[i].t == pytest.approx(scene["frames"][i].t, abs=1e-9)
    # K transposed from column-major and rescaled to the depth grid
    assert np.abs(seq.K("0") - scene["cameras"]["0"].K).max() < 1e-6
    # float meters -> inverse-depth codes -> meters within quantization error
    dm = seq.depth_meters(0)
    ref = scene["depth_m"][0]
    both = ~np.isnan(dm) & ~np.isnan(ref)
    assert both.sum() > 1000
    assert np.abs(dm[both] - ref[both]).max() < 0.05
    assert np.isnan(dm[np.isnan(ref)]).all()  # NaN invalids preserved as invalid
    conf_meta = seq.signal_meta("confidence")
    assert conf_meta is not None and conf_meta.value_map["labels"]["2"] == "high"
    assert np.all(seq.signal("confidence") == 2)
    assert seq.world["gravity_in_world"] == [0.0, -1.0, 0.0]
    vm = seq.signal_meta("depth").value_map
    assert vm["type"] == "inverse_depth" and 0 < vm["near"] < vm["far"]


def test_record3d_at_rgb(r3d_fixture, scene, tmp_path):
    out = tmp_path / "rgb.wurld.webm"
    record3d.from_record3d(r3d_fixture, out, at="rgb")
    seq = wl.read(out)
    H, W = scene["rgb"].shape[1:3]
    assert (seq.probe["width"], seq.probe["height"]) == (W * 2, H * 2)
    dm = seq.depth_meters(0)
    ref = scene["depth_m"][0]
    both = ~np.isnan(dm[::2, ::2]) & ~np.isnan(ref)
    assert np.abs(dm[::2, ::2][both] - ref[both]).max() < 0.05


def test_record3d_float16_depth_variant(r3d_fixture, scene, tmp_path):
    # rewrite one depth entry as float16 (the "compressed export" variant)
    src = zipfile.ZipFile(r3d_fixture)
    p = tmp_path / "f16.r3d"
    with zipfile.ZipFile(p, "w") as z:
        for name in src.namelist():
            data = src.read(name)
            if name.endswith(".depth"):
                d = np.frombuffer(liblzfse.decompress(data), np.float32)
                data = liblzfse.compress(d.astype(np.float16).tobytes())
            z.writestr(name, data)
    out = tmp_path / "f16.wurld.webm"
    record3d.from_record3d(p, out)
    seq = wl.read(out)
    dm = seq.depth_meters(0)
    ref = scene["depth_m"][0]
    both = ~np.isnan(dm) & ~np.isnan(ref)
    assert np.abs(dm[both] - ref[both]).max() < 0.1  # f16 + quantization


def test_device_uptime_timestamps_are_rebased(tmp_path, r3d_fixture, scene):
    """A real capture starts at ARKit's device uptime, not at zero.

    Record3D stores the raw clock, so a phone awake for four days produces a take
    whose first frame is at t≈330000 s — the shape of the first real WurldCam
    recording. SPEC §3 allows any epoch and such a file is valid, but nothing
    downstream wants one: it prints as nonsense, and a consumer treating t as an
    offset into the media has to work the origin out for itself. Only differences
    ever carry meaning, so the importer rebases to the first frame.

    The .r3d is left exactly as Record3D writes it; the rebase happens on the way
    into wurld.
    """
    import json
    import zipfile

    UPTIME = 330_973.252  # taken from an actual capture

    shifted = tmp_path / "uptime.r3d"
    with zipfile.ZipFile(r3d_fixture) as src, zipfile.ZipFile(shifted, "w") as dst:
        for name in src.namelist():
            data = src.read(name)
            if name == "metadata":
                meta = json.loads(data)
                meta["frameTimestamps"] = [t + UPTIME for t in meta["frameTimestamps"]]
                data = json.dumps(meta).encode()
            dst.writestr(name, data)

    out = tmp_path / "uptime.wurld.webm"
    record3d.from_record3d(shifted, out)
    seq = wl.read(out)

    assert seq.frames[0].t == pytest.approx(0.0, abs=1e-9), "the take starts at zero"
    # Every interval is untouched — rebasing must shift, never rescale.
    want = [f.t - scene["frames"][0].t for f in scene["frames"]]
    got = [f.t for f in seq.frames]
    assert got == pytest.approx(want, abs=1e-6)
    assert got == sorted(got), "still monotonic (SPEC §3)"
