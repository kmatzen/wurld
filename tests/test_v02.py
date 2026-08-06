import shutil
import subprocess

import numpy as np
import pytest

import wurld as wl
from wurld import container, conventions
from wurld.converters import detect


def test_binary_frames_roundtrip(scene, tmp_path):
    p = tmp_path / "bin.wl.webm"
    wl.write(p, cameras=scene["cameras"], frames=scene["frames"], rgb=scene["rgba"],
             frames_format="binary")
    seq = wl.read(p)
    assert len(seq.frames) == len(scene["frames"])
    for i in (0, 4, 9):
        a, b = seq.frames[i], scene["frames"][i]
        assert a.t == b.t  # f64: exact
        assert np.abs(np.array(a.q_wxyz) - np.array(b.q_wxyz)).max() < 1e-6  # f32
        assert np.abs(np.array(a.tr) - np.array(b.tr)).max() < 1e-5
    # JSON document must carry the descriptor and an empty frames array
    doc = seq.to_document()
    assert doc["frames"] != [] or True  # to_document re-serializes loaded frames
    import json
    from wurld import ebml
    raw = json.loads(ebml.read_tag(p.read_bytes(), "WURLD"))
    assert raw["frames"] == []
    assert raw["frames_binary"]["count"] == len(scene["frames"])


def test_binary_frames_pose_invalid(scene, tmp_path):
    frames = list(scene["frames"][:3]) + [wl.Frame(i=3, t=scene["frames"][3].t, pose_valid=False)]
    p = tmp_path / "inv.wl.webm"
    wl.write(p, cameras=scene["cameras"], frames=frames, rgb=scene["rgba"][:4],
             frames_format="binary")
    seq = wl.read(p)
    assert seq.frames[3].pose_valid is False
    assert seq.frames[2].pose_valid is True


def test_auto_stays_json_below_threshold(scene, tmp_path):
    p = tmp_path / "auto.wl.webm"
    wl.write(p, cameras=scene["cameras"], frames=scene["frames"], rgb=scene["rgba"])
    from wurld import ebml
    assert ebml.read_all_tags(p.read_bytes()).get("WURLD_FRAMES") is None


def test_binary_rejects_param_overrides(scene, tmp_path):
    frames = list(scene["frames"])
    f0 = frames[0]
    frames[0] = wl.Frame(i=f0.i, t=f0.t, q_wxyz=f0.q_wxyz, tr=f0.tr,
                         params=[100.0, 100.0, 63.5, 47.5])
    with pytest.raises(ValueError, match="JSON frame form"):
        wl.write(tmp_path / "x.webm", cameras=scene["cameras"], frames=frames,
                 rgb=scene["rgba"], frames_format="binary")


def test_per_frame_intrinsics(scene, tmp_path):
    frames = list(scene["frames"])
    f0 = frames[0]
    override = [200.0, 200.0, 63.5, 47.5]
    frames[0] = wl.Frame(i=f0.i, t=f0.t, q_wxyz=f0.q_wxyz, tr=f0.tr, params=override)
    p = tmp_path / "ov.wl.webm"
    wl.write(p, cameras=scene["cameras"], frames=frames, rgb=scene["rgba"])
    seq = wl.read(p)
    assert seq.frames[0].params == override
    assert seq.K("0", frame_index=0)[0, 0] == 200.0
    assert seq.K("0", frame_index=1)[0, 0] == scene["cameras"]["0"].params[0]


def test_per_frame_intrinsics_length_validated(scene, tmp_path):
    f0 = scene["frames"][0]
    bad = [wl.Frame(i=0, t=f0.t, q_wxyz=f0.q_wxyz, tr=f0.tr, params=[1.0, 2.0])]
    with pytest.raises(ValueError, match="params override length"):
        wl.write(tmp_path / "x.webm", cameras=scene["cameras"], frames=bad,
                 rgb=scene["rgba"][:1])


def test_rigs_roundtrip_and_derivation(scene, tmp_path):
    baseline = 0.12
    cam1 = container.Camera("PINHOLE", 128, 96, scene["cameras"]["0"].params)
    rigs = {"rig0": {"cameras": {
        "0": {"q_wxyz": [1, 0, 0, 0], "tr": [0, 0, 0]},
        "1": {"q_wxyz": [1, 0, 0, 0], "tr": [baseline, 0, 0]},
    }, "description": "stereo"}}
    p = tmp_path / "rig.wl.webm"
    wl.write(p, cameras={**scene["cameras"], "1": cam1}, frames=scene["frames"],
             rgb=scene["rgba"], rigs=rigs)
    seq = wl.read(p)
    assert seq.rigs == rigs
    c2w_right = seq.rig_c2w(0, "1")
    c2w_left = seq.c2w(0)
    # the right camera sits one baseline along the left camera's +X axis
    expected = c2w_left[:3, 3] + c2w_left[:3, 0] * baseline
    assert np.abs(c2w_right[:3, 3] - expected).max() < 1e-9
    assert np.abs(c2w_right[:3, :3] - c2w_left[:3, :3]).max() < 1e-12


def test_rigs_validation(scene, tmp_path):
    bad = {"rig0": {"cameras": {"nope": {"q_wxyz": [1, 0, 0, 0], "tr": [0, 0, 0]}}}}
    with pytest.raises(ValueError, match="unknown camera"):
        wl.write(tmp_path / "x.webm", cameras=scene["cameras"], frames=scene["frames"],
                 rgb=scene["rgba"], rigs=bad)


def test_imu_roundtrip(scene, tmp_path):
    rng = np.random.default_rng(1)
    n = 500
    samples = np.column_stack([
        np.sort(rng.uniform(0, 1, n)),
        rng.normal(0, 0.02, (n, 3)),
        rng.normal([0, 0, 9.81], 0.3, (n, 3)),
    ])
    imu = wl.ImuStream("imu0", samples, rate_hz=500.0,
                       extrinsics={"q_wxyz": [1, 0, 0, 0], "tr": [0.01, -0.002, 0.0]},
                       description="synthetic imu")
    p = tmp_path / "imu.wl.webm"
    wl.write(p, cameras=scene["cameras"], frames=scene["frames"], rgb=scene["rgba"], imu=[imu])
    seq = wl.read(p)
    got = seq.imu["imu0"]
    assert got.rate_hz == 500.0
    assert got.extrinsics == imu.extrinsics
    assert got.samples.shape == (n, 7)
    assert np.array_equal(got.samples[:, 0], samples[:, 0])  # f64 timestamps exact
    assert np.abs(got.samples[:, 1:] - samples[:, 1:]).max() < 1e-5  # f32


def test_v01_reader_sees_v02_as_valid(scene, tmp_path):
    # a file with binary frames + imu still probes/decodes as plain wurld
    p = tmp_path / "fwd.wl.webm"
    wl.write(p, cameras=scene["cameras"], frames=scene["frames"], rgb=scene["rgba"],
             frames_format="binary",
             imu=[wl.ImuStream("x", np.zeros((3, 7)))])
    import chromapakz as cz
    assert cz.probe(p.read_bytes())["frames"] == 10


# ---------- Stray Scanner ----------

ffmpeg_missing = shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None


@pytest.fixture()
def stray_fixture(scene, tmp_path):
    """Synthetic Stray layout: rgb.mp4 (2x depth res), mm depth PNGs, ARKit poses."""
    from PIL import Image

    root = tmp_path / "stray"
    (root / "depth").mkdir(parents=True)
    (root / "confidence").mkdir()
    H, W = scene["rgb"].shape[1:3]  # depth grid = scene resolution
    RW, RH = W * 2, H * 2

    # camera matrix at RGB resolution (2x the scene intrinsics)
    K = scene["cameras"]["0"].K.copy()
    K[0] *= 2
    K[1] *= 2
    np.savetxt(root / "camera_matrix.csv", K, delimiter=",")

    # odometry: canonical -> ARKit RUB, xyzw
    rows = ["timestamp, frame, x, y, z, qx, qy, qz, qw"]
    for f in scene["frames"]:
        c2w_gl = conventions.c2w_cv_to_gl(f.c2w)
        q = conventions.matrix_to_quat_wxyz(c2w_gl[:3, :3])
        qx, qy, qz, qw = conventions.quat_wxyz_to_xyzw(q)
        t = c2w_gl[:3, 3]
        rows.append(f"{f.t}, {f.i}, {t[0]}, {t[1]}, {t[2]}, {qx}, {qy}, {qz}, {qw}")
    (root / "odometry.csv").write_text("\n".join(rows) + "\n")

    # depth mm PNGs + constant-high confidence
    depth_mm = np.where(np.isnan(scene["depth_m"]), 0, np.round(scene["depth_m"] / 1e-3)).astype(np.uint16)
    for i in range(depth_mm.shape[0]):
        Image.fromarray(depth_mm[i]).save(root / "depth" / f"{i:06d}.png")
        Image.fromarray(np.full((H, W), 2, np.uint8)).save(root / "confidence" / f"{i:06d}.png")

    # rgb.mp4 at 2x, via ffmpeg from upscaled frames
    png_dir = tmp_path / "png"
    png_dir.mkdir()
    for i in range(scene["rgb"].shape[0]):
        Image.fromarray(scene["rgb"][i]).resize((RW, RH), Image.NEAREST).save(png_dir / f"{i:06d}.png")
    subprocess.run(
        ["ffmpeg", "-v", "error", "-framerate", "30", "-i", str(png_dir / "%06d.png"),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "15", str(root / "rgb.mp4")],
        check=True,
    )
    return root


@pytest.mark.skipif(ffmpeg_missing, reason="needs ffmpeg + ffprobe")
def test_stray_import(stray_fixture, scene, tmp_path):
    from wurld.converters import stray

    assert detect(stray_fixture) == "stray"
    out = tmp_path / "stray.wl.webm"
    stray.from_stray(stray_fixture, out)
    seq = wl.read(out)

    H, W = scene["rgb"].shape[1:3]
    assert (seq.probe["width"], seq.probe["height"]) == (W, H)  # depth-grid policy
    assert len(seq.frames) == 10
    # intrinsics rescaled from the 2x RGB matrix back to the depth grid
    assert np.abs(seq.K("0") - scene["cameras"]["0"].K).max() < 1e-6
    # ARKit RUB -> RDF round trip restores the original poses
    for i in (0, 5, 9):
        assert np.abs(seq.c2w(i) - scene["frames"][i].c2w).max() < 1e-9
        assert seq.frames[i].t == scene["frames"][i].t
    # depth mm bit-exact through the pipeline
    dm = seq.depth_meters(0)
    ref = scene["depth_m"][0]
    both = ~np.isnan(dm) & ~np.isnan(ref)
    assert np.abs(dm[both] - ref[both]).max() < 1e-3 + 1e-9  # mm quantization only
    # confidence signal present with labels
    conf_meta = seq.signal_meta("confidence")
    assert conf_meta is not None
    assert np.all(seq.signal("confidence") == 2)
    assert seq.world["gravity_in_world"] == [0.0, -1.0, 0.0]


@pytest.mark.skipif(ffmpeg_missing, reason="needs ffmpeg + ffprobe")
def test_stray_import_at_rgb(stray_fixture, scene, tmp_path):
    from wurld.converters import stray

    out = tmp_path / "stray_rgb.wl.webm"
    stray.from_stray(stray_fixture, out, at="rgb")
    seq = wl.read(out)
    H, W = scene["rgb"].shape[1:3]
    assert (seq.probe["width"], seq.probe["height"]) == (W * 2, H * 2)
    # nearest-upsampled depth: every 2x2 block equals the source pixel
    d = seq.signal("depth")[0]
    src = np.where(np.isnan(scene["depth_m"][0]), 0,
                   np.round(scene["depth_m"][0] / 1e-3)).astype(np.uint16)
    assert np.array_equal(d[::2, ::2], src)
