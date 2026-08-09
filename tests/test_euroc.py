import numpy as np
import pytest
from PIL import Image

import wurld as wl
from wurld import conventions
from wurld.converters import detect, euroc


def _yaml_cam(intr, dist, res, t_bs):
    data = ", ".join(repr(float(v)) for v in t_bs.flatten())
    return (
        "sensor_type: camera\n"
        "T_BS:\n  cols: 4\n  rows: 4\n"
        f"  data: [{data}]\n"
        f"rate_hz: 20\nresolution: [{res[0]}, {res[1]}]\ncamera_model: pinhole\n"
        f"intrinsics: [{', '.join(str(v) for v in intr)}]\n"
        "distortion_model: radial-tangential\n"
        f"distortion_coefficients: [{', '.join(str(v) for v in dist)}]\n"
    )


@pytest.fixture()
def euroc_fixture(scene, tmp_path):
    """Synthetic mav0: cam0 frames + stereo yaml + body ground truth + imu."""
    base = tmp_path / "seq" / "mav0"
    (base / "cam0" / "data").mkdir(parents=True)
    (base / "cam1").mkdir()
    (base / "imu0").mkdir()
    (base / "state_groundtruth_estimate0").mkdir()

    H, W = scene["rgb"].shape[1:3]
    K = scene["cameras"]["0"].K
    intr = [K[0, 0], K[1, 1], K[0, 2], K[1, 2]]

    # camera-to-body transforms: cam0 slightly offset/rotated, cam1 a stereo baseline away
    t_bs0 = conventions.pose_to_matrix(
        conventions.matrix_to_quat_wxyz(conventions.quat_wxyz_to_matrix([0.999, 0.02, -0.01, 0.03])),
        [0.015, -0.002, 0.01])
    t_bs1 = t_bs0 @ conventions.pose_to_matrix([1, 0, 0, 0], [0.11, 0, 0])
    (base / "cam0" / "sensor.yaml").write_text(_yaml_cam(intr, [0.01, -0.002, 0.0001, 0.0002], (W, H), t_bs0))
    (base / "cam1" / "sensor.yaml").write_text(_yaml_cam(intr, [0.011, -0.002, 0.0, 0.0], (W, H), t_bs1))

    cam_rows = ["#timestamp [ns],filename"]
    gt_rows = ["#timestamp, p_RS_R_x [m], p_RS_R_y [m], p_RS_R_z [m], q_RS_w [], q_RS_x [], q_RS_y [], q_RS_z []"]
    for f in scene["frames"]:
        ns = int(round(f.t * 1e9))
        name = f"{ns}.png"
        Image.fromarray(scene["rgb"][f.i]).convert("L").save(base / "cam0" / "data" / name)
        cam_rows.append(f"{ns},{name}")
        # body pose from the known camera pose: T_WB = c2w @ inv(T_BS0)
        t_wb = f.c2w @ conventions.invert_pose(t_bs0)
        q, p = conventions.matrix_to_pose(t_wb)
        gt_rows.append(f"{ns},{p[0]},{p[1]},{p[2]},{q[0]},{q[1]},{q[2]},{q[3]}")
    (base / "cam0" / "data.csv").write_text("\n".join(cam_rows) + "\n")
    (base / "state_groundtruth_estimate0" / "data.csv").write_text("\n".join(gt_rows) + "\n")

    t_bs_imu = conventions.pose_to_matrix([1, 0, 0, 0], [0.001, 0.002, -0.003])
    (base / "imu0" / "sensor.yaml").write_text(
        "sensor_type: imu\nT_BS:\n  cols: 4\n  rows: 4\n"
        f"  data: [{', '.join(repr(float(v)) for v in t_bs_imu.flatten())}]\nrate_hz: 200\n")
    imu_rows = ["#timestamp [ns],w_x,w_y,w_z,a_x,a_y,a_z"]
    rng = np.random.default_rng(3)
    t0 = int(round(scene["frames"][0].t * 1e9))
    for k in range(40):
        g, a = rng.normal(0, 0.01, 3), rng.normal([0, 0, 9.81], 0.1, 3)
        imu_rows.append(f"{t0 + k * 5_000_000},{g[0]},{g[1]},{g[2]},{a[0]},{a[1]},{a[2]}")
    (base / "imu0" / "data.csv").write_text("\n".join(imu_rows) + "\n")
    return base.parent, t_bs0, t_bs1, t_bs_imu


def test_euroc_import(euroc_fixture, scene, tmp_path):
    seq_dir, t_bs0, t_bs1, t_bs_imu = euroc_fixture
    assert detect(seq_dir) == "euroc"
    out = tmp_path / "euroc.wl.webm"
    euroc.from_euroc(seq_dir, out)
    seq = wl.read(out)

    # cam0 poses reconstruct through gt -> T_WB -> T_BS0.
    #
    # Tolerance is float32, not float64: the importer streams now (a real EuRoC
    # sequence is 8.4 GB of RGBA if materialised), and the streaming layout
    # stores poses in the binary frame table, which is float32 by SPEC §7. That
    # is ~6e-8 on a metre-scale translation — 60 nanometres. Timestamps are
    # unaffected; the table stores them as float64.
    assert len(seq.frames) == 10
    for i in (0, 5, 9):
        assert np.abs(seq.c2w(i) - scene["frames"][i].c2w).max() < 1e-6
        assert seq.frames[i].t == pytest.approx(scene["frames"][i].t, abs=1e-9)

    # both cameras' calibration present; OPENCV distortion carried
    assert set(seq.cameras) == {"0", "1"}
    assert seq.cameras["0"].model == "OPENCV"
    assert seq.cameras["0"].params[4] == pytest.approx(0.01)

    # rig: camera-to-body extrinsics round trip; derived cam1 pose = c2w0 @ inv(TBS0) @ TBS1
    rig = seq.rigs["body"]["cameras"]
    assert np.allclose(conventions.pose_to_matrix(rig["0"]["q_wxyz"], rig["0"]["tr"]), t_bs0)
    c2w1 = seq.rig_c2w(0, "1", "body")
    expected = scene["frames"][0].c2w @ conventions.invert_pose(t_bs0) @ t_bs1
    # float32, as above: the derived pose inherits the stored pose's precision.
    assert np.abs(c2w1 - expected).max() < 1e-6

    # imu: 40 samples, extrinsics = inv(TBS0) @ TBS_imu
    imu = seq.imu["imu0"]
    assert imu.samples.shape == (40, 7)
    assert imu.rate_hz == pytest.approx(200.0)
    got = conventions.pose_to_matrix(imu.extrinsics["q_wxyz"], imu.extrinsics["tr"])
    assert np.allclose(got, conventions.invert_pose(t_bs0) @ t_bs_imu, atol=1e-12)

    assert "cam0 only" in seq.world["description"]
