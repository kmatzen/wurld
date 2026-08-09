"""EuRoC stereo streams, pose interpolation and ground-truth coverage.

Complements `test_euroc.py`, which covers the single-track path, IMU extrinsics
and format detection against the shared `scene` fixture. This file uses a
standalone fixture instead, because what it checks needs things that one cannot
express: a realistic nanosecond epoch, a ground-truth clock deliberately offset
from the shutter, and coverage that stops short of the images at both ends.

We cannot ship EuRoC itself (CC BY-NC-SA 3.0), and the ETH host was unreachable
when this was written, so the converter is exercised against a directory tree in
the real layout — same filenames, same csv columns, same nanosecond-scale epoch,
and the genuine V1_01 calibration values. It has not been run against the real
download here.

The load-bearing test is `test_camera_pose_composition`: it builds ground truth
*backwards* from a known camera trajectory, so if the converter forgot to compose
T_WB with T_BS the recovered poses would be wrong by that fixed transform. A test
that merely checked "poses exist" would pass with the bug in place.
"""

import numpy as np
import pytest

import wurld as wl
from wurld import conventions, validate as v
from wurld.converters import euroc

# Genuine V1_01_easy calibration, so the fixture is not a fiction of its own.
CAM0_T_BS = [0.0148655429818, -0.999880929698, 0.00414029679422, -0.0216401454975,
             0.999557249008, 0.0149672133247, 0.025715529948, -0.064676986768,
             -0.0257744366974, 0.00375618835797, 0.999660727178, 0.00981073058949,
             0.0, 0.0, 0.0, 1.0]
CAM1_T_BS = [0.0125552670891, -0.999755099723, 0.0182237714554, -0.0198435579556,
             0.999598781151, 0.0130119051815, 0.0251588363115, 0.0453689425024,
             -0.0253898008918, 0.0179005838253, 0.999517347078, 0.00786212447038,
             0.0, 0.0, 0.0, 1.0]
CAM0_INTR = [458.654, 457.296, 367.215, 248.375]
CAM1_INTR = [457.587, 456.134, 379.999, 255.238]
CAM0_DIST = [-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05]
CAM1_DIST = [-0.28368365, 0.07451284, -0.00010473, -3.55590700e-05]

W, H = 752, 480
N_IMG = 8
IMG_HZ, GT_HZ, IMU_HZ = 20.0, 200.0, 200.0
T0_NS = 1403715273262142976          # a realistic EuRoC epoch, not zero
# Ground truth starts after the images begin and ends before they do, so frames
# fall off both ends — EuRoC really does this, and clamping would invent a still
# camera. The start is deliberately NOT a multiple of the image period: EuRoC's
# ground-truth clock is unrelated to the shutter, so samples land between frames.
# With aligned clocks, nearest-neighbour and interpolation agree and the pose
# tests below cannot tell them apart.
GT_START_S, GT_END_S = 0.0823, 0.3123


def _sensor_yaml(t_bs, intr, dist, comment):
    rows = ",\n         ".join(
        ", ".join(f"{v!r}" for v in t_bs[i * 4:i * 4 + 4]) for i in range(4))
    return (f"#Default sensor yaml file\n"
            f"sensor_type: camera\n"
            f"comment: {comment}\n"
            f"T_BS:\n  cols: 4\n  rows: 4\n  data: [{rows}]\n"
            f"rate_hz: 20\n"
            f"resolution: [{W}, {H}]\n"
            f"camera_model: pinhole\n"
            f"intrinsics: {list(intr)} #fu, fv, cu, cv\n"
            f"distortion_model: radial-tangential\n"
            f"distortion_coefficients: {list(dist)}\n")


def _true_camera_poses(n):
    """A known T_WC0 trajectory: gentle yaw while translating."""
    out = []
    for i in range(n):
        t = i / IMG_HZ
        ang = 0.3 * t
        r = np.array([[np.cos(ang), -np.sin(ang), 0.0],
                      [np.sin(ang), np.cos(ang), 0.0],
                      [0.0, 0.0, 1.0]])
        m = np.eye(4)
        m[:3, :3] = r
        m[:3, 3] = [0.4 * t, 0.1 * t, 1.2 + 0.05 * t]
        out.append(m)
    return out


def _pose_at(t):
    """The same trajectory as a continuous function, for 200 Hz ground truth."""
    ang = 0.3 * t
    m = np.eye(4)
    m[:3, :3] = [[np.cos(ang), -np.sin(ang), 0.0],
                 [np.sin(ang), np.cos(ang), 0.0],
                 [0.0, 0.0, 1.0]]
    m[:3, 3] = [0.4 * t, 0.1 * t, 1.2 + 0.05 * t]
    return m


def _quat(m):
    return conventions.matrix_to_pose(m)[0]


def _write_fixture(root, frames=None):
    from PIL import Image

    mav0 = root / "mav0"
    for cid, t_bs, intr, dist in (("cam0", CAM0_T_BS, CAM0_INTR, CAM0_DIST),
                                  ("cam1", CAM1_T_BS, CAM1_INTR, CAM1_DIST)):
        d = mav0 / cid
        (d / "data").mkdir(parents=True)
        (d / "sensor.yaml").write_text(_sensor_yaml(t_bs, intr, dist, f"VI-Sensor {cid}"))

    rng = np.random.default_rng(0)
    n_img = N_IMG if frames is None else frames
    stamps = [T0_NS + int(round(i / IMG_HZ * 1e9)) for i in range(n_img)]
    lines0, lines1 = ["#timestamp [ns],filename"], ["#timestamp [ns],filename"]
    for i, ts in enumerate(stamps):
        # Structured, not noise: VP9 on pure noise is slow and pointless here.
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        g = (0.5 + 0.4 * np.sin(xx / 23 + i * 0.4) * np.cos(yy / 19)) * 255
        img = np.clip(g, 0, 255).astype(np.uint8)
        Image.fromarray(img).save(mav0 / "cam0" / "data" / f"{ts}.png")
        Image.fromarray(np.roll(img, -11, axis=1)).save(mav0 / "cam1" / "data" / f"{ts}.png")
        lines0.append(f"{ts},{ts}.png")
        lines1.append(f"{ts},{ts}.png")
    (mav0 / "cam0" / "data.csv").write_text("\n".join(lines0) + "\n")
    (mav0 / "cam1" / "data.csv").write_text("\n".join(lines1) + "\n")

    # Ground truth is the BODY pose: T_WB = T_WC0 @ inv(T_BS0).
    t_bs0 = np.array(CAM0_T_BS).reshape(4, 4)
    inv_t_bs0 = np.linalg.inv(t_bs0)
    gt = mav0 / "state_groundtruth_estimate0"
    gt.mkdir(parents=True)
    rows = ["#timestamp,p_RS_R_x [m],p_RS_R_y [m],p_RS_R_z [m],q_RS_w [],q_RS_x [],"
            "q_RS_y [],q_RS_z [],v_x,v_y,v_z,bw_x,bw_y,bw_z,ba_x,ba_y,ba_z"]
    n_gt = int(round((GT_END_S - GT_START_S) * GT_HZ)) + 1
    for k in range(n_gt):
        t = GT_START_S + k / GT_HZ
        t_wb = _pose_at(t) @ inv_t_bs0
        q = _quat(t_wb)
        p = t_wb[:3, 3]
        rows.append(",".join([str(T0_NS + int(round(t * 1e9)))]
                             + [repr(float(x)) for x in p]
                             + [repr(float(x)) for x in q]
                             + ["0.0"] * 9))
    (gt / "data.csv").write_text("\n".join(rows) + "\n")

    imu = mav0 / "imu0"
    imu.mkdir(parents=True)
    (imu / "sensor.yaml").write_text(
        "sensor_type: imu\ncomment: VI-Sensor IMU\n"
        "T_BS:\n  cols: 4\n  rows: 4\n  data: ["
        + ", ".join(repr(float(v)) for v in np.eye(4).ravel()) + "]\n"
        "rate_hz: 200\n")
    irows = ["#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y,w_RS_S_z,"
             "a_RS_S_x [m s^-2],a_RS_S_y,a_RS_S_z"]
    n_imu = int(round((n_img - 1) / IMG_HZ * IMU_HZ)) + 1
    for k in range(n_imu):
        t = k / IMU_HZ
        irows.append(",".join([str(T0_NS + int(round(t * 1e9))), "0.0", "0.0", "0.3",
                               repr(float(rng.normal(0, 0.02))), "0.0", "9.81"]))
    (imu / "data.csv").write_text("\n".join(irows) + "\n")
    return mav0


@pytest.fixture(scope="module")
def converted(tmp_path_factory):
    root = tmp_path_factory.mktemp("euroc")
    mav0 = _write_fixture(root)
    out = root / "euroc.wl.webm"
    euroc.from_euroc(mav0, out)
    return wl.read(out), out


@pytest.fixture(scope="module")
def mono(tmp_path_factory):
    """stereo=False must still produce the pre-multi-stream single-track file."""
    root = tmp_path_factory.mktemp("euroc_mono")
    mav0 = _write_fixture(root)
    out = root / "mono.wl.webm"
    euroc.from_euroc(mav0, out, stereo=False)
    return wl.read(out), out


def test_conforms_to_spec(converted):
    _, out = converted
    assert v.validate(out) == []


def test_stereo_streams_and_rig(converted):
    seq, _ = converted
    assert seq.rgb_streams == ["0", "1"]
    # SPEC 4.4: stream ids are camera ids, which is how a reader knows which
    # intrinsics apply to which pixels.
    assert set(seq.rgb_streams) <= set(seq.cameras)
    # Real EuRoC stereo baseline is ~11 cm; it must come out of the calibration.
    baseline = np.linalg.norm(seq.rig_c2w(3, "1")[:3, 3] - seq.rig_c2w(3, "0")[:3, 3])
    assert abs(baseline - 0.110) < 0.002, baseline
    # Both eyes must actually be there, and differ.
    assert not np.array_equal(seq.rgb_for("0"), seq.rgb_for("1"))


def test_mono_still_writes_one_track(mono):
    seq, out = mono
    assert v.validate(out) == []
    # The single track keeps its legacy name, so pre-multi-stream readers are
    # unaffected by anything above.
    assert seq.rgb_streams == ["rgb"]
    # Calibration for the second camera survives even when its pixels do not.
    assert set(seq.cameras) == {"0", "1"}
    assert "1" in seq.rigs["body"]["cameras"]


def test_camera_model_is_distorted(converted):
    seq, _ = converted
    # PINHOLE here would silently discard the distortion EuRoC images carry.
    assert seq.cameras["0"].model == "OPENCV"
    assert len(seq.cameras["0"].params) == 8
    assert seq.cameras["0"].params[4] == pytest.approx(CAM0_DIST[0])
    assert (seq.cameras["0"].width, seq.cameras["0"].height) == (W, H)


def test_camera_pose_composition(converted):
    """Recovered camera poses must match the trajectory the fixture was built from.

    Ground truth was written as T_WB = T_WC0 @ inv(T_BS0). If the converter
    skipped the T_BS composition, every pose would be off by that transform —
    about 7 cm and a near-90-degree rotation, which this catches.
    """
    seq, _ = converted
    truth = _true_camera_poses(N_IMG)
    checked = 0
    for f in seq.frames:
        if not f.pose_valid:
            continue
        i = int(round((f.t - seq.frames[0].t) * IMG_HZ))
        got, want = f.c2w, truth[i]
        assert np.linalg.norm(got[:3, 3] - want[:3, 3]) < 1e-6, f"frame {i} translation"
        assert np.abs(got[:3, :3] - want[:3, :3]).max() < 1e-6, f"frame {i} rotation"
        checked += 1
    assert checked >= 4


def test_frames_outside_ground_truth_are_unposed(converted):
    seq, _ = converted
    unposed = [f.i for f in seq.frames if not f.pose_valid]
    # GT covers 0.0823..0.3123. Images at t=0.00 and 0.05 precede it by more
    # than max_dt=0.02; t=0.35 follows it. The rest are bracketed.
    assert unposed == [0, 1, 7], unposed
    assert all(seq.frames[i].pose_valid for i in (2, 3, 4, 5, 6))


def test_imu_keeps_its_own_rate(converted):
    seq, _ = converted
    s = seq.imu["imu0"].samples
    assert s.shape[1] == 7
    rate = 1.0 / np.median(np.diff(s[:, 0]))
    assert abs(rate - IMU_HZ) < 1.0
    assert s.shape[0] > len(seq.frames) * 5


def test_timestamps_spaced_correctly(converted):
    seq, _ = converted
    ts = [f.t for f in seq.frames]
    # Absolute EuRoC epoch is preserved; only the spacing is asserted here.
    assert ts[0] > 1.4e9
    assert np.allclose(np.diff(ts), 1 / IMG_HZ, atol=1e-6)


def test_no_depth_signal(converted):
    seq, _ = converted
    # EuRoC has no depth, so there are no signal planes; a point cloud would
    # have to be reconstructed rather than read.
    assert seq.signals == []


def test_interpolation_beats_nearest_neighbour(converted):
    """The reason poses are interpolated rather than snapped.

    Ground truth is 200 Hz on a clock unrelated to the shutter, so the nearest
    sample is up to 2.5 ms away. At this fixture's speed that is about a
    millimetre of avoidable error on every frame — small, systematic, and
    exactly the kind of thing that quietly sets the floor of a SLAM benchmark.
    """
    seq, _ = converted
    truth = _true_camera_poses(N_IMG)
    t_bs0 = np.array(CAM0_T_BS).reshape(4, 4)
    inv_t_bs0 = np.linalg.inv(t_bs0)

    gt_t = np.array([GT_START_S + k / GT_HZ
                     for k in range(int(round((GT_END_S - GT_START_S) * GT_HZ)) + 1)])
    interp_err, nearest_err = [], []
    for f in seq.frames:
        if not f.pose_valid:
            continue
        i = int(round((f.t - seq.frames[0].t) * IMG_HZ))
        want = truth[i][:3, 3]
        interp_err.append(np.linalg.norm(f.c2w[:3, 3] - want))
        # What the old nearest-sample code would have produced.
        k = int(np.argmin(np.abs(gt_t - i / IMG_HZ)))
        snapped = (_pose_at(gt_t[k]) @ inv_t_bs0) @ t_bs0
        nearest_err.append(np.linalg.norm(snapped[:3, 3] - want))

    # The interpolation floor is timestamp quantisation, not the maths: an
    # absolute EuRoC epoch (~1.4e9 s) in float64 resolves to ~238 ns, which at
    # this speed is ~1e-7 m. Nearest-neighbour is four orders of magnitude worse.
    assert max(interp_err) < 1e-6, max(interp_err)
    assert max(nearest_err) > 1e-4, max(nearest_err)
    assert max(nearest_err) > 100 * max(interp_err)


def test_max_frames_converts_a_prefix(tmp_path):
    """The escape hatch for sequences too large to hold at once."""
    root = tmp_path / "prefix"
    mav0 = _write_fixture(root)
    out = root / "prefix.wl.webm"
    euroc.from_euroc(mav0, out, max_frames=4)
    seq = wl.read(out)
    assert len(seq.frames) == 4
    assert v.validate(out) == []


def test_a_full_length_sequence_does_not_need_to_fit_in_memory(tmp_path):
    """The ceiling this importer used to have, asserted rather than described.

    A real EuRoC run is 2912 stereo frames at 752x480 — 8.4 GB of RGBA if the
    sequence is materialised, which is more than the machine this was written on
    has. Streaming holds one frame, so peak memory tracks the frame size and not
    the sequence length. The fixture is small; what it checks is the *shape* of
    the curve, by converting twice at different lengths.
    """
    import tracemalloc

    def peak_for(n):
        root = tmp_path / f"len{n}"
        mav0 = _write_fixture(root, frames=n)
        tracemalloc.start()
        # Forced: the fixture is small enough that the size heuristic would
        # choose the batch path, which materialises by design.
        euroc.from_euroc(mav0, root / "out.wl.webm", streaming=True)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return peak

    short, long = peak_for(4), peak_for(N_IMG)
    assert N_IMG >= 2 * 4, "the two lengths must differ enough to be informative"
    # Materialising would make peak scale with frame count; streaming must not.
    assert long < 1.6 * short, (
        f"peak grew from {short/1e6:.1f} MB at 4 frames to {long/1e6:.1f} MB at "
        f"{N_IMG} — memory is tracking sequence length, not frame size")


def test_the_writer_choice_is_visible_and_forcible(tmp_path):
    """Streaming and batch differ in pose precision, so the choice must be one."""
    root = tmp_path / "choice"
    mav0 = _write_fixture(root)

    batch = root / "batch.wl.webm"
    streamed = root / "stream.wl.webm"
    euroc.from_euroc(mav0, batch, streaming=False)
    euroc.from_euroc(mav0, streamed, streaming=True)
    assert v.validate(batch) == [] and v.validate(streamed) == []

    from wurld import ebml
    # The batch path keeps poses in the document; streaming puts them in the
    # binary table. Both read back as the same poses, to float32.
    assert "WURLD_FRAMES" not in ebml.read_all_tags(batch.read_bytes())
    assert isinstance(ebml.read_all_tags(streamed.read_bytes()).get("WURLD_FRAMES"), bytes)

    a, b = wl.read(batch), wl.read(streamed)
    assert len(a.frames) == len(b.frames)
    for x, y in zip(a.frames, b.frames):
        assert x.pose_valid == y.pose_valid
        if x.pose_valid:
            assert np.abs(np.asarray(x.c2w) - np.asarray(y.c2w)).max() < 1e-6
