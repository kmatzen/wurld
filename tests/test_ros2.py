"""The ROS 2 bridge: a real rosbag2, not a Foxglove-shaped log.

The two things most likely to be wrong in any bridge like this are silent, and
both are checked head-on:

**Quaternion order.** ROS is xyzw, wurld is wxyz. Swapping them yields a
perfectly valid rotation that is simply wrong, and no schema check catches it.
`test_pose_round_trips_exactly` compares recovered rotation matrices against the
source, and `test_quaternion_order_is_xyzw_on_the_wire` reads the raw message
fields rather than trusting our own reader.

**Frame convention.** ROS body frames are FLU, optical frames are RDF (REP 145).
wurld poses are RDF, so `world -> <cam>_optical_frame` is `c2w` unchanged — but
only if the frame is named as an optical one. A dropped suffix is a 90-degree
error a consumer would apply silently.

`rosbags` is used to write and read; it does not need ROS installed. Where a
check could be satisfied by our own conventions agreeing with themselves, the
raw deserialized message is inspected instead.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import wurld as wl
from wurld import conventions, validate as v

ROOT = Path(__file__).resolve().parents[1]
pytest.importorskip("rosbags", reason="the ROS 2 bridge needs rosbags")

from wurld.converters import ros2  # noqa: E402
from rosbags.rosbag2 import Reader  # noqa: E402
from rosbags.typesys import Stores, get_typestore  # noqa: E402

TS = get_typestore(Stores.ROS2_HUMBLE)


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    """A real example file: two cameras, IMU, rig, depth."""
    d = tmp_path_factory.mktemp("ros2src")
    out = d / "rig.wurld.webm"
    r = subprocess.run([sys.executable, str(ROOT / "examples" / "04_robot_rig_imu.py"),
                        str(out)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return out


@pytest.fixture(scope="module")
def bag(source, tmp_path_factory):
    return ros2.to_rosbag2(source, tmp_path_factory.mktemp("ros2bag") / "bag")


def read_all(bag_dir):
    """Every message, grouped by topic, deserialized from CDR."""
    out = {}
    with Reader(bag_dir) as reader:
        for conn, stamp, raw in reader.messages():
            out.setdefault(conn.topic, []).append(
                (conn.msgtype, stamp, TS.deserialize_cdr(raw, conn.msgtype)))
    return out


# ---------------------------------------------------------------------- shape

def test_bag_is_a_rosbag2_with_ros_types(bag):
    assert (bag / "metadata.yaml").exists(), "rosbag2 needs metadata.yaml"
    assert (bag / "bag.mcap").exists()
    msgs = read_all(bag)
    types = {t for topic in msgs.values() for t, _, _ in topic}
    # Real ROS 2 types, not Foxglove schemas: this is what makes it replayable.
    assert "tf2_msgs/msg/TFMessage" in types
    assert "sensor_msgs/msg/Image" in types
    assert "sensor_msgs/msg/CameraInfo" in types
    assert "sensor_msgs/msg/Imu" in types
    assert any(t.startswith("/camera/") and t.endswith("/image_raw") for t in msgs)
    assert "/tf" in msgs


def test_optical_frame_suffix_is_present(bag):
    """The suffix is the only signal that the frame is RDF, not FLU."""
    msgs = read_all(bag)
    for _, _, m in msgs["/tf"]:
        for tr in m.transforms:
            assert tr.header.frame_id == "world"
            assert tr.child_frame_id.endswith("_optical_frame"), tr.child_frame_id


# ---------------------------------------------------------------------- poses

def test_quaternion_order_is_xyzw_on_the_wire(source, bag):
    """Read the raw fields: our reader agreeing with our writer proves nothing."""
    seq = wl.read(source)
    msgs = read_all(bag)
    posed = [f for f in seq.frames if f.pose_valid]
    tfs = [tr for _, _, m in msgs["/tf"] for tr in m.transforms]
    assert len(tfs) == len(posed)

    for f, tr in zip(posed, tfs):
        q = tr.transform.rotation
        # wurld (w, x, y, z) -> ROS (x, y, z, w). If these were passed through
        # unswapped, w would hold wurld's x and the rotation would be wrong.
        assert q.w == pytest.approx(f.q_wxyz[0], abs=1e-9)
        assert q.x == pytest.approx(f.q_wxyz[1], abs=1e-9)
        assert q.y == pytest.approx(f.q_wxyz[2], abs=1e-9)
        assert q.z == pytest.approx(f.q_wxyz[3], abs=1e-9)
        t = tr.transform.translation
        assert (t.x, t.y, t.z) == pytest.approx(tuple(f.tr), abs=1e-9)


def test_pose_round_trips_exactly(source, bag):
    """world -> optical_frame must reconstruct c2w with no axis conversion."""
    seq = wl.read(source)
    msgs = read_all(bag)
    posed = [f for f in seq.frames if f.pose_valid]
    tfs = [tr for _, _, m in msgs["/tf"] for tr in m.transforms]

    for f, tr in zip(posed, tfs):
        q, t = tr.transform.rotation, tr.transform.translation
        got = conventions.pose_to_matrix([q.w, q.x, q.y, q.z], [t.x, t.y, t.z])
        assert np.abs(got - f.c2w).max() < 1e-9


def test_unposed_frames_emit_no_transform(tmp_path):
    """A gap in tf is honest; an identity transform is a camera that never was."""
    src = tmp_path / "ff.wurld.webm"
    r = subprocess.run([sys.executable,
                        str(ROOT / "examples" / "01_feedforward_reconstruction.py"),
                        str(src)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    seq = wl.read(src)
    n_posed = sum(1 for f in seq.frames if f.pose_valid)
    assert n_posed < len(seq.frames), "fixture should contain unposed frames"

    msgs = read_all(ros2.to_rosbag2(src, tmp_path / "bag"))
    tfs = [tr for _, _, m in msgs["/tf"] for tr in m.transforms]
    assert len(tfs) == n_posed


# --------------------------------------------------------------- calibration

def test_camera_info_matches_the_intrinsics(source, bag):
    seq = wl.read(source)
    msgs = read_all(bag)
    infos = [m for topic, items in msgs.items() if topic.endswith("/camera_info")
             for _, _, m in items]
    assert infos
    cam = seq.cameras[sorted(seq.cameras)[0]]
    fx, fy, cx, cy = cam.params[:4]
    info = infos[0]
    assert (info.width, info.height) == (cam.width, cam.height)
    k = np.asarray(info.k, dtype=float)
    assert k[0] == pytest.approx(fx) and k[4] == pytest.approx(fy)
    assert k[2] == pytest.approx(cx) and k[5] == pytest.approx(cy)
    assert list(np.asarray(info.p, dtype=float))[:3] == pytest.approx([fx, 0.0, cx])
    assert info.distortion_model in ("plumb_bob", "equidistant")


def test_distortion_maps_to_the_right_ros_model():
    f = 100.0
    opencv = wl.Camera("OPENCV", 64, 48, [f, f, 32, 24, -0.28, 0.07, 1e-4, 2e-5])
    model, d = ros2._distortion(opencv)
    assert model == "plumb_bob"
    assert d == [-0.28, 0.07, 1e-4, 2e-5, 0.0]        # k1 k2 p1 p2 k3

    fisheye = wl.Camera("OPENCV_FISHEYE", 64, 48, [f, f, 32, 24, -0.01, 2e-3, -3e-4, 4e-5])
    model, d = ros2._distortion(fisheye)
    assert model == "equidistant"
    assert len(d) == 4

    pinhole = wl.Camera("PINHOLE", 64, 48, [f, f, 32, 24])
    assert ros2._distortion(pinhole) == ("plumb_bob", [0.0] * 5)

    class Weird:
        model, params = "MADE_UP", [1, 2, 3]
    # Falling back to pinhole here would leave a consumer undistorting with zeros.
    with pytest.raises(ValueError, match="no ROS distortion model"):
        ros2._distortion(Weird())


# --------------------------------------------------------------------- images

def test_images_are_bit_exact(source, bag):
    seq = wl.read(source)
    msgs = read_all(bag)
    topic = next(t for t in msgs if t.endswith("/image_raw") and "depth" not in t)
    for idx, (_, _, m) in enumerate(msgs[topic]):
        assert m.encoding == "rgb8"
        got = np.asarray(m.data, dtype=np.uint8).reshape(m.height, m.width, 3)
        assert np.array_equal(got, seq.rgb[idx][..., :3])
        assert m.step == m.width * 3


def test_depth_is_metric_float_and_keeps_nan(tmp_path, source):
    """32FC1 metres, not 16UC1 millimetres: 0 must not mean two things."""
    msgs = read_all(ros2.to_rosbag2(source, tmp_path / "bagd"))
    topic = next(t for t in msgs if t.endswith("/depth/image_raw"))
    seq = wl.read(source)
    _, _, m = msgs[topic][0]
    assert m.encoding == "32FC1"
    got = np.asarray(m.data, dtype=np.uint8).view(np.float32).reshape(m.height, m.width)
    want = seq.depth_meters(0).astype(np.float32)
    assert np.array_equal(np.nan_to_num(got, nan=-1), np.nan_to_num(want, nan=-1))


# ------------------------------------------------------------------------ imu

def test_imu_declares_that_it_has_no_orientation(source, bag):
    """REP 145: covariance[0] = -1. A default identity would be believed."""
    msgs = read_all(bag)
    topic = next(t for t in msgs if t.startswith("/imu/"))
    seq = wl.read(source)
    stream = seq.imu[topic.rsplit("/", 1)[-1]]
    items = msgs[topic]
    assert len(items) == stream.samples.shape[0]

    for (_, _, m), row in zip(items, stream.samples):
        assert m.orientation_covariance[0] == -1.0
        assert m.angular_velocity.x == pytest.approx(row[1], rel=1e-6, abs=1e-7)
        assert m.linear_acceleration.z == pytest.approx(row[6], rel=1e-6, abs=1e-7)


# -------------------------------------------------------------------- reading

def test_import_round_trip(source, bag, tmp_path):
    """bag -> wurld must recover poses, calibration and pixels."""
    out = ros2.from_rosbag2(bag, tmp_path / "back.wurld.webm")
    assert v.validate(out) == []

    src, back = wl.read(source), wl.read(out)
    assert len(back.frames) == len(src.frames)
    cam = back.cameras[sorted(back.cameras)[0]]
    src_cam = src.cameras[sorted(src.cameras)[0]]
    assert (cam.width, cam.height) == (src_cam.width, src_cam.height)
    assert cam.params[:4] == pytest.approx(list(src_cam.params[:4]))

    for a, b in zip(src.frames, back.frames):
        assert b.pose_valid == a.pose_valid
        if a.pose_valid:
            assert np.abs(b.c2w - a.c2w).max() < 1e-6, "pose changed through ROS"
    # The display track is lossy VP9, so a round trip re-encodes it. Measured on
    # this fixture: mean |delta| ~2.9/255, max 27. Asserting equality here would
    # be asserting something the format never promised.
    d = np.abs(back.rgb[..., :3].astype(int) - src.rgb[..., :3].astype(int))
    assert d.mean() < 6.0, d.mean()

    # Depth is requantized on import (near/far are derived from the data, not
    # carried by the bag), so it is not bit-exact either — but it stays metric
    # to well under a millimetre. Measured max error here: 4.6e-06 m.
    a0, b0 = src.depth_meters(0), back.depth_meters(0)
    m = np.isfinite(a0) & np.isfinite(b0)
    assert np.abs(a0[m] - b0[m]).max() < 1e-4
    assert np.isnan(a0).sum() == np.isnan(b0).sum(), "invalid pixels must stay invalid"

    assert back.imu["imu0"].samples.shape == src.imu["imu0"].samples.shape


def test_import_preserves_unposed_frames(tmp_path):
    src = tmp_path / "ff.wurld.webm"
    subprocess.run([sys.executable,
                    str(ROOT / "examples" / "01_feedforward_reconstruction.py"),
                    str(src)], capture_output=True, check=True)
    bag = ros2.to_rosbag2(src, tmp_path / "ffbag")
    back = wl.read(ros2.from_rosbag2(bag, tmp_path / "ffback.wurld.webm"))
    lost = [f.i for f in back.frames if not f.pose_valid]
    assert lost == [f.i for f in wl.read(src).frames if not f.pose_valid]


def test_sqlite3_storage_also_works(source, tmp_path):
    """rosbag2's other storage plugin, since not every stack uses MCAP."""
    bag = ros2.to_rosbag2(source, tmp_path / "sq", storage="sqlite3")
    assert (bag / "metadata.yaml").exists()
    assert any(p.suffix == ".db3" for p in bag.iterdir())
    assert read_all(bag)["/tf"]


def test_rejects_an_unknown_storage(source, tmp_path):
    with pytest.raises(ValueError, match="storage must be"):
        ros2.to_rosbag2(source, tmp_path / "nope", storage="parquet")


def test_negative_timestamps_are_refused(source, tmp_path):
    """ROS time is unsigned; silently clamping would reorder a capture."""
    with pytest.raises(ValueError, match="negative"):
        ros2._stamp(TS, -0.5)


def test_cli_export_and_import(source, tmp_path):
    """The CLI path, since that is how most people will touch this."""
    bag = tmp_path / "clibag"
    r = subprocess.run([sys.executable, "-m", "wurld.cli", "ros2", "export",
                        str(source), str(bag)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "rosbag2" in r.stdout and "/tf" in r.stdout

    back = tmp_path / "cliback.wurld.webm"
    r = subprocess.run([sys.executable, "-m", "wurld.cli", "ros2", "import",
                        str(bag), str(back)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "posed" in r.stdout
    assert v.validate(back) == []


def test_poses_only_export_skips_pixels(source, tmp_path):
    """A trajectory-only bag: much smaller, and still replayable for tf."""
    full = ros2.to_rosbag2(source, tmp_path / "full")
    lean = ros2.to_rosbag2(source, tmp_path / "lean", images=False, depth=False)
    msgs = read_all(lean)
    assert "/tf" in msgs
    assert not any(t.endswith("image_raw") for t in msgs)
    size = lambda d: sum(p.stat().st_size for p in d.iterdir())
    assert size(lean) < 0.2 * size(full)


# --------------------------------------------------------------- multi-camera

@pytest.fixture(scope="module")
def stereo_bag(tmp_path_factory):
    """A real stereo file: both eyes must survive the bridge."""
    d = tmp_path_factory.mktemp("ros2stereo")
    src = d / "stereo.wurld.webm"
    r = subprocess.run([sys.executable, str(ROOT / "examples" / "06_stereo_rig.py"),
                        str(src)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return src, ros2.to_rosbag2(src, d / "bag")


def test_every_display_stream_is_exported(stereo_bag):
    """A stereo file exported as one camera looks like a success until it isn't."""
    src, bag = stereo_bag
    seq = wl.read(src)
    assert len(seq.rgb_streams) == 2, "the fixture must be genuinely stereo"

    msgs = read_all(bag)
    for cam_id in seq.rgb_streams:
        topic = f"/camera/{cam_id}/image_raw"
        assert topic in msgs, f"{cam_id} was dropped; topics are {sorted(msgs)}"
        assert len(msgs[topic]) == len(seq.frames)
        assert f"/camera/{cam_id}/camera_info" in msgs


def test_the_two_exported_eyes_are_not_the_same_image(stereo_bag):
    """The failure this would otherwise hide: one stream written twice."""
    src, bag = stereo_bag
    seq = wl.read(src)
    msgs = read_all(bag)
    a = msgs[f"/camera/{seq.rgb_streams[0]}/image_raw"]
    b = msgs[f"/camera/{seq.rgb_streams[1]}/image_raw"]
    left = np.asarray(a[0][2].data, dtype=np.uint8)
    right = np.asarray(b[0][2].data, dtype=np.uint8)
    assert not np.array_equal(left, right)


def test_tf_carries_a_frame_for_every_camera(stereo_bag):
    """A consumer needs to place both eyes, not just the posed one."""
    src, bag = stereo_bag
    seq = wl.read(src)
    msgs = read_all(bag)

    positions = {}
    for _, _, m in msgs["/tf"]:
        for tr in m.transforms:
            t = tr.transform.translation
            positions.setdefault(tr.child_frame_id, []).append([t.x, t.y, t.z])

    for cam_id in seq.rgb_streams:
        assert ros2.optical_frame(cam_id) in positions, f"no tf for {cam_id}"

    # The second camera's transform is derived from the rig, so the distance
    # between the two frames must be the calibrated baseline, on every frame.
    a = np.array(positions[ros2.optical_frame(seq.rgb_streams[0])])
    b = np.array(positions[ros2.optical_frame(seq.rgb_streams[1])])
    d = np.linalg.norm(a - b, axis=1)
    assert abs(d.mean() - 0.12) < 1e-4, d.mean()
    assert d.std() < 1e-9, "a rigid baseline must not vary frame to frame"


def test_export_memory_does_not_track_sequence_length(tmp_path):
    """seq.rgb on a real EuRoC sequence is 4.2 GB per stream.

    Measured as the *shape* of the curve — export two lengths and compare —
    rather than against a byte count. On a fixture small enough to keep the
    suite fast, the bag writer's fixed cost swamps any absolute bound, which
    would make the assertion about the fixture instead of the code.
    """
    import tracemalloc

    import chromapakz as cz

    def build(n, path):
        w, h = 96, 72
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        rgb = np.stack([
            np.dstack([np.clip((0.5 + 0.4 * np.sin(xx / 5 + i * 0.3)) * 255, 0, 255)] * 3
                      + [np.full((h, w), 255, np.float32)]).astype(np.uint8)
            for i in range(n)])
        f = 1.1 * w
        wl.write(path,
                 cameras={"0": wl.Camera("PINHOLE", w, h, [f, f, w / 2, h / 2])},
                 frames=[wl.Frame(i=i, t=i / 30, camera="0", q_wxyz=(1.0, 0.0, 0.0, 0.0),
                                  tr=(0.01 * i, 0.0, 0.5)) for i in range(n)],
                 rgb=rgb, world={"metric_scale": True}, fps=30)
        return path

    def peak_for(n):
        src = build(n, tmp_path / f"src{n}.wurld.webm")
        tracemalloc.start()
        ros2.to_rosbag2(src, tmp_path / f"bag{n}")
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return peak

    n_short, n_long = 20, 120
    short, long = peak_for(n_short), peak_for(n_long)
    length_ratio = n_long / n_short          # 6x

    # Some growth is unavoidable and is not ours: an MCAP writer keeps a message
    # index, which scales with message count. What must not scale is the pixel
    # buffer — materialising would put the memory ratio at the length ratio.
    # Measured here: ~1.9x for 6x the frames.
    assert long < 0.5 * length_ratio * short, (
        f"peak grew {long/short:.1f}x for {length_ratio:.0f}x the frames "
        f"({short/1e6:.1f} -> {long/1e6:.1f} MB) — the exporter looks like it is "
        "holding the sequence")


# ------------------------------------------------------------------------ hdr

@pytest.fixture(scope="module")
def hdr_file(tmp_path_factory):
    """A 10-bit PQ display track: seq.rgb comes back uint16, not uint8."""
    W2, H2, N2 = 64, 48, 6
    out = tmp_path_factory.mktemp("ros2hdr") / "hdr.wurld.webm"
    codes = np.stack([np.full((H2, W2, 4), 400 + 40 * i, np.uint16) for i in range(N2)])
    f = 1.1 * W2
    wl.write(out,
             cameras={"0": wl.Camera("PINHOLE", W2, H2, [f, f, W2 / 2, H2 / 2])},
             frames=[wl.Frame(i=i, t=i / 30, camera="0", q_wxyz=(1.0, 0.0, 0.0, 0.0),
                              tr=(0.01 * i, 0.0, 1.0)) for i in range(N2)],
             rgb=codes, hdr={"transfer": "pq", "max_cll": 1000},
             world={"metric_scale": True}, fps=30)
    return out


def test_hdr_images_are_labelled_rgb16_not_rgb8(hdr_file, tmp_path, caplog):
    """An rgb8 label on uint16 data is a corrupted double-width image.

    Nothing raises: the message is well-formed as far as CDR is concerned, and a
    consumer simply reads nonsense. The declared layout has to match the bytes.
    """
    import logging

    seq = wl.read(hdr_file)
    assert seq.rgb.dtype == np.uint16, "the fixture must actually be HDR"

    with caplog.at_level(logging.WARNING):
        bag = ros2.to_rosbag2(hdr_file, tmp_path / "hdrbag")
    assert any("HDR display track" in r.getMessage() for r in caplog.records), \
        "exporting PQ codes as if they were colour said nothing"

    msgs = read_all(bag)
    topic = next(t for t in msgs if t.endswith("/image_raw") and "depth" not in t)
    _, _, m = msgs[topic][0]
    assert m.encoding == "rgb16"
    assert m.step == m.width * 3 * 2
    assert len(m.data) == m.width * m.height * 3 * 2
    assert m.step * m.height == len(m.data), "declared layout does not match the bytes"


def test_importing_rgb16_is_refused_rather_than_misread(hdr_file, tmp_path):
    """A bag cannot record the transfer function, so the round trip is not one."""
    bag = ros2.to_rosbag2(hdr_file, tmp_path / "hdrbag2")
    with pytest.raises(ValueError, match="transfer function"):
        ros2.from_rosbag2(bag, tmp_path / "back.wurld.webm")


def test_rgb_encoding_rejects_types_it_cannot_describe():
    assert ros2._rgb_encoding(np.zeros((2, 2, 3), np.uint8)) == "rgb8"
    assert ros2._rgb_encoding(np.zeros((2, 2, 3), np.uint16)) == "rgb16"
    with pytest.raises(ValueError, match="no ROS encoding"):
        ros2._rgb_encoding(np.zeros((2, 2, 3), np.float32))
