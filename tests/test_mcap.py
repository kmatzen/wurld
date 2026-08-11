import base64
import json

import numpy as np
import pytest

import wurld as wl

mcap = pytest.importorskip("mcap")
from mcap.reader import make_reader  # noqa: E402

from wurld.converters import mcap_export  # noqa: E402


@pytest.fixture(scope="module")
def mcap_file(scene, tmp_path_factory):
    # reuse the standard scene but add an IMU stream
    import chromapakz as cz
    from tests.conftest import FAR, NEAR

    p = tmp_path_factory.mktemp("mcap") / "scene.wurld.webm"
    imu = wl.ImuStream("imu0", np.column_stack([
        np.linspace(0.0, 0.3, 20), np.full((20, 3), 0.01), np.full((20, 3), 3.27)
    ]), rate_hz=66.0)
    wl.write(p, cameras=scene["cameras"], frames=scene["frames"], rgb=scene["rgba"],
             signals={"depth": scene["d16"]}, specs={"depth": cz.inverse_depth_spec(NEAR, FAR)},
             signal_meta=[wl.SignalMeta("depth", "depth",
                 {"type": "inverse_depth", "near": NEAR, "far": FAR, "levels": 65536, "invalid": 0})],
             imu=[imu])
    out = tmp_path_factory.mktemp("mcap") / "scene.mcap"
    mcap_export.to_mcap(p, out)
    return p, out


def test_mcap_channels_and_counts(mcap_file, scene):
    _, out = mcap_file
    with open(out, "rb") as f:
        reader = make_reader(f)
        summary = reader.get_summary()
        topics = {c.topic: summary.statistics.channel_message_counts[cid]
                  for cid, c in summary.channels.items()}
    assert topics["/camera/pose"] == 10
    assert topics["/tf"] == 10
    assert topics["/camera/image"] == 10
    assert topics["/camera/depth"] == 10
    assert topics["/camera/calibration"] == 1
    assert topics["/imu/imu0"] == 20


def test_mcap_pose_and_depth_round_trip(mcap_file, scene):
    wl_path, out = mcap_file
    seq = wl.read(wl_path)
    poses, depths = [], []
    with open(out, "rb") as f:
        for schema, channel, message in make_reader(f).iter_messages(
                topics=["/camera/pose", "/camera/depth"]):
            msg = json.loads(message.data)
            (poses if channel.topic == "/camera/pose" else depths).append((message.log_time, msg))
    poses.sort(); depths.sort()

    p0 = poses[0][1]["pose"]
    f0 = seq.frames[0]
    assert np.allclose([p0["position"][k] for k in "xyz"], f0.tr, atol=1e-12)
    qx, qy, qz, qw = (p0["orientation"][k] for k in "xyzw")
    assert np.allclose([qw, qx, qy, qz], f0.q_wxyz, atol=1e-12)
    assert poses[0][0] == int(round(f0.t * 1e9))

    d0 = depths[0][1]
    raw = np.frombuffer(base64.b64decode(d0["data"]), dtype=np.uint16).reshape(
        d0["height"], d0["width"])
    assert np.array_equal(raw, scene["d16"][0])  # bit-exact codes in 16UC1
    assert d0["encoding"] == "16UC1" and d0["step"] == d0["width"] * 2


def test_mcap_metadata_preserves_document(mcap_file):
    wl_path, out = mcap_file
    seq = wl.read(wl_path)
    with open(out, "rb") as f:
        reader = make_reader(f)
        metas = [m for m in reader.iter_metadata() if m.name == "wurld"]
    assert metas
    doc = json.loads(metas[0].metadata["document"])
    assert doc["format"] == "wurld"
    assert doc["cameras"] == {k: c.to_json() for k, c in seq.cameras.items()}
    assert doc["signals"][0]["value_map"]["type"] == "inverse_depth"


def test_cli_extract_mcap(scene, wl_file, tmp_path, capsys):
    from wurld.cli import main

    out = tmp_path / "x.mcap"
    assert main(["extract", str(wl_file), str(out), "--format", "mcap"]) == 0
    assert out.stat().st_size > 1000
