import shutil
import subprocess

import chromapakz as cz
import numpy as np
import pytest

import wurld as wl
from wurld.stream import StreamReader, StreamWriter

pytestmark = pytest.mark.skipif(
    not hasattr(cz, "create_encoder"), reason="needs chromapakz streaming encode (PR #43)"
)

NEAR, FAR = 0.5, 40.0


def _record(scene, out_parts, imu=False, frames=None):
    n = frames if frames is not None else len(scene["frames"])
    w = StreamWriter(
        out_parts.append,
        cameras=scene["cameras"],
        signal_meta=[
            wl.SignalMeta(
                "depth", "depth",
                {"type": "inverse_depth", "near": NEAR, "far": FAR, "levels": 65536, "invalid": 0},
            )
        ],
        world={"metric_scale": True, "gravity_in_world": [0, 0, -1], "description": "live test"},
        imu={"imu0": {"rate_hz": 100.0, "description": "test imu"}} if imu else None,
        fps=30,
        has_rgb=True,
    )
    rng = np.random.default_rng(0)
    for i in range(n):
        f = scene["frames"][i]
        if imu:
            t0 = f.t
            samples = np.column_stack([
                t0 + np.arange(3) / 100.0, rng.normal(0, 0.01, (3, 3)),
                rng.normal([0, 0, 9.81], 0.1, (3, 3)),
            ])
            w.add_imu("imu0", samples)
        w.add_frame(
            f,
            rgb=scene["rgba"][i],
            signals={"depth": {"u16": scene["d16"][i]}},
        )
    return w, w.finish()


def test_streamwriter_roundtrip(scene, tmp_path):
    parts = []
    _, summary = _record(scene, parts)
    assert summary["frames"] == 10
    p = tmp_path / "live.wl.webm"
    p.write_bytes(b"".join(parts))

    seq = wl.read(p)
    assert len(seq.frames) == 10
    for i in (0, 5, 9):
        assert seq.frames[i].t == scene["frames"][i].t
        assert np.allclose(seq.frames[i].c2w, scene["frames"][i].c2w, atol=1e-6)
    assert np.array_equal(seq.signal("depth"), scene["d16"])  # bit-exact through live path
    assert seq.signal_meta("depth").value_map["near"] == NEAR


def test_streamwriter_imu(scene, tmp_path):
    parts = []
    _record(scene, parts, imu=True)
    p = tmp_path / "imu.wl.webm"
    p.write_bytes(b"".join(parts))
    seq = wl.read(p)
    got = seq.imu["imu0"]
    assert got.samples.shape == (30, 7)  # 3 samples x 10 frames, concatenated chunks
    assert got.rate_hz == 100.0
    assert np.all(np.diff(got.samples[:, 0]) >= 0)


def test_streamwriter_progressive_and_crash_safe(scene, tmp_path):
    parts = []
    _record(scene, parts)
    stream = b"".join(parts)

    # progressive parse: poses precede or accompany their clusters
    r = StreamReader()
    poses_at_cluster = []
    for off in range(0, len(stream), 333):
        for ev in r.feed(stream[off : off + 333]):
            if ev[0] == "cluster":
                poses_at_cluster.append(len(r.frames))
    assert poses_at_cluster and poses_at_cluster[0] >= 1
    assert len(r.frames) == 10

    # crash: drop the last 40% of the stream; poses flushed so far must survive
    cut = stream[: int(len(stream) * 0.6)]
    r2 = StreamReader()
    for off in range(0, len(cut), 4096):
        r2.feed(cut[off : off + 4096])
    assert r2.doc is not None
    assert len(r2.frames) >= 1  # everything up to the last flushed chunk

    # a truncated file is still a readable wurld file (chunk precedence)
    p = tmp_path / "crash.wl.webm"
    p.write_bytes(cut)
    seq = wl.read(p)
    assert 1 <= len(seq.frames) <= 10


def test_streamwriter_ffmpeg_valid(scene, tmp_path):
    if shutil.which("ffmpeg") is None:
        pytest.skip("no ffmpeg")
    parts = []
    _record(scene, parts)
    p = tmp_path / "live.webm"
    p.write_bytes(b"".join(parts))
    res = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(p), "-f", "null", "-"],
        capture_output=True, text=True,
    )
    assert res.returncode == 0, res.stderr
    assert res.stderr.strip() == ""


def test_streamwriter_rejects_bad_quaternion(scene):
    parts = []
    # note: chromapakz create_encoder requires >=1 signal even with has_rgb
    w = StreamWriter(
        parts.append, cameras=scene["cameras"], has_rgb=True,
        signal_meta=[wl.SignalMeta("depth", "depth", {"type": "identity"})],
    )
    bad = wl.Frame(i=0, t=0.0, q_wxyz=(2.0, 0, 0, 0), tr=(0, 0, 0))
    with pytest.raises(ValueError, match="not unit"):
        w.add_frame(bad, rgb=scene["rgba"][0], signals={"depth": {"u16": scene["d16"][0]}})
