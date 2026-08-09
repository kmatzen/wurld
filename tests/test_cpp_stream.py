"""The C++ streaming writer, against real encoder chunks.

`attach()` rebuilds a whole file in memory, which is the finalise-a-clip case
and useless to a robot recording for hours. `StreamWriter` is the incremental
form: metadata woven between the encoder's Clusters as they arrive.

The chunks here come from chromapakz's real streaming encoder, captured in
Python, because the C++ side deliberately cannot encode video — that is the
dependency split the whole C++ layer exists to preserve. Feeding recorded chunks
is exactly what a robot's encoder callback does, with the callback replaced by a
directory.

Correctness is defined against the Python StreamWriter: given the same frames,
the two must produce files that read identically. A C++ writer that only agrees
with the C++ reader would prove nothing.
"""

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

import wurld as wl
from wurld import ebml, validate as v
from wurld.stream import StreamWriter

ROOT = Path(__file__).resolve().parents[1]
CPP = ROOT / "cpp"
W, H, N, FPS = 64, 48, 100, 30

pytestmark = pytest.mark.skipif(
    shutil.which("cmake") is None, reason="cmake is needed to build the C++ writer")


@pytest.fixture(scope="module")
def weave_tool(tmp_path_factory):
    build = tmp_path_factory.mktemp("cppstream")
    r = subprocess.run(["cmake", "-S", str(CPP), "-B", str(build),
                        "-DCMAKE_BUILD_TYPE=Release"], capture_output=True, text=True)
    if r.returncode != 0:
        pytest.skip(f"cmake configure failed:\n{r.stderr}")
    r = subprocess.run(["cmake", "--build", str(build), "-j", "4"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    return build / "wurld_weave"


def _frames():
    out = []
    for i in range(N):
        ang = 0.08 * i
        out.append(wl.Frame(i=i, t=i / FPS, camera="0",
                            q_wxyz=(float(np.cos(ang / 2)), 0.0, float(np.sin(ang / 2)), 0.0),
                            tr=(0.03 * i, -0.01 * i, 1.2)))
    return out


def _pixels():
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    return np.stack([
        np.dstack([np.clip((0.5 + 0.4 * np.sin(xx / 5 + i * 0.3)) * 255, 0, 255)] * 3
                  + [np.full((H, W), 255, np.float32)]).astype(np.uint8)
        for i in range(N)])


CAMERAS = {"0": wl.Camera("PINHOLE", W, H, [70.4, 70.4, W / 2, H / 2])}
WORLD = {"metric_scale": True, "description": "c++ streaming writer"}


@pytest.fixture(scope="module")
def captured(tmp_path_factory):
    """Real chromapakz encoder chunks, plus the file Python wove from them."""
    import chromapakz as cz

    d = tmp_path_factory.mktemp("chunks")
    chunkdir = d / "chunks"
    chunkdir.mkdir()

    frames, rgb = _frames(), _pixels()

    # 1. Capture the encoder's chunk sequence on its own.
    chunks = []
    enc = cz.create_encoder(W, H, signals=[], fps=FPS, has_rgb=True,
                            on_chunk=chunks.append, cues=False)
    for i in range(N):
        enc.add_frame(rgb=rgb[i], signals={})
    enc.finish()
    for k, c in enumerate(chunks):
        (chunkdir / f"chunk_{k:05d}.bin").write_bytes(c)

    # 2. The Python StreamWriter's output for the same frames, as the reference.
    py_out = d / "python.wl.webm"
    with open(py_out, "wb") as fh:
        w = StreamWriter(fh.write, cameras=CAMERAS, world=WORLD, fps=FPS, has_rgb=True)
        for i in range(N):
            w.add_frame(frames[i], rgb=rgb[i])
        w.finish()

    spec = {
        "cameras": {"0": {"model": "PINHOLE", "width": W, "height": H,
                          "params": [70.4, 70.4, W / 2, H / 2]}},
        "frames": [{"i": f.i, "t": f.t, "camera": f.camera,
                    "q_wxyz": list(f.q_wxyz), "tr": list(f.tr)} for f in frames],
        "world": WORLD,
        # Chunks after the first are Clusters; the encoder groups frames into
        # them, and the writer must see poses before the Cluster carrying them.
        "frames_per_chunk": max(1, N // max(1, len(chunks) - 1)),
    }
    (d / "spec.json").write_text(json.dumps(spec))
    return d, chunkdir, py_out, frames, rgb, len(chunks)


@pytest.fixture(scope="module")
def cpp_out(weave_tool, captured):
    d, chunkdir, _py, _frames_, _rgb, _n = captured
    out = d / "cpp.wl.webm"
    r = subprocess.run([str(weave_tool), str(chunkdir), str(out), str(d / "spec.json")],
                       capture_output=True, text=True)
    assert r.returncode == 0, f"wurld_weave failed:\n{r.stdout}\n{r.stderr}"
    return out, r.stdout


def test_the_encoder_really_produced_several_chunks(captured):
    """One chunk means nothing is being interleaved and the rest is vacuous."""
    *_, n_chunks = captured
    assert n_chunks >= 3, f"only {n_chunks} encoder chunks; interleaving is untested"


def test_it_conforms(cpp_out):
    out, _ = cpp_out
    assert v.validate(out) == []


def test_poses_survive(cpp_out, captured):
    _, _, _, frames, _, _ = captured
    seq = wl.read(cpp_out[0])
    assert len(seq.frames) == N
    for got, want in zip(seq.frames, frames):
        assert got.i == want.i
        assert got.t == pytest.approx(want.t)
        assert got.camera == want.camera
        assert list(got.q_wxyz) == pytest.approx(list(want.q_wxyz), abs=1e-6)
        assert list(got.tr) == pytest.approx(list(want.tr), abs=1e-6)


def test_pixels_are_bit_exact(cpp_out, captured):
    """Weaving metadata must not disturb the encoder's Clusters.

    Compared against decoding the raw chunks — the encoder's own output, before
    any weaving — rather than against the source pixels, which the lossy display
    track never promised to reproduce.
    """
    import chromapakz as cz

    _, chunkdir, _py, _frames_, _rgb, n_chunks = captured
    raw = b"".join((chunkdir / f"chunk_{k:05d}.bin").read_bytes() for k in range(n_chunks))
    straight = cz.decode(raw)["rgb"]

    seq = wl.read(cpp_out[0])
    assert seq.rgb.shape == straight.shape
    assert np.array_equal(seq.rgb, straight), "weaving changed the decoded pixels"


def test_cpp_and_python_writers_agree(cpp_out, captured):
    """Same frames, same encoder chunks: the two files must read the same."""
    _, _, py_out, _, _, _ = captured
    a, b = wl.read(py_out), wl.read(cpp_out[0])

    assert len(a.frames) == len(b.frames)
    for x, y in zip(a.frames, b.frames):
        assert (x.i, x.camera, x.pose_valid) == (y.i, y.camera, y.pose_valid)
        assert x.t == pytest.approx(y.t)
        assert list(x.q_wxyz) == pytest.approx(list(y.q_wxyz))
        assert list(x.tr) == pytest.approx(list(y.tr))
    assert a.world == b.world
    assert sorted(a.cameras) == sorted(b.cameras)
    assert np.array_equal(a.rgb, b.rgb), "pixels differ between the two writers"


def test_layout_is_the_live_form(cpp_out):
    """Poses ahead of their Clusters, a consolidated table at the end, no Cues."""
    data = cpp_out[0].read_bytes()
    _, ps, pe = ebml._segment_bounds(data)
    order = []
    for eid, _es, pstart, pend in ebml._top_level(data, ps, pe):
        if eid == ebml.CLUSTER:
            order.append("cluster")
        elif eid == ebml.TAGS:
            names = [n for n, _ in ebml.collect_simple_tags(data, pstart, pend)]
            order.append("+".join(sorted(names)))
        elif eid == ebml.CUES:
            order.append("cues")

    assert "cues" not in order, "interleaved tags move Clusters; Cues would be stale"
    assert order.count("WURLD_FRAMES") == 1, "expected one consolidated table"
    assert order[-1] == "WURLD_FRAMES", "the table must come last"
    assert "WURLD_POSES" in order, "no streamed pose chunks — nothing was interleaved"
    # Every pose chunk precedes a Cluster, which is what makes an interrupted
    # recording readable.
    for k, item in enumerate(order):
        if item == "WURLD_POSES":
            assert "cluster" in order[k + 1:k + 2], f"pose chunk at {k} not before a Cluster"


def test_an_interrupted_recording_still_reads(cpp_out):
    """The point of streaming poses ahead of Clusters: a killed process leaves data.

    Truncating after the last Cluster removes the consolidated table, so a reader
    must fall back to the WURLD_POSES chunks (SPEC §9 precedence).
    """
    data = cpp_out[0].read_bytes()
    _, ps, pe = ebml._segment_bounds(data)
    last_cluster_end = max(pend for eid, _es, _p, pend in ebml._top_level(data, ps, pe)
                           if eid == ebml.CLUSTER)
    cut = cpp_out[0].with_name("killed.wl.webm")
    cut.write_bytes(data[:last_cluster_end])

    tags = ebml.read_all_tags(cut.read_bytes())
    assert "WURLD_FRAMES" not in tags, "the fixture should have lost the table"
    assert isinstance(tags.get("WURLD_POSES"), bytes)
    seq = wl.read(cut)
    # Poses recorded before the interruption survive.
    assert len(seq.frames) > 0
    assert all(f.pose_valid for f in seq.frames)


def test_imu_is_interleaved_too(weave_tool, captured, tmp_path):
    d, chunkdir, _py, frames, _rgb, _n = captured
    rows = [[i * 0.005, 0.1, 0.0, 0.3, 0.0, 0.0, 9.81] for i in range(40)]
    spec = json.loads((d / "spec.json").read_text())
    spec["imu"] = {"imu0": rows}
    p = tmp_path / "imu.spec.json"
    p.write_text(json.dumps(spec))
    out = tmp_path / "imu.wl.webm"
    r = subprocess.run([str(weave_tool), str(chunkdir), str(out), str(p)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr

    seq = wl.read(out)
    assert seq.imu["imu0"].samples.shape == (40, 7)
    want = wl.ImuStream("imu0", np.array(rows)).pack()
    assert ebml.read_all_tags(out.read_bytes())["WURLD_IMU_imu0"] == want


def test_finish_without_any_chunk_is_refused(weave_tool, tmp_path):
    """Nothing was recorded; producing a file would be a lie."""
    (tmp_path / "empty").mkdir()
    spec = {"cameras": {"0": {"model": "PINHOLE", "width": W, "height": H,
                              "params": [70.4, 70.4, W / 2, H / 2]}},
            "frames": [], "world": WORLD}
    p = tmp_path / "s.json"
    p.write_text(json.dumps(spec))
    r = subprocess.run([str(weave_tool), str(tmp_path / "empty"),
                        str(tmp_path / "x.wl.webm"), str(p)],
                       capture_output=True, text=True)
    assert r.returncode == 1
    assert "no chunks" in r.stderr
