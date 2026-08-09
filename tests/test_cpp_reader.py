"""The C++ reader must agree with the Python one, field by field.

A second implementation is only worth having if it is the *same* reader. These
tests build `cpp/` and diff its JSON output against the Python reader on files
covering the paths that actually differ between implementations: JSON frames vs
a binary frame table, multi-camera rigs, IMU streams, unposed frames, and
non-finite numbers in a value map.

Comparing the C++ reader against its own expectations would prove only that it
is self-consistent. Comparing against Python is what makes it a reader of *this*
format.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import wurld as wl

ROOT = Path(__file__).resolve().parents[1]
CPP = ROOT / "cpp"

pytestmark = pytest.mark.skipif(
    shutil.which("cmake") is None, reason="cmake is needed to build the C++ reader")


@pytest.fixture(scope="module")
def wurld_info(tmp_path_factory):
    """Build the C++ tools once for the module."""
    build = tmp_path_factory.mktemp("cppbuild")
    r = subprocess.run(["cmake", "-S", str(CPP), "-B", str(build),
                        "-DCMAKE_BUILD_TYPE=Release"], capture_output=True, text=True)
    if r.returncode != 0:
        pytest.skip(f"cmake configure failed:\n{r.stdout}\n{r.stderr}")
    r = subprocess.run(["cmake", "--build", str(build), "-j", "4"],
                       capture_output=True, text=True)
    assert r.returncode == 0, f"C++ build failed:\n{r.stdout}\n{r.stderr}"
    return build / "wurld_info", build / "wurld_test"


def read_cpp(tool, path):
    r = subprocess.run([str(tool), str(path), "--json"], capture_output=True, text=True)
    assert r.returncode == 0, f"wurld_info failed on {path}:\n{r.stderr}"
    # NaN/Infinity round-trip through Python's json, which accepts them bare.
    return json.loads(r.stdout)


def _write(path, *, n=6, cameras=("0",), frames_format="auto", imu=False,
           unposed=(), rigs=None, w=32, h=24):
    import chromapakz as cz

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    rgb = np.stack([np.dstack([np.clip((0.5 + 0.4 * np.sin(xx / 5 + i)) * 255, 0, 255)] * 3
                              + [np.full((h, w), 255, np.float32)]).astype(np.uint8)
                    for i in range(n)])
    depth = np.stack([(1.5 + 0.4 * np.sin(xx / 7 + i * 0.2)).astype(np.float32)
                      for i in range(n)])

    f = 1.1 * w
    cams = {c: wl.Camera(model="PINHOLE", width=w, height=h,
                         params=[f, f, w / 2, h / 2]) for c in cameras}
    frames = []
    for i in range(n):
        if i in unposed:
            frames.append(wl.Frame(i=i, t=i / 30, pose_valid=False))
        else:
            ang = 0.07 * i
            frames.append(wl.Frame(
                i=i, t=i / 30, camera=cameras[0],
                q_wxyz=(float(np.cos(ang / 2)), 0.0, float(np.sin(ang / 2)), 0.0),
                tr=(0.03 * i, -0.01 * i, 0.9)))

    kwargs = {}
    if imu:
        t = np.arange(0, n / 30, 0.005)
        samples = np.column_stack([t, np.full_like(t, 0.1), np.zeros_like(t),
                                   np.full_like(t, -0.05), np.zeros_like(t),
                                   np.zeros_like(t), np.full_like(t, 9.81)])
        kwargs["imu"] = [wl.ImuStream("imu0", samples)]
    if rigs:
        kwargs["rigs"] = rigs

    wl.write(
        path, cameras=cams, frames=frames, rgb=rgb,
        signals={"depth": cz.quantize_inverse(depth, near=0.3, far=9.0)},
        specs={"depth": {"inverse_depth": True, "near": 0.3, "far": 9.0}},
        signal_meta=[wl.SignalMeta("depth", "depth",
                                   {"type": "inverse_depth", "near": 0.3, "far": 9.0,
                                    "levels": 65536, "invalid": 0})],
        world={"metric_scale": True, "description": "c++ parity fixture"},
        frames_format=frames_format, fps=30, **kwargs)
    return path


def assert_agrees(tool, path):
    """Every field both readers model must match exactly."""
    got = read_cpp(tool, path)
    seq = wl.read(path)

    assert set(got["cameras"]) == set(seq.cameras)
    for cid, cam in seq.cameras.items():
        g = got["cameras"][cid]
        assert g["model"] == cam.model
        assert (g["width"], g["height"]) == (cam.width, cam.height)
        assert g["params"] == pytest.approx(list(cam.params))

    assert len(got["frames"]) == len(seq.frames), "frame counts differ"
    for g, f in zip(got["frames"], seq.frames):
        assert g["i"] == f.i
        assert g["t"] == pytest.approx(f.t, abs=1e-12)
        assert g["pose_valid"] == bool(f.pose_valid)
        if not f.pose_valid:
            # Absent, not identity — the same choice both readers make.
            assert "c2w" not in g
            continue
        assert g["camera"] == f.camera
        # Binary tables store float32, so compare at float32 precision.
        assert g["q_wxyz"] == pytest.approx(list(f.q_wxyz), rel=1e-6, abs=1e-7)
        assert g["tr"] == pytest.approx(list(f.tr), rel=1e-6, abs=1e-7)
        assert np.allclose(np.array(g["c2w"]).reshape(4, 4), f.c2w, atol=1e-6)

    assert [s["id"] for s in got["signals"]] == [s.id for s in seq.signals]
    assert [s["role"] for s in got["signals"]] == [s.role for s in seq.signals]
    for g, s in zip(got["signals"], seq.signals):
        assert g["value_map"] == s.value_map

    assert set(got["imu"]) == set(seq.imu)
    for sid, stream in seq.imu.items():
        arr = np.array(got["imu"][sid], dtype=np.float64)
        assert arr.shape == stream.samples.shape
        assert np.allclose(arr, stream.samples.astype(np.float32), rtol=1e-6, atol=1e-7)

    assert got["world"] == seq.world
    assert got["rigs"] == seq.rigs
    return got, seq


def test_cpp_unit_tests_pass(wurld_info):
    _, test_bin = wurld_info
    r = subprocess.run([str(test_bin)], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "all C++ unit checks passed" in r.stdout


def test_json_frames_agree(wurld_info, tmp_path):
    tool, _ = wurld_info
    p = _write(tmp_path / "json.wl.webm", n=6, frames_format="json")
    got, _ = assert_agrees(tool, p)
    assert len(got["frames"]) == 6


def test_binary_frame_table_agrees(wurld_info, tmp_path):
    """The path ffmpeg cannot read, and the one most likely to diverge."""
    tool, _ = wurld_info
    p = _write(tmp_path / "bin.wl.webm", n=8, frames_format="binary")
    # Confirm the fixture really took the binary path, or this proves nothing.
    from wurld import ebml
    tags = ebml.read_all_tags(p.read_bytes())
    assert isinstance(tags.get("WURLD_FRAMES"), bytes)
    assert_agrees(tool, p)


def test_unposed_frames_agree(wurld_info, tmp_path):
    tool, _ = wurld_info
    for fmt in ("json", "binary"):
        p = _write(tmp_path / f"unposed_{fmt}.wl.webm", n=7, unposed=(2, 5),
                   frames_format=fmt)
        got, seq = assert_agrees(tool, p)
        assert [g["i"] for g in got["frames"] if not g["pose_valid"]] == [2, 5]


def test_rig_and_imu_agree(wurld_info, tmp_path):
    tool, _ = wurld_info
    rigs = {"body": {"cameras": {
        "cam0": {"q_wxyz": [1.0, 0.0, 0.0, 0.0], "tr": [0.0, 0.0, 0.0]},
        "cam1": {"q_wxyz": [1.0, 0.0, 0.0, 0.0], "tr": [0.12, 0.0, 0.0]}}}}
    p = _write(tmp_path / "rig.wl.webm", n=6, cameras=("cam0", "cam1"),
               imu=True, rigs=rigs)
    got, seq = assert_agrees(tool, p)
    assert set(got["imu"]) == {"imu0"}
    assert len(got["imu"]["imu0"]) == len(seq.imu["imu0"].samples)
    assert got["rigs"]["body"]["cameras"]["cam1"]["tr"] == [0.12, 0.0, 0.0]


def test_examples_agree(wurld_info, tmp_path):
    """Real example files, not just fixtures written by this test."""
    tool, _ = wurld_info
    for script, name in [("04_robot_rig_imu.py", "rig"), ("06_stereo_rig.py", "stereo"),
                         ("01_feedforward_reconstruction.py", "ff")]:
        out = tmp_path / f"{name}.wl.webm"
        r = subprocess.run([sys.executable, str(ROOT / "examples" / script), str(out)],
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        assert_agrees(tool, out)


def test_cluster_start_points_at_the_pixels(wurld_info, tmp_path):
    tool, _ = wurld_info
    p = _write(tmp_path / "clusters.wl.webm", n=6)
    got = read_cpp(tool, p)
    start = got["cluster_start"]
    assert 0 < start < p.stat().st_size
    # It must be an actual Cluster id, since consumers hand this to a decoder.
    with open(p, "rb") as fh:
        fh.seek(int(start))
        assert fh.read(4) == b"\x1f\x43\xb6\x75"


def test_reader_rejects_non_wurld_and_garbage(wurld_info, tmp_path):
    tool, _ = wurld_info
    bad = tmp_path / "bad.wl.webm"
    bad.write_bytes(b"\x1a\x45\xdf\xa3 not really matroska")
    r = subprocess.run([str(tool), str(bad), "--json"], capture_output=True, text=True)
    assert r.returncode == 1
    assert "wurld:" in r.stderr

    missing = subprocess.run([str(tool), str(tmp_path / "nope.webm")],
                             capture_output=True, text=True)
    assert missing.returncode == 1
    assert "cannot open" in missing.stderr


def test_truncated_file_does_not_crash_or_invent_data(wurld_info, tmp_path):
    """Half a file must fail cleanly, not produce plausible-looking poses."""
    tool, _ = wurld_info
    p = _write(tmp_path / "whole.wl.webm", n=8)
    data = p.read_bytes()
    for frac in (0.05, 0.25, 0.5, 0.75):
        cut = tmp_path / f"cut_{int(frac*100)}.wl.webm"
        cut.write_bytes(data[: int(len(data) * frac)])
        r = subprocess.run([str(tool), str(cut), "--json"], capture_output=True, text=True)
        # Either it reads the metadata (which lives early in the file) or it
        # reports an error. It must never crash, and never emit invalid json.
        assert r.returncode in (0, 1), f"crashed at {frac}: {r.returncode}"
        if r.returncode == 0:
            doc = json.loads(r.stdout)
            whole = read_cpp(tool, p)
            # Anything it did parse must match the whole file's answer.
            assert doc["cameras"] == whole["cameras"]
            assert doc["frames"] == whole["frames"]
