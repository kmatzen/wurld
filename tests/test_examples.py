"""The runnable examples in USE_CASES.md must keep running.

A worked example that has rotted is worse than none: it is a claim the format
does something, sitting in the repository, untrue. These execute each script and
check the artefact it produces conforms to SPEC.
"""

import subprocess
import sys
from pathlib import Path

import pytest

import wurld as wl
from wurld import validate as v

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def _run(script, *args):
    r = subprocess.run([sys.executable, str(EXAMPLES / script), *map(str, args)],
                       capture_output=True, text=True)
    assert r.returncode == 0, f"{script} failed:\n{r.stdout}\n{r.stderr}"
    return r.stdout


def test_feedforward_reconstruction(tmp_path):
    out = tmp_path / "ff.wl.webm"
    stdout = _run("01_feedforward_reconstruction.py", out)
    assert v.validate(out) == []

    seq = wl.read(out)
    lost = [f.i for f in seq.frames if not f.pose_valid]
    # The scenario's whole point: unlocalised frames survive as unlocalised.
    assert lost == [9, 10, 11]
    assert seq.world["metric_scale"] is False
    assert seq.signal("confidence") is not None
    assert "up to scale" in stdout or "metric_scale = False" in stdout


def test_gaussian_splatting_export(tmp_path):
    src = Path("docs/samples/synthetic-orbit.wl.webm")
    if not src.exists():
        pytest.skip("hosted sample not present")
    outdir = tmp_path / "gs"
    stdout = _run("02_gaussian_splatting.py", src, outdir)
    # The axis assertion lives inside the example; if it had failed the run
    # would have raised, but check the artefacts landed too.
    assert (outdir / "transforms.json").exists()
    assert len(list((outdir / "depth").glob("*.npy"))) > 0
    assert "verified" in stdout


def test_slam_trajectory_export(tmp_path):
    src = Path("docs/samples/synthetic-orbit.wl.webm")
    if not src.exists():
        pytest.skip("hosted sample not present")
    outdir = tmp_path / "slam"
    _run("03_slam_evaluation.py", src, outdir)
    lines = [l for l in (outdir / "trajectory.txt").read_text().splitlines()
             if not l.startswith("#")]
    assert lines
    # TUM convention: 8 fields, scalar-last quaternion.
    for line in lines[:5]:
        assert len(line.split()) == 8


def test_robot_rig_and_imu(tmp_path):
    out = tmp_path / "rig.wl.webm"
    _run("04_robot_rig_imu.py", out)
    assert v.validate(out) == []

    seq = wl.read(out)
    assert set(seq.cameras) == {"cam0", "cam1"}
    # cam1 has no stored poses; its pose is derived from the rig calibration.
    assert all(f.camera == "cam0" for f in seq.frames)
    import numpy as np
    baseline = np.linalg.norm(seq.rig_c2w(5, "cam1")[:3, 3] - seq.c2w(5)[:3, 3])
    assert abs(baseline - 0.12) < 1e-4
    # IMU stays at its own rate rather than being resampled to the frame rate.
    assert seq.imu["imu0"].samples.shape[0] > len(seq.frames) * 5
