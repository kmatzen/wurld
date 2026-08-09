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


def test_stereo_rig(tmp_path):
    out = tmp_path / "stereo.wl.webm"
    stdout = _run("06_stereo_rig.py", out)
    assert v.validate(out) == []

    import numpy as np
    seq = wl.read(out)
    assert seq.rgb_streams == ["cam0", "cam1"]
    # Stream ids are camera ids — that binding is the whole mechanism (SPEC §4.4).
    assert set(seq.rgb_streams) <= set(seq.cameras)
    # Two genuinely different views, not the same buffer twice.
    assert not np.array_equal(seq.rgb_for("cam0"), seq.rgb_for("cam1"))
    # cam1 never had a pose stored; it comes from the rig.
    assert all(f.camera == "cam0" for f in seq.frames)
    baseline = np.linalg.norm(seq.rig_c2w(5, "cam1")[:3, 3] - seq.c2w(5)[:3, 3])
    assert abs(baseline - 0.12) < 1e-4
    assert "primary first" in stdout


def test_collection_training(tmp_path):
    stdout = _run("08_collection_training.py", tmp_path)
    # The example verifies its own sharding and would exit 1 on a failure, but
    # assert the conclusion is actually printed rather than trusting the code.
    assert "0 duplicated, 0 missing" in stdout
    assert "every frame exactly once" in stdout

    from wurld import collection as col
    c = col.Collection.read(tmp_path / "collection.json")
    assert len(c.members) == 4
    assert len(c) == c.manifest.total_frames > 0


def test_hdr_exr_render(tmp_path):
    out = tmp_path / "hdr.wl.webm"
    stdout = _run("05_hdr_exr_render.py", out)
    assert v.validate(out) == []

    import numpy as np
    seq = wl.read(out)
    # float16_bits is a reinterpretation, not a conversion: the codes are the float.
    vals = seq.signal_values("hdr_r")
    assert vals.dtype == np.float16
    assert float(np.nanmax(vals)) > 1000.0          # HDR range survived, not clipped
    assert "bit-exact round trip: True" in stdout
    # The honest framing must stay in the output — it can lose to EXR.
    assert "LARGER than EXR" in stdout


def test_float16_bits_carries_the_float_edge_cases():
    """NaN, infinities, -0.0 and denormals are all just bit patterns here."""
    import numpy as np
    from wurld.container import SignalMeta
    m = SignalMeta("x", "custom", {"type": "float16_bits"})
    src = np.array([np.nan, np.inf, -np.inf, -0.0, 0.0, 6e-8, 65504.0], np.float16)
    back = m.apply(src.view(np.uint16))
    assert back.dtype == np.float16
    assert (back.view(np.uint16) == src.view(np.uint16)).all()


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


def _exporters():
    from wurld.converters import colmap, nerfstudio, tum
    return [("tum", tum.to_tum), ("nerfstudio", nerfstudio.to_transforms),
            ("colmap", colmap.to_colmap)]


def test_feedforward_output_can_reach_every_exporter(tmp_path, caplog):
    """Scenario 1 must be able to feed scenario 2.

    A feed-forward pass leaves frames it could not localise, and exporting those
    used to raise "frame 9: pose not valid" from nerfstudio and COLMAP — so the
    format's flagship producer could not reach its flagship consumer.
    """
    import logging

    src = tmp_path / "ff.wl.webm"
    _run("01_feedforward_reconstruction.py", src)
    seq = wl.read(src)
    unposed = sum(1 for f in seq.frames if not f.pose_valid)
    assert unposed > 0, "the fixture must contain unposed frames"

    for name, fn in _exporters():
        with caplog.at_level(logging.WARNING):
            fn(src, tmp_path / f"out_{name}")

    # transforms.json must contain the posed frames and only those.
    import json
    doc = json.loads((tmp_path / "out_nerfstudio" / "transforms.json").read_text())
    assert len(doc["frames"]) == len(seq.frames) - unposed
    # And the drop must be reported, not silent.
    assert any("had no pose" in r.getMessage() for r in caplog.records)


def test_a_signals_only_file_is_refused_with_a_reason(tmp_path):
    """rgb=None is legal (scene-referred HDR); 'NoneType is not subscriptable' is not."""
    src = tmp_path / "norgb.wl.webm"
    _run("05_hdr_exr_render.py", src)
    assert wl.read(src).rgb is None

    for name, fn in _exporters():
        with pytest.raises(ValueError) as exc:
            fn(src, tmp_path / f"none_{name}")
        assert "no display track" in str(exc.value), name
