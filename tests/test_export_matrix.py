"""Every exporter, against every kind of legal wurld file.

Each side of this was covered and the combination was not, which is how a
feed-forward capture came to be unexportable to nerfstudio and COLMAP, a
signals-only file died with "'NoneType' object is not subscriptable", and an HDR
file died inside PIL. Three separate defects, one missing test.

The invariant is deliberately weak, because some combinations genuinely cannot
work: an exporter may succeed, or refuse with a `ValueError` that names the
problem. What it may not do is fail with an internal error — a TypeError from
PIL or an IndexError from numpy tells the caller nothing and reads as a bug in
wurld even when the request was impossible.
"""

import logging
import subprocess
import sys
from pathlib import Path

import pytest

import wurld as wl

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"

# One example per shape of file that has bitten us.
KINDS = {
    "unposed": "01_feedforward_reconstruction.py",   # pose_valid=False frames
    "rig_imu": "04_robot_rig_imu.py",                # rigs + IMU + depth
    "no_rgb": "05_hdr_exr_render.py",                # signals only, rgb=None
    "stereo": "06_stereo_rig.py",                    # two display streams
}

# An HDR display track decodes to uint16 and has no example script, so it is
# built here. It is the shape that broke four exporters inside PIL.
HDR_KIND = "hdr"


def _exporters():
    from wurld.converters import colmap, nerfstudio, tum
    out = [("tum", lambda s, d: tum.to_tum(s, d)),
           ("nerfstudio", lambda s, d: nerfstudio.to_transforms(s, d)),
           ("colmap", lambda s, d: colmap.to_colmap(s, d))]
    try:
        from wurld.converters import mcap_export
        def _mcap(s, d):
            Path(d).mkdir(parents=True, exist_ok=True)
            return mcap_export.to_mcap(s, Path(d) / "out.mcap")

        out.append(("mcap", _mcap))
    except ImportError:
        pass
    try:
        import rosbags  # noqa: F401

        from wurld.converters import ros2
        out.append(("rosbag2", lambda s, d: ros2.to_rosbag2(s, d)))
    except ImportError:
        pass
    return out


@pytest.fixture(scope="module")
def sources(tmp_path_factory):
    import numpy as np

    d = tmp_path_factory.mktemp("matrix")
    made = {}
    for kind, script in KINDS.items():
        out = d / f"{kind}.wl.webm"
        r = subprocess.run([sys.executable, str(EXAMPLES / script), str(out)],
                           capture_output=True, text=True)
        assert r.returncode == 0, f"{script} failed:\n{r.stderr}"
        made[kind] = out

    w, h, n = 32, 24, 4
    hdr = d / "hdr.wl.webm"
    f = 1.1 * w
    wl.write(hdr,
             cameras={"0": wl.Camera("PINHOLE", w, h, [f, f, w / 2, h / 2])},
             frames=[wl.Frame(i=i, t=i / 30, camera="0", q_wxyz=(1.0, 0.0, 0.0, 0.0),
                              tr=(0.01 * i, 0.0, 1.0)) for i in range(n)],
             rgb=np.stack([np.full((h, w, 4), 400 + 40 * i, np.uint16) for i in range(n)]),
             hdr={"transfer": "pq"}, world={"metric_scale": True}, fps=30)
    made[HDR_KIND] = hdr
    return made


def test_the_fixtures_really_are_different_shapes(sources):
    """Otherwise the matrix below is one case run four times."""
    unposed = wl.read(sources["unposed"])
    assert any(not f.pose_valid for f in unposed.frames)
    assert wl.read(sources["no_rgb"]).rgb is None
    assert len(wl.read(sources["stereo"]).rgb_streams) == 2
    assert wl.read(sources["rig_imu"]).imu
    import numpy as np
    assert wl.read(sources[HDR_KIND]).rgb.dtype == np.uint16


@pytest.mark.parametrize("kind", sorted(list(KINDS) + [HDR_KIND]))
def test_no_exporter_fails_with_an_internal_error(kind, sources, tmp_path, caplog):
    src = sources[kind]
    problems = []
    for name, fn in _exporters():
        dest = tmp_path / f"{kind}_{name}"
        try:
            with caplog.at_level(logging.WARNING):
                fn(src, dest)
        except ValueError as e:
            # A refusal is fine, but it has to explain itself.
            if len(str(e)) < 30:
                problems.append(f"{name}: terse refusal {e!r}")
        except NotImplementedError as e:
            if len(str(e)) < 30:
                problems.append(f"{name}: terse refusal {e!r}")
        except Exception as e:                       # noqa: BLE001 - that is the point
            problems.append(f"{name}: {type(e).__name__}: {str(e)[:120]}")
    assert not problems, (
        f"exporting a {kind!r} file failed with internal errors:\n  "
        + "\n  ".join(problems))
