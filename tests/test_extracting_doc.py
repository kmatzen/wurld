"""EXTRACTING.md's no-install recipe, executed rather than believed.

That document tells people how to read a wurld file with nothing but ffmpeg, so
a wrong instruction there costs more than a wrong docstring: the reader has no
wurld install to check against, and the failure is silent. Extracting the wrong
track yields a colour plane reinterpreted as depth codes, which dequantizes into
plausible nonsense rather than an error.

This runs the documented commands and compares the result against the Python
reader. It also runs them on a *stereo* file, because the original instructions
hardcoded `0:v:1` for the depth plane — which is the second camera's colour when
there are two cameras, and shifts again when a file carries confidence.
"""

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import wurld as wl

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None or
                                shutil.which("ffprobe") is None,
                                reason="ffmpeg/ffprobe not installed")


def _track_index(path, title):
    """The documented way to find a plane: by title, never by position."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=index:stream_tags=title",
         "-of", "csv=p=0", str(path)], capture_output=True, text=True, check=True).stdout
    for line in out.strip().splitlines():
        idx, _, name = line.partition(",")
        if name.strip() == title:
            return int(idx)
    raise AssertionError(f"{path}: no track titled {title!r}; got\n{out}")


def _plane(path, index, h, w, tmp):
    raw = tmp / f"p{index}.raw"
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-map", f"0:{index}",
                    "-frames:v", "1", "-pix_fmt", "gray", "-f", "rawvideo",
                    str(raw), "-y"], check=True)
    return np.fromfile(raw, np.uint8).reshape(h, w).astype(np.uint16)


def _depth_via_ffmpeg(path, tmp):
    seq = wl.read(path)
    h, w = seq.probe["height"], seq.probe["width"]
    hi = _plane(path, _track_index(path, "signal-depth-hi"), h, w, tmp)
    lo = _plane(path, _track_index(path, "signal-depth-lo"), h, w, tmp)

    # tri-fold-8+8, exactly as EXTRACTING.md states it.
    low = np.where(hi & 1, 255 - lo, lo)
    code = (hi << 8) | low

    vm = next(s.value_map for s in seq.signals if s.role == "depth")
    near, far = vm["near"], vm["far"]
    levels = vm.get("levels", 65536)
    out = np.full(code.shape, np.nan, np.float64)
    ok = code != vm.get("invalid", 0)
    c = code[ok].astype(np.float64)
    out[ok] = 1.0 / (((c - 1) / (levels - 2)) * (1 / near - 1 / far) + 1 / far)
    return out, seq


@pytest.mark.parametrize("script,name", [
    ("01_feedforward_reconstruction.py", "mono_with_confidence"),
    ("06_stereo_rig.py", "stereo"),
])
def test_the_documented_extraction_reproduces_the_reader(script, name, tmp_path):
    src = tmp_path / f"{name}.wl.webm"
    r = subprocess.run([sys.executable, str(EXAMPLES / script), str(src)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr

    got, seq = _depth_via_ffmpeg(src, tmp_path)
    want = seq.depth_meters(0)

    both = np.isfinite(got) & np.isfinite(want)
    assert both.sum() > 0
    # float32 storage of the dequantized value is the floor here.
    assert np.abs(got[both] - want[both]).max() < 1e-5
    assert np.array_equal(np.isnan(got), np.isnan(want)), "invalid pixels differ"


def test_depth_is_not_at_a_fixed_track_index(tmp_path):
    """Why the recipe resolves by title: the position genuinely moves.

    A mono capture with confidence puts depth-hi at v:1; a stereo capture puts
    the second camera's colour there. Hardcoding the index gives no error, just
    wrong numbers.
    """
    positions = {}
    for script, name in [("01_feedforward_reconstruction.py", "mono"),
                         ("06_stereo_rig.py", "stereo")]:
        src = tmp_path / f"{name}.wl.webm"
        subprocess.run([sys.executable, str(EXAMPLES / script), str(src)],
                       capture_output=True, check=True)
        positions[name] = _track_index(src, "signal-depth-hi")

    assert positions["mono"] != positions["stereo"], (
        f"depth-hi sits at {positions} — if these ever match, the warning in "
        "EXTRACTING.md is still right in principle but this test proves nothing")
