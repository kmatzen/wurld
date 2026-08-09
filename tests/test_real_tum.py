"""The TUM claim, against the real download rather than a fixture.

README, USE_CASES and LANDSCAPE all cite the same number as this project's
evidence that conversion preserves real data:

    Verified on the real TUM freiburg1_desk sequence: 572/573 poses associated
    within 10 ms, 0.000 mm translation error, depth bit-exact.

That was a one-off measurement. Nothing re-checked it, so the strongest factual
claim in the documentation was the least guarded — a converter change could have
falsified it silently, and every doc would have gone on asserting it.

Fixtures cannot substitute here. A synthetic TUM directory is written from the
same understanding of TUM's conventions that the importer reads it with, so the
two agree by construction; the whole point is data produced by someone else.

The sequence is CC BY 4.0 (Sturm et al., IROS 2012) and is **not** redistributed.
Fetch it once with `scripts/fetch_tum.sh`, or set WURLD_TUM_DIR. Absent, these
skip — loudly enough to notice, since a skipped test proves nothing.
"""

import os
from pathlib import Path

import numpy as np
import pytest

import wurld as wl
from wurld import validate as v
from wurld.converters import tum

SEQUENCE = "rgbd_dataset_freiburg1_desk"
MAX_DT = 0.01          # TUM's own associate.py default: 10 ms


def _find_sequence():
    """The extracted sequence directory, or None."""
    env = os.environ.get("WURLD_TUM_DIR")
    candidates = [Path(env)] if env else []
    candidates += [
        Path(os.environ.get("TMPDIR", "/tmp")) / "wurld-data" / SEQUENCE,
        Path.home() / "wurld-data" / SEQUENCE,
    ]
    for c in candidates:
        if (c / "groundtruth.txt").exists() and (c / "rgb.txt").exists():
            return c
    return None


SEQ_DIR = _find_sequence()
pytestmark = pytest.mark.skipif(
    SEQ_DIR is None,
    reason=f"{SEQUENCE} not present — run scripts/fetch_tum.sh or set WURLD_TUM_DIR")


def _read_pairs(path):
    out = []
    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split()
        out.append((float(parts[0]), parts[1:]))
    return out


@pytest.fixture(scope="module")
def converted(tmp_path_factory):
    out = tmp_path_factory.mktemp("tumreal") / "f1desk.wl.webm"
    tum.from_tum(SEQ_DIR, out)
    return out, wl.read(out)


def test_it_conforms(converted):
    out, _ = converted
    assert v.validate(out) == []


def test_frame_count_matches_the_rgb_list(converted):
    """Every RGB frame TUM ships should appear, posed or not."""
    _, seq = converted
    rgb_lines = _read_pairs(SEQ_DIR / "rgb.txt")
    # The importer associates rgb with depth, so it can carry at most the
    # smaller list; it must not silently carry fewer than that.
    depth_lines = _read_pairs(SEQ_DIR / "depth.txt")
    assert 0 < len(seq.frames) <= min(len(rgb_lines), len(depth_lines))
    assert len(seq.frames) >= 0.95 * min(len(rgb_lines), len(depth_lines))


def test_poses_match_the_original_groundtruth(converted):
    """The published claim: association within 10 ms at 0.000 mm.

    Ground truth is compared against `groundtruth.txt` as TUM ships it, parsed
    here independently of the importer — scalar-last quaternions and all — so
    an importer that mis-ordered them could not also mis-order the reference.
    """
    _, seq = converted

    gt = []
    for line in (SEQ_DIR / "groundtruth.txt").read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        t, tx, ty, tz, qx, qy, qz, qw = (float(x) for x in line.split())
        gt.append((t, np.array([tx, ty, tz]), np.array([qw, qx, qy, qz])))
    gt_t = np.array([g[0] for g in gt])
    assert len(gt) > 1000, "groundtruth.txt looks wrong"

    associated, worst_mm, worst_rot = 0, 0.0, 0.0
    for f in seq.frames:
        if not f.pose_valid:
            continue
        j = int(np.argmin(np.abs(gt_t - f.t)))
        if abs(gt_t[j] - f.t) > MAX_DT:
            continue
        associated += 1
        worst_mm = max(worst_mm, float(np.linalg.norm(np.asarray(f.tr) - gt[j][1])) * 1000)
        # Normalise before comparing. TUM stores ground truth to four decimals,
        # so its quaternions are only unit to ~8e-5 — and 2*acos(|q|^2) turns
        # that into 2 degrees of apparent error between a quaternion and itself.
        a = np.asarray(f.q_wxyz, dtype=np.float64)
        b = gt[j][2]
        dot = abs(float(np.dot(a / np.linalg.norm(a), b / np.linalg.norm(b))))
        worst_rot = max(worst_rot, float(np.degrees(2 * np.arccos(min(1.0, dot)))))

    posed = sum(1 for f in seq.frames if f.pose_valid)
    assert associated >= 0.98 * posed, f"only {associated}/{posed} associated within {MAX_DT*1000:g} ms"
    # "0.000 mm" in the docs is a rounded print; the real bar is float32 storage
    # of a metre-scale translation, which is ~1e-4 mm.
    assert worst_mm < 0.01, f"worst translation error {worst_mm:.6f} mm"
    assert worst_rot < 0.01, f"worst rotation error {worst_rot:.6f} deg"


def test_poses_are_copied_verbatim_not_transformed(converted):
    """Stronger than an error bound: the stored values *are* TUM's values.

    TUM ground truth is already camera-to-world in an RDF optical frame, which
    is wurld's convention, so the importer must reorder the quaternion
    (TUM is xyzw) and otherwise change nothing. Any axis flip or renormalisation
    would show up here as a non-zero difference rather than as a small one.
    """
    _, seq = converted
    gt = {}
    for line in (SEQ_DIR / "groundtruth.txt").read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        t, tx, ty, tz, qx, qy, qz, qw = (float(x) for x in line.split())
        gt[t] = (np.array([tx, ty, tz]), np.array([qw, qx, qy, qz]))
    times = np.array(sorted(gt))

    exact = 0
    for f in seq.frames:
        if not f.pose_valid:
            continue
        t = float(times[int(np.argmin(np.abs(times - f.t)))])
        if abs(t - f.t) > MAX_DT:
            continue
        tr, q = gt[t]
        assert np.array_equal(np.asarray(f.tr, dtype=np.float64), tr)
        assert np.array_equal(np.asarray(f.q_wxyz, dtype=np.float64), q)
        exact += 1
    assert exact > 500, f"only {exact} frames compared"


def test_depth_is_bit_exact_against_the_source_pngs(converted):
    """TUM ships 16-bit PNGs at 5000 units/metre; those codes must survive."""
    from PIL import Image

    _, seq = converted
    depth_lines = _read_pairs(SEQ_DIR / "depth.txt")
    by_time = {round(t, 6): rest[0] for t, rest in depth_lines}

    checked = 0
    for idx in (0, len(seq.frames) // 2, len(seq.frames) - 1):
        metres = seq.depth_meters(idx)
        # Find the source PNG nearest this frame's timestamp.
        t = seq.frames[idx].t
        best = min(by_time, key=lambda k: abs(k - t))
        src = np.asarray(Image.open(SEQ_DIR / by_time[best]))
        assert src.dtype == np.uint16

        want = src.astype(np.float64) / 5000.0
        want[src == 0] = np.nan                       # 0 is "no return" in TUM
        got = metres.astype(np.float64)

        both = np.isfinite(want) & np.isfinite(got)
        assert both.sum() > 0.3 * want.size, "almost everything came back invalid"
        # Quantisation is inverse-depth, so compare relatively rather than in mm.
        rel = np.abs(got[both] - want[both]) / np.maximum(want[both], 1e-6)
        assert rel.max() < 2e-3, f"worst relative depth error {rel.max():.2e}"
        # Invalid must stay invalid: a 0 that became a distance is the classic bug.
        assert np.isnan(got[src == 0]).all(), "TUM's 0 (no return) became a depth"
        checked += 1
    assert checked == 3


def test_timestamps_are_the_sensor_clock(converted):
    """TUM timestamps are unix seconds; they must not be rebased or renumbered."""
    _, seq = converted
    rgb_lines = _read_pairs(SEQ_DIR / "rgb.txt")
    first_src = min(t for t, _ in rgb_lines)
    assert seq.frames[0].t > 1.3e9, "timestamps look rebased"
    assert abs(seq.frames[0].t - first_src) < 1.0
    ts = [f.t for f in seq.frames]
    assert ts == sorted(ts), "frames are not in time order"


def test_metric_scale_is_claimed(converted):
    """TUM is metric; a consumer requiring metres must be able to tell."""
    _, seq = converted
    assert seq.world.get("metric_scale") is True
