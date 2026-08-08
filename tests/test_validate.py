"""The validator has to catch things, not just say OK.

Every case below is a file that is wrong in one specific way, paired with the
SPEC section it violates. A check that cannot be made to fire is not a check.
"""

import json

import numpy as np
import pytest

import wurld as wl
from wurld import ebml, validate as v


def _findings(doc_mutation=None, tag_mutation=None, base=None, tmp_path=None):
    """Rebuild a real file with a mutated WURLD document and/or tags, then validate."""
    data = base.read_bytes()
    tags = dict(ebml.read_all_tags(data))
    doc = json.loads(tags["WURLD"])
    if doc_mutation:
        doc_mutation(doc)
    tags["WURLD"] = json.dumps(doc, separators=(",", ":"))
    if tag_mutation:
        tag_mutation(tags)
    out = tmp_path / "mutated.wl.webm"
    out.write_bytes(ebml.insert_header_tags(data, tags))
    return v.validate(out)


def _has(findings, severity, section, needle):
    return any(f.severity == severity and f.section == section and needle in f.message
               for f in findings)


@pytest.fixture(scope="module")
def good(tmp_path_factory):
    """A small, valid file: the baseline every mutation starts from."""
    path = tmp_path_factory.mktemp("val") / "good.wl.webm"
    H, W, N = 32, 40, 6
    cams = {"0": wl.Camera(model="PINHOLE", width=W, height=H, params=[30.0, 30.0, W / 2, H / 2])}
    frames = [wl.Frame(i=i, t=i / 30, camera="0", q_wxyz=(1.0, 0.0, 0.0, 0.0), tr=(0.01 * i, 0.0, 0.5))
              for i in range(N)]
    rng = np.random.default_rng(0)
    wl.write(path, cameras=cams, frames=frames,
             rgb=rng.integers(0, 255, (N, H, W, 4), dtype=np.uint8),
             signals={"depth": np.full((N, H, W), 3000, np.uint16)},
             specs={"depth": {"inverse_depth": True, "near": 0.3, "far": 9.0}},
             signal_meta=[wl.SignalMeta("depth", "depth",
                                        {"type": "inverse_depth", "near": 0.3, "far": 9.0,
                                         "levels": 65536, "invalid": 0})],
             fps=30)
    return path


def test_a_valid_file_has_no_findings(good):
    assert v.validate(good) == []


def test_conventions_must_not_be_redefined(good, tmp_path):
    f = _findings(lambda d: d["conventions"].update(camera_axes="RUB"),
                  base=good, tmp_path=tmp_path)
    assert _has(f, v.ERROR, "3", "camera_axes")


def test_format_field_must_say_wurld(good, tmp_path):
    f = _findings(lambda d: d.update(format="worldline"), base=good, tmp_path=tmp_path)
    assert _has(f, v.ERROR, "5", "must be \"wurld\"")


def test_camera_resolution_must_match_the_video_track(good, tmp_path):
    # SPEC §4.1: intrinsics would otherwise be applied at the wrong scale.
    f = _findings(lambda d: d["cameras"]["0"].update(width=1280, height=720),
                  base=good, tmp_path=tmp_path)
    assert _has(f, v.ERROR, "4.1", "calibrated")


def test_wrong_parameter_count_for_the_model(good, tmp_path):
    f = _findings(lambda d: d["cameras"]["0"].update(params=[30.0, 30.0]),
                  base=good, tmp_path=tmp_path)
    assert _has(f, v.ERROR, "4.1", "needs 4 params")


def test_unknown_camera_model(good, tmp_path):
    f = _findings(lambda d: d["cameras"]["0"].update(model="MAGIC"),
                  base=good, tmp_path=tmp_path)
    assert _has(f, v.ERROR, "4.1", "unknown model")


def test_inverse_depth_with_far_below_near(good, tmp_path):
    def bad(d):
        d["signals"][0]["value_map"].update(near=9.0, far=0.3)
    f = _findings(bad, base=good, tmp_path=tmp_path)
    assert _has(f, v.ERROR, "6", "far > near")


def test_inverse_depth_levels_below_three_divides_by_zero(good, tmp_path):
    def bad(d):
        d["signals"][0]["value_map"].update(levels=2)
    f = _findings(bad, base=good, tmp_path=tmp_path)
    assert _has(f, v.ERROR, "6", "levels >= 3")


def test_non_unit_quaternion(good, tmp_path):
    def bad(d):
        d["frames"][2]["q_wxyz"] = [0.5, 0.5, 0.5, 0.5001]  # norm ~1.0 but nudge it
        d["frames"][3]["q_wxyz"] = [2.0, 0.0, 0.0, 0.0]     # clearly not unit
    f = _findings(bad, base=good, tmp_path=tmp_path)
    assert _has(f, v.ERROR, "4.2", "quaternion norm")


def test_timestamps_going_backwards(good, tmp_path):
    def bad(d):
        d["frames"][3]["t"] = d["frames"][1]["t"] - 1.0
    f = _findings(bad, base=good, tmp_path=tmp_path)
    assert _has(f, v.ERROR, "4.2", "backwards")


def test_frame_indices_out_of_order(good, tmp_path):
    def bad(d):
        d["frames"][1]["i"], d["frames"][2]["i"] = d["frames"][2]["i"], d["frames"][1]["i"]
    f = _findings(bad, base=good, tmp_path=tmp_path)
    assert _has(f, v.ERROR, "9", "ascending")


def test_pose_referencing_a_camera_that_does_not_exist(good, tmp_path):
    def bad(d):
        d["frames"][0]["camera"] = "left"
    f = _findings(bad, base=good, tmp_path=tmp_path)
    assert _has(f, v.ERROR, "4.2", "not in cameras")


def test_rig_referencing_a_camera_that_does_not_exist(good, tmp_path):
    def bad(d):
        d["rigs"] = {"body": {"cameras": {"1": {"q_wxyz": [1, 0, 0, 0], "tr": [0, 0, 0]}}}}
    f = _findings(bad, base=good, tmp_path=tmp_path)
    assert _has(f, v.ERROR, "8.1", "not declared")


def test_binary_table_truncated_mid_record(good, tmp_path):
    def bad_doc(d):
        d["frames"] = []
        d["frames_binary"] = {"version": 1, "count": 3, "cameras": ["0"]}

    def bad_tags(t):
        t["WURLD_FRAMES"] = bytes(45 * 3 + 7)      # a partial record on the end
    f = _findings(bad_doc, bad_tags, base=good, tmp_path=tmp_path)
    assert _has(f, v.ERROR, "7", "not a multiple of 45")


def test_frames_binary_count_disagrees_with_the_table(good, tmp_path):
    def bad_doc(d):
        d["frames"] = []
        d["frames_binary"] = {"version": 1, "count": 99, "cameras": ["0"]}

    def bad_tags(t):
        t["WURLD_FRAMES"] = bytes(45 * 3)
    f = _findings(bad_doc, bad_tags, base=good, tmp_path=tmp_path)
    assert _has(f, v.ERROR, "7", "count says 99")


def test_frames_binary_declared_but_no_table(good, tmp_path):
    def bad(d):
        d["frames"] = []
        d["frames_binary"] = {"version": 1, "count": 3, "cameras": ["0"]}
    f = _findings(bad, base=good, tmp_path=tmp_path)
    assert _has(f, v.ERROR, "7", "no WURLD_FRAMES tag")


def test_imu_declared_without_its_tag(good, tmp_path):
    def bad(d):
        d["imu"] = {"imu0": {"rate_hz": 100.0}}
    f = _findings(bad, base=good, tmp_path=tmp_path)
    assert _has(f, v.ERROR, "8.3", "is missing")


def test_a_plain_chromapakz_file_is_not_an_error(tmp_path):
    """SPEC §10: no WURLD tag means a plain chromapakz file, which is legitimate."""
    import chromapakz as cz
    H, W, N = 32, 40, 4
    parts = []
    enc = cz.create_encoder(W, H, signals=[{"id": "depth", "near": 0.3, "far": 9.0}],
                            fps=30, has_rgb=True, on_chunk=parts.append)
    for _ in range(N):
        enc.add_frame(rgb=np.zeros((H, W, 4), np.uint8),
                      signals={"depth": {"u16": np.full((H, W), 3000, np.uint16)}})
    enc.finish()
    out = tmp_path / "plain.webm"
    out.write_bytes(b"".join(parts))
    f = v.validate(out)
    assert not any(x.severity == v.ERROR for x in f)
    assert _has(f, v.NOTE, "10", "plain chromapakz")


def test_garbage_is_reported_not_raised(tmp_path):
    p = tmp_path / "garbage.wl.webm"
    p.write_bytes(b"this is not a matroska file at all")
    f = v.validate(p)
    assert any(x.severity == v.ERROR for x in f)


def test_real_shipped_samples_conform():
    """The files we publish must pass our own checker."""
    from pathlib import Path
    sample = Path("docs/samples/synthetic-orbit.wl.webm")
    if not sample.exists():
        pytest.skip("sample not present")
    assert v.validate(sample) == []
