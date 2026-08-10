"""The validator has to catch things, not just say OK.

Every case below is a file that is wrong in one specific way, paired with the
SPEC section it violates. A check that cannot be made to fire is not a check.
"""

import json

import numpy as np
import pytest

import wurld as wl
from wurld import container, ebml, validate as v


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


# ── display streams and HDR (SPEC §4.4, §4.5) ──

def _stereo(tmp_path, cam_ids=("cam0", "cam1"), hdr=None, n=4):
    H, W = 32, 40
    cams = {c: wl.Camera(model="PINHOLE", width=W, height=H,
                         params=[30.0, 30.0, W / 2, H / 2]) for c in cam_ids}
    frames = [wl.Frame(i=i, t=i / 30, camera=cam_ids[0], q_wxyz=(1.0, 0, 0, 0),
                       tr=(0.01 * i, 0, 0.5)) for i in range(n)]
    rng = np.random.default_rng(0)
    dt = np.uint16 if hdr else np.uint8
    hi = 1024 if hdr else 255
    path = tmp_path / "streams.wl.webm"
    wl.write(path, cameras=cams, frames=frames,
             rgb={c: rng.integers(0, hi, (n, H, W, 4), dtype=dt) for c in cam_ids},
             signals={"depth": np.full((n, H, W), 3000, np.uint16)},
             specs={"depth": {"inverse_depth": True, "near": 0.3, "far": 9.0}},
             signal_meta=[wl.SignalMeta("depth", "depth",
                                        {"type": "inverse_depth", "near": 0.3, "far": 9.0,
                                         "levels": 65536, "invalid": 0})],
             hdr=hdr, fps=30)
    return path


def test_stereo_streams_bind_to_cameras_by_id(tmp_path):
    p = _stereo(tmp_path)
    assert v.validate(p) == []
    seq = wl.read(p)
    assert seq.rgb_streams == ["cam0", "cam1"]
    # Distinct pixels per camera, not the same buffer handed back twice.
    assert not np.array_equal(seq.rgb_for("cam0"), seq.rgb_for("cam1"))


def test_a_stream_with_no_matching_camera_is_refused_at_write(tmp_path):
    H, W, n = 32, 40, 3
    cams = {"cam0": wl.Camera(model="PINHOLE", width=W, height=H,
                              params=[30.0, 30.0, W / 2, H / 2])}
    frames = [wl.Frame(i=i, t=i / 30, camera="cam0", q_wxyz=(1.0, 0, 0, 0), tr=(0, 0, 0.5))
              for i in range(n)]
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="no matching camera"):
        wl.write(tmp_path / "bad.wl.webm", cameras=cams, frames=frames,
                 rgb={"cam0": rng.integers(0, 255, (n, H, W, 4), np.uint8),
                      "ghost": rng.integers(0, 255, (n, H, W, 4), np.uint8)}, fps=30)


def test_streams_must_agree_on_frame_count(tmp_path):
    H, W = 32, 40
    cams = {c: wl.Camera(model="PINHOLE", width=W, height=H,
                         params=[30.0, 30.0, W / 2, H / 2]) for c in ("cam0", "cam1")}
    frames = [wl.Frame(i=i, t=i / 30, camera="cam0", q_wxyz=(1.0, 0, 0, 0), tr=(0, 0, 0.5))
              for i in range(3)]
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="disagree on frame count"):
        wl.write(tmp_path / "bad.wl.webm", cameras=cams, frames=frames,
                 rgb={"cam0": rng.integers(0, 255, (3, H, W, 4), np.uint8),
                      "cam1": rng.integers(0, 255, (5, H, W, 4), np.uint8)}, fps=30)


def test_hdr_display_track(tmp_path):
    p = _stereo(tmp_path, cam_ids=("cam0",), hdr={"transfer": "pq", "max_cll": 1000})
    assert v.validate(p) == []
    seq = wl.read(p)
    hdr = seq.hdr
    assert hdr is not None and hdr["transfer"] == "pq" and hdr["bits"] == 10
    # HDR display pixels come back as 10-bit codes, not truncated to uint8.
    assert seq.rgb.dtype == np.uint16
    assert int(seq.rgb.max()) > 255


def test_hdr_without_any_rgb_is_refused(tmp_path):
    H, W, n = 32, 40, 3
    cams = {"0": wl.Camera(model="PINHOLE", width=W, height=H,
                           params=[30.0, 30.0, W / 2, H / 2])}
    frames = [wl.Frame(i=i, t=i / 30, camera="0", q_wxyz=(1.0, 0, 0, 0), tr=(0, 0, 0.5))
              for i in range(n)]
    with pytest.raises(ValueError, match="no RGB stream"):
        wl.write(tmp_path / "bad.wl.webm", cameras=cams, frames=frames,
                 signals={"depth": np.full((n, H, W), 3000, np.uint16)},
                 specs={"depth": {"inverse_depth": True, "near": 0.3, "far": 9.0}},
                 hdr={"transfer": "pq"}, fps=30)


def test_new_files_declare_the_current_format_version(good):
    doc = json.loads(ebml.read_all_tags(good.read_bytes())["WURLD"])
    assert doc["version"] == container.FORMAT_VERSION == "1.2"


def test_older_format_versions_still_read_and_validate(good, tmp_path):
    """0.4 files predate the version bump and must keep working.

    SPEC §10 makes `version` additive and ungated, so a reader must not start
    rejecting the files every earlier release produced — including the WurldCam
    build that was in App Store review when the bump landed.
    """
    data = good.read_bytes()
    tags = dict(ebml.read_all_tags(data))
    doc = json.loads(tags["WURLD"])
    doc["version"] = "0.4"
    tags["WURLD"] = json.dumps(doc, separators=(",", ":"))
    aged = tmp_path / "aged.wl.webm"
    aged.write_bytes(ebml.insert_header_tags(data, tags))

    assert v.validate(aged) == []
    assert len(container.read(aged).frames) == len(container.read(good).frames)


def _multi_cluster(path, n=150, w=64, h=48):
    """Long enough to span several Clusters, so one can be removed."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    rgb = np.stack([
        np.dstack([np.clip((0.5 + 0.4 * np.sin(xx / 5 + i * 0.3)) * 255, 0, 255)] * 3
                  + [np.full((h, w), 255, np.float32)]).astype(np.uint8) for i in range(n)])
    f = 1.1 * w
    wl.write(path, cameras={"0": wl.Camera("PINHOLE", w, h, [f, f, w / 2, h / 2])},
             frames=[wl.Frame(i=i, t=i / 30, camera="0", q_wxyz=(1.0, 0.0, 0.0, 0.0),
                              tr=(0.01 * i, 0.0, 0.5)) for i in range(n)],
             rgb=rgb, world={"metric_scale": True}, fps=30)
    return path


def _drop_clusters_after_the_first(src, dest):
    """Keep the header, one Cluster and the trailing tags: a plausible truncation."""
    data = src.read_bytes()
    seg, ps, pe = ebml._segment_bounds(data)
    cl = [(es, pend) for eid, es, _p, pend in ebml._top_level(data, ps, pe)
          if eid == ebml.CLUSTER]
    assert len(cl) > 2, "the fixture must span several Clusters"
    payload = data[ps:cl[0][1]] + data[cl[-1][1]:pe]
    size = len(payload).to_bytes(8, "big")
    size = bytes([size[0] | 0x01]) + size[1:]
    dest.write_bytes(data[:seg] + b"\x18\x53\x80\x67" + size + payload + data[pe:])
    return dest


def test_a_truncated_file_is_reported(tmp_path):
    """Metadata and poses intact, most of the video gone — this used to pass.

    Nothing else notices: the document is self-consistent, every pose is well
    formed, and a reader reports the full frame count. Only the pixels are
    missing.
    """
    src = _multi_cluster(tmp_path / "full.wl.webm")
    assert v.validate(src) == [], "the healthy fixture must be clean"

    cut = _drop_clusters_after_the_first(src, tmp_path / "cut.wl.webm")
    findings = v.validate(cut)
    assert _has(findings, "error", "9", "video are missing"), findings
    # The poses survive, which is exactly why this was invisible.
    assert len(container.read(cut).frames) == 150


def test_a_healthy_file_is_not_accused_of_truncation(tmp_path):
    """A check that fires on intact files would be worse than none."""
    for n in (4, 40, 150):
        src = _multi_cluster(tmp_path / f"ok{n}.wl.webm", n=n)
        assert not [f for f in v.validate(src) if "missing" in f.message], n
