"""Files written by C++ must be indistinguishable from files written by Python.

The writer is the half that can corrupt data rather than merely misread it, so
the bar here is higher than "our reader can read it back". A C++-written file
must:

  * satisfy `wurld validate` — conformance to SPEC, judged by the same code that
    judges Python-written files;
  * read back through the Python reader with the poses that went in;
  * produce byte-identical binary tables to the Python packers, since those are
    the parts two implementations are most likely to disagree on;
  * survive the round trip through all three readers.

`attach` is checked against a real chromapakz WebM, not a synthetic container,
because the failure mode that matters is mangling someone else's muxing —
shifting Clusters without rebuilding Cues leaves a file that looks fine until
you seek.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import wurld as wl
from wurld import ebml, validate as v

ROOT = Path(__file__).resolve().parents[1]
CPP = ROOT / "cpp"
W, H = 32, 24

pytestmark = pytest.mark.skipif(
    shutil.which("cmake") is None, reason="cmake is needed to build the C++ writer")


@pytest.fixture(scope="module")
def tools(tmp_path_factory):
    build = tmp_path_factory.mktemp("cppwrite")
    r = subprocess.run(["cmake", "-S", str(CPP), "-B", str(build),
                        "-DCMAKE_BUILD_TYPE=Release"], capture_output=True, text=True)
    if r.returncode != 0:
        pytest.skip(f"cmake configure failed:\n{r.stderr}")
    r = subprocess.run(["cmake", "--build", str(build), "-j", "4"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    return {"attach": build / "wurld_attach", "info": build / "wurld_info"}


def bare_webm(path, n=6, *, signals=True):
    """A chromapakz WebM with the wurld tags stripped: the writer's input.

    Encoding is chromapakz's job in C++ too, so the writer's input is exactly
    this — someone else's muxed file.
    """
    import chromapakz as cz

    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    rgb = np.stack([np.dstack([np.clip((0.5 + 0.4 * np.sin(xx / 4 + i * 0.3)) * 255, 0, 255)] * 3
                              + [np.full((H, W), 255, np.float32)]).astype(np.uint8)
                    for i in range(n)])
    kwargs = {}
    if signals:
        depth = np.stack([(1.5 + 0.4 * np.sin(xx / 6 + i * 0.2)).astype(np.float32)
                          for i in range(n)])
        kwargs = {"signals": {"depth": cz.quantize_inverse(depth, near=0.3, far=9.0)},
                  "specs": {"depth": {"inverse_depth": True, "near": 0.3, "far": 9.0}}}
    data = cz.encode(kwargs.get("signals", {}), specs=kwargs.get("specs"),
                     rgb=rgb, fps=30)
    path.write_bytes(data)
    return path


def spec(n=6, *, cameras=None, frames_format="auto", imu=None, rigs=None,
         unposed=(), signals=True):
    cams = cameras or {"0": {"model": "PINHOLE", "width": W, "height": H,
                             "params": [35.2, 35.2, W / 2, H / 2]}}
    first_cam = sorted(cams)[0]
    frames = []
    for i in range(n):
        if i in unposed:
            frames.append({"i": i, "t": i / 30, "pose_valid": False})
        else:
            ang = 0.09 * i
            frames.append({"i": i, "t": i / 30, "camera": first_cam,
                           "q_wxyz": [float(np.cos(ang / 2)), 0.0,
                                      float(np.sin(ang / 2)), 0.0],
                           "tr": [0.04 * i, -0.01 * i, 1.1]})
    out = {"cameras": cams, "frames": frames, "frames_format": frames_format,
           "world": {"metric_scale": True, "description": "written by the C++ writer"}}
    if signals:
        out["signals"] = [{"id": "depth", "role": "depth",
                           "value_map": {"type": "inverse_depth", "near": 0.3,
                                         "far": 9.0, "levels": 65536, "invalid": 0}}]
    if imu:
        out["imu"] = imu
    if rigs:
        out["rigs"] = rigs
    return out


def attach(tools, tmp_path, name, sp, *, n=6, signals=True):
    src = bare_webm(tmp_path / f"{name}_src.webm", n, signals=signals)
    out = tmp_path / f"{name}.wurld.webm"
    spec_path = tmp_path / f"{name}.spec.json"
    spec_path.write_text(json.dumps(sp))
    r = subprocess.run([str(tools["attach"]), str(src), str(out), str(spec_path)],
                       capture_output=True, text=True)
    assert r.returncode == 0, f"wurld_attach failed:\n{r.stdout}\n{r.stderr}"
    return out, r.stdout


# --------------------------------------------------------------------- tests

def test_written_file_passes_the_validator(tools, tmp_path):
    """The same conformance check Python-written files face."""
    out, _ = attach(tools, tmp_path, "basic", spec())
    assert v.validate(out) == []


def test_poses_survive_the_round_trip(tools, tmp_path):
    sp = spec(n=6)
    out, _ = attach(tools, tmp_path, "poses", sp)
    seq = wl.read(out)
    assert len(seq.frames) == 6
    for got, want in zip(seq.frames, sp["frames"]):
        assert got.i == want["i"]
        assert got.t == pytest.approx(want["t"])
        assert got.camera == want["camera"]
        assert list(got.q_wxyz) == pytest.approx(want["q_wxyz"], abs=1e-6)
        assert list(got.tr) == pytest.approx(want["tr"], abs=1e-6)


def test_pixels_are_untouched(tools, tmp_path):
    """Attaching metadata must not disturb the video it was attached to."""
    src = bare_webm(tmp_path / "px_src.webm", 6)
    import chromapakz as cz
    before = cz.decode(src.read_bytes())

    out, _ = attach(tools, tmp_path, "px", spec())
    seq = wl.read(out)
    assert np.array_equal(seq.rgb, before["rgb"])
    assert np.array_equal(seq.signal("depth"), before["signals"]["depth"])


def test_binary_table_is_byte_identical_to_python(tools, tmp_path):
    """Two writers, one byte layout. This is where they would drift."""
    sp = spec(n=8, frames_format="binary")
    out, _ = attach(tools, tmp_path, "bin", sp, n=8)
    tags = ebml.read_all_tags(out.read_bytes())
    got = tags["WURLD_FRAMES"]
    assert isinstance(got, bytes)

    frames = [wl.Frame(i=f["i"], t=f["t"], camera=f.get("camera", "0"),
                       q_wxyz=tuple(f.get("q_wxyz", (1, 0, 0, 0))),
                       tr=tuple(f.get("tr", (0, 0, 0))),
                       pose_valid=f.get("pose_valid", True))
              for f in sp["frames"]]
    from wurld.container import pack_frames
    assert got == pack_frames(frames, ["0"]), "C++ and Python frame tables differ"


def test_imu_is_byte_identical_to_python(tools, tmp_path):
    t = np.round(np.arange(0, 0.2, 0.005), 6)
    rows = [[float(x), 0.1, -0.02, 0.3, 0.01, 0.0, 9.81] for x in t]
    out, _ = attach(tools, tmp_path, "imu", spec(imu={"imu0": rows}))

    tags = ebml.read_all_tags(out.read_bytes())
    got = tags["WURLD_IMU_imu0"]
    want = wl.ImuStream("imu0", np.array(rows)).pack()
    assert got == want, "C++ and Python IMU records differ"

    seq = wl.read(out)
    assert seq.imu["imu0"].samples.shape == (len(rows), 7)
    assert np.allclose(seq.imu["imu0"].samples, np.array(rows, dtype=np.float32),
                       rtol=1e-6, atol=1e-7)


def test_unposed_frames_are_written_as_unposed(tools, tmp_path):
    for fmt in ("json", "binary"):
        out, _ = attach(tools, tmp_path, f"unposed_{fmt}",
                        spec(n=7, unposed=(2, 5), frames_format=fmt), n=7)
        assert v.validate(out) == []
        seq = wl.read(out)
        assert [f.i for f in seq.frames if not f.pose_valid] == [2, 5]


def test_cues_are_rebuilt_at_the_new_offsets(tools, tmp_path):
    """Inserting tags shifts every Cluster; stale Cues seek into the middle of one.

    This is the writer's most dangerous failure: the file opens, plays from the
    start, and only misbehaves when something seeks.
    """
    out, _ = attach(tools, tmp_path, "cues", spec(n=8), n=8)
    data = out.read_bytes()
    _, payload_start, payload_end = ebml._segment_bounds(data)

    # Cues live where the SeekHead says; read_cues parses one element at a
    # position rather than searching a file.
    seeks = ebml.read_seek_head(data, payload_start, payload_end)
    assert ebml.CUES in seeks, "no Cues entry in the SeekHead"
    cues = ebml.read_cues(data, seeks[ebml.CUES])
    assert cues, "no Cues written"

    starts = {elem_start - payload_start
              for eid, elem_start, _, _ in ebml._top_level(data, payload_start, payload_end)
              if eid == ebml.CLUSTER}
    assert len(cues) == len(starts)
    for _, offset in cues:
        assert offset in starts, f"cue points at {offset}, which is not a Cluster start"


def test_seek_head_entries_resolve(tools, tmp_path):
    out, _ = attach(tools, tmp_path, "seek", spec())
    data = out.read_bytes()
    _, payload_start, payload_end = ebml._segment_bounds(data)
    seeks = ebml.read_seek_head(data, payload_start, payload_end)
    assert ebml.TAGS in seeks and ebml.CLUSTER in seeks and ebml.CUES in seeks
    for eid, absolute in seeks.items():
        # Every SeekHead entry must land on an element of the id it claims.
        found = ebml._read_vint(data, absolute, keep_marker=True)[0]
        assert found == eid, f"seek entry for {eid:#x} lands on {found:#x}"


def test_ranged_header_read_works_on_a_cpp_written_file(tools, tmp_path):
    """The layout must support fetch_header, which needs a SeekHead."""
    from wurld import remote
    # Big enough to exceed remote's 8 KiB first probe, or "read less than the
    # whole file" is not a claim this file can make.
    out, _ = attach(tools, tmp_path, "ranged", spec(n=40), n=40)
    assert out.stat().st_size > 3 * 8192
    h = remote.fetch_header(remote.file_fetcher(out))
    assert len(h.frames) == 40
    assert h.header_extent < out.stat().st_size
    assert h.bytes_fetched < out.stat().st_size


def test_all_three_readers_agree_on_a_cpp_written_file(tools, tmp_path):
    rigs = {"body": {"cameras": {
        "cam0": {"q_wxyz": [1.0, 0.0, 0.0, 0.0], "tr": [0.0, 0.0, 0.0]},
        "cam1": {"q_wxyz": [1.0, 0.0, 0.0, 0.0], "tr": [0.12, 0.0, 0.0]}}}}
    cams = {c: {"model": "PINHOLE", "width": W, "height": H,
                "params": [35.2, 35.2, W / 2, H / 2]} for c in ("cam0", "cam1")}
    rows = [[i * 0.005, 0.1, 0.0, 0.3, 0.0, 0.0, 9.81] for i in range(40)]
    out, _ = attach(tools, tmp_path, "three",
                    spec(cameras=cams, rigs=rigs, imu={"imu0": rows}))
    assert v.validate(out) == []

    sys.path.insert(0, str(ROOT / "tests"))
    from test_conformance import approx_equal, run_cpp, run_js, run_python

    py = run_python(out)
    cpp = run_cpp(tools["info"], out)
    for field in ("cameras", "frames", "signals", "world", "rigs", "imu"):
        assert not approx_equal(cpp[field], py[field], field), \
            f"c++ and python disagree on {field} of a C++-written file"
    if shutil.which("node"):
        js = run_js(out)
        for field in ("cameras", "frames", "signals", "world", "rigs", "imu"):
            assert not approx_equal(js[field], py[field], field), \
                f"javascript and python disagree on {field}"


def test_auto_picks_json_for_small_files(tools, tmp_path):
    out, _ = attach(tools, tmp_path, "auto", spec(n=6, frames_format="auto"))
    tags = ebml.read_all_tags(out.read_bytes())
    assert "WURLD_FRAMES" not in tags
    assert json.loads(tags["WURLD"])["frames"]


def test_writer_rejects_a_frame_naming_an_undeclared_camera(tools, tmp_path):
    sp = spec(n=3, frames_format="binary")
    sp["frames"][1]["camera"] = "ghost"
    src = bare_webm(tmp_path / "ghost_src.webm", 3)
    spec_path = tmp_path / "ghost.spec.json"
    spec_path.write_text(json.dumps(sp))
    r = subprocess.run([str(tools["attach"]), str(src),
                        str(tmp_path / "ghost.wurld.webm"), str(spec_path)],
                       capture_output=True, text=True)
    # Writing index 0 instead would silently attribute the frame to the wrong
    # camera, which no reader could detect.
    assert r.returncode == 1
    assert "not declared" in r.stderr


def test_writer_rejects_malformed_embedded_json(tools, tmp_path):
    src = bare_webm(tmp_path / "badjson_src.webm", 3)
    spec_path = tmp_path / "bad.spec.json"
    spec_path.write_text('{"cameras":{},"frames":[],"world":"not an object"}')
    r = subprocess.run([str(tools["attach"]), str(src),
                        str(tmp_path / "bad.wurld.webm"), str(spec_path)],
                       capture_output=True, text=True)
    assert r.returncode == 1
    assert "json object" in r.stderr or "not valid json" in r.stderr


def test_attaching_twice_replaces_rather_than_accumulates(tools, tmp_path):
    """Re-attaching must not leave two WURLD documents or two Cues elements."""
    once, _ = attach(tools, tmp_path, "once", spec(n=5), n=5)
    twice = tmp_path / "twice.wurld.webm"
    spec_path = tmp_path / "twice.spec.json"
    spec_path.write_text(json.dumps(spec(n=5)))
    r = subprocess.run([str(tools["attach"]), str(once), str(twice), str(spec_path)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr

    def count_named(path, name):
        """How many SimpleTags carry this name — dict-based readers hide dupes."""
        data = path.read_bytes()
        _, ps, pe = ebml._segment_bounds(data)
        n = 0
        for eid, _, pstart, pend in ebml._top_level(data, ps, pe):
            if eid != ebml.TAGS:
                continue
            n += sum(1 for tag, _ in ebml.collect_simple_tags(data, pstart, pend)
                     if tag == name)
        return n

    def structure(path):
        data = path.read_bytes()
        _, ps, pe = ebml._segment_bounds(data)
        ids = [eid for eid, _, _, _ in ebml._top_level(data, ps, pe)]
        return (ids.count(ebml.CUES), ids.count(ebml.SEEK_HEAD), ids.count(ebml.TAGS))

    # Exactly one document, not two: which of two WURLD tags a reader picks is a
    # coin toss, and both would parse.
    assert count_named(once, "WURLD") == 1
    assert count_named(twice, "WURLD") == 1
    # chromapakz's own tag element must survive both passes untouched.
    assert count_named(twice, "CHROMAPAKZ") == 1
    # Re-attaching is idempotent in structure, not merely valid.
    assert structure(twice) == structure(once)
    assert structure(twice)[0] == 1 and structure(twice)[1] == 1

    assert v.validate(twice) == []
    assert len(wl.read(twice).frames) == 5
