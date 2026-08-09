"""Every reader must agree with the corpus, and therefore with each other.

There are three wurld readers: Python, JavaScript and C++. Before this, each was
tested against its own author's expectations, which is three separate claims
rather than one. Drift would not have failed anything — it would have meant a
phone recording read one way in a browser and another on a robot.

`conformance/vectors/` holds small files plus what a reader must see, generated
from *intent* rather than captured from a reader's output (see
conformance/generate.py). One comparison function judges all three
implementations, so "conforming" means the same thing for each.

A reader may legitimately not model a field — the C++ reader does not parse the
payload layer's stream list — so each runner declares what it supports. The
core (cameras, frames, signals, world, rigs, imu) is required from everyone;
anything omitted must be omitted *loudly*, or a silent gap would read as a pass.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import wurld as wl

ROOT = Path(__file__).resolve().parents[1]
CONF = ROOT / "conformance"
VECTORS = CONF / "vectors"

CORE_FIELDS = {"cameras", "frames", "signals", "world", "rigs", "imu"}
# Binary tables store float32, so no reader can be held to more than that.
TOL = 1e-6

pytestmark = pytest.mark.skipif(not VECTORS.exists(),
                                reason="conformance vectors not generated")


def vector_names():
    index = json.loads((VECTORS / "index.json").read_text())
    return [v["name"] for v in index["vectors"]]


def expected_for(name):
    return json.loads((VECTORS / f"{name}.expected.json").read_text())


# ------------------------------------------------------------------- runners

def run_python(path):
    seq = wl.read(path)
    frames = []
    for f in seq.frames:
        rec = {"i": f.i, "t": f.t, "pose_valid": bool(f.pose_valid)}
        if f.pose_valid:
            rec["camera"] = f.camera
            rec["q_wxyz"] = [float(v) for v in f.q_wxyz]
            rec["tr"] = [float(v) for v in f.tr]
        frames.append(rec)
    rgbs = (seq.probe or {}).get("rgbs") or []
    return {
        "supports": sorted(CORE_FIELDS | {"rgb_streams"}),
        "cameras": {k: {"model": c.model, "width": c.width, "height": c.height,
                        "params": [float(p) for p in c.params]}
                    for k, c in seq.cameras.items()},
        "frames": frames,
        "signals": [{"id": s.id, "role": s.role, "value_map": s.value_map}
                    for s in seq.signals],
        "world": seq.world,
        "rigs": seq.rigs,
        "imu": {k: [[float(x) for x in row] for row in v.samples]
                for k, v in seq.imu.items()},
        "rgb_streams": [r["id"] for r in rgbs if isinstance(r, dict) and r.get("id")],
    }


def run_js(path):
    r = subprocess.run(["node", str(CONF / "run_js.mjs"), str(path)],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, f"js runner failed on {path.name}:\n{r.stderr}"
    return json.loads(r.stdout)


def run_cpp(tool, path):
    r = subprocess.run([str(tool), str(path), "--json"], capture_output=True, text=True)
    assert r.returncode == 0, f"cpp runner failed on {path.name}:\n{r.stderr}"
    got = json.loads(r.stdout)
    frames = []
    for f in got["frames"]:
        rec = {"i": f["i"], "t": f["t"], "pose_valid": f["pose_valid"]}
        if f["pose_valid"]:
            rec["camera"] = f["camera"]
            rec["q_wxyz"] = f["q_wxyz"]
            rec["tr"] = f["tr"]
        frames.append(rec)
    return {
        # The stream list comes from the payload layer's CHROMAPAKZ tag, which
        # the dependency-free C++ header deliberately does not model.
        "supports": sorted(CORE_FIELDS),
        "cameras": got["cameras"],
        "frames": frames,
        "signals": got["signals"],
        "world": got["world"],
        "rigs": got["rigs"],
        "imu": got["imu"],
    }


# ---------------------------------------------------------------- comparison

def approx_equal(got, want, path=""):
    """Structural comparison with a float tolerance. Returns a list of diffs."""
    diffs = []
    if isinstance(want, dict):
        if not isinstance(got, dict):
            return [f"{path}: expected object, got {type(got).__name__}"]
        for k in want:
            if k not in got:
                diffs.append(f"{path}.{k}: missing")
            else:
                diffs += approx_equal(got[k], want[k], f"{path}.{k}")
        for k in got:
            if k not in want:
                diffs.append(f"{path}.{k}: unexpected")
    elif isinstance(want, list):
        if not isinstance(got, list):
            return [f"{path}: expected array, got {type(got).__name__}"]
        if len(got) != len(want):
            return [f"{path}: length {len(got)} != {len(want)}"]
        for k, (g, w) in enumerate(zip(got, want)):
            diffs += approx_equal(g, w, f"{path}[{k}]")
    elif isinstance(want, bool) or isinstance(got, bool):
        # Before the number branch: True == 1 in Python, and a reader returning
        # 1 where the spec says true should not quietly pass.
        if bool(got) != bool(want) or isinstance(got, bool) != isinstance(want, bool):
            diffs.append(f"{path}: {got!r} != {want!r}")
    elif isinstance(want, (int, float)) and isinstance(got, (int, float)):
        if want != want:                       # NaN in the expectation
            if got == got:
                diffs.append(f"{path}: {got!r} is not NaN")
        elif abs(got - want) > TOL * max(1.0, abs(want)):
            diffs.append(f"{path}: {got!r} != {want!r}")
    elif got != want:
        diffs.append(f"{path}: {got!r} != {want!r}")
    return diffs


def check(reader_name, got, want):
    supports = set(got.get("supports", []))
    missing_core = CORE_FIELDS - supports
    assert not missing_core, f"{reader_name} does not support core fields {missing_core}"

    diffs = []
    for field in sorted(supports):
        if field not in want:
            continue
        diffs += approx_equal(got.get(field), want[field], field)
    assert not diffs, f"{reader_name} diverges from the corpus:\n  " + "\n  ".join(diffs)


# -------------------------------------------------------------------- fixtures

@pytest.fixture(scope="module")
def cpp_tool(tmp_path_factory):
    if shutil.which("cmake") is None:
        pytest.skip("cmake is needed to build the C++ reader")
    build = tmp_path_factory.mktemp("conf_cpp")
    r = subprocess.run(["cmake", "-S", str(ROOT / "cpp"), "-B", str(build),
                        "-DCMAKE_BUILD_TYPE=Release"], capture_output=True, text=True)
    if r.returncode != 0:
        pytest.skip(f"cmake configure failed:\n{r.stderr}")
    r = subprocess.run(["cmake", "--build", str(build), "-j", "4"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    return build / "wurld_info"


# ----------------------------------------------------------------------- tests

@pytest.mark.parametrize("name", vector_names())
def test_python_reader_conforms(name):
    check("python", run_python(VECTORS / f"{name}.wl.webm"), expected_for(name))


@pytest.mark.parametrize("name", vector_names())
@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_js_reader_conforms(name):
    check("javascript", run_js(VECTORS / f"{name}.wl.webm"), expected_for(name))


@pytest.mark.parametrize("name", vector_names())
def test_cpp_reader_conforms(cpp_tool, name):
    check("c++", run_cpp(cpp_tool, VECTORS / f"{name}.wl.webm"), expected_for(name))


def test_corpus_is_up_to_date():
    """The vectors must match what generate.py would produce today.

    Without this the corpus rots into a record of a format that used to exist,
    and every reader keeps passing against it.
    """
    r = subprocess.run([sys.executable, str(CONF / "generate.py"), "--check"],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stdout + r.stderr


def test_corpus_covers_the_paths_that_matter():
    """Guard against a vector being quietly dropped from the corpus."""
    names = set(vector_names())
    for required in ("v03_binary_frames", "v04_unposed", "v05_stereo", "v06_rig_imu",
                     "v08_float16_signal", "v10_single_frame"):
        assert required in names, f"{required} is missing from the corpus"
    # Every vector must have both a file and an expectation.
    for name in names:
        assert (VECTORS / f"{name}.wl.webm").exists()
        assert (VECTORS / f"{name}.expected.json").exists()


def test_readers_agree_with_each_other_directly(cpp_tool):
    """Belt and braces: compare the readers pairwise, not only via the corpus.

    Conforming to the corpus already implies agreement, but only over the fields
    the corpus models. This catches a field all three report and none is checked
    on — the shape of gap that hides between a spec and its tests.
    """
    if shutil.which("node") is None:
        pytest.skip("node is not installed")
    for name in vector_names():
        path = VECTORS / f"{name}.wl.webm"
        py, js, cpp = run_python(path), run_js(path), run_cpp(cpp_tool, path)
        for field in sorted(CORE_FIELDS):
            assert not approx_equal(js[field], py[field], field), \
                f"{name}: javascript and python disagree on {field}"
            assert not approx_equal(cpp[field], py[field], field), \
                f"{name}: c++ and python disagree on {field}"
