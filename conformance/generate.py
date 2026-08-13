"""Generate the conformance corpus: small files plus what a reader must see.

Why this exists. There are three wurld readers now — Python, JavaScript and C++
— and "they all work" was, until this corpus, three separate claims each checked
against its own author's expectations. Drift between them would not fail any
test; it would just mean a phone recording read one way in a browser and another
on a robot.

**Expectations come from intent, not from a reader.** Each vector declares what
it contains, the file is written from that declaration, and `expected.json` is
emitted from the same declaration. Nothing is captured from a reader's output,
so the corpus cannot enshrine a bug that all three happen to share — a golden
file dumped from the Python reader would do exactly that.

The vectors are deliberately tiny (16x16, a handful of frames) so the corpus is
small enough to live in the repository and a third-party implementer can run it
in a second.

    python conformance/generate.py            # rewrite vectors/
    python conformance/generate.py --check    # fail if they would change
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import wurld as wl  # noqa: E402

HERE = Path(__file__).resolve().parent
VECTORS = HERE / "vectors"
W, H = 16, 16


def _rgb(n, shift=0):
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    out = []
    for i in range(n):
        g = np.clip((0.5 + 0.4 * np.sin(xx / 3 + i * 0.4 + shift) * np.cos(yy / 3)) * 255,
                    0, 255)
        out.append(np.dstack([g, g, g, np.full_like(g, 255)]).astype(np.uint8))
    return np.stack(out)


def _depth(n):
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    return np.stack([(1.5 + 0.4 * np.sin(xx / 4 + i * 0.2)).astype(np.float32)
                     for i in range(n)])


def _cam(model="PINHOLE", params=None):
    f = 1.1 * W
    return wl.Camera(model=model, width=W, height=H,
                     params=params or [f, f, W / 2, H / 2])


def _pose(i):
    ang = 0.11 * i
    return ((float(np.cos(ang / 2)), 0.0, float(np.sin(ang / 2)), 0.0),
            (0.05 * i, -0.02 * i, 1.25))


def _frames(n, *, camera="0", unposed=()):
    out = []
    for i in range(n):
        if i in unposed:
            out.append(wl.Frame(i=i, t=i / 30, pose_valid=False))
        else:
            q, tr = _pose(i)
            out.append(wl.Frame(i=i, t=i / 30, camera=camera, q_wxyz=q, tr=tr))
    return out


def _expect_frames(frames):
    """What every reader must report, at the precision the container preserves.

    Binary tables store float32, so a conforming reader is not required to
    return more precision than that; the tolerance lives in the runners.
    """
    out = []
    for f in frames:
        rec = {"i": f.i, "t": f.t, "pose_valid": bool(f.pose_valid)}
        if f.pose_valid:
            rec["camera"] = f.camera
            rec["q_wxyz"] = [float(v) for v in f.q_wxyz]
            rec["tr"] = [float(v) for v in f.tr]
        out.append(rec)
    return out


def _expect(cameras, frames, *, signals=(), world=None, rigs=None, imu=None,
            rgb_streams=()):
    return {
        "cameras": {k: {"model": c.model, "width": c.width, "height": c.height,
                        "params": [float(p) for p in c.params]}
                    for k, c in cameras.items()},
        "frames": _expect_frames(frames),
        "signals": [{"id": s.id, "role": s.role, "value_map": s.value_map}
                    for s in signals],
        "world": world or {},
        "rigs": rigs or {},
        "imu": imu or {},
        "rgb_streams": list(rgb_streams),
    }


# --------------------------------------------------------------------- vectors

def v01_minimal(path):
    """Poses and calibration only: no signals, no pixels beyond the RGB track."""
    cams = {"0": _cam()}
    frames = _frames(4)
    wl.write(path, cameras=cams, frames=frames, rgb=_rgb(4),
             world={"metric_scale": True}, frames_format="json", fps=30)
    return _expect(cams, frames, world={"metric_scale": True}, rgb_streams=["rgb"])


def v02_depth(path):
    """The common case: inverse-depth signal beside the display track."""
    import chromapakz as cz
    cams = {"0": _cam()}
    frames = _frames(4)
    vm = {"type": "inverse_depth", "near": 0.3, "far": 9.0, "levels": 65536,
          "invalid": 0}
    sig = wl.SignalMeta("depth", "depth", vm)
    wl.write(path, cameras=cams, frames=frames, rgb=_rgb(4),
             signals={"depth": cz.quantize_inverse(_depth(4), near=0.3, far=9.0)},
             specs={"depth": {"inverse_depth": True, "near": 0.3, "far": 9.0}},
             signal_meta=[sig], world={"metric_scale": True},
             frames_format="json", fps=30)
    return _expect(cams, frames, signals=[sig], world={"metric_scale": True},
                   rgb_streams=["rgb"])


def v03_binary_frames(path):
    """Poses in the binary table — the form ffmpeg cannot see at all."""
    cams = {"0": _cam()}
    frames = _frames(6)
    wl.write(path, cameras=cams, frames=frames, rgb=_rgb(6),
             world={"metric_scale": True}, frames_format="binary", fps=30)
    return _expect(cams, frames, world={"metric_scale": True}, rgb_streams=["rgb"])


def v04_unposed(path):
    """Frames the producer could not localise, in both storage forms.

    A reader must report these as unposed rather than substituting identity;
    that substitution is silent and trains on a camera that never existed.
    """
    cams = {"0": _cam()}
    frames = _frames(6, unposed=(1, 4))
    wl.write(path, cameras=cams, frames=frames, rgb=_rgb(6),
             world={"metric_scale": False}, frames_format="binary", fps=30)
    return _expect(cams, frames, world={"metric_scale": False}, rgb_streams=["rgb"])


def v05_stereo(path):
    """Two display streams keyed by camera id (SPEC §4.4)."""
    import chromapakz as cz
    cams = {"cam0": _cam(), "cam1": _cam()}
    frames = _frames(4, camera="cam0")
    vm = {"type": "inverse_depth", "near": 0.3, "far": 9.0, "levels": 65536,
          "invalid": 0}
    sig = wl.SignalMeta("depth", "depth", vm)
    rigs = {"body": {"cameras": {
        "cam0": {"q_wxyz": [1.0, 0.0, 0.0, 0.0], "tr": [0.0, 0.0, 0.0]},
        "cam1": {"q_wxyz": [1.0, 0.0, 0.0, 0.0], "tr": [0.12, 0.0, 0.0]}}}}
    wl.write(path, cameras=cams, frames=frames,
             rgb={"cam0": _rgb(4), "cam1": _rgb(4, shift=1.0)},
             signals={"depth": cz.quantize_inverse(_depth(4), near=0.3, far=9.0)},
             specs={"depth": {"inverse_depth": True, "near": 0.3, "far": 9.0}},
             signal_meta=[sig], rigs=rigs, world={"metric_scale": True},
             frames_format="json", fps=30)
    return _expect(cams, frames, signals=[sig], rigs=rigs,
                   world={"metric_scale": True}, rgb_streams=["cam0", "cam1"])


def v06_rig_imu(path):
    """Rig extrinsics plus an IMU at its own rate."""
    cams = {"cam0": _cam(), "cam1": _cam()}
    frames = _frames(4, camera="cam0")
    rigs = {"body": {"cameras": {
        "cam0": {"q_wxyz": [1.0, 0.0, 0.0, 0.0], "tr": [0.0, 0.0, 0.0]},
        "cam1": {"q_wxyz": [1.0, 0.0, 0.0, 0.0], "tr": [0.12, 0.0, 0.0]}}}}
    t = np.round(np.arange(0, 4 / 30, 0.005), 6)
    samples = np.column_stack([
        t, np.full_like(t, 0.10), np.full_like(t, -0.02), np.full_like(t, 0.30),
        np.full_like(t, 0.01), np.full_like(t, 0.00), np.full_like(t, 9.81)])
    wl.write(path, cameras=cams, frames=frames, rgb=_rgb(4),
             imu=[wl.ImuStream("imu0", samples)], rigs=rigs,
             world={"metric_scale": True, "gravity_in_world": [0.0, 0.0, -1.0]},
             frames_format="json", fps=30)
    return _expect(cams, frames, rigs=rigs,
                   world={"metric_scale": True, "gravity_in_world": [0.0, 0.0, -1.0]},
                   imu={"imu0": [[float(x) for x in row] for row in samples]},
                   rgb_streams=["rgb"])


def v07_camera_models(path):
    """A distorted camera model: OPENCV with eight parameters, not PINHOLE."""
    cams = {"0": _cam("OPENCV", [17.6, 17.6, 8.0, 8.0,
                                 -0.283408, 0.073959, 0.000194, 1.7619e-05]),
            "fisheye": _cam("OPENCV_FISHEYE", [17.6, 17.6, 8.0, 8.0,
                                               -0.01, 0.002, -0.0003, 4e-05])}
    frames = _frames(3)
    wl.write(path, cameras=cams, frames=frames, rgb=_rgb(3),
             world={"metric_scale": True}, frames_format="json", fps=30)
    return _expect(cams, frames, world={"metric_scale": True}, rgb_streams=["rgb"])


def v08_float16_signal(path):
    """A float16_bits signal (SPEC §6.1): the codes *are* the floats."""
    src = np.stack([np.full((H, W), 3.5, np.float16) + np.float16(i)
                    for i in range(3)])
    cams = {"0": _cam()}
    frames = _frames(3)
    sig = wl.SignalMeta("hdr_r", "custom", {"type": "float16_bits"})
    wl.write(path, cameras=cams, frames=frames, rgb=None,
             signals={"hdr_r": np.ascontiguousarray(src).view(np.uint16)},
             signal_meta=[sig], world={"metric_scale": True},
             frames_format="json", fps=30)
    return _expect(cams, frames, signals=[sig], world={"metric_scale": True})


def v09_awkward_strings(path):
    """Unicode and escapes in the document, where a JSON parser earns its keep."""
    cams = {"caméra/0": _cam()}
    frames = _frames(3, camera="caméra/0")
    world = {"metric_scale": True,
             "description": 'quote " backslash \\ newline \n tab \t emoji 😀 ünïcode'}
    wl.write(path, cameras=cams, frames=frames, rgb=_rgb(3), world=world,
             frames_format="json", fps=30)
    return _expect(cams, frames, world=world, rgb_streams=["rgb"])


def v10_single_frame(path):
    """One frame. Off-by-one handling in a reader shows up here first."""
    cams = {"0": _cam()}
    frames = _frames(1)
    wl.write(path, cameras=cams, frames=frames, rgb=_rgb(1),
             world={"metric_scale": True}, frames_format="binary", fps=30)
    return _expect(cams, frames, world={"metric_scale": True}, rgb_streams=["rgb"])


def v11_mixed_resolution(path):
    """Depth stored at its own resolution beside the RGB (SPEC §4.6, v1.3).

    Half-size here; the motivating case is a 256x192 LiDAR map beside
    full-resolution RGB. Pre-v4 chromapakz readers must fail loudly on the
    depth stream, not return misshapen data.
    """
    import chromapakz as cz
    cams = {"0": _cam()}
    frames = _frames(4)
    vm = {"type": "inverse_depth", "near": 0.3, "far": 9.0, "levels": 65536,
          "invalid": 0}
    sig = wl.SignalMeta("depth", "depth", vm)
    yy, xx = np.mgrid[0:H // 2, 0:W // 2].astype(np.float32)
    depth = np.stack([(1.5 + 0.4 * np.sin(xx / 2 + i * 0.2)).astype(np.float32)
                      for i in range(4)])
    wl.write(path, cameras=cams, frames=frames, rgb=_rgb(4),
             signals={"depth": cz.quantize_inverse(depth, near=0.3, far=9.0)},
             specs={"depth": {"inverse_depth": True, "near": 0.3, "far": 9.0}},
             signal_meta=[sig], world={"metric_scale": True},
             frames_format="json", fps=30)
    return _expect(cams, frames, signals=[sig], world={"metric_scale": True},
                   rgb_streams=["rgb"])


VECTOR_FNS = [v01_minimal, v02_depth, v03_binary_frames, v04_unposed, v05_stereo,
              v06_rig_imu, v07_camera_models, v08_float16_signal,
              v09_awkward_strings, v10_single_frame, v11_mixed_resolution]


def build(out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    index = {"format": "wurld-conformance", "version": 1, "vectors": []}
    for fn in VECTOR_FNS:
        name = fn.__name__
        path = out_dir / f"{name}.wurld.webm"
        expected = fn(path)
        (out_dir / f"{name}.expected.json").write_text(
            json.dumps(expected, indent=2, sort_keys=True) + "\n")
        index["vectors"].append({
            "name": name,
            "file": path.name,
            "expected": f"{name}.expected.json",
            "description": (fn.__doc__ or "").strip().split("\n")[0],
            "bytes": path.stat().st_size,
        })
    (out_dir / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    return index


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="regenerate into a temp dir and fail if anything differs")
    args = ap.parse_args()

    if not args.check:
        index = build(VECTORS)
        total = sum(v["bytes"] for v in index["vectors"])
        print(f"wrote {len(index['vectors'])} vectors to {VECTORS} ({total/1024:.1f} KiB)")
        return 0

    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        fresh = Path(tmp) / "vectors"
        build(fresh)
        differences = []
        for f in sorted(fresh.glob("*.expected.json")) + [fresh / "index.json"]:
            old = VECTORS / f.name
            if not old.exists():
                differences.append(f"{f.name}: missing from vectors/")
            elif old.read_text() != f.read_text():
                differences.append(f"{f.name}: differs")
    # Only the expectations are compared: the .webm bytes depend on the encoder
    # build, so byte-identity across libvpx versions is not something to demand.
    if differences:
        print("conformance expectations are stale:", file=sys.stderr)
        for d in differences:
            print(f"  {d}", file=sys.stderr)
        print("run: python conformance/generate.py", file=sys.stderr)
        return 1
    print("conformance expectations are up to date")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
