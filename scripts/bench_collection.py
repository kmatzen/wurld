"""Measure a collection at scale, because the docs claim a scale nothing tested.

`wurld/collection.py` opens with "Training holds ten thousand of them". The test
suite's largest corpus is five files. This closes that gap with numbers rather
than confidence.

Each case runs in its own process so peak RSS is attributable to it
(`ru_maxrss` of the child), and every case prints what it measured rather than
whether it passed — a benchmark that only says "fast" is not evidence.

    python scripts/bench_collection.py --members 200 --frames 12
    python scripts/bench_collection.py --case memory --frames 600 --width 320
    python scripts/bench_collection.py --members 10000 --frames 4 --width 32

`--case` runs one measurement in-process; without it the script forks a child
per case and prints a table.
"""

import argparse
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import wurld as wl  # noqa: E402
from wurld import collection as col  # noqa: E402

CASES = ["index", "load", "locate", "stream_meta", "stream_pixels", "verify",
         "peak_collection", "peak_sequence"]


def build_corpus(root: Path, members: int, frames: int, w: int, h: int) -> Path:
    """Write `members` small wurld files. Reused across runs when it already fits."""
    import chromapakz as cz

    root.mkdir(parents=True, exist_ok=True)
    marker = root / ".corpus.json"
    want = {"members": members, "frames": frames, "w": w, "h": h}
    if marker.exists() and json.loads(marker.read_text()) == want:
        return root

    for p in root.glob("*.wurld.webm"):
        p.unlink()
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    for k in range(members):
        rgb = np.stack([
            np.dstack([np.clip((0.5 + 0.4 * np.sin(xx / 5 + i * 0.3 + k)) * 255, 0, 255)] * 3
                      + [np.full((h, w), 255, np.float32)]).astype(np.uint8)
            for i in range(frames)])
        depth = np.stack([(1.5 + 0.4 * np.sin(xx / 7 + i * 0.2)).astype(np.float32)
                          for i in range(frames)])
        f = 1.1 * w
        wl.write(
            root / f"take{k:06d}.wurld.webm",
            cameras={"0": wl.Camera("PINHOLE", w, h, [f, f, w / 2, h / 2])},
            frames=[wl.Frame(i=i, t=i / 30, camera="0", q_wxyz=(1.0, 0.0, 0.0, 0.0),
                             tr=(0.01 * i, 0.0, 0.5)) for i in range(frames)],
            rgb=rgb,
            signals={"depth": cz.quantize_inverse(depth, near=0.3, far=9.0)},
            specs={"depth": {"inverse_depth": True, "near": 0.3, "far": 9.0}},
            signal_meta=[wl.SignalMeta("depth", "depth",
                                       {"type": "inverse_depth", "near": 0.3, "far": 9.0,
                                        "levels": 65536, "invalid": 0})],
            world={"metric_scale": True}, fps=30)
        if members > 200 and k % 500 == 0 and k:
            print(f"    ... {k}/{members} written", file=sys.stderr)
    marker.write_text(json.dumps(want))
    return root


def peak_rss_mb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KiB, macOS bytes.
    return r / (1024 * 1024) if sys.platform == "darwin" else r / 1024


def run_case(case: str, root: Path, members: int, frames: int) -> dict:
    manifest_path = root / "collection.json"
    out: dict = {}

    if case == "index":
        t0 = time.perf_counter()
        m, failures = col.build_manifest(root, relative_to=root)
        dt = time.perf_counter() - t0
        m.write(manifest_path)
        out = {"seconds": dt, "members": len(m.members),
               "per_member_ms": dt / max(1, len(m.members)) * 1000,
               "manifest_kib": manifest_path.stat().st_size / 1024,
               "failures": len(failures)}

    elif case == "load":
        t0 = time.perf_counter()
        c = col.Collection.read(manifest_path)
        dt = time.perf_counter() - t0
        out = {"seconds": dt, "frames": len(c), "peak_rss_mb": peak_rss_mb()}

    elif case == "locate":
        c = col.Collection.read(manifest_path)
        n = len(c)
        idx = np.random.default_rng(0).integers(0, n, size=min(200_000, n * 20))
        t0 = time.perf_counter()
        for i in idx:
            c.locate(int(i))
        dt = time.perf_counter() - t0
        out = {"lookups": len(idx), "seconds": dt, "per_lookup_us": dt / len(idx) * 1e6}

    elif case in ("stream_meta", "stream_pixels"):
        fields = () if case == "stream_meta" else ("rgb", "depth")
        c = col.Collection.read(manifest_path)
        t0 = time.perf_counter()
        n = sum(1 for _ in c.iter_frames(fields=fields))
        dt = time.perf_counter() - t0
        out = {"frames": n, "seconds": dt, "fps": n / dt if dt else 0,
               "peak_rss_mb": peak_rss_mb()}

    elif case == "verify":
        c = col.Collection.read(manifest_path)
        t0 = time.perf_counter()
        drift = c.verify()
        dt = time.perf_counter() - t0
        out = {"seconds": dt, "members": len(c.members), "drift": len(drift),
               "per_member_ms": dt / max(1, len(c.members)) * 1000}

    elif case == "peak_collection":
        # ru_maxrss is monotonic, so the two paths must be compared in separate
        # processes; measuring both in one only reports the larger.
        c = col.Collection.read(manifest_path)
        base = peak_rss_mb()
        n = sum(1 for _ in c.iter_frames(fields=("rgb", "depth")))
        out = {"frames": n, "rss_start_mb": base, "peak_rss_mb": peak_rss_mb()}

    elif case == "peak_sequence":
        # The bounded-memory reference: container.Sequence.iter_frames decodes
        # one Cluster at a time. A collection streaming the same bytes should
        # not cost dramatically more.
        c = col.Collection.read(manifest_path)
        base = peak_rss_mb()
        total = 0
        for mi in range(len(c.members)):
            seq = wl.read(c.resolve(c.members[mi]))
            total += sum(1 for _ in seq.iter_frames())
            del seq
        out = {"frames": total, "rss_start_mb": base, "peak_rss_mb": peak_rss_mb()}

    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--members", type=int, default=200)
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--height", type=int, default=24)
    ap.add_argument("--root", default=None, help="corpus directory (kept, for reuse)")
    ap.add_argument("--case", choices=CASES, help="run one case in this process")
    args = ap.parse_args()

    root = Path(args.root or (Path(os.environ.get("TMPDIR", "/tmp")) /
                              f"wurld-bench-{args.members}x{args.frames}"
                              f"-{args.width}x{args.height}"))

    if args.case:
        print(json.dumps(run_case(args.case, root, args.members, args.frames)))
        return 0

    print(f"corpus: {args.members} members x {args.frames} frames "
          f"@ {args.width}x{args.height} -> {root}")
    t0 = time.perf_counter()
    build_corpus(root, args.members, args.frames, args.width, args.height)
    on_disk = sum(p.stat().st_size for p in root.glob("*.wurld.webm"))
    print(f"  built in {time.perf_counter() - t0:.1f}s, {on_disk / 1024 / 1024:.1f} MiB "
          f"on disk\n")

    for case in CASES:
        if case.startswith("peak_") and args.frames < 100:
            continue           # only meaningful with a member big enough to hurt
        r = subprocess.run([sys.executable, __file__, "--case", case,
                            "--members", str(args.members), "--frames", str(args.frames),
                            "--width", str(args.width), "--height", str(args.height),
                            "--root", str(root)], capture_output=True, text=True)
        if r.returncode != 0:
            print(f"{case:14} FAILED\n{r.stderr.strip()[-400:]}")
            continue
        vals = json.loads(r.stdout)
        child_peak = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        vals["child_peak_rss_mb"] = round(
            child_peak / (1024 * 1024) if sys.platform == "darwin" else child_peak / 1024, 1)
        print(f"{case:14} " + "  ".join(
            f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in vals.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
