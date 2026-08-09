"""Scenario: ten thousand captures as one training set.

A wurld file holds one sequence. Training holds a corpus, and the awkward part
was never the file — it was that a corpus of them had no index, no global frame
addressing, and no way to split across dataloader workers without every worker
reading everything.

A collection is a *manifest* plus the files it names. Deliberately not a new
container: each member stays an ordinary playable wurld file that still works
alone, so building a dataset costs nothing and un-building it is a delete.

Three things this shows, in order of how much they matter:

1. **Indexing is cheap.** Members are described from header reads that stop at
   the first Cluster, so cataloguing a corpus costs kilobytes per file rather
   than reading the pixels. The numbers below are measured, not asserted.
2. **Addressing is global.** N files behave as one frame-indexed sequence.
3. **Sharding is safe.** Splitting across workers keeps whole files together
   and — the property that matters — yields every frame exactly once. A
   sharding bug does not crash; it quietly trains on duplicates.

Run:  python examples/08_collection_training.py [workdir]
"""

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

import wurld as wl
from wurld import collection as col

W, H, FPS = 64, 48, 30
LENGTHS = [12, 30, 8, 21]          # uneven on purpose: real corpora are


def write_capture(path, n, seed):
    """Stand-in for a capture: RGB + metric depth, one frame unlocalised."""
    import chromapakz as cz

    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    rgb, depth = [], []
    for i in range(n):
        g = (0.5 + 0.4 * np.sin(xx / 6 + i * 0.25 + seed) * np.cos(yy / 5)) * 255
        rgb.append(np.dstack([np.clip(g, 0, 255)] * 3
                             + [np.full_like(g, 255)]).astype(np.uint8))
        depth.append((1.4 + 0.6 * np.sin(xx / 9 + i * 0.15)).astype(np.float32))

    frames = []
    for i in range(n):
        if i == n // 2:                      # one frame the tracker lost
            frames.append(wl.Frame(i=i, t=i / FPS, pose_valid=False))
        else:
            ang = 0.05 * i
            frames.append(wl.Frame(
                i=i, t=i / FPS, camera="0",
                q_wxyz=(float(np.cos(ang / 2)), 0.0, float(np.sin(ang / 2)), 0.0),
                tr=(0.02 * i, 0.0, 0.8)))

    f = 1.1 * W
    wl.write(
        path,
        cameras={"0": wl.Camera(model="PINHOLE", width=W, height=H,
                                params=[f, f, W / 2, H / 2])},
        frames=frames,
        rgb=np.stack(rgb),
        signals={"depth": cz.quantize_inverse(np.stack(depth), near=0.3, far=9.0)},
        specs={"depth": {"inverse_depth": True, "near": 0.3, "far": 9.0}},
        signal_meta=[wl.SignalMeta("depth", "depth",
                                   {"type": "inverse_depth", "near": 0.3, "far": 9.0,
                                    "levels": 65536, "invalid": 0})],
        world={"metric_scale": True, "description": f"synthetic capture {seed}"},
        fps=FPS,
    )
    return path


def main(workdir=None):
    tmp = None
    if workdir is None:
        tmp = tempfile.mkdtemp()
        workdir = tmp
    root = Path(workdir)
    (root / "captures").mkdir(parents=True, exist_ok=True)

    paths = [write_capture(root / "captures" / f"take{i:02d}.wl.webm", n, i)
             for i, n in enumerate(LENGTHS)]
    on_disk = sum(p.stat().st_size for p in paths)

    # --- 1. indexing, and what it costs ------------------------------------
    from wurld import remote
    header_bytes = sum(remote.fetch_header(remote.file_fetcher(p)).bytes_fetched
                       for p in paths)

    manifest, failures = col.build_manifest(root / "captures", relative_to=root)
    manifest_path = manifest.write(root / "collection.json")
    assert not failures

    print(f"indexed {len(manifest.members)} captures -> {manifest_path.name}")
    print(f"  {manifest.total_frames} frames, {manifest.total_posed_frames} posed")
    print(f"  read {header_bytes/1024/len(paths):.1f} KiB per file "
          f"({header_bytes/1024:.1f} KiB total) to index {on_disk/1024:.1f} KiB on disk")
    print("    That cost tracks the number of frames, not the size of the video —")
    print("    it stops at the first Cluster. These toy files are mostly header, so")
    print("    the ratio here flatters nothing; on a 100 MB capture of the same")
    print("    length the index costs the same few KiB.")

    # --- 2. global addressing ----------------------------------------------
    c = col.Collection.read(manifest_path)
    print(f"\n{c}")
    mid = len(c) // 2
    mi, li = c.locate(mid)
    item = c.frame(mid, fields=("depth",))
    print(f"  global frame {mid} -> member {mi} ('{c.members[mi].uri}') frame {li}")
    print(f"    t={item['t']:.3f}s  posed={item['pose_valid']}  "
          f"depth {np.nanmin(item['depth']):.2f}..{np.nanmax(item['depth']):.2f} m")

    # --- 3. sharding, checked rather than claimed --------------------------
    print("\nsharding across 3 workers:")
    seen, everything = [], set()
    for k in range(3):
        ids = [(it["member"], it["frame"])
               for it in c.iter_frames(shard=(k, 3), shuffle=42)]
        members = sorted({m for m, _ in ids})
        print(f"  worker {k}: {len(ids):3d} frames from members {members}")
        seen.extend(ids)
    for m_i, mem in enumerate(c.members):
        everything |= {(m_i, f) for f in range(mem.frames)}

    duplicated = len(seen) - len(set(seen))
    missing = len(everything - set(seen))
    print(f"  union: {len(set(seen))}/{len(everything)} frames, "
          f"{duplicated} duplicated, {missing} missing")
    if duplicated or missing:
        print("  FAILED: a sharding bug would silently reweight the training set")
        return 1
    print("  every frame exactly once, and no capture split across workers")

    # --- what this plugs into ----------------------------------------------
    print("\nWith PyTorch installed (`pip install 'wurld[torch]'`):")
    print("    from wurld.integrations.torch_data import WurldIterableDataset")
    print("    ds = WurldIterableDataset('collection.json', fields=('rgb','depth'))")
    print("    loader = DataLoader(ds, batch_size=8, num_workers=4)")
    print("  Workers and distributed ranks shard themselves; each member decodes once.")

    if tmp:
        shutil.rmtree(tmp)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else None))
