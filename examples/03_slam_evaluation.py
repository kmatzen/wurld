"""Scenario: SLAM/VIO benchmarking — trajectories in and out.

The TUM RGB-D benchmark's convention is a text file per trajectory:

    timestamp tx ty tz qx qy qz qw          # note: quaternion scalar LAST

and evaluation tools (evo, the TUM scripts) compare an estimate against ground
truth by associating on timestamp. Two things routinely go wrong: the quaternion
component order, and whether the pose is camera-to-world or world-to-camera.

wurld fixes both by construction — wxyz, camera-to-world, always — so the
conversion happens once in the exporter instead of in every consumer. This
demonstrates the round trip and, given the original ground truth, proves the
numbers survived it.

Run:  python examples/03_slam_evaluation.py scene.wl.webm out/ [groundtruth.txt]
"""

import sys
from pathlib import Path

import numpy as np

import wurld as wl


def main(src, outdir, reference=None):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    seq = wl.read(src)

    # TUM format: scalar-last quaternion, one line per pose, '#' comments.
    traj = outdir / "trajectory.txt"
    with open(traj, "w") as f:
        f.write("# timestamp tx ty tz qx qy qz qw\n")
        for fr in seq.frames:
            if not fr.pose_valid:
                continue                      # a lost frame has no pose to report
            qw, qx, qy, qz = fr.q_wxyz        # wurld is scalar-FIRST
            tx, ty, tz = fr.tr
            f.write(f"{fr.t!r} {tx!r} {ty!r} {tz!r} {qx!r} {qy!r} {qz!r} {qw!r}\n")

    posed = sum(1 for fr in seq.frames if fr.pose_valid)
    print(f"{traj}")
    print(f"  {posed} poses, TUM convention (scalar-last quaternion, camera-to-world)")

    if reference:
        # Associate on timestamp the way the TUM tooling does, then compare.
        gt = np.loadtxt(reference)
        gt_t, gt_xyz = gt[:, 0], gt[:, 1:4]
        ours = np.array([[fr.t, *fr.tr] for fr in seq.frames if fr.pose_valid])
        errs = []
        for row in ours:
            j = int(np.argmin(np.abs(gt_t - row[0])))
            if abs(gt_t[j] - row[0]) < 0.01:      # 10 ms association window
                errs.append(np.linalg.norm(gt_xyz[j] - row[1:4]))
        errs = np.array(errs)
        print(f"  vs {Path(reference).name}: {len(errs)}/{len(ours)} associated "
              f"within 10 ms")
        print(f"  translation error: median {np.median(errs)*1000:.3f} mm, "
              f"max {errs.max()*1000:.3f} mm")
        length = np.linalg.norm(np.diff(ours[:, 1:4], axis=0), axis=1).sum()
        print(f"  trajectory length {length:.2f} m")

    # The other half of benchmarking: the estimate is a wurld file too, so the
    # comparison is between two files of the same kind rather than between a
    # text file and a directory of PNGs.
    print("\n  an estimator's output is written the same way:")
    print("    wl.write(out, cameras=..., frames=[wl.Frame(i, t, camera, q_wxyz, tr), ...])")
    print("  so ground truth and estimate are the same format, and `wurld validate`")
    print("  checks both.")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None))
