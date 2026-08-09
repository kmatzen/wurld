"""Scenario: a stereo rig storing both cameras' pixels in one file.

Until ChromaPakZ 0.7.0 a stereo pair meant two files or one camera's pixels.
Now both eyes live in the same clusters on the same timeline, and wurld binds
them to calibration by id: a display stream named `cam1` carries the pixels
`cameras["cam1"]` calibrates (SPEC §4.4). That binding is the point — without
it a reader has two image sequences and no way to know which intrinsics apply.

Poses stay single-camera. `cam0` has the trajectory; `cam1`'s pose is derived
through `rigs`, because a rigid baseline belongs in the calibration once rather
than restated on every frame where it can drift away from the trajectory.

Run:  python examples/06_stereo_rig.py out.wl.webm
"""

import sys

import numpy as np

import wurld as wl

W, H, N, FPS = 160, 120, 20, 30
BASELINE = 0.12          # metres between the two cameras


def stereo_pair(i, rng):
    """A scene shifted by disparity, so the two views genuinely differ."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    depth = 1.5 + 0.8 * np.exp(-((xx - 80 + i) ** 2 + (yy - 60) ** 2) / 900)
    shade = (0.45 + 0.4 * np.sin(xx / 9 + i * 0.2) * np.cos(yy / 7)) * 255
    left = np.clip(shade, 0, 255).astype(np.uint8)
    # Horizontal disparity: closer surfaces shift further between the eyes.
    fx = 1.1 * W
    disp = np.clip(fx * BASELINE / depth, 0, W - 1).astype(np.int32)
    cols = np.clip(xx.astype(np.int32) - disp, 0, W - 1)
    right = left[yy.astype(np.int32), cols]

    def rgba(g):
        return np.dstack([g, g, g, np.full_like(g, 255)])

    return rgba(left), rgba(right), depth.astype(np.float32)


def main(out_path):
    import chromapakz as cz
    rng = np.random.default_rng(0)

    fx = 1.1 * W
    cams = {
        cid: wl.Camera(model="PINHOLE", width=W, height=H, params=[fx, fx, W / 2, H / 2])
        for cid in ("cam0", "cam1")
    }

    left, right, depth = [], [], []
    for i in range(N):
        l, r, d = stereo_pair(i, rng)
        left.append(l); right.append(r); depth.append(d)

    frames = [wl.Frame(i=i, t=i / FPS, camera="cam0",
                       q_wxyz=(1.0, 0.0, 0.0, 0.0), tr=(0.02 * i, 0.0, 0.0))
              for i in range(N)]

    wl.write(
        out_path,
        cameras=cams,
        frames=frames,
        # Two display streams, keyed by camera id. cam0 is declared first and so
        # is the primary: it keeps track 1 and the name "rgb", which is what an
        # older reader decodes while ignoring the rest.
        rgb={"cam0": np.stack(left), "cam1": np.stack(right)},
        signals={"depth": cz.quantize_inverse(np.stack(depth), near=0.3, far=9.0)},
        specs={"depth": {"inverse_depth": True, "near": 0.3, "far": 9.0}},
        signal_meta=[wl.SignalMeta("depth", "depth",
                                   {"type": "inverse_depth", "near": 0.3, "far": 9.0,
                                    "levels": 65536, "invalid": 0})],
        rigs={"body": {"cameras": {
            "cam0": {"q_wxyz": [1.0, 0.0, 0.0, 0.0], "tr": [0.0, 0.0, 0.0]},
            "cam1": {"q_wxyz": [1.0, 0.0, 0.0, 0.0], "tr": [BASELINE, 0.0, 0.0]},
        }}},
        world={"metric_scale": True, "description": "stereo rig, both eyes stored"},
        fps=FPS,
    )

    seq = wl.read(out_path)
    print(f"wrote {out_path}")
    print(f"  streams: {seq.rgb_streams}  (primary first)")
    print(f"  cameras: {sorted(seq.cameras)} — ids match the streams, which is how a "
          "reader knows\n           which intrinsics apply to which pixels")

    l0, r0 = seq.rgb_for("cam0"), seq.rgb_for("cam1")
    differ = float(np.abs(l0[5].astype(int) - r0[5].astype(int)).mean())
    print(f"  the two eyes really differ: mean |L-R| = {differ:.1f}/255")

    # cam1 never had a pose stored; it comes from the rig.
    c0, c1 = seq.c2w(5), seq.rig_c2w(5, "cam1")
    print(f"  poses: stored for {seq.frames[0].camera} only; cam1 derived, "
          f"baseline {np.linalg.norm(c1[:3, 3] - c0[:3, 3]) * 100:.1f} cm")
    print(f"  depth: {seq.depth_meters(5).shape}, shared by both views")
    print("\n  An older reader sees one RGB track and decodes it unchanged.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "stereo.wl.webm"))
