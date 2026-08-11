"""Scenario: a feed-forward reconstruction model's output, written as wurld.

DUSt3R / MASt3R / VGGT-style models predict camera poses and dense depth in one
pass, with no SfM stage. That output has properties a ground-truth dataset does
not: poses are *estimates* with varying reliability, depth is *predicted* rather
than measured, and some frames fail outright.

The interesting question for a container is whether it can carry the uncertainty
honestly rather than flattening it into "poses". wurld can:

  - per-frame `pose_valid=False` for frames the model could not localise, so a
    consumer skips them instead of training on a wrong pose;
  - a `confidence` signal alongside depth, which is how these models report
    per-pixel reliability, carried at full resolution rather than as a scalar;
  - `world.metric_scale=False`, because a monocular feed-forward model recovers
    geometry only up to scale — a consumer that needs metres must know that.

Run:  python examples/01_feedforward_reconstruction.py out.wurld.webm
"""

import sys

import numpy as np

import wurld as wl

W, H, N = 160, 120, 24


def fake_model_output(rng):
    """Stand-in for a VGGT-style forward pass: poses, depth, per-pixel confidence.

    The numbers are synthetic; the *shape* of the output is what matters here —
    every field below is something these models actually emit.
    """
    yy, xx = np.mgrid[0:H, 0:W]
    depth, conf, poses, valid = [], [], [], []
    for i in range(N):
        # A plane receding from the camera, plus a bump, in arbitrary units.
        z = 2.0 + 0.6 * np.sin(xx / 25 + i * 0.05) + 0.4 * (yy / H)
        # Models are least certain at image borders and on distant surfaces.
        c = np.clip(1.0 - 0.6 * (z - z.min()) / (np.ptp(z) + 1e-6), 0, 1)
        c[:8, :] = c[-8:, :] = c[:, :8] = c[:, -8:] = 0.15
        # Frames 9..11: the model lost tracking. It still returns *something*,
        # and a naive pipeline would happily train on it.
        ok = not (9 <= i <= 11)
        ang = i * 0.04
        q = np.array([np.cos(ang / 2), 0.0, np.sin(ang / 2), 0.0])
        t = np.array([0.05 * i, 0.0, 0.0])
        depth.append(z.astype(np.float32))
        conf.append(c.astype(np.float32))
        poses.append((q, t))
        valid.append(ok)
    return depth, conf, poses, valid


def main(out_path):
    rng = np.random.default_rng(0)
    depth, conf, poses, valid = fake_model_output(rng)

    NEAR, FAR = 0.5, 12.0
    # wl.write() takes uint16 codes, not metres: quantisation is the caller's
    # decision (near/far bound the usable range), and `specs` records it so the
    # file says how to invert it.
    import chromapakz as cz
    metres = np.stack([np.clip(z, NEAR, FAR) for z in depth]).astype(np.float32)
    codes = cz.quantize_inverse(metres, near=NEAR, far=FAR)
    # Confidence is continuous here (0..1), unlike a LiDAR's 3-level flag, so it
    # is quantised linearly into the uint16 signal and the value_map says so.
    conf_codes = np.stack([np.clip(c * 65535, 0, 65535).astype(np.uint16) for c in conf])

    frames = []
    for i, ((q, t), ok) in enumerate(zip(poses, valid)):
        frames.append(wl.Frame(
            i=i, t=i / 30.0, camera="0",
            q_wxyz=tuple(q) if ok else None,
            tr=tuple(t) if ok else None,
            pose_valid=ok,
        ))

    f = 0.9 * W
    wl.write(
        out_path,
        cameras={"0": wl.Camera(model="PINHOLE", width=W, height=H,
                                params=[f, f, W / 2, H / 2])},
        frames=frames,
        rgb=rng.integers(60, 200, (N, H, W, 4), dtype=np.uint8),
        signals={"depth": codes, "confidence": conf_codes},
        specs={"depth": {"inverse_depth": True, "near": NEAR, "far": FAR}},
        signal_meta=[
            wl.SignalMeta("depth", "depth",
                          {"type": "inverse_depth", "near": NEAR, "far": FAR,
                           "levels": 65536, "invalid": 0}),
            wl.SignalMeta("confidence", "confidence",
                          {"type": "linear", "scale": 1.0 / 65535, "offset": 0.0}),
        ],
        world={
            # The claim a monocular feed-forward model cannot make.
            "metric_scale": False,
            "description": "feed-forward reconstruction; geometry up to scale",
        },
        fps=30,
    )

    seq = wl.read(out_path)
    lost = [fr.i for fr in seq.frames if not fr.pose_valid]
    print(f"wrote {out_path}")
    print(f"  {len(seq.frames)} frames, {len(lost)} without a pose: {lost}")
    print(f"  metric_scale = {seq.world.get('metric_scale')} "
          "(a consumer needing metres must refuse or rescale)")
    c = seq.signal("confidence")
    print(f"  confidence: {c.shape}, {c.min()}..{c.max()} raw codes")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "feedforward.wurld.webm"))
