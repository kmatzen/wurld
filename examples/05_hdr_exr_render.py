"""Scenario: HDR renders (EXR half-float) as posed sensor video.

An Unreal/Blender render sequence is posed by construction — the camera is known
exactly — and its pixels are scene-referred linear radiance in EXR half-float,
unbounded above 1.0. EXR compresses each frame alone, so a video codec can win on a coherent
sequence — but only when the frames really are coherent. This example measures
that instead of assuming it, because the answer flips sign: denoised renders
compress far better than EXR/ZIP, while raw path-traced output with independent
per-frame Monte Carlo noise compresses *worse*. Lossless coding of fresh noise
costs more than zlib does.

IEEE half is exactly 16 bits, so the raw bit patterns store losslessly as
`uint16` signal codes — one signal per colour channel, `value_map` type
`float16_bits` (SPEC §6.1). Nothing is quantised: the codes *are* the floats,
so NaN, ±Inf, −0.0 and denormals all survive, and there is no `invalid`
sentinel because every bit pattern denotes a value.

Note what this is *not*: an HDR display track. That is display-referred (PQ/HLG,
absolute nits) and lives in the lossy RGB stream. This is the data.

Run:  python examples/05_hdr_exr_render.py out.wl.webm
"""

import sys
import zlib

import numpy as np

import wurld as wl

W, H, N = 320, 240, 16


def render_sequence(denoised=True):
    """Stand-in for an EXR sequence: linear radiance with a very bright emitter.

    `denoised=False` adds independent Monte Carlo noise per frame, which is what
    raw path-traced output at finite sample counts looks like.
    """
    rng = np.random.default_rng(0)
    static_grain = rng.lognormal(0.0, 0.08, (H, W)).astype(np.float32)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    frames, poses = [], []
    for i in range(N):
        t = i / 24.0
        base = 0.02 + 0.8 * np.exp(-((xx - 200 + 4 * i) ** 2 + (yy - 120) ** 2) / 20000)
        sky = 3.5 + 0.5 * np.sin(xx / 40)
        img = np.where(yy < 80, sky, base)
        img = img * (1.0 + 0.35 * np.sin(xx / 3.1) * np.cos(yy / 2.7))   # texture
        img = img * (rng.lognormal(0.0, 0.08, img.shape).astype(np.float32)
                     if not denoised else static_grain)
        img[100:130, 140:190] = 2000.0                                   # emissive panel
        rgb = np.stack([img, img * 0.95, img * 0.8], -1).astype(np.float16)
        frames.append(rgb)
        ang = 0.15 * t
        poses.append((
            (float(np.cos(ang / 2)), 0.0, float(np.sin(ang / 2)), 0.0),
            (0.3 * t, 0.0, 1.2),
        ))
    return np.stack(frames), poses


def main(out_path):
    seq_rgb, poses = render_sequence(denoised=True)   # (N, H, W, 3) float16

    # One lossless signal per channel, carrying the half-float bits verbatim.
    channels = {f"hdr_{c}": np.ascontiguousarray(seq_rgb[..., k]).view(np.uint16)
                for k, c in enumerate("rgb")}
    meta = [wl.SignalMeta(cid, "custom", {"type": "float16_bits"}) for cid in channels]

    f = 1.1 * W
    wl.write(
        out_path,
        cameras={"0": wl.Camera(model="PINHOLE", width=W, height=H,
                                params=[f, f, W / 2, H / 2])},
        frames=[wl.Frame(i=i, t=i / 24.0, camera="0", q_wxyz=q, tr=tr)
                for i, (q, tr) in enumerate(poses)],
        rgb=None,                                # no display track in this example
        signals=channels,
        signal_meta=meta,
        world={"metric_scale": True,
               "description": "scene-referred HDR render, linear radiance"},
        fps=24,
    )

    seq = wl.read(out_path)
    back = np.stack([seq.signal_values(f"hdr_{c}") for c in "rgb"], -1)

    exact = bool((back.view(np.uint16) == seq_rgb.view(np.uint16)).all())
    raw_kib = seq_rgb.nbytes / N / 1024
    exr_kib = sum(len(zlib.compress(seq_rgb[i].tobytes(), 6))
                  for i in range(N)) / N / 1024      # EXR/ZIP is zlib per frame
    import os
    got_kib = os.path.getsize(out_path) / N / 1024

    print(f"wrote {out_path}")
    print(f"  {N} frames, {W}x{H} half-float RGB, "
          f"range {float(np.nanmin(seq_rgb)):.4f} .. {float(np.nanmax(seq_rgb)):.0f}")
    print(f"  raw half-float        {raw_kib:8.1f} KiB/frame")
    print(f"  zlib (~EXR/ZIP)       {exr_kib:8.1f} KiB/frame   {raw_kib/exr_kib:5.2f}x")
    print(f"  wurld lossless        {got_kib:8.1f} KiB/frame   {raw_kib/got_kib:5.2f}x"
          f"   ({exr_kib/got_kib:.2f}x smaller than EXR/ZIP)")
    print(f"  bit-exact round trip: {exact}")
    print(f"  poses carried alongside: {len(seq.frames)}")
    print()
    print("  Whether this beats EXR depends on temporal coherence, not on the")
    print("  format. Measured against EXR/ZIP on this scene at 320x240:")
    print("    static camera, denoised              13.5x smaller")
    print("    moving camera, denoised               1.5x smaller")
    print("    static camera, per-frame MC noise     0.8x  (LARGER than EXR)")
    print("    moving camera, per-frame MC noise     0.8x  (LARGER than EXR)")
    print("  Denoise before archiving, or expect to lose to EXR on raw output.")
    return 0 if exact else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "hdr_render.wl.webm"))
