"""Scenario: feeding a 3DGS / NeRF trainer from a wurld file.

nerfstudio-family trainers want a `transforms.json` — intrinsics, a per-image
camera-to-world matrix, and image paths on disk. Depth-supervised variants
(DN-Splatter and friends) additionally want a depth map per image, which is
normally a second directory the user has to align by filename and hope the
scales match.

The point here is not that wurld replaces `transforms.json` — trainers read
that, so wurld emits it. The point is that one file carries everything the
trainer needs, and the depth it emits is the *same* depth the capture recorded,
bit-exact, in metres, with the invalid pixels still marked.

Run:  python examples/02_gaussian_splatting.py scene.wurld.webm outdir/
"""

import json
import sys
from pathlib import Path

import numpy as np

import wurld as wl


def main(src, outdir):
    outdir = Path(outdir)
    seq = wl.read(src)

    # The nerfstudio exporter writes transforms.json + images; ask for it directly.
    from wurld.converters import nerfstudio
    nerfstudio.to_transforms(src, outdir)

    tf = json.loads((outdir / "transforms.json").read_text())
    print(f"{outdir}/transforms.json")
    print(f"  {len(tf['frames'])} posed frames, "
          f"fl_x={tf.get('fl_x'):.2f} cx={tf.get('cx'):.2f} w={tf.get('w')} h={tf.get('h')}")

    # Depth supervision: write the metric depth beside the images, as the .npy a
    # trainer can load directly. NaN marks "no return" — a trainer must mask
    # those rather than treat them as zero distance, which is the usual bug when
    # depth arrives as a 16-bit PNG with 0 meaning both "invalid" and "at the
    # sensor origin".
    depth_dir = outdir / "depth"
    depth_dir.mkdir(exist_ok=True)
    n_valid = 0
    for k, frame in enumerate(seq.frames):
        if not frame.pose_valid:
            continue
        d = seq.depth_meters(frame.i)
        np.save(depth_dir / f"{frame.i:05d}.npy", d)
        n_valid += 1
    print(f"{depth_dir}/  {n_valid} metric depth maps (NaN = no return)")

    d0 = seq.depth_meters(seq.frames[0].i)
    print(f"  depth[0]: {np.isfinite(d0).mean()*100:.0f}% valid, "
          f"{np.nanmin(d0):.2f}..{np.nanmax(d0):.2f} m")

    # The convention that actually bites people. nerfstudio's transforms.json is
    # OpenGL-style (camera looks down -Z, +Y up); wurld stores RDF (+Z forward,
    # +Y down). The exporter converts on the way out — assert it did, because a
    # silent axis flip trains a model that renders mirrored.
    m = np.array(tf["frames"][0]["transform_matrix"])
    c2w = seq.c2w(seq.frames[0].i)
    flip = np.diag([1.0, -1.0, -1.0, 1.0])          # RDF -> OpenGL
    assert np.allclose(m, c2w @ flip, atol=1e-5), "exporter did not convert axes"
    print("  axis convention: RDF (wurld) -> OpenGL (nerfstudio), verified")

    # What a trainer still has to be told out-of-band, and wurld records it:
    print(f"  metric_scale = {seq.world.get('metric_scale')}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1], sys.argv[2]))
