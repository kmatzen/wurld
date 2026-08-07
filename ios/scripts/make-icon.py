#!/usr/bin/env python3
"""Render WurldCam's app icon: a camera frustum and the depth samples it sees.

Geometry is a real pinhole projection rather than a drawing, so the icon says
what the app does. Supersampled 4x and downsampled for antialiasing; output is
opaque sRGB with square corners, as the App Store requires (iOS masks it).

    python ios/scripts/make-icon.py [out.png]
"""

from __future__ import annotations

import math
import sys

import numpy as np
from PIL import Image, ImageDraw

SS = 4                      # supersample factor
SIZE = 1024
N = SIZE * SS

BG_TOP = (13, 20, 34)       # deep navy, matching the viewer's clear colour
BG_BOT = (5, 8, 16)
FRUSTUM = (255, 191, 64)    # viewer frustum amber
NEAR = (166, 214, 255)      # near depth samples
FAR = (104, 146, 233)       # far depth samples


def project(pts, fx, cx, cy, eye, look):
    """Pinhole-project world points; returns pixel coords and camera depth."""
    fwd = look - eye
    fwd /= np.linalg.norm(fwd)
    up0 = np.array([0.0, 0.0, 1.0])
    right = np.cross(fwd, up0)
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    rel = pts - eye
    z = rel @ fwd
    x = rel @ right
    y = rel @ up
    ok = z > 1e-3
    u = np.full(len(pts), np.nan)
    v = np.full(len(pts), np.nan)
    u[ok] = fx * x[ok] / z[ok] + cx
    v[ok] = -fx * y[ok] / z[ok] + cy
    return u, v, z


def main() -> None:
    out = sys.argv[1] if len(sys.argv) > 1 else "AppIcon.png"

    img = Image.new("RGB", (N, N))
    d = ImageDraw.Draw(img)
    for row in range(N):                       # vertical gradient backdrop
        f = row / (N - 1)
        d.line([(0, row), (N, row)],
               fill=tuple(int(a + (b - a) * f) for a, b in zip(BG_TOP, BG_BOT)))

    # The viewing camera looks along +y; the depicted frustum points along +x,
    # so we see the pyramid side-on rather than down its axis (which collapses
    # it to a rectangle). Keep the two axes near-perpendicular.
    eye = np.array([0.10, -1.85, 0.30])
    look = np.array([0.16, 2.70, -0.02])
    # Principal point offset so the frustum-plus-cloud composition lands centred
    # in the square and fills ~80% of it; iOS icons read poorly with wide margins.
    fx, cx, cy = 1.22 * N, N * 0.384, N * 0.514

    # A curved sheet of depth samples beyond the frustum's far plane, spanning
    # the image plane (x across, z up) so it reads as a broad patch rather than
    # a foreshortened sliver. Spacing is kept wide enough that the samples stay
    # discrete dots at small icon sizes instead of merging into a blob.
    rng = np.random.default_rng(7)
    gu, gv = np.meshgrid(np.linspace(-1.0, 1.0, 25), np.linspace(-1.0, 1.0, 27))
    gu = gu.ravel(); gv = gv.ravel()
    surf = np.stack([
        1.34 + gu * 0.72,
        2.74 + 0.30 * np.cos(gu * 1.5) + 0.16 * np.sin(gv * 1.9),
        0.04 + gv * 1.02,
    ], axis=1)
    surf += rng.normal(0, 0.004, surf.shape)

    u, v, z = project(surf, fx, cx, cy, eye, look)
    order = np.argsort(-z)                      # far points first
    zmin, zmax = np.nanmin(z), np.nanmax(z)
    for i in order:
        if not (np.isfinite(u[i]) and np.isfinite(v[i])):
            continue
        t = (z[i] - zmin) / max(1e-6, zmax - zmin)
        col = tuple(int(a + (b - a) * t) for a, b in zip(NEAR, FAR))
        r = (1.0 - 0.32 * t) * 6.6 * SS
        d.ellipse([u[i] - r, v[i] - r, u[i] + r, v[i] + r], fill=col)

    # The camera frustum, projected through the same virtual lens.
    apex = np.array([-0.86, 2.62, 0.02])
    fwd = np.array([1.0, 0.06, 0.02]); fwd /= np.linalg.norm(fwd)
    rt = np.cross(fwd, [0, 0, 1.0]); rt /= np.linalg.norm(rt)
    up = np.cross(rt, fwd)
    depth, hw, hh = 1.30, 0.62, 0.56
    corners = np.array([apex + fwd * depth + rt * sx * hw + up * sy * hh
                        for sx, sy in ((-1, 1), (1, 1), (1, -1), (-1, -1))])
    pts = np.vstack([apex[None, :], corners])
    pu, pv, _ = project(pts, fx, cx, cy, eye, look)
    w = int(8.5 * SS)
    for a, b in ((0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4), (4, 1)):
        d.line([(pu[a], pv[a]), (pu[b], pv[b])], fill=FRUSTUM, width=w)
    r = 11.0 * SS
    d.ellipse([pu[0] - r, pv[0] - r, pu[0] + r, pv[0] + r], fill=FRUSTUM)

    img.resize((SIZE, SIZE), Image.LANCZOS).save(out, "PNG")
    print(f"wrote {out} ({SIZE}x{SIZE}, opaque sRGB)")


if __name__ == "__main__":
    main()
