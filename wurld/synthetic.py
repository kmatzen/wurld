"""Synthetic posed-RGBD sequence: analytic raytrace of a simple scene.

A camera orbits a scene containing a matte sphere and a checkered ground
plane. Depth (along camera +Z) and RGB are computed analytically, so every
pixel has exact ground truth for tests and demos.
"""

from __future__ import annotations

import numpy as np

from . import conventions
from .container import Camera, Frame


def look_at_c2w(eye: np.ndarray, target: np.ndarray, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """Camera-to-world for an RDF camera at eye looking at target (world +Z up)."""
    fwd = target - eye
    fwd = fwd / np.linalg.norm(fwd)
    right = np.cross(fwd, np.asarray(up, dtype=np.float64))
    right = right / np.linalg.norm(right)
    down = np.cross(fwd, right)  # RDF: +Y is down = forward x right
    c2w = np.eye(4)
    c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = right, down, fwd, eye
    return c2w


def render_frame(c2w: np.ndarray, K: np.ndarray, width: int, height: int):
    """Return (rgb uint8 HxWx3, depth float64 HxW meters along +Z, 0=no hit)."""
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    dirs_cam = np.stack([(u - cx) / fx, (v - cy) / fy, np.ones_like(u, dtype=np.float64)], -1)
    R, eye = c2w[:3, :3], c2w[:3, 3]
    dirs = dirs_cam @ R.T  # world-space ray directions (unnormalized, z_cam=1)

    t_hit = np.full((height, width), np.inf)
    rgb = np.zeros((height, width, 3), dtype=np.float64)

    # Sphere: center (0,0,1), r=1, warm matte shaded by normal.
    center, radius = np.array([0.0, 0.0, 1.0]), 1.0
    oc = eye - center
    a = np.sum(dirs * dirs, -1)
    b = 2 * np.sum(dirs * oc, -1)
    c = float(oc @ oc) - radius * radius
    disc = b * b - 4 * a * c
    hit = disc > 0
    t_s = np.where(hit, (-b - np.sqrt(np.maximum(disc, 0))) / (2 * a), np.inf)
    t_s = np.where(t_s > 1e-6, t_s, np.inf)
    p = eye + dirs * t_s[..., None]
    n = (p - center) / radius
    shade = np.clip(n @ np.array([0.5, 0.3, 0.8]), 0, 1)[..., None]
    sphere_rgb = np.array([0.9, 0.45, 0.2]) * (0.25 + 0.75 * shade)
    mask = t_s < t_hit
    t_hit = np.where(mask, t_s, t_hit)
    rgb = np.where(mask[..., None], sphere_rgb, rgb)

    # Ground plane z=0, checkerboard.
    dz = dirs[..., 2]
    t_p = np.where(np.abs(dz) > 1e-9, -eye[2] / dz, np.inf)
    t_p = np.where(t_p > 1e-6, t_p, np.inf)
    pp = eye + dirs * np.where(np.isfinite(t_p), t_p, 0.0)[..., None]
    checker = ((np.floor(pp[..., 0]) + np.floor(pp[..., 1])) % 2).astype(bool)
    plane_rgb = np.where(checker[..., None], np.array([0.85, 0.85, 0.9]), np.array([0.2, 0.25, 0.35]))
    # fade with distance
    fade = np.clip(1.0 / (1.0 + 0.02 * t_p * t_p), 0, 1)[..., None]
    plane_rgb = plane_rgb * fade
    mask = t_p < t_hit
    t_hit = np.where(mask, t_p, t_hit)
    rgb = np.where(mask[..., None], plane_rgb, rgb)

    # Background: vertical sky gradient.
    sky = np.array([0.05, 0.07, 0.12]) + np.clip(dirs[..., 2:3] / np.linalg.norm(dirs, axis=-1, keepdims=True), 0, 1) * np.array([0.1, 0.15, 0.3])
    none = ~np.isfinite(t_hit)
    rgb = np.where(none[..., None], sky, rgb)

    # Depth along camera +Z: rays have z_cam=1, so depth = t.
    depth = np.where(np.isfinite(t_hit), t_hit, 0.0)
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8), depth


def make_sequence(n_frames: int = 60, width: int = 320, height: int = 240, fps: float = 30.0):
    """Return (rgb u8 TxHxWx3, depth_m f64 TxHxW with 0=invalid, cameras, frames)."""
    fx = fy = 0.75 * width
    K = np.array([[fx, 0, (width - 1) / 2], [0, fy, (height - 1) / 2], [0, 0, 1.0]])
    camera = Camera("PINHOLE", width, height, [fx, fy, (width - 1) / 2, (height - 1) / 2])

    rgb = np.empty((n_frames, height, width, 3), dtype=np.uint8)
    depth = np.empty((n_frames, height, width), dtype=np.float64)
    frames = []
    for i in range(n_frames):
        ang = 2 * np.pi * i / n_frames
        eye = np.array([3.2 * np.cos(ang), 3.2 * np.sin(ang), 1.6 + 0.4 * np.sin(2 * ang)])
        c2w = look_at_c2w(eye, np.array([0.0, 0.0, 0.8]))
        rgb[i], depth[i] = render_frame(c2w, K, width, height)
        q, tr = conventions.matrix_to_pose(c2w)
        frames.append(Frame(i=i, t=i / fps, camera="0", q_wxyz=tuple(q), tr=tuple(tr)))
    return rgb, depth, {"0": camera}, frames
