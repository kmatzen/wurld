"""Coordinate conventions and pose math.

Wurld files always use the canonical convention (SPEC.md §3):
RDF camera axes (OpenCV/COLMAP), camera-to-world poses, wxyz quaternions,
meters, seconds. This module converts everything else into and out of it.
"""

from __future__ import annotations

import numpy as np

# Right-multiplying a c2w matrix by this flips camera +Y/+Z, converting between
# RDF (OpenCV: x right, y down, z forward) and RUB (OpenGL/Blender/nerfstudio:
# x right, y up, z backward). The operation is its own inverse.
_FLIP_YZ = np.diag([1.0, -1.0, -1.0, 1.0])


def quat_wxyz_to_matrix(q: np.ndarray) -> np.ndarray:
    """Unit quaternion (w, x, y, z) -> 3x3 rotation matrix."""
    w, x, y, z = np.asarray(q, dtype=np.float64)
    n = np.sqrt(w * w + x * x + y * y + z * z)
    if n == 0:
        raise ValueError("zero quaternion")
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def matrix_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix -> unit quaternion (w, x, y, z), w >= 0."""
    R = np.asarray(R, dtype=np.float64)
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        q = np.array(
            [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
        )
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = np.sqrt(R[i, i] - R[j, j] - R[k, k] + 1.0) * 2
        q = np.empty(4)
        q[0] = (R[k, j] - R[j, k]) / s
        q[1 + i] = 0.25 * s
        q[1 + j] = (R[j, i] + R[i, j]) / s
        q[1 + k] = (R[k, i] + R[i, k]) / s
    if q[0] < 0:
        q = -q
    return q / np.linalg.norm(q)


def pose_to_matrix(q_wxyz, tr) -> np.ndarray:
    """(quaternion, translation) -> 4x4 camera-to-world matrix."""
    m = np.eye(4)
    m[:3, :3] = quat_wxyz_to_matrix(q_wxyz)
    m[:3, 3] = np.asarray(tr, dtype=np.float64)
    return m


def matrix_to_pose(c2w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """4x4 camera-to-world matrix -> (q_wxyz, tr)."""
    c2w = np.asarray(c2w, dtype=np.float64)
    return matrix_to_quat_wxyz(c2w[:3, :3]), c2w[:3, 3].copy()


def invert_pose(m: np.ndarray) -> np.ndarray:
    """Invert a rigid 4x4 transform (c2w <-> w2c)."""
    R, t = m[:3, :3], m[:3, 3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def c2w_gl_to_cv(c2w_gl: np.ndarray) -> np.ndarray:
    """OpenGL/Blender/nerfstudio (RUB) c2w -> canonical RDF c2w."""
    return np.asarray(c2w_gl, dtype=np.float64) @ _FLIP_YZ


def c2w_cv_to_gl(c2w_cv: np.ndarray) -> np.ndarray:
    """Canonical RDF c2w -> OpenGL/Blender/nerfstudio (RUB) c2w."""
    return np.asarray(c2w_cv, dtype=np.float64) @ _FLIP_YZ


def quat_xyzw_to_wxyz(q) -> np.ndarray:
    x, y, z, w = np.asarray(q, dtype=np.float64)
    return np.array([w, x, y, z])


def quat_wxyz_to_xyzw(q) -> np.ndarray:
    w, x, y, z = np.asarray(q, dtype=np.float64)
    return np.array([x, y, z, w])


def validate_frames(frames) -> list[str]:
    """Return a list of human-readable problems (empty when clean)."""
    problems = []
    last_t = None
    for f in frames:
        if last_t is not None and f.t < last_t:
            problems.append(f"frame {f.i}: timestamp decreases ({f.t} < {last_t})")
        last_t = f.t
        if f.pose_valid:
            q = np.asarray(f.q_wxyz, dtype=np.float64)
            n = np.linalg.norm(q)
            if abs(n - 1.0) > 1e-3:
                problems.append(f"frame {f.i}: quaternion norm {n:.6f} (not unit)")
    return problems
