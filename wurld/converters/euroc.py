"""EuRoC MAV (ASL) -> wurld: stereo rig calibration + IMU + posed video.

Layout (per sequence, e.g. ``MH_01_easy/mav0``):

    cam0/sensor.yaml   intrinsics [fu fv cu cv], distortion_coefficients
                       [k1 k2 p1 p2], resolution, T_BS (camera-to-body, 4x4)
    cam0/data.csv      "#timestamp [ns],filename"
    cam0/data/*.png    grayscale frames
    cam1/...           the second stereo camera
    imu0/sensor.yaml   T_BS (imu-to-body); imu0/data.csv  ns, gyro xyz, accel xyz
    state_groundtruth_estimate0/data.csv   ns, p_RS_R xyz, q_RS wxyz, ...

EuRoC cameras use the standard CV convention (RDF, z forward), so no axis flip:
``c2w(cam0) = T_WB @ T_BS(cam0)``. Ground truth is the *body* pose, so that
composition is not optional — omitting it leaves a fixed ~7 cm offset and a
rotation that no trajectory metric will flag as an error.

Both cameras' pixels are carried as display streams keyed by camera id (SPEC
§4.4), with calibration and camera-to-body extrinsics in a ``body`` rig. Poses
are stored for camera "0" only; camera "1" derives through the rig, so the
baseline cannot drift away from the trajectory. Pass ``stereo=False`` for cam0
alone, which is what this converter did before multi-stream existed.

Ground truth runs at 200 Hz against 20 Hz images on an unrelated clock, so poses
are interpolated to image timestamps — linear on position, SLERP on rotation —
rather than snapped to the nearest sample. At the very ends, and across gaps in
the ground truth, a frame is accepted only if a sample lies within ``max_dt``
(20 ms by default) — otherwise it is written ``pose_valid: false``. Clamping an
uncovered stretch to the nearest pose would invent a stationary camera that
never existed.

The IMU stream ships with imu-to-cam0 extrinsics, at its own 200 Hz rate.

Timestamps keep EuRoC's absolute epoch. Note the floor that implies: ~1.4e9
seconds in float64 resolves to about 238 ns, so times carry that quantisation no
matter how they are parsed. It is far below any of these sensors' accuracy, but
it is why exact timestamp equality holds only in the integer nanoseconds, which
is what the cam0/cam1 pairing below compares.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np
from PIL import Image

from .. import container, conventions


def _yaml_list(text: str, key: str) -> list[float] | None:
    """Parse ``key: [a, b, ...]`` (possibly spanning lines) from simple YAML."""
    m = re.search(rf"^{re.escape(key)}:\s*\[(.*?)\]", text, re.S | re.M)
    if not m:
        return None
    return [float(v) for v in m.group(1).replace("\n", " ").split(",") if v.strip()]


def _yaml_t_bs(text: str) -> np.ndarray:
    """The 4x4 row-major ``T_BS.data`` matrix (sensor-to-body)."""
    m = re.search(r"^T_BS:.*?data:\s*\[(.*?)\]", text, re.S | re.M)
    if not m:
        raise ValueError("sensor.yaml has no T_BS block")
    vals = [float(v) for v in m.group(1).replace("\n", " ").split(",") if v.strip()]
    if len(vals) != 16:
        raise ValueError(f"T_BS.data has {len(vals)} values, expected 16")
    return np.array(vals, dtype=np.float64).reshape(4, 4)


def _read_cam_yaml(path: Path) -> tuple[container.Camera, np.ndarray]:
    text = path.read_text()
    intr = _yaml_list(text, "intrinsics")
    dist = _yaml_list(text, "distortion_coefficients") or [0.0, 0.0, 0.0, 0.0]
    res = _yaml_list(text, "resolution")
    if intr is None or res is None:
        raise ValueError(f"{path}: missing intrinsics/resolution")
    cam = container.Camera(
        "OPENCV", int(res[0]), int(res[1]),
        [intr[0], intr[1], intr[2], intr[3], dist[0], dist[1], dist[2], dist[3]],
    )
    return cam, _yaml_t_bs(text)


def _slerp(q0: np.ndarray, q1: np.ndarray, u: float) -> np.ndarray:
    """Shortest-arc quaternion interpolation; lerps when the arc is negligible."""
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:                      # same rotation, opposite hemisphere
        q1, dot = -q1, -dot
    if dot > 0.9995:
        q = q0 + u * (q1 - q0)
        return q / np.linalg.norm(q)
    th = np.arccos(dot) * u
    q2 = q1 - q0 * dot
    q2 /= np.linalg.norm(q2)
    return q0 * np.cos(th) + q2 * np.sin(th)


def _interpolate_body_pose(gt_t, gt_p, gt_q, t, max_dt):
    """T_WB at time t, or None when the ground truth does not cover it."""
    j = int(np.searchsorted(gt_t, t))
    if j == 0 or j >= len(gt_t):
        # Outside the bracket entirely: accept only if a sample is close enough,
        # so a sequence whose GT starts a hair late is not thrown away.
        k = 0 if j == 0 else len(gt_t) - 1
        if abs(gt_t[k] - t) > max_dt:
            return None
        return conventions.pose_to_matrix(gt_q[k], gt_p[k])
    i = j - 1
    if gt_t[j] - gt_t[i] > 2 * max_dt:  # a real gap in the ground truth
        return None
    span = gt_t[j] - gt_t[i]
    u = 0.0 if span <= 0 else (t - gt_t[i]) / span
    return conventions.pose_to_matrix(_slerp(gt_q[i], gt_q[j], u),
                                      gt_p[i] + u * (gt_p[j] - gt_p[i]))


def _read_csv(path: Path) -> list[list[str]]:
    rows = []
    with open(path) as f:
        for row in csv.reader(f):
            if not row or row[0].lstrip().startswith("#"):
                continue
            rows.append([c.strip() for c in row])
    return rows


def _mav_dir(path: Path) -> Path:
    for base in (path, path / "mav0"):
        if (base / "cam0" / "sensor.yaml").exists():
            return base
    raise FileNotFoundError(f"no EuRoC mav0/cam0 under {path}")


def from_euroc(
    seq_dir: str | Path,
    out_path: str | Path,
    max_dt: float = 0.02,
    rgb_kbps: int = 4000,
    stereo: bool = True,
) -> Path:
    base = _mav_dir(Path(seq_dir))

    cam0, t_bs0 = _read_cam_yaml(base / "cam0" / "sensor.yaml")
    cameras = {"0": cam0}
    rig_cams: dict[str, dict] = {}
    q0, tr0 = conventions.matrix_to_pose(t_bs0)
    rig_cams["0"] = {"q_wxyz": [float(v) for v in q0], "tr": [float(v) for v in tr0]}
    if (base / "cam1" / "sensor.yaml").exists():
        cam1, t_bs1 = _read_cam_yaml(base / "cam1" / "sensor.yaml")
        cameras["1"] = cam1
        q1, tr1 = conventions.matrix_to_pose(t_bs1)
        rig_cams["1"] = {"q_wxyz": [float(v) for v in q1], "tr": [float(v) for v in tr1]}

    # ground truth: t_ns, p xyz, q wxyz (body-to-world)
    gt_rows = _read_csv(base / "state_groundtruth_estimate0" / "data.csv")
    # int() first: EuRoC stamps are ~1.4e18 ns, past float64's exact-integer
    # range, and the pairing below compares them for equality.
    gt_t = np.array([int(r[0]) * 1e-9 for r in gt_rows])
    gt_p = np.array([[float(v) for v in r[1:4]] for r in gt_rows])
    gt_q = np.array([[float(v) for v in r[4:8]] for r in gt_rows])

    cam1_by_ns = {}
    want_stereo = stereo and "1" in cameras and (base / "cam1" / "data.csv").exists()
    if want_stereo:
        cam1_by_ns = {int(r[0]): r[1] for r in _read_csv(base / "cam1" / "data.csv")}

    frames, rgb, rgb1, skipped = [], [], [], 0
    for row in _read_csv(base / "cam0" / "data.csv"):
        ns = int(row[0])
        if want_stereo and ns not in cam1_by_ns:
            skipped += 1          # unpaired frame; a stereo stream needs both eyes
            continue
        t = ns * 1e-9
        rgb.append(np.asarray(Image.open(base / "cam0" / "data" / row[1]).convert("RGBA")))
        if want_stereo:
            rgb1.append(np.asarray(
                Image.open(base / "cam1" / "data" / cam1_by_ns[ns]).convert("RGBA")))

        i = len(frames)
        t_wb = _interpolate_body_pose(gt_t, gt_p, gt_q, t, max_dt)
        if t_wb is None:
            frames.append(container.Frame(i=i, t=t, pose_valid=False))
        else:
            c2w = t_wb @ t_bs0  # EuRoC cameras are already RDF: no axis flip
            q, tr = conventions.matrix_to_pose(c2w)
            frames.append(container.Frame(i=i, t=t, camera="0", q_wxyz=tuple(q), tr=tuple(tr)))

    imu = []
    if (base / "imu0" / "data.csv").exists():
        rows = _read_csv(base / "imu0" / "data.csv")
        samples = np.array(
            [[int(r[0]) * 1e-9, *(float(v) for v in r[1:7])] for r in rows], dtype=np.float64
        )
        t_bs_imu = _yaml_t_bs((base / "imu0" / "sensor.yaml").read_text())
        imu_to_cam0 = conventions.invert_pose(t_bs0) @ t_bs_imu
        qi, ti = conventions.matrix_to_pose(imu_to_cam0)
        rate = None
        if samples.shape[0] > 1:
            dt = np.median(np.diff(samples[:, 0]))
            rate = round(1.0 / dt, 1) if dt > 0 else None
        imu.append(container.ImuStream(
            "imu0", samples, rate_hz=rate,
            extrinsics={"q_wxyz": [float(v) for v in qi], "tr": [float(v) for v in ti]},
            description='EuRoC imu0; extrinsics = imu-to-camera "0"',
        ))

    fps = 20.0
    ts = [f.t for f in frames]
    if len(ts) > 1 and ts[-1] > ts[0]:
        fps = round((len(ts) - 1) / (ts[-1] - ts[0]), 3)

    return container.write(
        out_path,
        cameras=cameras,
        frames=frames,
        rgb=({"0": np.stack(rgb), "1": np.stack(rgb1)} if want_stereo else np.stack(rgb)),
        rigs={"body": {"cameras": rig_cams,
                       "description": "EuRoC T_BS calibration: camera-to-body transforms"}},
        imu=imu,
        fps=fps,
        rgb_kbps=rgb_kbps,
        world={
            "metric_scale": True,
            "gravity_in_world": [0.0, 0.0, -1.0],  # EuRoC world (Leica/Vicon) is z-up
            "description": (
                f"EuRoC MAV import from {base}; "
                + ("both eyes carried as display streams \"0\"/\"1\""
                   if want_stereo else
                   "video carries cam0 only (stereo=False)")
                + "; poses stored for camera \"0\", camera \"1\" derives from the body rig"
            ),
        },
    )
