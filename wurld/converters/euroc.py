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
``c2w(cam0) = T_WB @ T_BS(cam0)``. Both cameras' calibration and the
camera-to-body extrinsics are recorded (cameras "0"/"1" + a ``body`` rig), but
only cam0's pixels are carried — SPEC v0.1 files have one RGB track. The IMU
stream ships with imu-to-cam0 extrinsics.
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
    gt_t = np.array([float(r[0]) * 1e-9 for r in gt_rows])
    gt = [(np.array([float(v) for v in r[1:4]]),
           np.array([float(v) for v in r[4:8]])) for r in gt_rows]

    frames, rgb = [], []
    for i, row in enumerate(_read_csv(base / "cam0" / "data.csv")):
        t = float(row[0]) * 1e-9
        img = Image.open(base / "cam0" / "data" / row[1]).convert("RGBA")
        rgb.append(np.asarray(img))
        gi = int(np.argmin(np.abs(gt_t - t)))
        if abs(gt_t[gi] - t) < max_dt:
            p, q_wxyz = gt[gi]
            t_wb = conventions.pose_to_matrix(q_wxyz, p)
            c2w = t_wb @ t_bs0  # EuRoC cameras are already RDF: no axis flip
            q, tr = conventions.matrix_to_pose(c2w)
            frames.append(container.Frame(i=i, t=t, camera="0", q_wxyz=tuple(q), tr=tuple(tr)))
        else:
            frames.append(container.Frame(i=i, t=t, pose_valid=False))

    imu = []
    if (base / "imu0" / "data.csv").exists():
        rows = _read_csv(base / "imu0" / "data.csv")
        samples = np.array(
            [[float(r[0]) * 1e-9, *(float(v) for v in r[1:7])] for r in rows], dtype=np.float64
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
        rgb=np.stack(rgb),
        rigs={"body": {"cameras": rig_cams,
                       "description": "EuRoC T_BS calibration: camera-to-body transforms"}},
        imu=imu,
        fps=fps,
        rgb_kbps=rgb_kbps,
        world={
            "metric_scale": True,
            "gravity_in_world": [0.0, 0.0, -1.0],  # EuRoC world (Leica/Vicon) is z-up
            "description": (
                f"EuRoC MAV import from {base}; video carries cam0 only "
                "(SPEC v0.1 single RGB track) — cam1 calibration and the "
                "camera-to-body rig are recorded for stereo consumers"
            ),
        },
    )
