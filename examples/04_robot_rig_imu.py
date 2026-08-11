"""Scenario: a robot rig — several cameras on one body, plus IMU.

A mobile robot or a VIO headset is not one camera. It is a rigid assembly whose
extrinsics are calibrated once, an IMU sampling far faster than the cameras, and
one clock. Handled badly, this becomes a directory of per-sensor files plus a
YAML nobody validates, which is roughly what EuRoC, KITTI and ROS bags each
invented separately.

wurld's answer (SPEC §8, §11):

  - `rigs` holds camera-to-body extrinsics once, not per frame. Poses are stored
    for one camera; `rig_c2w(i, "cam1")` derives the others, so the calibration
    cannot drift out of sync with the trajectory.
  - `imu` is a separate stream at its own rate on the same clock, not resampled
    to the video rate — resampling IMU to 30 Hz destroys exactly the signal a
    VIO consumer wants.
  - One file is one rig on one clock. Two robots means two files; see §11 and
    the "misfits" section of USE_CASES.md.

Run:  python examples/04_robot_rig_imu.py out.wurld.webm
"""

import sys

import numpy as np

import wurld as wl

W, H, N, FPS = 128, 96, 30, 30
IMU_HZ = 200


def main(out_path):
    rng = np.random.default_rng(3)
    import chromapakz as cz

    # Two cameras, 12 cm apart — a stereo pair on a body frame.
    cams = {
        "cam0": wl.Camera(model="PINHOLE", width=W, height=H,
                          params=[110.0, 110.0, W / 2, H / 2]),
        "cam1": wl.Camera(model="PINHOLE", width=W, height=H,
                          params=[110.0, 110.0, W / 2, H / 2]),
    }

    # Poses are stored for cam0 only. The rig says where the others sit.
    frames = []
    for i in range(N):
        t = i / FPS
        ang = 0.3 * np.sin(t)
        frames.append(wl.Frame(
            i=i, t=t, camera="cam0",
            q_wxyz=(float(np.cos(ang / 2)), 0.0, float(np.sin(ang / 2)), 0.0),
            tr=(0.4 * t, 0.0, 0.9),
        ))

    rigs = {
        "body": {
            "cameras": {
                # camera-to-rig, wxyz + metres, same conventions as poses
                "cam0": {"q_wxyz": [1.0, 0.0, 0.0, 0.0], "tr": [0.0, 0.0, 0.0]},
                "cam1": {"q_wxyz": [1.0, 0.0, 0.0, 0.0], "tr": [0.12, 0.0, 0.0]},
            }
        }
    }

    # IMU at 200 Hz on the same clock as the frames — not resampled to 30.
    n_imu = int(N / FPS * IMU_HZ)
    imu_t = np.arange(n_imu) / IMU_HZ
    gyro = np.column_stack([0.3 * np.cos(imu_t), rng.normal(0, 0.01, n_imu),
                            rng.normal(0, 0.01, n_imu)])
    accel = np.column_stack([rng.normal(0, 0.05, n_imu), rng.normal(0, 0.05, n_imu),
                             rng.normal(9.81, 0.05, n_imu)])
    imu = [wl.ImuStream("imu0", np.column_stack([imu_t, gyro, accel]),
                        rate_hz=float(IMU_HZ),
                        description="body-frame IMU, gravity along +z")]

    depth_m = np.repeat((1.5 + 0.5 * np.mgrid[0:H, 0:W][0] / H)[None], N, 0).astype(np.float32)
    wl.write(
        out_path,
        cameras=cams,
        frames=frames,
        rgb=rng.integers(40, 220, (N, H, W, 4), dtype=np.uint8),
        signals={"depth": cz.quantize_inverse(depth_m, near=0.3, far=9.0)},
        specs={"depth": {"inverse_depth": True, "near": 0.3, "far": 9.0}},
        signal_meta=[wl.SignalMeta("depth", "depth",
                                   {"type": "inverse_depth", "near": 0.3, "far": 9.0,
                                    "levels": 65536, "invalid": 0})],
        rigs=rigs,
        imu=imu,
        world={"metric_scale": True, "gravity_in_world": [0, 0, -1],
               "description": "two-camera rig with IMU"},
        fps=FPS,
    )

    seq = wl.read(out_path)
    print(f"wrote {out_path}")
    print(f"  cameras: {', '.join(sorted(seq.cameras))}; poses stored for "
          f"{seq.frames[0].camera} only")

    # The second camera's pose is derived, not stored — one source of truth.
    c0 = seq.c2w(5)
    c1 = seq.rig_c2w(5, "cam1")
    baseline = np.linalg.norm(c1[:3, 3] - c0[:3, 3])
    print(f"  rig_c2w(5, 'cam1') derived; baseline {baseline*100:.1f} cm "
          "(matches the calibration, not a stored pose)")

    s = seq.imu["imu0"]
    print(f"  imu0: {s.samples.shape[0]} samples at {s.rate_hz:.0f} Hz over "
          f"{s.samples[-1,0]-s.samples[0,0]:.2f} s "
          f"({s.samples.shape[0] / len(seq.frames):.1f}x the frame rate)")
    print(f"  frames: {len(seq.frames)} at {FPS} fps — IMU kept at its own rate")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "rig.wurld.webm"))
