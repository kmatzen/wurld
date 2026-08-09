"""wurld <-> rosbag2: real ROS 2 messages, not a Foxglove-shaped log.

`mcap_export` already writes an MCAP that Foxglove Studio understands, but it
uses Foxglove jsonschema channels with JSON payloads. No ROS 2 node can
subscribe to that and `ros2 bag play` will not touch it. This module writes a
genuine rosbag2 — CDR-encoded `sensor_msgs` and `tf2_msgs` with ROS 2 type
definitions — so a wurld capture becomes something a robot stack can replay, and
a rosbag2 recording becomes a wurld file.

Needs `rosbags` (`pip install 'wurld[ros2]'`); it does not need ROS installed.

**The frame convention is the part worth reading.** ROS body frames are
x-forward, y-left, z-up (REP 103), but *optical* frames are z-forward, x-right,
y-down (REP 145) — which is exactly wurld's RDF. So the transform
`world -> <camera>_optical_frame` **is** wurld's `c2w`, with no axis conversion
at all. The `_optical_frame` suffix is not decoration: it is the signal to a ROS
consumer that this frame is RDF rather than FLU, and dropping it invites exactly
the silent 90-degree error that the suffix exists to prevent.

Topics written:

    /camera/<id>/image_raw          sensor_msgs/msg/Image        rgb8
    /camera/<id>/camera_info        sensor_msgs/msg/CameraInfo
    /camera/<id>/depth/image_raw    sensor_msgs/msg/Image        32FC1, metres
    /tf                             tf2_msgs/msg/TFMessage       world -> optical
    /imu/<id>                       sensor_msgs/msg/Imu

Depth goes out as `32FC1` in metres rather than `16UC1` in millimetres because
NaN survives it. In the 16-bit convention 0 means both "no return" and "at the
sensor", which is the exact ambiguity wurld exists to avoid re-introducing.

**Round-trip fidelity, stated rather than implied.** wurld -> rosbag2 is exact:
poses are float64, depth goes out as the metres it already was, images are the
decoded pixels. The return leg is not. A bag carries no quantization range, so
`from_rosbag2` derives near/far from the data and requantizes — measured at
under 5 micrometres on a 1.5-2.0 m scene, but it scales with the range. Images
re-encode through lossy VP9 (measured mean |delta| ~3/255). Use this to move
data between ecosystems, not as an archival round trip.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .. import container, conventions

# Frames that ROS 2 consumers expect to exist.
WORLD_FRAME = "world"


def _require_rosbags():
    try:
        from rosbags.rosbag2 import Reader, Writer  # noqa: F401
        from rosbags.typesys import Stores, get_typestore  # noqa: F401
    except ImportError as exc:  # pragma: no cover - depends on the user's env
        raise SystemExit(
            "the ROS 2 bridge needs rosbags:  pip install 'wurld[ros2]'") from exc


def optical_frame(camera_id: str) -> str:
    """REP 145 optical frame name: the suffix declares the axis convention."""
    return f"{camera_id}_optical_frame"


def _stamp(ts, seconds: float):
    """builtin_interfaces/Time from wurld seconds.

    wurld timestamps may be relative to the start of a capture rather than an
    epoch. That is carried through unchanged rather than shifted to wall clock:
    inventing an epoch would make two captures look simultaneous.
    """
    if seconds < 0:
        raise ValueError("ROS timestamps cannot be negative; rebase the capture first")
    sec = int(seconds)
    nanosec = int(round((seconds - sec) * 1e9))
    if nanosec >= 1_000_000_000:       # rounding can carry
        sec += 1
        nanosec -= 1_000_000_000
    return ts.types["builtin_interfaces/msg/Time"](sec=sec, nanosec=nanosec)


def _header(ts, seconds: float, frame_id: str):
    return ts.types["std_msgs/msg/Header"](stamp=_stamp(ts, seconds), frame_id=frame_id)


def _distortion(cam: container.Camera) -> tuple[str, list[float]]:
    """ROS distortion model and coefficients for a wurld camera model.

    ROS `plumb_bob` is [k1, k2, p1, p2, k3]; `equidistant` is [k1..k4]. A model
    whose parameters cannot be expressed is an error rather than a silent
    truncation to pinhole, which would leave a consumer undistorting with zeros.
    """
    p = list(cam.params)
    if cam.model == "PINHOLE":
        return "plumb_bob", [0.0, 0.0, 0.0, 0.0, 0.0]
    if cam.model == "SIMPLE_PINHOLE":
        return "plumb_bob", [0.0, 0.0, 0.0, 0.0, 0.0]
    if cam.model == "OPENCV":
        return "plumb_bob", [p[4], p[5], p[6], p[7], 0.0]
    if cam.model == "OPENCV_FISHEYE":
        return "equidistant", [p[4], p[5], p[6], p[7]]
    if cam.model == "SIMPLE_RADIAL":
        return "plumb_bob", [p[3], 0.0, 0.0, 0.0, 0.0]
    if cam.model == "RADIAL":
        return "plumb_bob", [p[3], p[4], 0.0, 0.0, 0.0]
    raise ValueError(f"no ROS distortion model for camera model {cam.model!r}")


def _intrinsics(cam: container.Camera) -> tuple[float, float, float, float]:
    p = list(cam.params)
    if cam.model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
        return p[0], p[0], p[1], p[2]
    return p[0], p[1], p[2], p[3]


def _camera_info(ts, cam: container.Camera, seconds: float, frame_id: str):
    fx, fy, cx, cy = _intrinsics(cam)
    model, d = _distortion(cam)
    return ts.types["sensor_msgs/msg/CameraInfo"](
        header=_header(ts, seconds, frame_id),
        height=cam.height, width=cam.width,
        distortion_model=model,
        d=np.array(d, dtype=np.float64),
        k=np.array([fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0], dtype=np.float64),
        r=np.eye(3, dtype=np.float64).ravel(),
        p=np.array([fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0],
                   dtype=np.float64),
        binning_x=0, binning_y=0,
        roi=ts.types["sensor_msgs/msg/RegionOfInterest"](
            x_offset=0, y_offset=0, height=0, width=0, do_rectify=False),
    )


def _image(ts, seconds: float, frame_id: str, array: np.ndarray, encoding: str):
    h, w = array.shape[:2]
    data = np.ascontiguousarray(array)
    return ts.types["sensor_msgs/msg/Image"](
        header=_header(ts, seconds, frame_id),
        height=h, width=w, encoding=encoding,
        is_bigendian=0, step=int(data.nbytes // h),
        data=data.view(np.uint8).reshape(-1),
    )


def to_rosbag2(
    src: str | Path,
    out_dir: str | Path,
    *,
    depth: bool = True,
    images: bool = True,
    storage: str = "mcap",
) -> Path:
    """Write a wurld file out as a rosbag2 directory."""
    _require_rosbags()
    from rosbags.rosbag2 import Writer
    from rosbags.rosbag2.writer import StoragePlugin
    from rosbags.typesys import Stores, get_typestore

    seq = container.read(src)
    ts = get_typestore(Stores.ROS2_HUMBLE)
    out_dir = Path(out_dir)

    if storage not in ("mcap", "sqlite3"):
        raise ValueError(f"storage must be 'mcap' or 'sqlite3', got {storage!r}")
    plugin = StoragePlugin.MCAP if storage == "mcap" else StoragePlugin.SQLITE3

    # Every display stream, not just the primary. A stereo file exported as one
    # camera looks like a successful conversion until somebody wants the other
    # eye. Streams are camera ids (SPEC §4.4), so each gets its own topic and
    # its own optical frame.
    stream_ids = seq.rgb_streams if images else []
    if stream_ids == ["rgb"]:
        # The conventional single-stream name is not a camera id; attribute it
        # to the camera the frames actually name.
        stream_ids = []
    want_depth = depth and any(s.role == "depth" for s in seq.signals)

    with Writer(out_dir, version=9, storage_plugin=plugin) as bag:
        conns = {}

        def conn(topic, msgtype):
            if topic not in conns:
                conns[topic] = bag.add_connection(topic, msgtype, typestore=ts)
            return conns[topic]

        tf_conn = conn("/tf", "tf2_msgs/msg/TFMessage")

        # Streamed rather than decoded whole: seq.rgb on a real EuRoC sequence is
        # 4.2 GB per stream, and an exporter that cannot open its input is not an
        # exporter.
        payloads = seq.iter_frames() if (images or want_depth) else iter(())
        depth_meta = next((s for s in seq.signals if s.role == "depth"), None)

        for idx, f in enumerate(seq.frames):
            ns = int(round(f.t * 1e9))
            pose_cam = f.camera or next(iter(seq.cameras), "0")

            payload = None
            if images or want_depth:
                try:
                    _i, payload = next(payloads)
                except StopIteration:
                    payload = None

            # Calibration for every camera whose pixels are going out, plus the
            # one the poses belong to.
            for cam_id in dict.fromkeys(list(stream_ids) + [pose_cam]):
                cam = seq.cameras.get(cam_id)
                if cam is None:
                    continue
                bag.write(conn(f"/camera/{cam_id}/camera_info",
                               "sensor_msgs/msg/CameraInfo"),
                          ns, ts.serialize_cdr(
                              _camera_info(ts, cam, f.t, optical_frame(cam_id)),
                              "sensor_msgs/msg/CameraInfo"))

            if payload is not None and images:
                planes = payload.get("rgbs")
                if not planes:
                    primary = stream_ids[0] if stream_ids else pose_cam
                    planes = {primary: payload.get("rgb")}
                for cam_id, plane in planes.items():
                    if plane is None:
                        continue
                    msg = _image(ts, f.t, optical_frame(cam_id),
                                 np.ascontiguousarray(plane[..., :3]), "rgb8")
                    bag.write(conn(f"/camera/{cam_id}/image_raw",
                                   "sensor_msgs/msg/Image"),
                              ns, ts.serialize_cdr(msg, "sensor_msgs/msg/Image"))

            if payload is not None and want_depth and depth_meta is not None:
                codes = (payload.get("signals") or {}).get(depth_meta.id)
                if codes is not None:
                    metres = np.asarray(depth_meta.apply(codes), dtype=np.float32)
                    msg = _image(ts, f.t, optical_frame(pose_cam), metres, "32FC1")
                    bag.write(conn(f"/camera/{pose_cam}/depth/image_raw",
                                   "sensor_msgs/msg/Image"),
                              ns, ts.serialize_cdr(msg, "sensor_msgs/msg/Image"))

            if not f.pose_valid:
                # No transform at all, rather than identity. tf2 interpolates
                # across gaps and will refuse to extrapolate past the end, which
                # is the honest behaviour for a frame nothing localised.
                continue

            transforms = [(pose_cam, f.c2w)]
            # Other cameras on the rig get their own transform, derived from the
            # calibration, so tf can place both eyes rather than only the one
            # the poses were stored for.
            for cam_id in stream_ids:
                if cam_id == pose_cam or not seq.rigs:
                    continue
                try:
                    transforms.append((cam_id, seq.rig_c2w(idx, cam_id)))
                except Exception:                      # noqa: BLE001 - no rig entry
                    pass

            stamped = []
            for cam_id, c2w in transforms:
                q, tvec = conventions.matrix_to_pose(np.asarray(c2w))
                stamped.append(ts.types["geometry_msgs/msg/TransformStamped"](
                    header=_header(ts, f.t, WORLD_FRAME),
                    child_frame_id=optical_frame(cam_id),
                    transform=ts.types["geometry_msgs/msg/Transform"](
                        translation=ts.types["geometry_msgs/msg/Vector3"](
                            x=float(tvec[0]), y=float(tvec[1]), z=float(tvec[2])),
                        # ROS quaternions are xyzw; wurld's are wxyz. This reorder
                        # is the single most common way to get a bridge subtly
                        # wrong, and nothing downstream would flag it.
                        rotation=ts.types["geometry_msgs/msg/Quaternion"](
                            x=float(q[1]), y=float(q[2]), z=float(q[3]), w=float(q[0])),
                    ),
                ))
            msg = ts.types["tf2_msgs/msg/TFMessage"](transforms=stamped)
            bag.write(tf_conn, ns, ts.serialize_cdr(msg, "tf2_msgs/msg/TFMessage"))

        for stream_id, stream in seq.imu.items():
            topic = f"/imu/{stream_id}"
            c = conn(topic, "sensor_msgs/msg/Imu")
            frame_id = f"{stream_id}_frame"
            unknown = np.zeros(9, dtype=np.float64)
            # REP 145: covariance[0] = -1 means "this field is not provided".
            # wurld IMU carries no orientation, and a default identity would be
            # read as a real measurement.
            no_orientation = unknown.copy()
            no_orientation[0] = -1.0
            for row in stream.samples:
                t = float(row[0])
                msg = ts.types["sensor_msgs/msg/Imu"](
                    header=_header(ts, t, frame_id),
                    orientation=ts.types["geometry_msgs/msg/Quaternion"](
                        x=0.0, y=0.0, z=0.0, w=1.0),
                    orientation_covariance=no_orientation,
                    angular_velocity=ts.types["geometry_msgs/msg/Vector3"](
                        x=float(row[1]), y=float(row[2]), z=float(row[3])),
                    angular_velocity_covariance=unknown.copy(),
                    linear_acceleration=ts.types["geometry_msgs/msg/Vector3"](
                        x=float(row[4]), y=float(row[5]), z=float(row[6])),
                    linear_acceleration_covariance=unknown.copy(),
                )
                bag.write(c, int(round(t * 1e9)),
                          ts.serialize_cdr(msg, "sensor_msgs/msg/Imu"))

    return out_dir


# ------------------------------------------------------------------- importing

def from_rosbag2(
    bag_dir: str | Path,
    out_path: str | Path,
    *,
    image_topic: str | None = None,
    rgb_kbps: int = 4000,
) -> Path:
    """Read a rosbag2 into a wurld file.

    Uses `/tf` transforms whose child is an `*_optical_frame` as the camera
    trajectory, the matching `camera_info` for calibration, and images
    associated by timestamp. Frames with no transform within half a frame
    interval are written `pose_valid: false` rather than dropped, because a
    recording with a tf gap is information about the recording.
    """
    _require_rosbags()
    from rosbags.rosbag2 import Reader
    from rosbags.typesys import Stores, get_typestore

    ts = get_typestore(Stores.ROS2_HUMBLE)
    bag_dir = Path(bag_dir)

    poses: list[tuple[float, str, tuple, tuple]] = []
    infos: dict[str, object] = {}
    images: dict[str, list[tuple[float, np.ndarray]]] = {}
    depths: dict[str, list[tuple[float, np.ndarray]]] = {}
    imu_rows: dict[str, list[list[float]]] = {}

    with Reader(bag_dir) as reader:
        for conn, _, raw in reader.messages():
            topic, mt = conn.topic, conn.msgtype
            if mt == "tf2_msgs/msg/TFMessage":
                msg = ts.deserialize_cdr(raw, mt)
                for tr in msg.transforms:
                    child = tr.child_frame_id
                    if not child.endswith("_optical_frame"):
                        continue
                    t = tr.header.stamp.sec + tr.header.stamp.nanosec * 1e-9
                    q, v = tr.transform.rotation, tr.transform.translation
                    poses.append((t, child[: -len("_optical_frame")],
                                  (q.w, q.x, q.y, q.z), (v.x, v.y, v.z)))
            elif mt == "sensor_msgs/msg/CameraInfo":
                msg = ts.deserialize_cdr(raw, mt)
                infos.setdefault(msg.header.frame_id, msg)
            elif mt == "sensor_msgs/msg/Image":
                msg = ts.deserialize_cdr(raw, mt)
                t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                cam = msg.header.frame_id
                if cam.endswith("_optical_frame"):
                    cam = cam[: -len("_optical_frame")]
                if msg.encoding == "32FC1":
                    arr = msg.data.view(np.float32).reshape(msg.height, msg.width)
                    depths.setdefault(cam, []).append((t, arr))
                elif msg.encoding in ("rgb8", "bgr8"):
                    arr = msg.data.reshape(msg.height, msg.width, 3)
                    if msg.encoding == "bgr8":
                        arr = arr[..., ::-1]
                    if image_topic in (None, topic):
                        images.setdefault(cam, []).append((t, np.ascontiguousarray(arr)))
                else:
                    continue
            elif mt == "sensor_msgs/msg/Imu":
                msg = ts.deserialize_cdr(raw, mt)
                t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                sid = topic.rsplit("/", 1)[-1]
                imu_rows.setdefault(sid, []).append([
                    t, msg.angular_velocity.x, msg.angular_velocity.y,
                    msg.angular_velocity.z, msg.linear_acceleration.x,
                    msg.linear_acceleration.y, msg.linear_acceleration.z])

    if not images:
        raise ValueError(f"{bag_dir}: no rgb8/bgr8 sensor_msgs/Image messages found")

    cam_id = sorted(images)[0]
    frames_rgb = sorted(images[cam_id], key=lambda x: x[0])
    stamps = [t for t, _ in frames_rgb]

    info = infos.get(optical_frame(cam_id)) or (next(iter(infos.values())) if infos else None)
    if info is None:
        raise ValueError(f"{bag_dir}: no CameraInfo; calibration cannot be recovered")
    k = np.asarray(info.k, dtype=np.float64)
    cameras = {cam_id: container.Camera(
        model="OPENCV", width=int(info.width), height=int(info.height),
        params=[k[0], k[4], k[2], k[5],
                *[float(x) for x in list(np.asarray(info.d, dtype=np.float64))[:4]]]
        if len(np.asarray(info.d)) >= 4 else
        [k[0], k[4], k[2], k[5], 0.0, 0.0, 0.0, 0.0])}

    pose_at = sorted((p for p in poses if p[1] == cam_id), key=lambda x: x[0])
    ptimes = np.array([p[0] for p in pose_at]) if pose_at else np.zeros(0)
    tol = 0.5 * (np.median(np.diff(stamps)) if len(stamps) > 1 else 1 / 30)

    frames = []
    for i, t in enumerate(stamps):
        j = int(np.argmin(np.abs(ptimes - t))) if len(ptimes) else -1
        if j < 0 or abs(ptimes[j] - t) > tol:
            frames.append(container.Frame(i=i, t=t, pose_valid=False))
        else:
            _, _, q, v = pose_at[j]
            frames.append(container.Frame(i=i, t=t, camera=cam_id,
                                          q_wxyz=tuple(float(x) for x in q),
                                          tr=tuple(float(x) for x in v)))

    signals, specs, signal_meta = {}, {}, []
    if depths.get(cam_id):
        import chromapakz as cz
        dmap = dict(sorted(depths[cam_id], key=lambda x: x[0]))
        dtimes = np.array(sorted(dmap))
        stack = []
        for t in stamps:
            j = int(np.argmin(np.abs(dtimes - t)))
            stack.append(dmap[dtimes[j]])
        arr = np.stack(stack)
        finite = arr[np.isfinite(arr) & (arr > 0)]
        near = float(max(0.05, finite.min())) if finite.size else 0.1
        far = float(finite.max()) if finite.size else 10.0
        signals["depth"] = cz.quantize_inverse(arr, near=near, far=far)
        specs["depth"] = {"inverse_depth": True, "near": near, "far": far}
        signal_meta = [container.SignalMeta(
            "depth", "depth", {"type": "inverse_depth", "near": near, "far": far,
                               "levels": 65536, "invalid": 0})]

    rgb = np.stack([np.dstack([a, np.full(a.shape[:2], 255, np.uint8)])
                    for _, a in frames_rgb])
    fps = 1.0 / np.median(np.diff(stamps)) if len(stamps) > 1 else 30.0

    imu = [container.ImuStream(sid, np.array(sorted(rows)))
           for sid, rows in sorted(imu_rows.items()) if rows]

    return container.write(
        out_path, cameras=cameras, frames=frames, rgb=rgb,
        signals=signals or None, specs=specs or None,
        signal_meta=signal_meta or None, imu=imu or None,
        world={"metric_scale": True,
               "description": f"imported from rosbag2 {bag_dir.name}"},
        fps=float(fps), rgb_kbps=rgb_kbps)
