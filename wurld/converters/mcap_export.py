"""wurld -> MCAP (Foxglove-ready).

Writes a self-describing MCAP log with jsonschema-encoded Foxglove channels, so a
wurld capture drops straight into Foxglove Studio / lichtblick:

    /camera/image        foxglove.CompressedImage   (jpeg)
    /camera/depth        foxglove.RawImage          (16UC1 raw codes)
    /camera/pose         foxglove.PoseInFrame       (world frame, canonical RDF camera)
    /camera/calibration  foxglove.CameraCalibration (per camera, once at start)
    /imu/<id>            wurld.ImuSample        (custom schema)

The complete WURLD document rides along as an MCAP metadata record named
``wurld``, so nothing (value maps, rigs, conventions) is lost in transit.
Requires the ``mcap`` package (``pip install wurld-video[mcap]``).
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .. import container, conventions

_TS = {"type": "object", "properties": {"sec": {"type": "integer"}, "nsec": {"type": "integer"}}}
_VEC3 = {"type": "object", "properties": {k: {"type": "number"} for k in "xyz"}}
_QUAT = {"type": "object", "properties": {k: {"type": "number"} for k in "xyzw"}}

SCHEMAS = {
    "foxglove.CompressedImage": {
        "type": "object",
        "properties": {"timestamp": _TS, "frame_id": {"type": "string"},
                       "data": {"type": "string", "contentEncoding": "base64"},
                       "format": {"type": "string"}},
    },
    "foxglove.RawImage": {
        "type": "object",
        "properties": {"timestamp": _TS, "frame_id": {"type": "string"},
                       "width": {"type": "integer"}, "height": {"type": "integer"},
                       "encoding": {"type": "string"}, "step": {"type": "integer"},
                       "data": {"type": "string", "contentEncoding": "base64"}},
    },
    "foxglove.PoseInFrame": {
        "type": "object",
        "properties": {"timestamp": _TS, "frame_id": {"type": "string"},
                       "pose": {"type": "object",
                                "properties": {"position": _VEC3, "orientation": _QUAT}}},
    },
    "foxglove.CameraCalibration": {
        "type": "object",
        "properties": {"timestamp": _TS, "frame_id": {"type": "string"},
                       "width": {"type": "integer"}, "height": {"type": "integer"},
                       "distortion_model": {"type": "string"},
                       "D": {"type": "array", "items": {"type": "number"}},
                       "K": {"type": "array", "items": {"type": "number"}},
                       "R": {"type": "array", "items": {"type": "number"}},
                       "P": {"type": "array", "items": {"type": "number"}}},
    },
    "wurld.ImuSample": {
        "type": "object",
        "properties": {"timestamp": _TS, "gyro": _VEC3, "accel": _VEC3},
    },
}


def _stamp(t: float) -> dict:
    sec = int(t)
    return {"sec": sec, "nsec": int(round((t - sec) * 1e9))}


def _calibration(cam: container.Camera, t: float, frame_id: str) -> dict:
    K = cam.K
    D = list(cam.params[4:]) if cam.model in ("OPENCV", "OPENCV_FISHEYE") else []
    model = {"OPENCV": "plumb_bob", "OPENCV_FISHEYE": "fisheye"}.get(cam.model, "")
    P = np.zeros((3, 4))
    P[:, :3] = K
    return {
        "timestamp": _stamp(t), "frame_id": frame_id,
        "width": cam.width, "height": cam.height,
        "distortion_model": model, "D": D,
        "K": K.flatten().tolist(), "R": np.eye(3).flatten().tolist(),
        "P": P.flatten().tolist(),
    }


def to_mcap(wl_path: str | Path, out_path: str | Path, jpeg_quality: int = 90) -> Path:
    try:
        from mcap.writer import Writer
    except ImportError as e:
        raise RuntimeError("MCAP export needs the 'mcap' package (pip install mcap)") from e

    seq = container.read(wl_path)
    out_path = Path(out_path)
    rgb = seq.rgb
    depth_meta = seq.signal_meta("depth")
    depth_raw = seq.signal(depth_meta.id) if depth_meta else None

    with open(out_path, "wb") as f:
        w = Writer(f)
        w.start(profile="", library="wurld")
        w.add_metadata("wurld", {"document": json.dumps(seq.to_document())})

        schema_ids = {
            name: w.register_schema(name=name, encoding="jsonschema",
                                    data=json.dumps(schema).encode())
            for name, schema in SCHEMAS.items()
        }

        def channel(topic, schema_name):
            return w.register_channel(topic=topic, message_encoding="json",
                                      schema_id=schema_ids[schema_name])

        ch_image = channel("/camera/image", "foxglove.CompressedImage") if rgb is not None else None
        ch_depth = channel("/camera/depth", "foxglove.RawImage") if depth_raw is not None else None
        ch_pose = channel("/camera/pose", "foxglove.PoseInFrame")
        ch_calib = channel("/camera/calibration", "foxglove.CameraCalibration")

        t0 = seq.frames[0].t if seq.frames else 0.0
        for key, cam in seq.cameras.items():
            _publish(w, ch_calib, t0, _calibration(cam, t0, f"camera/{key}"))

        for fr in seq.frames:
            ns = int(round(fr.t * 1e9))
            if fr.pose_valid:
                q = conventions.quat_wxyz_to_xyzw(fr.q_wxyz)
                _publish(w, ch_pose, fr.t, {
                    "timestamp": _stamp(fr.t), "frame_id": "world",
                    "pose": {"position": dict(zip("xyz", map(float, fr.tr))),
                             "orientation": dict(zip("xyzw", map(float, q)))},
                }, ns)
            if ch_image is not None:
                buf = io.BytesIO()
                Image.fromarray(np.asarray(rgb[fr.i])[..., :3]).save(
                    buf, format="JPEG", quality=jpeg_quality)
                _publish(w, ch_image, fr.t, {
                    "timestamp": _stamp(fr.t), "frame_id": f"camera/{fr.camera}",
                    "data": base64.b64encode(buf.getvalue()).decode(), "format": "jpeg",
                }, ns)
            if ch_depth is not None:
                plane = np.ascontiguousarray(depth_raw[fr.i])
                _publish(w, ch_depth, fr.t, {
                    "timestamp": _stamp(fr.t), "frame_id": f"camera/{fr.camera}",
                    "width": plane.shape[1], "height": plane.shape[0],
                    "encoding": "16UC1", "step": plane.shape[1] * 2,
                    "data": base64.b64encode(plane.tobytes()).decode(),
                }, ns)

        for stream_id, imu in seq.imu.items():
            ch = channel(f"/imu/{stream_id}", "wurld.ImuSample")
            for row in imu.samples:
                _publish(w, ch, row[0], {
                    "timestamp": _stamp(row[0]),
                    "gyro": dict(zip("xyz", map(float, row[1:4]))),
                    "accel": dict(zip("xyz", map(float, row[4:7]))),
                })
        w.finish()
    return out_path


def _publish(w, channel_id: int, t: float, msg: dict, ns: int | None = None) -> None:
    ns = int(round(t * 1e9)) if ns is None else ns
    w.add_message(channel_id=channel_id, log_time=ns, publish_time=ns,
                  data=json.dumps(msg).encode())
