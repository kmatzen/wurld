"""Pre-rename files (WURLD* tags, format "wurld") must read forever."""

import json

import chromapakz as cz
import numpy as np

import wurld as wl
from wurld import ebml
from wurld.container import pack_frames
from wurld.stream import StreamReader


def _legacy_file(scene, tmp_path, binary_frames: bool):
    """Write a file exactly as pre-1.1 wurld did: legacy tag names throughout."""
    data = cz.encode({"depth": scene["d16"]}, rgb=scene["rgba"])
    doc = {
        "format": "wurld", "version": "1.0",
        "conventions": {"camera_axes": "RDF", "pose_direction": "camera_to_world",
                        "quaternion_order": "wxyz", "units": "meters",
                        "timestamp_units": "seconds"},
        "world": {"metric_scale": True},
        "cameras": {k: c.to_json() for k, c in scene["cameras"].items()},
        "signals": [{"id": "depth", "role": "depth", "value_map": {"type": "identity"}}],
        "frames": [] if binary_frames else [f.to_json() for f in scene["frames"]],
    }
    tags = {}
    if binary_frames:
        doc["frames_binary"] = {"version": 1, "count": len(scene["frames"]), "cameras": ["0"]}
        tags["WURLD_FRAMES"] = pack_frames(scene["frames"], ["0"])
    tags["WURLD"] = json.dumps(doc)
    imu = np.column_stack([np.linspace(0, 0.3, 12), np.full((12, 3), 0.01), np.full((12, 3), 9.0)])
    doc["imu"] = {"imu0": {"rate_hz": 40.0}}
    tags["WURLD"] = json.dumps(doc)  # rebuild with imu descriptor
    from wurld.container import ImuStream

    tags["WURLD_IMU_imu0"] = ImuStream("imu0", imu).pack()
    p = tmp_path / ("legacy_bin.webm" if binary_frames else "legacy_json.webm")
    p.write_bytes(ebml.insert_header_tags(data, tags))
    return p, imu


def test_legacy_json_frames_read(scene, tmp_path):
    p, imu = _legacy_file(scene, tmp_path, binary_frames=False)
    seq = wl.read(p)
    assert len(seq.frames) == len(scene["frames"])
    assert np.allclose(seq.c2w(3), scene["frames"][3].c2w)
    assert np.array_equal(seq.signal("depth"), scene["d16"])
    assert seq.imu["imu0"].samples.shape == (12, 7)
    assert np.array_equal(seq.imu["imu0"].samples[:, 0], imu[:, 0])


def test_legacy_binary_table_read(scene, tmp_path):
    p, _ = _legacy_file(scene, tmp_path, binary_frames=True)
    seq = wl.read(p)
    assert len(seq.frames) == len(scene["frames"])
    assert seq.frames[5].t == scene["frames"][5].t


def test_legacy_streamreader(scene, tmp_path):
    p, _ = _legacy_file(scene, tmp_path, binary_frames=False)
    r = StreamReader()
    data = p.read_bytes()
    events = []
    for off in range(0, len(data), 777):
        events.extend(e[0] for e in r.feed(data[off : off + 777]))
    assert "wurld" not in str(r.doc.get("format")) or r.doc["format"] == "wurld"
    assert len(r.frames) == len(scene["frames"])
    assert "imu" in events


def test_legacy_remote_header(scene, tmp_path):
    from wurld import remote

    p, _ = _legacy_file(scene, tmp_path, binary_frames=True)
    hdr = remote.fetch_header(remote.file_fetcher(p))
    assert len(hdr.frames) == len(scene["frames"])
    assert hdr.imu["imu0"].samples.shape == (12, 7)
