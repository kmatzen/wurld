import json

import chromapakz as cz
import numpy as np
import pytest

import wurld as wl
from wurld import ebml
from wurld.container import pack_frames
from wurld.stream import StreamReader


def _feed_in_chunks(data: bytes, chunk_size: int):
    r = StreamReader()
    events = []
    for off in range(0, len(data), chunk_size):
        events.extend(r.feed(data[off : off + chunk_size]))
    r.finish()
    return r, events


@pytest.mark.parametrize("chunk_size", [7, 977, 65536, 10**9])
def test_streamreader_matches_batch_read(wl_file, scene, chunk_size):
    data = wl_file.read_bytes()
    ref = wl.read(wl_file)
    r, events = _feed_in_chunks(data, chunk_size)

    kinds = [e[0] for e in events]
    assert "wurld" in kinds
    assert kinds.count("cluster") >= 1
    # header-first: metadata must arrive before the first cluster event
    assert kinds.index("wurld") < kinds.index("cluster")

    assert r.doc["cameras"].keys() == {"0"}
    assert len(r.frames) == len(ref.frames)
    for a, b in zip(r.frames, ref.frames):
        assert a.t == b.t and a.i == b.i
        assert np.allclose(a.q_wxyz, b.q_wxyz) and np.allclose(a.tr, b.tr)


def test_streamreader_binary_table(scene, tmp_path):
    p = tmp_path / "bin.wl.webm"
    wl.write(p, cameras=scene["cameras"], frames=scene["frames"], rgb=scene["rgba"],
             frames_format="binary")
    r, events = _feed_in_chunks(p.read_bytes(), 4096)
    assert any(e[0] == "frames_table" for e in events)
    assert len(r.frames) == len(scene["frames"])


def _make_live_style_stream(scene) -> bytes:
    """Simulate the JS live recorder's output: header tags + per-cluster pose chunks.

    Built from a chromapakz batch encode re-cut so that each Cluster is preceded
    by a WURLD_POSES chunk for its frames — the SPEC §9 chunked form.
    """
    rgba, frames, cameras = scene["rgba"], scene["frames"], scene["cameras"]
    data = cz.encode({"depth": scene["d16"]}, rgb=rgba)
    _, ps, pe = ebml._segment_bounds(data)

    doc = {
        "format": "wurld", "version": "0.3",
        "conventions": {"camera_axes": "RDF", "pose_direction": "camera_to_world",
                        "quaternion_order": "wxyz", "units": "meters",
                        "timestamp_units": "seconds"},
        "world": {"metric_scale": True},
        "cameras": {k: c.to_json() for k, c in cameras.items()},
        "signals": [], "frames": [],
    }
    camera_keys = sorted(cameras)

    head, clusters, tail = [], [], []
    for eid, es, pstart, pend in ebml._top_level(data, ps, pe):
        raw = data[es:pend]
        if eid == ebml.CLUSTER:
            clusters.append(raw)
        elif eid == ebml.CUES:
            continue
        elif not clusters:
            head.append(raw)
        else:
            tail.append(raw)

    # split frames evenly across clusters (approximates timestamp assignment)
    per = max(1, -(-len(frames) // len(clusters)))
    payload = b"".join(head) + ebml.build_tags({"WURLD": json.dumps(doc)})
    for ci, cl in enumerate(clusters):
        chunk = frames[ci * per : (ci + 1) * per]
        if chunk:
            payload += ebml.build_tags({"WURLD_POSES": pack_frames(chunk, camera_keys)})
        payload += cl
    payload += b"".join(tail)
    # live stream: unknown-size segment, no cues
    seg_start, _, _ = ebml._segment_bounds(data)
    unknown = (0xFF).to_bytes(1, "big")  # 1-byte all-ones vint = unknown size
    return data[:seg_start] + ebml._encode_id(ebml.SEGMENT) + unknown + payload


def test_streamreader_live_chunked_form(scene):
    stream = _make_live_style_stream(scene)
    r = StreamReader()
    got_poses_before_end = False
    seen_frames_at_cluster = []
    for off in range(0, len(stream), 512):
        for ev in r.feed(stream[off : off + 512]):
            if ev[0] == "cluster":
                seen_frames_at_cluster.append(len(r.frames))
            if ev[0] == "poses":
                got_poses_before_end = True
    assert got_poses_before_end
    # chunked form: each cluster's poses arrived at or before the cluster itself
    assert seen_frames_at_cluster[0] >= 1
    assert len(r.frames) == len(scene["frames"])
    for a, b in zip(r.frames, scene["frames"]):
        assert a.t == b.t
        assert np.allclose(a.q_wxyz, b.q_wxyz, atol=1e-6)


def test_batch_reader_accepts_live_chunked_file(scene, tmp_path):
    """A crash-truncated live file (no consolidated table) reads via chunk concat."""
    stream = _make_live_style_stream(scene)
    p = tmp_path / "live.wl.webm"
    p.write_bytes(stream)
    seq = wl.read(p)
    assert len(seq.frames) == len(scene["frames"])
    assert np.array_equal(seq.signal("depth"), scene["d16"])