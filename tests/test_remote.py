import numpy as np
import pytest

import wurld as wl
from wurld import remote


def counting_fetcher(path):
    inner = remote.file_fetcher(path)
    stats = {"bytes": 0, "calls": 0}

    def fetch(offset, size):
        data = inner(offset, size)
        stats["bytes"] += len(data)
        stats["calls"] += 1
        return data

    return fetch, stats


def test_fetch_header_reads_poses_without_video(wl_file, scene):
    file_size = wl_file.stat().st_size
    fetch, stats = counting_fetcher(wl_file)
    hdr = remote.fetch_header(fetch)

    # complete metadata, exactly matching a full read
    ref = wl.read(wl_file)
    assert len(hdr.frames) == len(ref.frames)
    for i in (0, 5, 9):
        assert hdr.frames[i].t == ref.frames[i].t
        assert np.allclose(hdr.frames[i].c2w, ref.frames[i].c2w)
    assert hdr.cameras["0"].K[0, 0] == ref.cameras["0"].K[0, 0]
    assert hdr.world == ref.world
    assert hdr.signals[0].role == "depth"
    assert hdr.video["width"] == 128 and hdr.video["frames"] == 10

    # ranged reads bounded by max(initial probe, header region) — cluster bytes
    # are only ever touched by the blind first probe on tiny files
    assert hdr.header_extent < file_size
    assert stats["bytes"] <= max(hdr.header_extent, remote._PROBE_SIZE)
    assert stats["bytes"] < file_size * 0.5
    assert hdr.bytes_fetched == stats["bytes"]
    assert stats["calls"] <= 2
    assert hdr.cues_offset is not None and hdr.cues_offset > hdr.header_extent


def test_fetch_header_on_large_file_is_a_tiny_fraction(tmp_path):
    import chromapakz as cz
    from wurld.synthetic import make_sequence

    rgb, depth_m, cameras, frames = make_sequence(n_frames=60, width=320, height=240)
    z = np.where(depth_m > 0, np.clip(depth_m, 0.5, 40.0), np.nan)
    d16 = cz.quantize_inverse(z, near=0.5, far=40.0)
    rgba = np.concatenate([rgb, np.full(rgb.shape[:3] + (1,), 255, np.uint8)], -1)
    p = tmp_path / "big.wl.webm"
    wl.write(p, cameras=cameras, frames=frames, rgb=rgba,
             signals={"depth": d16}, specs={"depth": cz.inverse_depth_spec(0.5, 40.0)},
             signal_meta=[wl.SignalMeta("depth", "depth",
                 {"type": "inverse_depth", "near": 0.5, "far": 40.0, "levels": 65536, "invalid": 0})])
    size = p.stat().st_size
    fetch, stats = counting_fetcher(p)
    hdr = remote.fetch_header(fetch)
    assert len(hdr.frames) == 60
    assert stats["bytes"] < size * 0.02, f"fetched {stats['bytes']} of {size}"


def test_fetch_header_rejects_live_streams(scene, tmp_path):
    from tests.test_stream import _make_live_style_stream

    p = tmp_path / "live.wl.webm"
    p.write_bytes(_make_live_style_stream(scene))
    with pytest.raises(ValueError, match="no SeekHead"):
        remote.fetch_header(remote.file_fetcher(p))
