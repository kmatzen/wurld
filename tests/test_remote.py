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


def test_fetch_frames_random_access(wl_file, scene):
    file_size = wl_file.stat().st_size
    fetch, stats = counting_fetcher(wl_file)
    hdr = remote.fetch_header(fetch)
    header_bytes = stats["bytes"]

    ref = wl.read(wl_file)
    full_depth = ref.signal("depth")
    full_rgb = ref.rgb

    result = remote.fetch_frames(fetch, [7], header=hdr)
    assert result["clusters_fetched"] == 1
    got = result["frames"][7]
    assert np.array_equal(got["signals"]["depth"], full_depth[7])  # bit-exact
    assert np.array_equal(got["rgb"], np.asarray(full_rgb[7]))
    # this fixture is single-cluster, so byte proportionality is exercised by
    # test_fetch_frames_multi_cluster; here just sanity-check the accounting
    assert 0 < result["bytes_fetched"] <= file_size + remote._CUES_FETCH
    assert header_bytes < file_size


def test_fetch_frames_multi_cluster(tmp_path):
    import chromapakz as cz
    from wurld.synthetic import make_sequence

    rgb, depth_m, cameras, frames = make_sequence(n_frames=90, width=160, height=120, fps=30)
    z = np.where(depth_m > 0, np.clip(depth_m, 0.5, 40.0), np.nan)
    d16 = cz.quantize_inverse(z, near=0.5, far=40.0)
    rgba = np.concatenate([rgb, np.full(rgb.shape[:3] + (1,), 255, np.uint8)], -1)
    p = tmp_path / "long.wl.webm"
    wl.write(p, cameras=cameras, frames=frames, rgb=rgba,
             signals={"depth": d16}, specs={"depth": cz.inverse_depth_spec(0.5, 40.0)},
             signal_meta=[wl.SignalMeta("depth", "depth",
                 {"type": "inverse_depth", "near": 0.5, "far": 40.0, "levels": 65536, "invalid": 0})])
    size = p.stat().st_size

    fetch, stats = counting_fetcher(p)
    hdr = remote.fetch_header(fetch)
    before = stats["bytes"]
    # frames 5 and 65 live in clusters 0 and 2 — cluster 1 must not be fetched
    result = remote.fetch_frames(fetch, [5, 65, 66], header=hdr)
    assert result["clusters_fetched"] == 2
    assert np.array_equal(result["frames"][5]["signals"]["depth"], d16[5])
    assert np.array_equal(result["frames"][65]["signals"]["depth"], d16[65])
    assert np.array_equal(result["frames"][66]["signals"]["depth"], d16[66])
    cluster_bytes = stats["bytes"] - before
    assert cluster_bytes < size * 0.85  # skipped at least one cluster + header


def test_fetch_frames_rejects_pre_cadence_files(tmp_path):
    # simulate an old-encoder file by clearing the keyframe flag on the first
    # depth block of a non-first cluster
    import chromapakz as cz

    from wurld import ebml
    from wurld.synthetic import make_sequence

    rgb, depth_m, cameras, frames = make_sequence(n_frames=60, width=96, height=72, fps=30)
    d16 = cz.quantize_inverse(np.where(depth_m > 0, np.clip(depth_m, 0.5, 40.0), np.nan),
                              near=0.5, far=40.0)
    rgba = np.concatenate([rgb, np.full(rgb.shape[:3] + (1,), 255, np.uint8)], -1)
    src = tmp_path / "multi.wl.webm"
    wl.write(src, cameras=cameras, frames=frames, rgb=rgba,
             signals={"depth": d16}, specs={"depth": cz.inverse_depth_spec(0.5, 40.0)})

    data = bytearray(src.read_bytes())
    _, ps, pe = ebml._segment_bounds(bytes(data))
    clusters = [(es, pstart, pend) for eid, es, pstart, pend
                in ebml._top_level(bytes(data), ps, pe) if eid == ebml.CLUSTER]
    if len(clusters) < 2:
        pytest.skip("test file has a single cluster")
    es, pstart, pend = clusters[1]
    for cid, cs, ce in ebml.iter_children(bytes(data), pstart, pend):
        if cid == 0xA3:
            track, tp = ebml._read_vint(bytes(data), cs, keep_marker=False)
            if track == 2:
                data[tp + 2] &= 0x7F  # clear keyframe bit
                break
    p = tmp_path / "old.wl.webm"
    p.write_bytes(bytes(data))
    with pytest.raises(ValueError, match="keyframe"):
        remote.fetch_frames(remote.file_fetcher(p), [35])  # a cluster-1 frame


def test_sequence_fetch_frames_local_partial_decode(tmp_path):
    import chromapakz as cz
    from wurld.synthetic import make_sequence

    rgb, depth_m, cameras, frames = make_sequence(n_frames=90, width=96, height=72, fps=30)
    d16 = cz.quantize_inverse(np.where(depth_m > 0, np.clip(depth_m, 0.5, 40.0), np.nan),
                              near=0.5, far=40.0)
    rgba = np.concatenate([rgb, np.full(rgb.shape[:3] + (1,), 255, np.uint8)], -1)
    p = tmp_path / "seq.wl.webm"
    wl.write(p, cameras=cameras, frames=frames, rgb=rgba,
             signals={"depth": d16}, specs={"depth": cz.inverse_depth_spec(0.5, 40.0)})

    seq = wl.read(p)
    got = seq.fetch_frames([3, 70])
    assert set(got) == {3, 70}
    full = seq.signal("depth")
    assert np.array_equal(got[3]["signals"]["depth"], full[3])
    assert np.array_equal(got[70]["signals"]["depth"], full[70])
    assert np.array_equal(got[70]["rgb"], np.asarray(seq.rgb[70]))
