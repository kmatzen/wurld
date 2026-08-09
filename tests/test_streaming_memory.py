"""Streaming iteration must cost a Cluster, not a file.

`Sequence.iter_frames` documents "bounded memory ... an hour-long file never
holds more than ~1 s of decoded frames". That was true of the *packets* and
false of the *buffers*: a spliced single-Cluster file still advertised the whole
sequence's frame count, and chromapakz sizes its output arrays from that. Every
"one Cluster" decode allocated for the entire file — measured at 277 MB per
Cluster on a 600-frame 320x240 sequence, against 14.7 MB once the count matched.

Nothing failed when it was wrong. The iteration produced correct frames at
correct times; it just used an order of magnitude more memory than it claimed,
which is invisible until something is killed for it. Hence these tests.
"""

import json

import numpy as np
import pytest

import wurld as wl
from wurld import container, ebml

W, H, N, FPS = 160, 120, 240, 30


@pytest.fixture(scope="module")
def long_file(tmp_path_factory):
    """Long enough to span many Clusters, small enough to stay quick."""
    import chromapakz as cz

    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    rgb = np.stack([
        np.dstack([np.clip((0.5 + 0.4 * np.sin(xx / 5 + i * 0.2)) * 255, 0, 255)] * 3
                  + [np.full((H, W), 255, np.float32)]).astype(np.uint8)
        for i in range(N)])
    depth = np.stack([(1.5 + 0.4 * np.sin(xx / 7 + i * 0.1)).astype(np.float32)
                      for i in range(N)])
    f = 1.1 * W
    out = tmp_path_factory.mktemp("stream") / "long.wl.webm"
    wl.write(out,
             cameras={"0": wl.Camera("PINHOLE", W, H, [f, f, W / 2, H / 2])},
             frames=[wl.Frame(i=i, t=i / FPS, camera="0", q_wxyz=(1.0, 0.0, 0.0, 0.0),
                              tr=(0.01 * i, 0.0, 0.5)) for i in range(N)],
             rgb=rgb,
             signals={"depth": cz.quantize_inverse(depth, near=0.3, far=9.0)},
             specs={"depth": {"inverse_depth": True, "near": 0.3, "far": 9.0}},
             signal_meta=[wl.SignalMeta("depth", "depth",
                                        {"type": "inverse_depth", "near": 0.3,
                                         "far": 9.0, "levels": 65536, "invalid": 0})],
             world={"metric_scale": True}, fps=FPS)
    return out


def cluster_count(path):
    data = path.read_bytes()
    _, ps, pe = ebml._segment_bounds(data)
    return sum(1 for eid, *_ in ebml._top_level(data, ps, pe) if eid == ebml.CLUSTER)


def test_the_fixture_really_spans_many_clusters(long_file):
    """Without several Clusters, every claim below is vacuous."""
    assert cluster_count(long_file) >= 4, "one Cluster means nothing is being bounded"


def test_decode_head_corrects_the_frame_count(long_file):
    """The unit fix: a per-Cluster header must not claim the whole sequence."""
    data = long_file.read_bytes()
    _, ps, _pe = ebml._segment_bounds(data)
    head_end = next(es for eid, es, _, _ in ebml._top_level(data, ps, _pe)
                    if eid == ebml.CLUSTER)
    head = data[ps:head_end]

    full = json.loads(dict(ebml.collect_simple_tags(
        head, *_tags_range(head)))["CHROMAPAKZ"])
    assert full["frames"] == N

    fixed = container._decode_head(head, 30)
    meta = json.loads(dict(ebml.collect_simple_tags(
        fixed, *_tags_range(fixed)))["CHROMAPAKZ"])
    assert meta["frames"] == 30
    # Everything else about the stream must survive, or the decode changes.
    for key in ("width", "height", "fps"):
        assert meta[key] == full[key]


def _tags_range(buf):
    for eid, _es, pstart, pend in ebml._top_level(buf, 0, len(buf)):
        if eid == ebml.TAGS:
            return pstart, pend
    raise AssertionError("no Tags element")


def test_wurld_tags_are_dropped_from_the_decode_header(long_file):
    """cz.decode never reads them, and copying a pose table per Cluster is waste."""
    data = long_file.read_bytes()
    _, ps, pe = ebml._segment_bounds(data)
    head_end = next(es for eid, es, _, _ in ebml._top_level(data, ps, pe)
                    if eid == ebml.CLUSTER)
    head = data[ps:head_end]
    fixed = container._decode_head(head, 30)
    names = {n for n, _ in ebml.collect_simple_tags(fixed, *_tags_range(fixed))}
    assert names == {"CHROMAPAKZ"}
    assert len(fixed) <= len(head)


def test_streamed_frames_are_bit_identical_to_a_full_decode(long_file):
    """Memory is worth nothing if the pixels changed."""
    full = wl.read(long_file)._decode()
    seq = wl.read(long_file)
    checked = 0
    for idx, payload in seq.iter_frames():
        if idx % 37:
            continue
        assert np.array_equal(payload["rgb"], full["rgb"][idx])
        assert np.array_equal(payload["signals"]["depth"], full["signals"]["depth"][idx])
        checked += 1
    assert checked >= 5


def test_iteration_holds_a_cluster_not_the_file(long_file):
    """The regression this file exists for.

    A whole-file decode of the fixture is ~2x N x W x H x 2 bytes of signals
    plus N x W x H x 4 of RGB. Iterating must stay far below that; the bug made
    the two equal.
    """
    import tracemalloc

    whole = N * W * H * 4 + N * W * H * 2 * 2         # rgb + hi/lo signal planes
    seq = wl.read(long_file)
    tracemalloc.start()
    count = sum(1 for _ in seq.iter_frames())
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert count == N
    # Generous: the real ratio measured is ~9x better than a full decode. This
    # only has to catch a return to whole-file allocation.
    assert peak < 0.5 * whole, f"peak {peak/1e6:.1f} MB vs whole-decode {whole/1e6:.1f} MB"


def test_collection_streaming_is_bounded_too(long_file, tmp_path):
    """The collection layer must use the bounded path, not a whole-member decode."""
    import tracemalloc

    from wurld import collection as col

    root = tmp_path / "corpus"
    root.mkdir()
    for k in range(2):
        (root / f"m{k}.wl.webm").write_bytes(long_file.read_bytes())
    m, failures = col.build_manifest(root, relative_to=root)
    assert not failures
    c = col.Collection(m, root=root)

    whole = N * W * H * 4 + N * W * H * 2 * 2
    tracemalloc.start()
    n = sum(1 for _ in c.iter_frames(fields=("rgb", "depth")))
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert n == 2 * N
    assert peak < 0.5 * whole, f"peak {peak/1e6:.1f} MB vs one member {whole/1e6:.1f} MB"


def test_metadata_only_iteration_does_not_read_pixels(long_file, tmp_path):
    """Reporting poses must not cost the size of the video."""
    import tracemalloc

    from wurld import collection as col

    root = tmp_path / "meta"
    root.mkdir()
    for k in range(3):
        (root / f"m{k}.wl.webm").write_bytes(long_file.read_bytes())
    m, _ = col.build_manifest(root, relative_to=root)
    c = col.Collection(m, root=root)

    tracemalloc.start()
    n = sum(1 for _ in c.iter_frames())
    _, meta_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert n == 3 * N

    tracemalloc.start()
    sum(1 for _ in c.iter_frames(fields=("rgb", "depth")))
    _, pixel_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Compared against the pixel path on the same corpus rather than against a
    # byte count: on a small fixture, fixed overhead swamps any absolute bound,
    # which would make the assertion about the fixture rather than the code.
    assert meta_peak < 0.1 * pixel_peak, (
        f"metadata {meta_peak/1e6:.2f} MB vs pixels {pixel_peak/1e6:.2f} MB — "
        "metadata iteration appears to be decoding")
