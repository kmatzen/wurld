"""Collections: manifests, global indexing, and sharded streaming.

The sharding tests are the strict ones on purpose. A wrong shard split does not
raise — it silently trains on duplicated or missing data, and the loss curve
looks fine. So every iteration path here asserts the two properties that matter:
shards are **disjoint**, and their union is **complete**.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import wurld as wl
from wurld import collection as col

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def _write_seq(path, n, *, w=32, h=24, unposed=(), metric=True, fps=30):
    """A small real wurld file: RGB + depth, some frames optionally unposed."""
    import chromapakz as cz

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    rgb, depth = [], []
    for i in range(n):
        g = np.clip((0.5 + 0.4 * np.sin(xx / 5 + i * 0.3) * np.cos(yy / 4)) * 255, 0, 255)
        rgb.append(np.dstack([g, g, g, np.full_like(g, 255)]).astype(np.uint8))
        depth.append((1.5 + 0.5 * np.sin(xx / 7 + i * 0.2)).astype(np.float32))

    frames = []
    for i in range(n):
        if i in unposed:
            frames.append(wl.Frame(i=i, t=i / fps, pose_valid=False))
        else:
            frames.append(wl.Frame(i=i, t=i / fps, camera="0",
                                   q_wxyz=(1.0, 0.0, 0.0, 0.0),
                                   tr=(0.01 * i, 0.0, 0.5)))
    f = 1.1 * w
    wl.write(
        path,
        cameras={"0": wl.Camera(model="PINHOLE", width=w, height=h,
                                params=[f, f, w / 2, h / 2])},
        frames=frames,
        rgb=np.stack(rgb),
        signals={"depth": cz.quantize_inverse(np.stack(depth), near=0.3, far=9.0)},
        specs={"depth": {"inverse_depth": True, "near": 0.3, "far": 9.0}},
        signal_meta=[wl.SignalMeta("depth", "depth",
                                   {"type": "inverse_depth", "near": 0.3, "far": 9.0,
                                    "levels": 65536, "invalid": 0})],
        world={"metric_scale": metric},
        fps=fps,
    )
    return path


COUNTS = [3, 5, 2, 7, 4]          # deliberately uneven: shards cannot be equal


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    root = tmp_path_factory.mktemp("corpus")
    (root / "sub").mkdir()
    paths = []
    for i, n in enumerate(COUNTS):
        d = root / "sub" if i % 2 else root
        paths.append(_write_seq(d / f"seq{i}.wl.webm", n,
                                unposed=(1,) if i == 3 else ()))
    return root, paths


@pytest.fixture(scope="module")
def manifest(corpus):
    root, _ = corpus
    m, failures = col.build_manifest(root, relative_to=root)
    assert failures == []
    return root, m


@pytest.fixture(scope="module")
def counts(manifest):
    """Frame counts in *manifest order*, which recursive globbing decides.

    Members live in two directories, so sorted rglob order is not the order the
    fixture wrote them in. Deriving this rather than assuming it keeps the tests
    honest about what the collection actually contains.
    """
    _, m = manifest
    return [mem.frames for mem in m.members]


@pytest.fixture(scope="module")
def unposed_at(manifest):
    """(member index, local index) of the one frame with no pose."""
    _, m = manifest
    mi = next(i for i, mem in enumerate(m.members) if mem.posed_frames < mem.frames)
    return mi, 1


def test_describe_reads_only_the_header(corpus):
    _, paths = corpus
    big = paths[3]
    m = col.describe(big)
    assert m.frames == COUNTS[3]
    assert m.posed_frames == COUNTS[3] - 1        # one unposed frame
    assert m.cameras == ["0"]
    assert m.signals == ["depth"]
    assert m.metric_scale is True
    assert (m.width, m.height) == (32, 24)
    assert m.bytes == big.stat().st_size


def test_header_read_does_not_scale_with_video_size(tmp_path):
    """Indexing cost tracks frame count, not bytes — the claim the docs make.

    Both files hold ten frames; one has a hundred times the pixels. If indexing
    ever started decoding, this is where it would show up.
    """
    from wurld import remote

    small = _write_seq(tmp_path / "small.wl.webm", 10, w=32, h=24)
    big = _write_seq(tmp_path / "big.wl.webm", 10, w=320, h=240)
    assert big.stat().st_size > 20 * small.stat().st_size

    read_big = remote.fetch_header(remote.file_fetcher(big)).bytes_fetched
    assert read_big < 0.1 * big.stat().st_size
    # Same frame count, so the header region is the same order of magnitude.
    read_small = remote.fetch_header(remote.file_fetcher(small)).bytes_fetched
    assert read_big <= 2 * max(read_small, 8192)


def test_manifest_round_trips(manifest, tmp_path):
    root, m = manifest
    assert len(m.members) == len(COUNTS)
    assert m.total_frames == sum(COUNTS)
    assert m.total_posed_frames == sum(COUNTS) - 1

    p = tmp_path / "collection.json"
    m.write(p)
    back = col.Manifest.read(p)
    assert [x.uri for x in back.members] == [x.uri for x in m.members]
    assert back.total_frames == m.total_frames
    assert back.root == p.parent


def test_manifest_rejects_foreign_and_future_documents(tmp_path):
    p = tmp_path / "x.json"
    p.write_text(json.dumps({"format": "something-else"}))
    with pytest.raises(ValueError, match="not a wurld collection"):
        col.Manifest.read(p)

    p.write_text(json.dumps({"format": "wurld-collection", "version": 99, "members": []}))
    with pytest.raises(ValueError, match="newer than this reader"):
        col.Manifest.read(p)


def test_relative_uris_make_the_collection_movable(manifest, tmp_path):
    root, m = manifest
    for mem in m.members:
        assert not Path(mem.uri).is_absolute()

    # Copy the tree elsewhere; the same manifest must still resolve.
    import shutil
    dest = tmp_path / "moved"
    shutil.copytree(root, dest)
    p = m.write(dest / "collection.json")
    c = col.Collection.read(p)
    assert len(c) == sum(COUNTS)
    assert c.frame(0)["t"] == 0.0


def test_global_index_resolves_by_bisection(manifest, counts):
    root, m = manifest
    c = col.Collection(m, root=root)
    assert len(c) == sum(counts)

    # Every global index maps to the right (member, local) pair.
    expected = [(mi, li) for mi, n in enumerate(counts) for li in range(n)]
    assert [c.locate(i) for i in range(len(c))] == expected
    # Boundaries specifically: the first and last frame of each member.
    base = 0
    for mi, n in enumerate(counts):
        assert c.locate(base) == (mi, 0)
        assert c.locate(base + n - 1) == (mi, n - 1)
        base += n
    assert c.locate(-1) == (len(counts) - 1, counts[-1] - 1)
    with pytest.raises(IndexError):
        c.locate(len(c))


def test_frame_returns_pose_and_pixels(manifest, counts, unposed_at):
    root, m = manifest
    c = col.Collection(m, root=root)
    item = c.frame(0, fields=("rgb", "depth"))
    assert item["pose_valid"] is True
    assert item["c2w"].shape == (4, 4)
    assert item["rgb"].shape[:2] == (24, 32)
    assert item["depth"].shape == (24, 32)
    assert np.isfinite(item["depth"]).all()

    umi, uli = unposed_at
    gi = sum(counts[:umi]) + uli
    assert c.locate(gi) == (umi, uli)
    lost = c.frame(gi)
    assert lost["pose_valid"] is False
    assert "c2w" not in lost                 # absent, not identity


def test_frame_matches_reading_the_member_directly(manifest, counts):
    root, m = manifest
    c = col.Collection(m, root=root)
    gi = sum(counts[:2]) + 1                 # member 2, frame 1
    item = c.frame(gi, fields=("rgb", "depth"))
    seq = wl.read(root / m.members[2].uri)
    assert np.array_equal(item["rgb"], seq.rgb[1])
    got, want = item["depth"], seq.depth_meters(1)
    assert np.array_equal(np.nan_to_num(got, nan=-1), np.nan_to_num(want, nan=-1))
    assert item["t"] == pytest.approx(seq.frames[1].t)


def test_iter_frames_covers_everything_in_order(manifest, counts):
    root, m = manifest
    c = col.Collection(m, root=root)
    got = [(it["member"], it["frame"]) for it in c.iter_frames()]
    assert got == [(mi, li) for mi, n in enumerate(counts) for li in range(n)]


def test_posed_only_skips_unposed(manifest, counts, unposed_at):
    root, m = manifest
    c = col.Collection(m, root=root)
    items = list(c.iter_frames(posed_only=True))
    assert len(items) == sum(counts) - 1
    assert all(it["pose_valid"] for it in items)
    assert unposed_at not in [(it["member"], it["frame"]) for it in items]


@pytest.mark.parametrize("shards", [1, 2, 3, 5, 8])
@pytest.mark.parametrize("shuffle", [None, 1234])
def test_shards_are_disjoint_and_complete(manifest, counts, shards, shuffle):
    """The property that silently corrupts training if it breaks."""
    root, m = manifest
    c = col.Collection(m, root=root)

    seen, per_shard = [], []
    for k in range(shards):
        ids = [(it["member"], it["frame"])
               for it in c.iter_frames(shard=(k, shards), shuffle=shuffle)]
        per_shard.append(ids)
        seen.extend(ids)

    everything = {(mi, li) for mi, n in enumerate(counts) for li in range(n)}
    assert len(seen) == len(everything), "a frame was duplicated or dropped"
    assert set(seen) == everything
    # More shards than members is legal; the extras are empty, not wrong.
    assert sum(len(s) for s in per_shard) == len(everything)
    # A member never splits across shards: whole files stay together.
    for ids in per_shard:
        for mi in {x for x, _ in ids}:
            assert sum(1 for x, _ in ids if x == mi) == counts[mi]


def test_shard_index_is_validated(manifest):
    root, m = manifest
    c = col.Collection(m, root=root)
    with pytest.raises(ValueError, match="out of range"):
        list(c.iter_members(shard=(3, 3)))


def test_shuffle_is_deterministic_and_actually_reorders(manifest):
    root, m = manifest
    c = col.Collection(m, root=root)
    a = [it["member"] for it in c.iter_frames(shuffle=7)]
    b = [it["member"] for it in c.iter_frames(shuffle=7)]
    plain = [it["member"] for it in c.iter_frames()]
    assert a == b, "same seed must give the same order"
    assert a != plain, "the seed did nothing"
    assert sorted(a) == sorted(plain)


def test_shuffle_buffer_reorders_within_a_member_without_losing_frames(manifest, counts):
    root, m = manifest
    c = col.Collection(m, root=root)
    ids = [(it["member"], it["frame"])
           for it in c.iter_frames(shuffle=3, shuffle_buffer=6)]
    everything = {(mi, li) for mi, n in enumerate(counts) for li in range(n)}
    assert set(ids) == everything
    assert len(ids) == len(everything)     # the buffer must drain, not drop


def test_build_manifest_reports_bad_members_instead_of_hiding_them(tmp_path):
    good = _write_seq(tmp_path / "good.wl.webm", 3)
    bad = tmp_path / "bad.wl.webm"
    bad.write_bytes(b"\x1a\x45\xdf\xa3not a real file")

    with pytest.raises(Exception):
        col.build_manifest(tmp_path)

    m, failures = col.build_manifest(tmp_path, on_error="skip")
    assert [x.uri for x in m.members] == [str(good)]
    assert len(failures) == 1 and failures[0][0] == str(bad)


def test_cli_index_and_collection(corpus, tmp_path):
    root, _ = corpus
    out = tmp_path / "c.json"
    r = subprocess.run([sys.executable, "-m", "wurld.cli", "index", str(root),
                        "-o", str(out)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert f"{len(COUNTS)} members" in r.stdout
    assert f"{sum(COUNTS)} frames" in r.stdout

    r = subprocess.run([sys.executable, "-m", "wurld.cli", "collection", str(out),
                        "--members"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "32x24" in r.stdout
    assert "depth" in r.stdout
    assert r.stdout.count("[") >= len(COUNTS)


def test_cli_index_exits_nonzero_on_a_bad_member(tmp_path):
    _write_seq(tmp_path / "good.wl.webm", 2)
    (tmp_path / "bad.wl.webm").write_bytes(b"nope")
    out = tmp_path / "c.json"
    r = subprocess.run([sys.executable, "-m", "wurld.cli", "index", str(tmp_path),
                        "-o", str(out), "--skip-errors"], capture_output=True, text=True)
    # A partial index is not a success, even though it was written.
    assert r.returncode == 1
    assert "skipped" in r.stderr
    assert col.Manifest.read(out).total_frames == 2


def test_mixed_metric_scale_is_flagged(tmp_path):
    _write_seq(tmp_path / "a.wl.webm", 2, metric=True)
    _write_seq(tmp_path / "b.wl.webm", 2, metric=False)
    out = tmp_path / "c.json"
    subprocess.run([sys.executable, "-m", "wurld.cli", "index", str(tmp_path),
                    "-o", str(out)], capture_output=True, text=True, check=True)
    r = subprocess.run([sys.executable, "-m", "wurld.cli", "collection", str(out)],
                       capture_output=True, text=True)
    # Mixing scaled and unscaled reconstructions is a real hazard, not a detail.
    assert "mixed metric_scale" in r.stderr


def test_checksum_is_recorded_when_asked(tmp_path):
    import hashlib
    p = _write_seq(tmp_path / "a.wl.webm", 2)
    m = col.describe(p, checksum=True)
    assert m.sha256 == hashlib.sha256(p.read_bytes()).hexdigest()
    assert col.describe(p).sha256 is None


# ---------------------------------------------------------------- verification

def test_verify_passes_on_an_untouched_corpus(manifest):
    root, m = manifest
    assert col.Collection(m, root=root).verify() == []


def test_verify_catches_a_member_whose_frame_count_changed(tmp_path):
    """The drift that silently shifts every global index after it."""
    _write_seq(tmp_path / "a.wl.webm", 4)
    _write_seq(tmp_path / "b.wl.webm", 5)
    m, _ = col.build_manifest(tmp_path, relative_to=tmp_path)
    c = col.Collection(m, root=tmp_path)
    assert len(c) == 9
    assert c.locate(6) == (1, 2)

    # Re-record the first member one frame shorter, as a re-export would.
    _write_seq(tmp_path / "a.wl.webm", 3)

    drift = c.verify()
    kinds = {d.kind for d in drift}
    assert "frames" in kinds
    assert any("every global index after this member is wrong" in d.detail
               for d in drift)
    # The stale collection still answers, which is exactly the danger.
    assert c.locate(6) == (1, 2)


def test_verify_catches_missing_and_unreadable_members(tmp_path):
    _write_seq(tmp_path / "a.wl.webm", 3)
    _write_seq(tmp_path / "b.wl.webm", 3)
    m, _ = col.build_manifest(tmp_path, relative_to=tmp_path)
    c = col.Collection(m, root=tmp_path)

    (tmp_path / "a.wl.webm").unlink()
    (tmp_path / "b.wl.webm").write_bytes(b"\x1a\x45\xdf\xa3 truncated")

    drift = {d.kind for d in c.verify()}
    assert drift == {"missing", "unreadable"}


def test_verify_catches_a_resolution_change(tmp_path):
    _write_seq(tmp_path / "a.wl.webm", 3, w=32, h=24)
    m, _ = col.build_manifest(tmp_path, relative_to=tmp_path)
    c = col.Collection(m, root=tmp_path)
    _write_seq(tmp_path / "a.wl.webm", 3, w=64, h=48)

    drift = c.verify()
    assert any(d.kind == "resolution" for d in drift), drift


def test_verify_checksum_catches_content_changes_the_header_hides(tmp_path):
    """Same frame count and size, different pixels: only a hash sees it."""
    p = _write_seq(tmp_path / "a.wl.webm", 3)
    m, _ = col.build_manifest(tmp_path, relative_to=tmp_path, checksum=True)
    c = col.Collection(m, root=tmp_path)
    assert c.verify(checksum=True) == []

    # Flip a byte deep in the cluster data, leaving the header untouched.
    data = bytearray(p.read_bytes())
    data[-8] ^= 0xFF
    p.write_bytes(bytes(data))

    plain = c.verify()
    assert not any(d.kind == "sha256" for d in plain), "header check cannot see this"
    hashed = c.verify(checksum=True)
    assert any(d.kind == "sha256" for d in hashed), hashed


def test_cli_collection_verify_exits_nonzero_on_drift(tmp_path):
    _write_seq(tmp_path / "a.wl.webm", 4)
    out = tmp_path / "c.json"
    subprocess.run([sys.executable, "-m", "wurld.cli", "index", str(tmp_path),
                    "-o", str(out)], capture_output=True, text=True, check=True)

    ok = subprocess.run([sys.executable, "-m", "wurld.cli", "collection", str(out),
                         "--verify"], capture_output=True, text=True)
    assert ok.returncode == 0
    assert "verified" in ok.stdout

    _write_seq(tmp_path / "a.wl.webm", 2)
    bad = subprocess.run([sys.executable, "-m", "wurld.cli", "collection", str(out),
                          "--verify"], capture_output=True, text=True)
    assert bad.returncode == 1
    assert "no longer describes" in bad.stderr
    assert "wurld index" in bad.stderr


def test_streaming_a_stereo_member_yields_both_eyes(tmp_path):
    """The defect fixed in the ROS bridge, checked here too.

    A collection is the dataset path, so a stereo member losing an eye would be
    silent right up until a model trained on half the cameras.
    """
    import shutil
    src = Path("conformance/vectors/v05_stereo.wl.webm")
    if not src.exists():
        pytest.skip("conformance vectors not generated")

    root = tmp_path / "stereo"
    root.mkdir()
    shutil.copy(src, root / "a.wl.webm")
    m, failures = col.build_manifest(root, relative_to=root)
    assert not failures
    assert m.members[0].rgb_streams == ["cam0", "cam1"], "fixture must be stereo"

    c = col.Collection(m, root=root)
    items = list(c.iter_frames(fields=("rgb",)))
    assert items
    for it in items:
        assert sorted(it["rgbs"]) == ["cam0", "cam1"], "a display stream was dropped"
        assert not np.array_equal(it["rgbs"]["cam0"], it["rgbs"]["cam1"])
        # The primary stays where a single-camera consumer expects it.
        assert np.array_equal(it["rgb"], it["rgbs"]["cam0"])


def test_a_heterogeneous_corpus_streams(tmp_path):
    """Members need not be uniform: mixed resolutions, stream counts, bit depths.

    A real corpus is assembled from whatever was captured, so this must work
    rather than merely not crash.
    """
    import shutil

    root = tmp_path / "mixed"
    root.mkdir()
    _write_seq(root / "mono.wl.webm", 4, w=32, h=24)
    _write_seq(root / "bigger.wl.webm", 3, w=64, h=48)
    stereo = Path("conformance/vectors/v05_stereo.wl.webm")
    if stereo.exists():
        shutil.copy(stereo, root / "stereo.wl.webm")

    m, failures = col.build_manifest(root, relative_to=root)
    assert not failures
    assert len({(x.width, x.height) for x in m.members}) > 1, "fixture is uniform"

    c = col.Collection(m, root=root)
    items = list(c.iter_frames(fields=("rgb",)))
    assert len(items) == len(c) == m.total_frames
    assert c.verify() == []


def test_mixing_metric_and_unscaled_members_warns(tmp_path, caplog):
    """Poses that do not share a unit is a training hazard, not a detail."""
    import logging

    root = tmp_path / "scales"
    root.mkdir()
    _write_seq(root / "metric.wl.webm", 3, metric=True)
    _write_seq(root / "unscaled.wl.webm", 3, metric=False)
    m, _ = col.build_manifest(root, relative_to=root)

    with caplog.at_level(logging.WARNING):
        col.Collection(m, root=root)
    assert any("metric_scale" in r.getMessage() for r in caplog.records), \
        "a collection mixing scaled and unscaled poses said nothing"


def test_a_uniform_collection_stays_quiet(tmp_path, caplog):
    """A warning that fires on everything is noise."""
    import logging

    root = tmp_path / "uniform"
    root.mkdir()
    _write_seq(root / "a.wl.webm", 3, metric=True)
    _write_seq(root / "b.wl.webm", 3, metric=True)
    m, _ = col.build_manifest(root, relative_to=root)
    with caplog.at_level(logging.WARNING):
        col.Collection(m, root=root)
    assert not [r for r in caplog.records if "metric_scale" in r.getMessage()]


def test_indexing_finds_both_legal_suffixes(tmp_path):
    """SPEC §2 allows plain `.webm`; globbing only `*.wl.webm` found none of them.

    A corpus named the other legal way indexed as nothing, silently.
    """
    root = tmp_path / "suffixes"
    root.mkdir()
    _write_seq(root / "a.wl.webm", 3)
    _write_seq(root / "b.webm", 4)

    m, failures = col.build_manifest(root, relative_to=root)
    assert failures == []
    assert sorted(x.uri for x in m.members) == ["a.wl.webm", "b.webm"]
    assert m.total_frames == 7


def test_a_plain_webm_is_not_a_member_and_not_a_failure(tmp_path):
    """A directory holding ordinary video beside captures is unremarkable.

    Distinguished by sniffing for the WURLD tag rather than by matching an error
    message: a non-wurld file can fail a header read at several points first.
    """
    import chromapakz as cz

    root = tmp_path / "mixedbag"
    root.mkdir()
    _write_seq(root / "capture.wl.webm", 3)
    (root / "holiday.webm").write_bytes(
        cz.encode({}, rgb=np.zeros((3, 16, 16, 4), np.uint8), fps=30))
    (root / "broken.wl.webm").write_bytes(b"\x1a\x45\xdf\xa3 WURLD but truncated")

    m, failures = col.build_manifest(root, relative_to=root, on_error="skip")
    assert [x.uri for x in m.members] == ["capture.wl.webm"]
    # The plain video is absent from both lists; the broken wurld file is reported.
    assert [Path(f[0]).name for f in failures] == ["broken.wl.webm"]
