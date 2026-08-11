"""The PyTorch datasets, with real DataLoader workers.

The point of these tests is `_shard_of`. Combining a distributed rank with a
DataLoader worker id is the one piece of this feature that fails *silently*:
get it wrong and every worker yields the same frames, so an epoch trains on
duplicates at N times the intended weight and nothing raises. So the multi-worker
tests below run an actual DataLoader with `num_workers>0` and assert the
collected output is exactly the collection, once.
"""

import numpy as np
import pytest

import wurld as wl
from wurld import collection as col

torch = pytest.importorskip("torch", reason="PyTorch is an optional extra")
from torch.utils.data import DataLoader                      # noqa: E402

from wurld.integrations.torch_data import (                  # noqa: E402
    WurldFrameDataset, WurldIterableDataset, _shard_of, to_tensors,
)

COUNTS = [3, 5, 2, 7, 4]


def _write_seq(path, n, *, w=32, h=24):
    import chromapakz as cz

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    rgb, depth = [], []
    for i in range(n):
        g = np.clip((0.5 + 0.4 * np.sin(xx / 5 + i * 0.3) * np.cos(yy / 4)) * 255, 0, 255)
        rgb.append(np.dstack([g, g, g, np.full_like(g, 255)]).astype(np.uint8))
        depth.append((1.5 + 0.5 * np.sin(xx / 7 + i * 0.2)).astype(np.float32))
    f = 1.1 * w
    wl.write(
        path,
        cameras={"0": wl.Camera(model="PINHOLE", width=w, height=h,
                                params=[f, f, w / 2, h / 2])},
        frames=[wl.Frame(i=i, t=i / 30, camera="0", q_wxyz=(1.0, 0.0, 0.0, 0.0),
                         tr=(0.01 * i, 0.0, 0.5)) for i in range(n)],
        rgb=np.stack(rgb),
        signals={"depth": cz.quantize_inverse(np.stack(depth), near=0.3, far=9.0)},
        specs={"depth": {"inverse_depth": True, "near": 0.3, "far": 9.0}},
        signal_meta=[wl.SignalMeta("depth", "depth",
                                   {"type": "inverse_depth", "near": 0.3, "far": 9.0,
                                    "levels": 65536, "invalid": 0})],
        world={"metric_scale": True}, fps=30,
    )
    return path


@pytest.fixture(scope="module")
def manifest_path(tmp_path_factory):
    root = tmp_path_factory.mktemp("torchcorpus")
    for i, n in enumerate(COUNTS):
        _write_seq(root / f"seq{i}.wurld.webm", n)
    m, failures = col.build_manifest(root, relative_to=root)
    assert failures == []
    return m.write(root / "collection.json")


def _ids(batchless_items):
    return sorted((int(it["member"]), int(it["frame"])) for it in batchless_items)


ALL = None  # filled by the first test that needs it


def _everything(manifest_path):
    c = col.Collection.read(manifest_path)
    return sorted((it["member"], it["frame"]) for it in c.iter_frames())


def test_shard_of_combines_rank_and_worker_without_overlap():
    """Pure arithmetic check across every rank/worker pair.

    torch.distributed is not initialised here, so the rank comes from the
    explicit `shard` argument — the same code path a caller uses when they
    manage ranks themselves.
    """
    for world in (1, 2, 3):
        for workers in (1, 2, 4):
            got = set()
            for rank in range(world):
                for wid in range(workers):
                    # Emulate what get_worker_info() would report.
                    class _Info:
                        id = wid
                        num_workers = workers

                    import wurld.integrations.torch_data as td
                    real = td.get_worker_info
                    td.get_worker_info = lambda _i=_Info: _i
                    try:
                        got.add(_shard_of((rank, world)))
                    finally:
                        td.get_worker_info = real
            # Every (rank, worker) pair must land on a distinct shard index,
            # and the shard count must be the product.
            assert len(got) == world * workers
            assert {n for _, n in got} == {world * workers}
            assert {i for i, _ in got} == set(range(world * workers))


def test_iterable_dataset_single_process_covers_everything(manifest_path):
    ds = WurldIterableDataset(manifest_path, fields=(), tensors=False)
    assert _ids(list(ds)) == _everything(manifest_path)


@pytest.mark.parametrize("workers", [1, 2, 3])
def test_dataloader_workers_partition_the_collection(manifest_path, workers):
    """The silent-duplication test: real workers, real DataLoader."""
    ds = WurldIterableDataset(manifest_path, fields=(), tensors=False)
    seen = []
    for batch in DataLoader(ds, batch_size=None, num_workers=workers):
        seen.append({"member": batch["member"], "frame": batch["frame"]})
    got = _ids(seen)
    assert got == _everything(manifest_path), "workers duplicated or dropped frames"
    assert len(got) == len(set(got))


def test_dataloader_with_more_workers_than_members(manifest_path):
    # Extra workers must be empty, not repeats of someone else's shard.
    ds = WurldIterableDataset(manifest_path, fields=(), tensors=False)
    seen = [{"member": b["member"], "frame": b["frame"]}
            for b in DataLoader(ds, batch_size=None, num_workers=8)]
    assert _ids(seen) == _everything(manifest_path)


def test_tensors_and_shapes(manifest_path):
    ds = WurldIterableDataset(manifest_path, fields=("rgb", "depth"), posed_only=True)
    item = next(iter(ds))
    assert torch.is_tensor(item["rgb"]) and item["rgb"].shape[:2] == (24, 32)
    assert torch.is_tensor(item["depth"]) and item["depth"].shape == (24, 32)
    assert torch.is_tensor(item["c2w"]) and item["c2w"].shape == (4, 4)
    assert item["depth"].dtype == torch.float32


def test_nan_survives_conversion():
    """Depth NaN means 'no return' — it must not become 0 on the way to torch."""
    d = np.array([[1.0, np.nan], [2.0, 3.0]], dtype=np.float32)
    out = to_tensors({"depth": d})["depth"]
    assert torch.isnan(out[0, 1])
    assert out[0, 0] == 1.0


def test_set_epoch_changes_the_order(manifest_path):
    ds = WurldIterableDataset(manifest_path, fields=(), shuffle=11, tensors=False)
    first = [(it["member"], it["frame"]) for it in ds]
    ds.set_epoch(1)
    second = [(it["member"], it["frame"]) for it in ds]
    assert sorted(first) == sorted(second)
    assert first != second, "set_epoch did not reshuffle"


def test_map_style_dataset_indexes_globally(manifest_path):
    ds = WurldFrameDataset(manifest_path, fields=("depth",))
    assert len(ds) == sum(COUNTS)
    item = ds[sum(COUNTS) - 1]
    assert int(item["member"]) == len(COUNTS) - 1
    assert int(item["frame"]) == COUNTS[-1] - 1
    assert item["depth"].shape == (24, 32)


def test_map_style_matches_direct_read(manifest_path):
    ds = WurldFrameDataset(manifest_path, fields=("rgb",), tensors=False)
    c = col.Collection.read(manifest_path)
    root = c.root
    gi = COUNTS[0] + 1                                  # member 1, frame 1
    item = ds[gi]
    seq = wl.read(root / c.members[1].uri)
    assert np.array_equal(item["rgb"], seq.rgb[1])
