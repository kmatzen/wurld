"""PyTorch datasets over a wurld collection.

``torch`` is an optional dependency; importing this module without it raises a
clear error rather than failing at first use.

Two shapes, because the two access patterns have genuinely different costs:

``WurldIterableDataset`` streams. It shards by member across DataLoader workers
*and* distributed ranks, so each member is decoded exactly once by exactly one
worker. This is the one to use for training.

``WurldFrameDataset`` is map-style, for evaluation and debugging where you need
``dataset[i]`` and index order matters. It uses partial decode per frame, which
is far slower per sample — a random index still costs a cluster decode. Do not
reach for it because it is familiar.
"""

from __future__ import annotations

from typing import Iterable

from ..collection import Collection, Manifest

try:  # pragma: no cover - exercised by the absence path in tests
    import torch
    from torch.utils.data import Dataset, IterableDataset, get_worker_info
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "wurld.integrations.torch_data needs PyTorch:  pip install 'wurld[torch]'"
    ) from exc


def _as_collection(source) -> Collection:
    if isinstance(source, Collection):
        return source
    if isinstance(source, Manifest):
        return Collection(source)
    return Collection.read(source)


def _shard_of(shard: tuple[int, int] | None) -> tuple[int, int]:
    """Combine the distributed rank and the DataLoader worker into one shard.

    Getting this wrong duplicates data instead of erroring, so it is computed in
    one place: rank-major, worker-minor, giving world_size * num_workers
    disjoint shards.
    """
    rank, world = 0, 1
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        rank, world = torch.distributed.get_rank(), torch.distributed.get_world_size()
    if shard is not None:
        rank, world = shard

    info = get_worker_info()
    if info is None:
        return rank, world
    return rank * info.num_workers + info.id, world * info.num_workers


def to_tensors(item: dict) -> dict:
    """Convert the numpy arrays in a frame item; leave scalars and strings."""
    out = {}
    for k, v in item.items():
        if hasattr(v, "dtype") and hasattr(v, "shape"):
            # NaN in depth is meaningful (no return) and must survive as NaN.
            out[k] = torch.from_numpy(v.copy())
        else:
            out[k] = v
    return out


class WurldIterableDataset(IterableDataset):
    """Streaming frames from a collection, sharded across workers and ranks."""

    def __init__(
        self,
        source,
        *,
        fields: Iterable[str] = ("rgb", "depth"),
        shuffle: int | None = None,
        shuffle_buffer: int = 0,
        posed_only: bool = True,
        shard: tuple[int, int] | None = None,
        tensors: bool = True,
        epoch: int = 0,
    ):
        self.source = source
        self.fields = tuple(fields)
        self.shuffle = shuffle
        self.shuffle_buffer = shuffle_buffer
        self.posed_only = posed_only
        self.shard = shard
        self.tensors = tensors
        self.epoch = epoch

    def set_epoch(self, epoch: int) -> None:
        """Reshuffle between epochs; without this every epoch sees one order."""
        self.epoch = epoch

    def __iter__(self):
        collection = _as_collection(self.source)   # per-worker, so no shared handles
        seed = None if self.shuffle is None else self.shuffle + self.epoch
        for item in collection.iter_frames(
            fields=self.fields,
            shard=_shard_of(self.shard),
            shuffle=seed,
            shuffle_buffer=self.shuffle_buffer,
            posed_only=self.posed_only,
        ):
            yield to_tensors(item) if self.tensors else item


class WurldFrameDataset(Dataset):
    """Map-style access by global frame index. Slower per sample; see module docs."""

    def __init__(self, source, *, fields: Iterable[str] = ("rgb", "depth"),
                 tensors: bool = True):
        self.collection = _as_collection(source)
        self.fields = tuple(fields)
        self.tensors = tensors

    def __len__(self) -> int:
        return len(self.collection)

    def __getitem__(self, index: int) -> dict:
        item = self.collection.frame(index, fields=self.fields)
        return to_tensors(item) if self.tensors else item
