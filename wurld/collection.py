"""Many wurld files as one dataset: manifest, global indexing, sharded streaming.

A wurld file holds one sequence. Training holds ten thousand of them, and the
gap between those two facts is this module. It is deliberately not a new
container — a collection is a *manifest* plus the files it points at, so every
member stays an ordinary playable wurld file that keeps working alone.

Three things make the difference:

**Indexing is cheap.** A manifest is built from header reads (``remote.fetch_header``),
which fetch only the bytes before the first Cluster. Describing a 100 MB file
costs kilobytes, not 100 MB, and the same path works for ``http(s)://`` members.

**Access is global.** ``Collection`` presents N files as one frame-indexed
sequence, resolving a global index to (member, local index) by bisection over
cumulative counts, so nothing scans.

**Iteration shards without overlap.** Sharding is by *member*, not by frame:
a worker that opens a file uses all of it, decoding once. Shuffling permutes
members deterministically from a seed and then shards, so shards stay disjoint
and their union is the whole collection — asserted in the tests, because a
sharding bug silently trains on duplicated data rather than crashing.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
from bisect import bisect_right
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

from . import container, ebml, remote

MANIFEST_VERSION = 1
_REMOTE_PREFIXES = ("http://", "https://")


def _is_remote(uri: str) -> bool:
    return uri.startswith(_REMOTE_PREFIXES)


@dataclass
class Member:
    """One file in a collection, described without decoding its pixels."""

    uri: str
    frames: int = 0
    posed_frames: int = 0
    cameras: list[str] = field(default_factory=list)
    rgb_streams: list[str] = field(default_factory=list)
    signals: list[str] = field(default_factory=list)
    metric_scale: bool | None = None
    t_start: float = 0.0
    t_end: float = 0.0
    width: int = 0
    height: int = 0
    bytes: int | None = None
    sha256: str | None = None

    def to_json(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}

    @classmethod
    def from_json(cls, d: dict) -> "Member":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


def _is_foreign_video(uri: str) -> bool:
    """A readable WebM that simply is not a wurld file.

    Distinguishes "not one of ours" from "one of ours, and broken", which
    deserve opposite treatment: a holiday video beside the captures should be
    passed over in silence, while a corrupt capture must be reported. The test
    is whether the container parses *and* lacks a WURLD tag — not whether the
    bytes happen to contain the string, since truncated rubbish can carry it,
    and not the error text, since a foreign file can fail at several points.
    """
    if _is_remote(uri):
        return False                    # cannot check cheaply; report the failure
    try:
        tags = ebml.read_all_tags(Path(uri).read_bytes())
    except Exception:                   # noqa: BLE001 - unparseable is a failure
        return False
    return "WURLD" not in tags


def describe(uri: str | Path, *, checksum: bool = False) -> Member:
    """Summarise one file from its header alone. Never decodes pixels."""
    uri = str(uri)
    if _is_remote(uri):
        fetch = remote.http_fetcher(uri)
        size = None
    else:
        p = Path(uri)
        fetch = remote.file_fetcher(p)
        size = p.stat().st_size

    h = remote.fetch_header(fetch)
    ts = [f.t for f in h.frames]
    video = h.video or {}
    rgbs = video.get("rgbs") or ([{"id": "rgb"}] if video.get("rgb") else [])
    stream_ids = [r["id"] if isinstance(r, dict) else str(r) for r in rgbs]

    digest = None
    if checksum and not _is_remote(uri):
        digest = hashlib.sha256(Path(uri).read_bytes()).hexdigest()

    return Member(
        uri=uri,
        frames=len(h.frames),
        posed_frames=sum(1 for f in h.frames if f.pose_valid),
        cameras=sorted(h.cameras),
        rgb_streams=stream_ids,
        signals=[s.id for s in h.signals],
        metric_scale=h.world.get("metric_scale"),
        t_start=float(min(ts)) if ts else 0.0,
        t_end=float(max(ts)) if ts else 0.0,
        width=int(video.get("width", 0)),
        height=int(video.get("height", 0)),
        bytes=size,
        sha256=digest,
    )


@dataclass
class Drift:
    """One way a manifest disagrees with the files it describes."""

    member: int
    uri: str
    kind: str        # missing | unreadable | frames | resolution | bytes | sha256 | cameras
    detail: str

    def __str__(self) -> str:
        return f"[{self.member}] {self.uri}: {self.kind} — {self.detail}"


@dataclass
class Manifest:
    """A collection: an ordered list of members plus what they add up to."""

    members: list[Member] = field(default_factory=list)
    version: int = MANIFEST_VERSION
    description: str = ""
    root: Path | None = field(default=None, repr=False)

    @property
    def total_frames(self) -> int:
        return sum(m.frames for m in self.members)

    @property
    def total_posed_frames(self) -> int:
        return sum(m.posed_frames for m in self.members)

    def to_json(self) -> dict:
        return {
            "format": "wurld-collection",
            "version": self.version,
            "description": self.description,
            "totals": {"members": len(self.members),
                       "frames": self.total_frames,
                       "posed_frames": self.total_posed_frames},
            "members": [m.to_json() for m in self.members],
        }

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.write_text(json.dumps(self.to_json(), indent=2) + "\n")
        return path

    @classmethod
    def from_json(cls, d: dict, root: Path | None = None) -> "Manifest":
        if d.get("format") != "wurld-collection":
            raise ValueError(f"not a wurld collection manifest: format={d.get('format')!r}")
        if int(d.get("version", 0)) > MANIFEST_VERSION:
            raise ValueError(
                f"manifest version {d['version']} is newer than this reader "
                f"understands ({MANIFEST_VERSION})")
        return cls(members=[Member.from_json(m) for m in d.get("members", [])],
                   version=int(d.get("version", MANIFEST_VERSION)),
                   description=d.get("description", ""),
                   root=root)

    @classmethod
    def read(cls, path: str | Path) -> "Manifest":
        path = Path(path)
        # Relative member uris resolve against the manifest, so a collection
        # stays movable as a directory.
        return cls.from_json(json.loads(path.read_text()), root=path.parent)


def build_manifest(
    sources: Iterable[str | Path] | str | Path,
    *,
    pattern: str = "*.webm",
    checksum: bool = False,
    relative_to: Path | None = None,
    on_error: str = "raise",
    description: str = "",
) -> tuple[Manifest, list[tuple[str, str]]]:
    """Describe every file in ``sources``; a directory is globbed recursively.

    The default pattern is ``*.webm``, which covers both the recommended
    ``.wurld.webm`` suffix and the plain ``.webm`` that SPEC §2 also allows. Globbing
    only ``*.wurld.webm`` meant a corpus named the other legal way indexed as
    nothing, without a word.

    A ``.webm`` with no ``WURLD`` tag is a plain video, not a broken member: it
    is skipped quietly rather than reported, because a directory holding both is
    ordinary. A file that *is* wurld and cannot be read is a failure.

    Returns ``(manifest, failures)``. ``on_error="skip"`` records unreadable
    files in ``failures`` instead of raising — a collection built over ten
    thousand files should not die on one truncated member, but it must not
    hide it either, so nothing is dropped silently.
    """
    if isinstance(sources, (str, Path)):
        sources = [sources]

    paths: list[str] = []
    for s in sources:
        if not isinstance(s, (str, Path)) or _is_remote(str(s)):
            paths.append(str(s))
            continue
        p = Path(s)
        if p.is_dir():
            paths.extend(sorted(str(q) for q in p.rglob(pattern)))
        else:
            paths.append(str(p))

    members, failures, not_wurld = [], [], 0
    for uri in paths:
        try:
            m = describe(uri, checksum=checksum)
        except Exception as exc:                       # noqa: BLE001 - reported, not swallowed
            if _is_foreign_video(uri):
                # A plain WebM sitting beside the captures. Not a member and not
                # a fault; counted so a surprising total can be explained.
                # Sniffed rather than matched on the error text, because a
                # non-wurld file can fail at several different points first.
                not_wurld += 1
                continue
            if on_error == "raise":
                raise
            failures.append((uri, f"{type(exc).__name__}: {exc}"))
            continue
        if relative_to is not None and not _is_remote(m.uri):
            try:
                m.uri = str(Path(m.uri).resolve().relative_to(Path(relative_to).resolve()))
            except ValueError:
                pass                                   # outside the root; keep it absolute
        members.append(m)

    if not_wurld:
        logging.getLogger(__name__).info(
            "skipped %d .webm file(s) with no WURLD tag (plain video, not members)",
            not_wurld)
    return Manifest(members=members, description=description), failures


class Collection:
    """N wurld files addressed as one frame-indexed dataset."""

    def __init__(self, manifest: Manifest, root: str | Path | None = None):
        self.manifest = manifest
        self.root = Path(root) if root is not None else manifest.root
        # Cumulative frame counts for O(log n) global -> (member, local).
        self._cum: list[int] = []
        total = 0
        for m in manifest.members:
            total += m.frames
            self._cum.append(total)
        self._open_index: int | None = None
        self._open_seq: container.Sequence | None = None
        self._headers: dict[int, remote.RemoteHeader] = {}

        # Members may legitimately differ in resolution, camera count or bit
        # depth — a corpus is not obliged to be uniform. Scale is different: a
        # collection mixing metric captures with up-to-scale reconstructions
        # will train on translations that do not share a unit, and nothing
        # downstream can tell. The CLI says so; anything using the API directly
        # should hear it too.
        scales = {m.metric_scale for m in manifest.members if m.metric_scale is not None}
        if len(scales) > 1:
            metric = sum(1 for m in manifest.members if m.metric_scale)
            logging.getLogger(__name__).warning(
                "collection mixes metric_scale: %d member(s) metric, %d not — "
                "poses do not share a unit; filter before training on them",
                metric, len(manifest.members) - metric)

    @classmethod
    def read(cls, manifest_path: str | Path) -> "Collection":
        return cls(Manifest.read(manifest_path))

    def __len__(self) -> int:
        return self._cum[-1] if self._cum else 0

    @property
    def members(self) -> list[Member]:
        return self.manifest.members

    def resolve(self, member: Member) -> str:
        if _is_remote(member.uri) or self.root is None or Path(member.uri).is_absolute():
            return member.uri
        return str(self.root / member.uri)

    def locate(self, index: int) -> tuple[int, int]:
        """Global frame index -> (member index, index within that member)."""
        n = len(self)
        if not -n <= index < n:
            raise IndexError(f"frame {index} out of range for {n} frames")
        if index < 0:
            index += n
        mi = bisect_right(self._cum, index)
        base = self._cum[mi - 1] if mi else 0
        return mi, index - base

    def sequence(self, member_index: int) -> container.Sequence:
        """Open a member, caching the most recent one.

        Sequential iteration touches one file at a time, so a single-slot cache
        turns "open per frame" into "open per file" without holding a whole
        collection's bytes.
        """
        if self._open_index != member_index:
            uri = self.resolve(self.manifest.members[member_index])
            if _is_remote(uri):
                raise ValueError(
                    f"{uri} is remote; use Collection.frame() for ranged access, "
                    "or download the member first")
            self._open_seq = container.read(uri)
            self._open_index = member_index
        return self._open_seq

    def _release(self) -> None:
        """Drop the cached member so the next one is not decoded alongside it."""
        self._open_index = None
        self._open_seq = None

    def frame(self, index: int, *, fields: Iterable[str] = ()) -> dict:
        """One frame by global index, decoding as little as the fields allow."""
        mi, li = self.locate(index)
        member = self.manifest.members[mi]
        want = set(fields)

        uri = self.resolve(member)
        if _is_remote(uri):
            fetch = remote.http_fetcher(uri)
            hdr = self._headers.get(mi)
            if hdr is None:
                hdr = self._headers[mi] = remote.fetch_header(fetch)
            item = _pose_fields(hdr.frames[li], member, mi, li)
            if want:
                # Pass the cached header so only the frame's clusters are fetched.
                got = remote.fetch_frames(fetch, [li], header=hdr)
                _fill_from_partial(item, got["frames"][li], hdr.signals, want)
            return item

        seq = self.sequence(mi)
        item = _pose_fields(seq.frames[li], member, mi, li)
        if want:
            # Partial decode: touch only the clusters this frame lives in.
            part = seq.fetch_frames([li])[li]
            _fill_from_partial(item, part, seq.signals, want)
        return item

    def iter_members(
        self,
        *,
        shard: tuple[int, int] | None = None,
        shuffle: int | None = None,
    ) -> Iterator[tuple[int, Member]]:
        """Member indices for one shard, optionally in a seeded random order.

        Shards are disjoint and cover everything: members are permuted first
        (so a shuffle does not change *which* shard owns a member relative to
        the permuted order) and then dealt round-robin.
        """
        order = list(range(len(self.manifest.members)))
        if shuffle is not None:
            random.Random(shuffle).shuffle(order)
        if shard is not None:
            idx, count = shard
            if not 0 <= idx < count:
                raise ValueError(f"shard {idx} out of range for {count} shards")
            order = order[idx::count]
        for mi in order:
            yield mi, self.manifest.members[mi]

    def iter_frames(
        self,
        *,
        fields: Iterable[str] = (),
        shard: tuple[int, int] | None = None,
        shuffle: int | None = None,
        shuffle_buffer: int = 0,
        posed_only: bool = False,
    ) -> Iterator[dict]:
        """Stream frames, decoding each member once.

        ``shuffle`` seeds both the member order and the within-buffer order.
        A shuffle buffer is used rather than global random access because
        decoding a file to reach one frame would dominate everything else.
        """
        want = set(fields)
        rng = random.Random(shuffle) if shuffle is not None else None
        buf: list[dict] = []

        def emit(item):
            if shuffle_buffer > 1 and rng is not None:
                buf.append(item)
                if len(buf) >= shuffle_buffer:
                    return [buf.pop(rng.randrange(len(buf)))]
                return []
            return [item]

        for mi, member in self.iter_members(shard=shard, shuffle=shuffle):
            uri = self.resolve(member)
            if _is_remote(uri):
                raise ValueError(f"{uri} is remote; iter_frames needs local members")

            if not want:
                # Metadata only: read the header region and stop before the
                # Clusters. Pulling a whole file into memory to report its poses
                # costs the file's size for nothing.
                hdr = remote.fetch_header(remote.file_fetcher(Path(uri)))
                for li, fr in enumerate(hdr.frames):
                    if posed_only and not fr.pose_valid:
                        continue
                    yield from emit(_pose_fields(fr, member, mi, li))
                continue

            # Release the previous member first: holding it while the next one
            # decodes doubles peak memory at every boundary.
            self._release()
            seq = self.sequence(mi)
            signals, frames = seq.signals, seq.frames

            # One Cluster at a time (SPEC §9.1 cluster independence) rather than
            # a whole-member decode, so memory tracks the cluster size, not the
            # length of the longest member.
            for li, payload in seq.iter_frames():
                if li >= len(frames):
                    break
                fr = frames[li]
                if posed_only and not fr.pose_valid:
                    continue
                item = _pose_fields(fr, member, mi, li)
                _fill_from_partial(item, payload, signals, want)
                yield from emit(item)

        if rng is not None:
            rng.shuffle(buf)
        yield from buf

    def verify(self, *, checksum: bool = False) -> list[Drift]:
        """Re-read every member's header and report where the manifest lies.

        Worth doing before a training run, and cheap enough to be routine: the
        header reads that built the manifest are the same ones that check it.

        Drift is not cosmetic. Global frame indexing is computed from the cached
        ``frames`` counts, so a member that gained or lost frames silently
        shifts every index after it — `locate()` keeps returning an answer, just
        the wrong one. That is the failure this exists to catch.
        """
        out: list[Drift] = []
        for mi, member in enumerate(self.manifest.members):
            uri = self.resolve(member)
            if not _is_remote(uri) and not Path(uri).exists():
                out.append(Drift(mi, member.uri, "missing", "file does not exist"))
                continue
            try:
                fresh = describe(uri, checksum=checksum and member.sha256 is not None)
            except Exception as exc:                   # noqa: BLE001 - reported below
                out.append(Drift(mi, member.uri, "unreadable", f"{type(exc).__name__}: {exc}"))
                continue

            if fresh.frames != member.frames:
                out.append(Drift(mi, member.uri, "frames",
                                 f"manifest says {member.frames}, file has {fresh.frames} "
                                 "— every global index after this member is wrong"))
            if fresh.posed_frames != member.posed_frames:
                out.append(Drift(mi, member.uri, "frames",
                                 f"posed {member.posed_frames} -> {fresh.posed_frames}"))
            if (fresh.width, fresh.height) != (member.width, member.height):
                out.append(Drift(mi, member.uri, "resolution",
                                 f"{member.width}x{member.height} -> "
                                 f"{fresh.width}x{fresh.height}"))
            if sorted(fresh.cameras) != sorted(member.cameras):
                out.append(Drift(mi, member.uri, "cameras",
                                 f"{member.cameras} -> {fresh.cameras}"))
            if member.bytes is not None and fresh.bytes is not None \
                    and fresh.bytes != member.bytes:
                out.append(Drift(mi, member.uri, "bytes",
                                 f"{member.bytes} -> {fresh.bytes}"))
            if checksum and member.sha256 and fresh.sha256 \
                    and fresh.sha256 != member.sha256:
                out.append(Drift(mi, member.uri, "sha256",
                                 "content changed while the header still matches"))
        return out

    def __repr__(self) -> str:
        return (f"Collection({len(self.manifest.members)} members, "
                f"{len(self)} frames, {self.manifest.total_posed_frames} posed)")


def _pose_fields(fr: container.Frame, member: Member, mi: int, li: int) -> dict:
    item = {
        "uri": member.uri,
        "member": mi,
        "frame": li,
        "t": fr.t,
        "camera": fr.camera,
        "pose_valid": bool(fr.pose_valid),
    }
    if fr.pose_valid:
        item["c2w"] = fr.c2w
        item["q_wxyz"] = fr.q_wxyz
        item["tr"] = fr.tr
    return item


def _wanted_signals(want: set[str], signals) -> list:
    if "signals" in want:
        return list(signals)
    return [s for s in signals if s.id in want or (s.role == "depth" and "depth" in want)]


def _fill_from_partial(item: dict, part: dict, signals, want: set[str]) -> None:
    if "rgb" in want and part.get("rgb") is not None:
        # Copy: the decoded array is a view into a whole Cluster, and a shuffle
        # buffer holding views would pin every cluster it has seen.
        item["rgb"] = np.array(part["rgb"], copy=True)
    # A stereo member has to hand over both eyes; `rgb` stays the primary so a
    # single-camera consumer is unaffected.
    if "rgb" in want and part.get("rgbs"):
        item["rgbs"] = {sid: np.array(plane, copy=True)
                        for sid, plane in part["rgbs"].items()}
    raw = part.get("signals") or {}
    for s in _wanted_signals(want, signals):
        codes = raw.get(s.id)
        if codes is None:
            continue
        item[s.id if s.role != "depth" else "depth"] = s.apply(codes)
