"""Minimal EBML/Matroska handling: read and append a named SimpleTag.

Wurld stores its JSON document as a Matroska Tags element containing a
SimpleTag named ``WURLD`` (SPEC.md §2), appended to the end of the Segment
payload so Cue/Cluster offsets (relative to the Segment payload start) survive.
"""

from __future__ import annotations

# Element IDs (stored with EBML marker bits, i.e. as they appear on disk).
EBML_HEADER = 0x1A45DFA3
SEGMENT = 0x18538067
TAGS = 0x1254C367
TAG = 0x7373
SIMPLE_TAG = 0x67C8
TAG_NAME = 0x45A3
TAG_STRING = 0x4487
TAG_BINARY = 0x4485
CLUSTER = 0x1F43B675
CLUSTER_TIMESTAMP = 0xE7
CUES = 0x1C53BB6B
CUE_POINT = 0xBB
CUE_TIME = 0xB3
CUE_TRACK_POSITIONS = 0xB7
CUE_TRACK = 0xF7
CUE_CLUSTER_POSITION = 0xF1
SEEK_HEAD = 0x114D9B74
SEEK = 0x4DBB
SEEK_ID = 0x53AB
SEEK_POSITION = 0x53AC
INFO = 0x1549A966
TRACKS = 0x1654AE6B


def _read_vint(data: bytes, pos: int, keep_marker: bool) -> tuple[int, int]:
    """Return (value, new_pos). keep_marker=True for element IDs."""
    first = data[pos]
    if first == 0:
        raise ValueError(f"invalid vint at {pos}")
    length = 8 - first.bit_length() + 1
    raw = int.from_bytes(data[pos : pos + length], "big")
    if not keep_marker:
        raw &= (1 << (7 * length)) - 1
    return raw, pos + length


def _unknown_size(size: int, length: int) -> bool:
    return size == (1 << (7 * length)) - 1


def _encode_id(element_id: int) -> bytes:
    return element_id.to_bytes((element_id.bit_length() + 7) // 8, "big")


def _encode_size(size: int, length: int | None = None) -> bytes:
    if length is None:
        length = 1
        while size >= (1 << (7 * length)) - 1:
            length += 1
    if size >= (1 << (7 * length)) - 1:
        raise ValueError("size does not fit vint length")
    return (size | (1 << (7 * length))).to_bytes(length, "big")


def _element(element_id: int, payload: bytes) -> bytes:
    return _encode_id(element_id) + _encode_size(len(payload)) + payload


def iter_children(data: bytes, pos: int, end: int):
    """Yield (element_id, payload_start, payload_end) for EBML children."""
    while pos < end:
        element_id, pos = _read_vint(data, pos, keep_marker=True)
        size_start = pos
        size, pos = _read_vint(data, pos, keep_marker=False)
        size_len = pos - size_start
        if _unknown_size(size, size_len):
            # Unknown-size element extends to end of parent.
            yield element_id, pos, end
            return
        yield element_id, pos, pos + size
        pos += size


def _segment_bounds(data: bytes) -> tuple[int, int, int]:
    """Return (segment_id_start, payload_start, payload_end)."""
    pos = 0
    while pos < len(data):
        id_start = pos
        eid, pos = _read_vint(data, pos, keep_marker=True)
        size_start = pos
        size, pos = _read_vint(data, pos, keep_marker=False)
        size_len = pos - size_start
        end = len(data) if _unknown_size(size, size_len) else pos + size
        if eid == SEGMENT:
            return id_start, pos, end
        pos = end
    raise ValueError("no Matroska Segment found")


def build_tags(tags: dict[str, str | bytes]) -> bytes:
    """One Tags element holding one SimpleTag per entry.

    str payloads become TagString (UTF-8); bytes payloads become TagBinary.
    """
    body = b""
    for name, payload in tags.items():
        if isinstance(payload, str):
            value = _element(TAG_STRING, payload.encode())
        else:
            value = _element(TAG_BINARY, bytes(payload))
        body += _element(TAG, _element(SIMPLE_TAG, _element(TAG_NAME, name.encode()) + value))
    return _element(TAGS, body)


def append_tags(webm: bytes, tags: dict[str, str | bytes]) -> bytes:
    """Return new WebM bytes with a Tags element appended to the Segment."""
    seg_start, payload_start, payload_end = _segment_bounds(webm)
    head = webm[:seg_start]
    payload = webm[payload_start:payload_end]
    trailer = webm[payload_end:]
    new_payload = payload + build_tags(tags)
    # Fixed 8-byte size vint keeps room for any file size.
    return (
        head
        + _encode_id(SEGMENT)
        + _encode_size(len(new_payload), length=8)
        + new_payload
        + trailer
    )


def append_tag(webm: bytes, name: str, text: str) -> bytes:
    """Back-compat convenience for a single string tag."""
    return append_tags(webm, {name: text})


def _encode_uint(element_id: int, value: int) -> bytes:
    n = max(1, (value.bit_length() + 7) // 8)
    return _element(element_id, value.to_bytes(n, "big"))


def _read_uint(data: bytes, start: int, end: int) -> int:
    return int.from_bytes(data[start:end], "big")


def _top_level(data: bytes, payload_start: int, payload_end: int):
    """Yield (eid, elem_start, payload_start, payload_end) inside the Segment."""
    pos = payload_start
    while pos < payload_end:
        elem_start = pos
        eid, pos = _read_vint(data, pos, keep_marker=True)
        size_start = pos
        size, pos = _read_vint(data, pos, keep_marker=False)
        size_len = pos - size_start
        end = payload_end if _unknown_size(size, size_len) else pos + size
        yield eid, elem_start, pos, end
        pos = end


def _build_cues(clusters: list[tuple[int, int]], track: int = 1) -> bytes:
    """clusters: (timestamp, offset-relative-to-segment-payload-start)."""
    body = b""
    for ts, offset in clusters:
        positions = _encode_uint(CUE_TRACK, track) + _encode_uint(CUE_CLUSTER_POSITION, offset)
        body += _element(
            CUE_POINT, _encode_uint(CUE_TIME, ts) + _element(CUE_TRACK_POSITIONS, positions)
        )
    return _element(CUES, body)


def _build_seek_head(entries: list[tuple[int, int]]) -> bytes:
    """entries: (element_id, position relative to Segment payload start).

    Positions are fixed 8-byte uints so the SeekHead's size does not depend on
    the values, letting callers compute positions in a single pass (SPEC §9.1).
    """
    body = b""
    for element_id, pos in entries:
        body += _element(
            SEEK,
            _element(SEEK_ID, _encode_id(element_id))
            + _element(SEEK_POSITION, pos.to_bytes(8, "big")),
        )
    return _element(SEEK_HEAD, body)


def seek_head_size(n_entries: int) -> int:
    """Byte length of a SeekHead built by _build_seek_head (value-independent)."""
    return len(_build_seek_head([(SEGMENT, 0)] * n_entries))


def read_seek_head(data: bytes, payload_start: int, payload_end: int) -> dict[int, int]:
    """{element_id: absolute file offset} from the first SeekHead child, or {}."""
    for eid, _, pstart, pend in _top_level(data, payload_start, payload_end):
        if eid != SEEK_HEAD:
            continue
        out = {}
        for sid, ss, se in iter_children(data, pstart, pend):
            if sid != SEEK:
                continue
            target = pos = None
            for fid, fs, fe in iter_children(data, ss, se):
                if fid == SEEK_ID:
                    target = _read_uint(data, fs, fe)
                elif fid == SEEK_POSITION:
                    pos = _read_uint(data, fs, fe)
            if target is not None and pos is not None:
                out[target] = payload_start + pos
        return out
    return {}


def read_cues(buf: bytes, pos: int = 0) -> list[tuple[int, int]]:
    """Parse a Cues element at ``pos`` -> [(time_ms, segment-relative position)]."""
    eid, p = _read_vint(buf, pos, keep_marker=True)
    if eid != CUES:
        raise ValueError(f"expected Cues element, found {eid:#x}")
    size, p = _read_vint(buf, p, keep_marker=False)
    out = []
    for cid, cs, ce in iter_children(buf, p, p + size):
        if cid != CUE_POINT:
            continue
        time_ms = position = None
        for fid, fs, fe in iter_children(buf, cs, ce):
            if fid == CUE_TIME:
                time_ms = _read_uint(buf, fs, fe)
            elif fid == CUE_TRACK_POSITIONS:
                for gid, gs, ge in iter_children(buf, fs, fe):
                    if gid == CUE_CLUSTER_POSITION:
                        position = _read_uint(buf, gs, ge)
        if time_ms is not None and position is not None:
            out.append((time_ms, position))
    return out


def cluster_first_block_keyframes(buf: bytes, pos: int = 0) -> dict[int, bool]:
    """First-block keyframe flag per track for the Cluster element at ``pos``."""
    eid, p = _read_vint(buf, pos, keep_marker=True)
    if eid != CLUSTER:
        raise ValueError(f"expected Cluster element, found {eid:#x}")
    size, p = _read_vint(buf, p, keep_marker=False)
    first: dict[int, bool] = {}
    for cid, cs, ce in iter_children(buf, p, p + size):
        if cid == 0xA3:  # SimpleBlock
            track, tp = _read_vint(buf, cs, keep_marker=False)
            if track not in first:
                first[track] = bool(buf[tp + 2] & 0x80)
    return first


def splice_file(prefix: bytes, seg_start: int, payload_parts: list[bytes]) -> bytes:
    """Rebuild a standalone file: pre-Segment bytes + Segment(payload_parts)."""
    payload = b"".join(payload_parts)
    return (
        prefix[:seg_start]
        + _encode_id(SEGMENT)
        + _encode_size(len(payload), length=8)
        + payload
    )


def insert_header_tags(webm: bytes, tags: dict[str, str | bytes]) -> bytes:
    """Rebuild into the batch streaming layout (SPEC §9/§9.1): a SeekHead, then the
    original header elements, the wurld Tags, Clusters, and rebuilt Cues.

    Cluster offsets shift, so Cues are rebuilt from the actual cluster
    timestamps at their new positions; a pre-existing Cues/SeekHead is replaced.
    """
    seg_start, payload_start, payload_end = _segment_bounds(webm)
    tags_bytes = build_tags(tags)

    head_parts: list[tuple[int, bytes]] = []  # (element id, raw) before the first Cluster
    body_parts: list[bytes] = []  # Clusters and anything between/after them
    seen_cluster = False
    cluster_offsets: list[tuple[int, int]] = []  # (timestamp, offset within body)
    body_len = 0
    for eid, elem_start, pstart, pend in _top_level(webm, payload_start, payload_end):
        raw = webm[elem_start:pend]
        if eid in (CUES, SEEK_HEAD):
            continue  # rebuilt / replaced below
        if eid == CLUSTER:
            seen_cluster = True
            ts = 0
            for cid, cs, ce in iter_children(webm, pstart, pend):
                if cid == CLUSTER_TIMESTAMP:
                    ts = _read_uint(webm, cs, ce)
                    break
            cluster_offsets.append((ts, body_len))
            body_parts.append(raw)
            body_len += len(raw)
        elif not seen_cluster:
            head_parts.append((eid, raw))
        else:
            body_parts.append(raw)
            body_len += len(raw)

    # SeekHead entries: every header element, the first Cluster, and Cues.
    n_entries = len(head_parts) + 1 + (2 if cluster_offsets else 1)
    sh_size = seek_head_size(n_entries)

    entries: list[tuple[int, int]] = []
    pos = sh_size
    head = b""
    for eid, raw in head_parts:
        entries.append((eid, pos))
        head += raw
        pos += len(raw)
    entries.append((TAGS, pos))
    head += tags_bytes
    pos += len(tags_bytes)
    first_cluster_pos = pos
    if cluster_offsets:
        entries.append((CLUSTER, first_cluster_pos))
    cues_pos = first_cluster_pos + body_len
    entries.append((CUES, cues_pos))

    seek_head = _build_seek_head(entries)
    assert len(seek_head) == sh_size, "SeekHead size must be value-independent"
    cues = _build_cues([(ts, first_cluster_pos + off) for ts, off in cluster_offsets])
    new_payload = seek_head + head + b"".join(body_parts) + cues
    return (
        webm[:seg_start]
        + _encode_id(SEGMENT)
        + _encode_size(len(new_payload), length=8)
        + new_payload
        + webm[payload_end:]
    )


def collect_simple_tags(data: bytes, start: int, end: int):
    """Yield (name, str|bytes) for each SimpleTag inside one Tags payload."""
    for tid, tstart, tend in iter_children(data, start, end):
        if tid != TAG:
            continue
        for sid, sstart, send in iter_children(data, tstart, tend):
            if sid != SIMPLE_TAG:
                continue
            tag_name, value = None, None
            for fid, fstart, fend in iter_children(data, sstart, send):
                if fid == TAG_NAME:
                    tag_name = data[fstart:fend].decode()
                elif fid == TAG_STRING:
                    value = data[fstart:fend].decode()
                elif fid == TAG_BINARY:
                    value = bytes(data[fstart:fend])
            if tag_name is not None and value is not None:
                yield tag_name, value


def read_all_tags(webm: bytes) -> dict[str, str | bytes]:
    """All SimpleTags in the file: TagString entries as str, TagBinary as bytes."""
    _, payload_start, payload_end = _segment_bounds(webm)
    out: dict[str, str | bytes] = {}
    for eid, start, end in iter_children(webm, payload_start, payload_end):
        if eid != TAGS:
            continue
        for tag_name, value in collect_simple_tags(webm, start, end):
            if isinstance(value, bytes) and isinstance(out.get(tag_name), bytes):
                out[tag_name] += value  # repeated binary tags concatenate (SPEC §10)
            else:
                out[tag_name] = value  # strings: last occurrence wins
    return out


def read_tag(webm: bytes, name: str) -> str | None:
    """Return the TagString of the SimpleTag with TagName == name, or None."""
    value = read_all_tags(webm).get(name)
    return value if isinstance(value, str) else None
