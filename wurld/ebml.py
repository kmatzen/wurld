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


def insert_header_tags(webm: bytes, tags: dict[str, str | bytes]) -> bytes:
    """Insert a Tags element before the first Cluster (streaming layout, SPEC §9).

    Cluster offsets shift, so Cues are rebuilt from the actual cluster
    timestamps at their new positions; a pre-existing Cues element is replaced.
    """
    seg_start, payload_start, payload_end = _segment_bounds(webm)
    tags_bytes = build_tags(tags)

    head_parts: list[bytes] = []  # everything before the first Cluster, minus old Cues
    body_parts: list[bytes] = []  # Clusters and anything between/after them, minus old Cues
    seen_cluster = False
    cluster_offsets: list[tuple[int, int]] = []  # (timestamp, offset within body_parts)
    body_len = 0
    for eid, elem_start, pstart, pend in _top_level(webm, payload_start, payload_end):
        raw = webm[elem_start:pend]
        if eid == CUES:
            continue  # rebuilt below
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
            head_parts.append(raw)
        else:
            body_parts.append(raw)
            body_len += len(raw)

    head = b"".join(head_parts) + tags_bytes
    cues = _build_cues([(ts, len(head) + off) for ts, off in cluster_offsets])
    new_payload = head + b"".join(body_parts) + cues
    return (
        webm[:seg_start]
        + _encode_id(SEGMENT)
        + _encode_size(len(new_payload), length=8)
        + new_payload
        + webm[payload_end:]
    )


def read_all_tags(webm: bytes) -> dict[str, str | bytes]:
    """All SimpleTags in the file: TagString entries as str, TagBinary as bytes."""
    _, payload_start, payload_end = _segment_bounds(webm)
    out: dict[str, str | bytes] = {}
    for eid, start, end in iter_children(webm, payload_start, payload_end):
        if eid != TAGS:
            continue
        for tid, tstart, tend in iter_children(webm, start, end):
            if tid != TAG:
                continue
            for sid, sstart, send in iter_children(webm, tstart, tend):
                if sid != SIMPLE_TAG:
                    continue
                tag_name, value = None, None
                for fid, fstart, fend in iter_children(webm, sstart, send):
                    if fid == TAG_NAME:
                        tag_name = webm[fstart:fend].decode()
                    elif fid == TAG_STRING:
                        value = webm[fstart:fend].decode()
                    elif fid == TAG_BINARY:
                        value = bytes(webm[fstart:fend])
                if tag_name is not None and value is not None:
                    if isinstance(value, bytes) and isinstance(out.get(tag_name), bytes):
                        out[tag_name] += value  # repeated binary tags concatenate (SPEC §10)
                    else:
                        out[tag_name] = value  # strings: last occurrence wins
    return out


def read_tag(webm: bytes, name: str) -> str | None:
    """Return the TagString of the SimpleTag with TagName == name, or None."""
    value = read_all_tags(webm).get(name)
    return value if isinstance(value, str) else None
