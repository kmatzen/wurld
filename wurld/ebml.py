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


def build_tags(name: str, text: str) -> bytes:
    simple = _element(TAG_NAME, name.encode()) + _element(TAG_STRING, text.encode())
    return _element(TAGS, _element(TAG, _element(SIMPLE_TAG, simple)))


def append_tag(webm: bytes, name: str, text: str) -> bytes:
    """Return new WebM bytes with a Tags/SimpleTag appended to the Segment."""
    seg_start, payload_start, payload_end = _segment_bounds(webm)
    head = webm[:seg_start]
    payload = webm[payload_start:payload_end]
    trailer = webm[payload_end:]
    new_payload = payload + build_tags(name, text)
    # Fixed 8-byte size vint keeps room for any file size.
    return (
        head
        + _encode_id(SEGMENT)
        + _encode_size(len(new_payload), length=8)
        + new_payload
        + trailer
    )


def read_tag(webm: bytes, name: str) -> str | None:
    """Return the TagString of the SimpleTag with TagName == name, or None."""
    _, payload_start, payload_end = _segment_bounds(webm)
    wanted = name.encode()
    for eid, start, end in iter_children(webm, payload_start, payload_end):
        if eid != TAGS:
            continue
        for tid, tstart, tend in iter_children(webm, start, end):
            if tid != TAG:
                continue
            for sid, sstart, send in iter_children(webm, tstart, tend):
                if sid != SIMPLE_TAG:
                    continue
                tag_name, tag_string = None, None
                for fid, fstart, fend in iter_children(webm, sstart, send):
                    if fid == TAG_NAME:
                        tag_name = webm[fstart:fend]
                    elif fid == TAG_STRING:
                        tag_string = webm[fstart:fend]
                if tag_name == wanted and tag_string is not None:
                    return tag_string.decode()
    return None
