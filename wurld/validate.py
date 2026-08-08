"""Conformance checking: SPEC.md's normative requirements, made executable.

The format's value is that a stranger's file reads correctly in someone else's
tool. Nothing enforced that — SPEC carries the MUSTs in prose, and every
interop bug found so far was a violation nobody could have caught without
reading the whole document. This turns the prose into checks a producer can run
before shipping a file, and CI can run against every implementation.

Findings carry the SPEC section they come from, because "poses are out of
order" is far less useful than knowing which rule that breaks and why it exists.

Severity mirrors the spec's own language: a MUST violation is an ERROR, a
SHOULD violation is a WARNING, and anything merely suspicious is a NOTE.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from . import container, ebml

# COLMAP model -> required parameter count (SPEC §4.1).
CAMERA_MODELS = {
    "PINHOLE": 4,
    "SIMPLE_RADIAL": 4,
    "RADIAL": 5,
    "OPENCV": 8,
    "OPENCV_FISHEYE": 8,
}
ROLES = {"depth", "confidence", "object_id", "semantic_id", "normal_packed", "custom"}
CONVENTIONS = {
    "camera_axes": "RDF",
    "pose_direction": "camera_to_world",
    "quaternion_order": "wxyz",
    "units": "meters",
    "timestamp_units": "seconds",
}

ERROR, WARNING, NOTE = "error", "warning", "note"


@dataclass(frozen=True)
class Finding:
    severity: str
    section: str
    message: str

    def __str__(self) -> str:
        return f"{self.severity.upper():7} [SPEC {self.section}] {self.message}"


class _Report:
    def __init__(self) -> None:
        self.findings: list[Finding] = []

    def error(self, section: str, msg: str) -> None:
        self.findings.append(Finding(ERROR, section, msg))

    def warn(self, section: str, msg: str) -> None:
        self.findings.append(Finding(WARNING, section, msg))

    def note(self, section: str, msg: str) -> None:
        self.findings.append(Finding(NOTE, section, msg))


def validate(path: str | Path) -> list[Finding]:
    """Check one file against SPEC. Returns findings, worst-first by severity."""
    r = _Report()
    data = Path(path).read_bytes()

    try:
        tags = ebml.read_all_tags(data)
    except Exception as e:  # noqa: BLE001 — a parse failure is itself the finding
        r.error("3", f"not a readable Matroska/WebM file: {type(e).__name__}: {e}")
        return r.findings

    doc_str = tags.get("WURLD")
    if doc_str is None:
        # Legitimate: SPEC §10 says a file with no WURLD tag is plain chromapakz.
        r.note("10", "no WURLD tag: this is a plain chromapakz file, not a wurld file")
        return r.findings
    if not isinstance(doc_str, str):
        r.error("5", "WURLD tag is binary; it must be a TagString holding JSON")
        return r.findings
    try:
        doc = json.loads(doc_str)
    except json.JSONDecodeError as e:
        r.error("5", f"WURLD tag is not valid JSON: {e}")
        return r.findings

    _check_document(r, doc)
    cameras = doc.get("cameras") or {}
    _check_cameras(r, cameras, data)
    _check_signals(r, doc)
    frames = _check_frames(r, doc, tags, cameras)
    _check_rigs(r, doc, cameras)
    _check_imu(r, doc, tags, frames)
    _check_layout(r, data, tags)

    order = {ERROR: 0, WARNING: 1, NOTE: 2}
    return sorted(r.findings, key=lambda f: (order[f.severity], f.section))


def _check_document(r: _Report, doc: dict) -> None:
    if doc.get("format") != "wurld":
        r.error("5", f'document "format" is {doc.get("format")!r}, must be "wurld"')
    if not doc.get("version"):
        r.error("5", 'document has no "version"')

    # SPEC §3: conventions are fixed, not negotiated. A file that declares
    # something else is not describing a dialect — consumers do not branch on
    # these, so a mismatch means the data is silently misinterpreted.
    conv = doc.get("conventions") or {}
    for key, expected in CONVENTIONS.items():
        got = conv.get(key)
        if got is None:
            r.warn("3", f'conventions omit "{key}" (fixed value is {expected!r})')
        elif got != expected:
            r.error("3", f'conventions.{key} is {got!r}, must be {expected!r} — '
                         "consumers do not branch on this, so the data will be misread")


def _check_cameras(r: _Report, cameras: dict, data: bytes) -> None:
    if not cameras:
        r.error("4.1", "no cameras declared")
        return
    try:
        probe = ebml.read_all_tags(data).get("CHROMAPAKZ")
        video = json.loads(probe) if isinstance(probe, str) else {}
    except Exception:  # noqa: BLE001
        video = {}
    vw, vh = video.get("width"), video.get("height")

    for cid, cam in cameras.items():
        model = cam.get("model")
        if model not in CAMERA_MODELS:
            r.error("4.1", f"camera {cid!r}: unknown model {model!r} "
                           f"(known: {', '.join(sorted(CAMERA_MODELS))})")
        else:
            want = CAMERA_MODELS[model]
            got = len(cam.get("params") or [])
            if got != want:
                r.error("4.1", f"camera {cid!r}: {model} needs {want} params, got {got}")
        w, h = cam.get("width"), cam.get("height")
        if not (isinstance(w, int) and isinstance(h, int) and w > 0 and h > 0):
            r.error("4.1", f"camera {cid!r}: width/height must be positive ints, got {w}x{h}")
        elif vw and vh and (w != vw or h != vh):
            # §4.1: calibrated resolution MUST equal the video track resolution.
            r.error("4.1", f"camera {cid!r} is calibrated {w}x{h} but the video track is "
                           f"{vw}x{vh}; intrinsics would be applied at the wrong scale")


def _check_signals(r: _Report, doc: dict) -> None:
    for sig in doc.get("signals") or []:
        sid = sig.get("id")
        if not sid:
            r.error("4.3", "a signal has no id")
        role = sig.get("role")
        if role not in ROLES:
            r.warn("4.3", f"signal {sid!r}: unknown role {role!r}")
        vm = sig.get("value_map") or {}
        kind = vm.get("type")
        if kind == "inverse_depth":
            near, far = vm.get("near"), vm.get("far")
            levels = vm.get("levels", 65536)
            if not (isinstance(near, (int, float)) and near > 0):
                r.error("6", f"signal {sid!r}: inverse_depth needs near > 0, got {near!r}")
            if not (isinstance(far, (int, float)) and far > (near or 0)):
                r.error("6", f"signal {sid!r}: inverse_depth needs far > near, got {far!r}")
            if not (isinstance(levels, int) and levels >= 3):
                # levels-2 is the divisor in the dequantize; < 3 divides by zero.
                r.error("6", f"signal {sid!r}: inverse_depth needs levels >= 3, got {levels!r}")
        elif kind == "linear":
            if not isinstance(vm.get("scale"), (int, float)):
                r.error("6", f"signal {sid!r}: linear value_map needs a numeric scale")
        elif kind == "labels":
            if not isinstance(vm.get("labels"), dict):
                r.error("6", f"signal {sid!r}: labels value_map needs a labels object")
        elif kind is None:
            r.warn("6", f"signal {sid!r}: no value_map; consumers cannot interpret the codes")
        else:
            r.note("6", f"signal {sid!r}: unrecognised value_map type {kind!r}")


def _check_frames(r: _Report, doc: dict, tags: dict, cameras: dict) -> list:
    """Resolve poses the way SPEC §9 says a reader must, then check them."""
    fb = doc.get("frames_binary")
    camera_keys = list(fb["cameras"]) if fb and "cameras" in fb else sorted(cameras)
    table, chunks = tags.get("WURLD_FRAMES"), tags.get("WURLD_POSES")

    frames, source = [], None
    if isinstance(table, bytes):
        source = "WURLD_FRAMES"
        if len(table) % container._FRAME_RECORD.size:
            r.error("7", f"WURLD_FRAMES is {len(table)} bytes, not a multiple of "
                         f"{container._FRAME_RECORD.size}")
            return []          # unpacking a partial record would raise, not inform
        try:
            frames = container.unpack_frames(table, camera_keys)
        except ValueError as e:
            # The reader rejects this file; the validator's job is to say why.
            r.error("7", f"WURLD_FRAMES: {e}")
            return []
        if fb and "count" in fb and len(frames) != fb["count"]:
            r.error("7", f'frames_binary.count says {fb["count"]} but the table holds '
                         f"{len(frames)}")
    elif isinstance(chunks, bytes):
        source = "WURLD_POSES"
        if len(chunks) % container._FRAME_RECORD.size:
            r.error("7", f"WURLD_POSES is {len(chunks)} bytes, not a multiple of "
                         f"{container._FRAME_RECORD.size}")
            return []
        try:
            frames = container.unpack_frames(chunks, camera_keys)
        except ValueError as e:
            r.error("7", f"WURLD_POSES: {e}")
            return []
    else:
        source = "JSON frames"
        frames = doc.get("frames") or []
        if fb is not None:
            r.error("7", "frames_binary is declared but no WURLD_FRAMES tag is present")

    if not frames:
        r.warn("4.2", "no poses in the file")
        return frames

    # §7: overrides need the JSON form, so a binary table plus per-frame params
    # means some intrinsics were silently dropped.
    if source != "JSON frames" and any(
        isinstance(f, dict) and f.get("params") for f in (doc.get("frames") or [])
    ):
        r.error("7", "per-frame intrinsics overrides require the JSON frames array; "
                     "writers must not use the binary table with overridden frames")

    idx = [f["i"] if isinstance(f, dict) else f.i for f in frames]
    ts = [f["t"] if isinstance(f, dict) else f.t for f in frames]

    # §9: records MUST be ordered by frame index across the whole file.
    if any(b < a for a, b in zip(idx, idx[1:])):
        bad = next(k for k, (a, b) in enumerate(zip(idx, idx[1:])) if b < a)
        r.error("9", f"{source}: frame indices are not ascending "
                     f"(index {idx[bad]} then {idx[bad+1]} at record {bad})")
    if len(set(idx)) != len(idx):
        r.error("9", f"{source}: duplicate frame indices")

    # §4.2: timestamps monotonic non-decreasing.
    if any(b < a for a, b in zip(ts, ts[1:])):
        bad = next(k for k, (a, b) in enumerate(zip(ts, ts[1:])) if b < a)
        r.error("4.2", f"timestamps go backwards at record {bad}: {ts[bad]} then {ts[bad+1]}")

    for n, f in enumerate(frames):
        valid = f.get("pose_valid", True) if isinstance(f, dict) else f.pose_valid
        if not valid:
            continue
        q = f["q_wxyz"] if isinstance(f, dict) else f.q_wxyz
        cam = f.get("camera") if isinstance(f, dict) else f.camera
        if q is None:
            r.error("4.2", f"record {n}: pose_valid but no q_wxyz")
            continue
        norm = math.sqrt(sum(float(v) ** 2 for v in q))
        if abs(norm - 1.0) > 1e-3:
            r.error("4.2", f"record {n}: quaternion norm {norm:.6f}, must be unit")
        if cameras and cam not in cameras:
            r.error("4.2", f"record {n}: camera {cam!r} is not in cameras "
                           f"({', '.join(sorted(cameras))})")
    return frames


def _check_rigs(r: _Report, doc: dict, cameras: dict) -> None:
    for rig_id, rig in (doc.get("rigs") or {}).items():
        for cam_id in (rig.get("cameras") or {}):
            if cam_id not in cameras:
                r.error("8.1", f"rig {rig_id!r} references camera {cam_id!r}, "
                               "which is not declared")


def _check_imu(r: _Report, doc: dict, tags: dict, frames: list) -> None:
    for stream_id in (doc.get("imu") or {}):
        buf = tags.get(f"WURLD_IMU_{stream_id}")
        if not isinstance(buf, bytes):
            r.error("8.3", f"imu stream {stream_id!r} is declared but WURLD_IMU_{stream_id} "
                           "is missing")
            continue
        if len(buf) % 32:
            r.error("8.3", f"WURLD_IMU_{stream_id} is {len(buf)} bytes, not a multiple of 32")


def _check_layout(r: _Report, data: bytes, tags: dict) -> None:
    """§9 streaming layout: metadata ahead of the media, and a truthful index."""
    try:
        _, payload_start, payload_end = ebml._segment_bounds(data)
    except Exception as e:  # noqa: BLE001
        r.error("9", f"cannot read the Segment: {e}")
        return

    first_cluster = None
    saw_wurld_tag_before_cluster = False
    for eid, start, end in ebml.iter_children(data, payload_start, payload_end):
        if eid == ebml.CLUSTER and first_cluster is None:
            first_cluster = start
        elif eid == ebml.TAGS and first_cluster is None:
            saw_wurld_tag_before_cluster = True

    if first_cluster is None:
        r.note("9", "no Clusters: header-only file")
        return
    if not saw_wurld_tag_before_cluster:
        r.warn("9", "no Tags element before the first Cluster; a progressive reader "
                    "has no calibration until it has fetched media")


def format_report(path: str | Path, findings: list[Finding]) -> str:
    lines = [f"{path}"]
    if not findings:
        lines.append("  conforms to SPEC (no findings)")
        return "\n".join(lines)
    for f in findings:
        lines.append(f"  {f}")
    counts = {}
    for f in findings:
        counts[f.severity] = counts.get(f.severity, 0) + 1
    lines.append("  " + ", ".join(f"{n} {s}{'s' if n != 1 else ''}"
                                  for s, n in sorted(counts.items())))
    return "\n".join(lines)
