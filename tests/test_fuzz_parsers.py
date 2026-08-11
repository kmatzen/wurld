"""Malformed input must be rejected, not crash, hang, or read past the buffer.

An interchange format receives files from strangers by definition, so these
paths are adversarial by default. The contract asserted here is narrow on
purpose: a parser may raise ValueError (or struct.error, which is the same thing
from struct.unpack) and nothing else. IndexError or TypeError would mean it
acted on a length it never checked — the difference between "this file is bad"
and "this library is bad".

Cases are mutations of a real file plus pure random bytes, with a fixed seed so
a failure reproduces. Wall-clock guards catch a parser that loops instead of
returning.
"""

import random
import signal
import struct

import numpy as np
import pytest

import wurld as wl
from wurld import container, ebml, validate as v

ACCEPTABLE = (ValueError, struct.error)
CASE_TIMEOUT_S = 5


class _Hang(Exception):
    pass


def _on_alarm(_sig, _frame):
    raise _Hang()


@pytest.fixture(scope="module")
def seed_bytes(tmp_path_factory):
    """A real file to mutate — random bytes alone rarely reach the deeper parsers."""
    path = tmp_path_factory.mktemp("fuzz") / "seed.wurld.webm"
    H, W, N = 32, 40, 8
    cams = {"0": wl.Camera(model="PINHOLE", width=W, height=H,
                           params=[30.0, 30.0, W / 2, H / 2])}
    frames = [wl.Frame(i=i, t=i / 30, camera="0", q_wxyz=(1.0, 0.0, 0.0, 0.0),
                       tr=(0.01 * i, 0.0, 0.5)) for i in range(N)]
    rng = np.random.default_rng(0)
    wl.write(path, cameras=cams, frames=frames,
             rgb=rng.integers(0, 255, (N, H, W, 4), dtype=np.uint8),
             signals={"depth": np.full((N, H, W), 3000, np.uint16)},
             specs={"depth": {"inverse_depth": True, "near": 0.3, "far": 9.0}},
             fps=30)
    return path.read_bytes()


def _mutate(rng, data):
    b = bytearray(data)
    for _ in range(rng.randint(1, 24)):
        if not b:
            break
        p = rng.randrange(len(b))
        kind = rng.random()
        if kind < 0.45:
            b[p] = rng.randrange(256)              # flip a byte
        elif kind < 0.65:
            b = b[:p]                              # truncate mid-element
        elif kind < 0.85:
            b[p:p + 8] = bytes([0x01] + [0xff] * 7)  # a wild 8-byte length vint
        else:
            q = min(len(b), p + rng.randint(1, 64))
            b[p:p] = b[p:q]                        # duplicate a run
    return bytes(b)


def _targets(buf):
    return {
        "read_all_tags": lambda: ebml.read_all_tags(buf),
        "segment_bounds": lambda: ebml._segment_bounds(buf),
        "read_cues": lambda: ebml.read_cues(buf),
        "iter_children": lambda: list(ebml.iter_children(buf, 0, len(buf))),
        "read_seek_head": lambda: ebml.read_seek_head(buf, 0, len(buf)),
        "unpack_frames": lambda: container.unpack_frames(buf, ["0"]),
    }


@pytest.mark.parametrize("name", sorted(_targets(b"")))
def test_parsers_reject_malformed_input_cleanly(name, seed_bytes):
    rng = random.Random(20260808)
    prev = signal.signal(signal.SIGALRM, _on_alarm)
    try:
        for n in range(250):
            buf = (bytes(rng.randrange(256) for _ in range(rng.randint(0, 200)))
                   if n % 5 == 0 else _mutate(rng, seed_bytes))
            signal.setitimer(signal.ITIMER_REAL, CASE_TIMEOUT_S)
            try:
                _targets(buf)[name]()
            except ACCEPTABLE:
                pass
            except _Hang:
                pytest.fail(f"{name} did not return within {CASE_TIMEOUT_S}s "
                            f"on a {len(buf)}-byte input")
            except Exception as e:  # noqa: BLE001
                pytest.fail(f"{name} raised {type(e).__name__} ({e}) instead of "
                            f"rejecting a {len(buf)}-byte input with ValueError")
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
    finally:
        signal.signal(signal.SIGALRM, prev)


def test_vint_bounds(seed_bytes):
    """The specific hole the fuzzer found: a vint reader indexing past the end."""
    with pytest.raises(ValueError):
        ebml._read_vint(b"", 0, keep_marker=True)
    with pytest.raises(ValueError):
        ebml._read_vint(b"\x1a", 5, keep_marker=True)
    # A 4-byte ID with only two bytes left must be refused, not read short.
    with pytest.raises(ValueError):
        ebml._read_vint(b"\x1a\x45", 0, keep_marker=True)


def test_camera_index_beyond_the_declared_cameras(seed_bytes):
    """A binary table may claim any camera index; the reader must not index with it."""
    rec = container._FRAME_RECORD.pack(0, 7, 0.0, 1, 0, 0, 0, 0, 0, 0, 1)
    with pytest.raises(ValueError, match="camera index 7"):
        container.unpack_frames(rec, ["0"])


def test_validator_reports_rather_than_raises(tmp_path, seed_bytes):
    """The tool for diagnosing broken files must survive them."""
    import json
    tags = dict(ebml.read_all_tags(seed_bytes))
    doc = json.loads(tags["WURLD"])
    doc["frames"] = []
    doc["frames_binary"] = {"version": 1, "count": 1, "cameras": ["0"]}
    tags["WURLD"] = json.dumps(doc, separators=(",", ":"))
    tags["WURLD_FRAMES"] = container._FRAME_RECORD.pack(0, 7, 0.0, 1, 0, 0, 0, 0, 0, 0, 1)
    p = tmp_path / "badcam.wurld.webm"
    p.write_bytes(ebml.insert_header_tags(seed_bytes, tags))

    findings = v.validate(p)          # must not raise
    assert any(f.severity == v.ERROR and "camera index 7" in f.message for f in findings)


def test_validator_survives_mutated_files(tmp_path, seed_bytes):
    rng = random.Random(99)
    prev = signal.signal(signal.SIGALRM, _on_alarm)
    p = tmp_path / "case.wurld.webm"
    try:
        for _ in range(120):
            p.write_bytes(_mutate(rng, seed_bytes))
            signal.setitimer(signal.ITIMER_REAL, CASE_TIMEOUT_S)
            try:
                v.validate(p)
            except _Hang:
                pytest.fail("validate() did not return within the timeout")
            except Exception as e:  # noqa: BLE001
                pytest.fail(f"validate() raised {type(e).__name__} ({e}) instead of "
                            f"reporting findings")
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
    finally:
        signal.signal(signal.SIGALRM, prev)
