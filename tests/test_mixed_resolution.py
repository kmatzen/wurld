"""Per-stream resolution (SPEC §4.6, chromapakz format v4, >= 0.10.0).

A signal may be stored at its own resolution — the motivating case is a
256x192 LiDAR depth map beside full-resolution RGB, with neither resampled
to the other. These tests cover both write paths, the reader accessors that
make such a file usable (per-signal resolution, scaled intrinsics), the
document self-description, and validation.
"""

import json

import chromapakz as cz
import numpy as np
import pytest

import wurld as wl
from wurld import ebml, validate
from wurld.container import Camera, Frame, SignalMeta
from wurld.stream import StreamWriter

NEAR, FAR = 0.5, 40.0
N, H, W = 6, 96, 128    # rgb / file geometry
SH, SW = 24, 32         # depth geometry


def _scene():
    rng = np.random.default_rng(7)
    rgba = rng.integers(0, 255, (N, H, W, 4), dtype=np.uint8)
    rgba[..., 3] = 255
    d16 = rng.integers(1, 60000, (N, SH, SW), dtype=np.uint16)
    cameras = {"0": Camera("PINHOLE", W, H, [100.0, 100.0, 63.5, 47.5])}
    frames = [Frame(i, i / 30.0, "0", (1.0, 0, 0, 0), (0, 0, float(i)))
              for i in range(N)]
    return rgba, d16, cameras, frames


def _meta(**geom):
    return [SignalMeta("depth", "depth",
                       {"type": "inverse_depth", "near": NEAR, "far": FAR},
                       **geom)]


@pytest.fixture(scope="module")
def mixed_file(tmp_path_factory):
    rgba, d16, cameras, frames = _scene()
    path = tmp_path_factory.mktemp("mixed") / "mixed.wurld.webm"
    wl.write(path, cameras=cameras, frames=frames, rgb=rgba,
             signals={"depth": d16},
             specs={"depth": cz.inverse_depth_spec(NEAR, FAR)},
             signal_meta=_meta())
    return path, rgba, d16


def test_batch_roundtrip_keeps_each_streams_shape(mixed_file):
    path, rgba, d16 = mixed_file
    seq = wl.read(path)
    assert seq.rgb.shape == (N, H, W, 4)
    assert seq.signal("depth").shape == (N, SH, SW)
    np.testing.assert_array_equal(seq.signal("depth"), d16)


def test_signal_resolution_comes_from_the_codec_metadata(mixed_file):
    seq = wl.read(mixed_file[0])
    assert seq.signal_resolution("depth") == (SW, SH)
    # The file pair stays the primary display resolution.
    assert (seq.probe["width"], seq.probe["height"]) == (W, H)
    with pytest.raises(KeyError):
        seq.signal_resolution("nope")


def test_K_scales_to_a_signals_own_grid(mixed_file):
    seq = wl.read(mixed_file[0])
    K = seq.K("0")
    Kd = seq.K("0", signal_id="depth")
    sx, sy = SW / W, SH / H
    np.testing.assert_allclose(Kd[0], K[0] * sx)
    np.testing.assert_allclose(Kd[1], K[1] * sy)
    assert Kd[2, 2] == 1.0


def test_write_fills_signal_geometry_into_the_document(mixed_file):
    """The caller never mentioned geometry; the document still states it."""
    doc = json.loads(ebml.read_all_tags(mixed_file[0].read_bytes())["WURLD"])
    (entry,) = doc["signals"]
    assert (entry["width"], entry["height"]) == (SW, SH)


def test_mixed_resolution_file_validates_clean(mixed_file):
    findings = validate.validate(mixed_file[0])
    assert [f for f in findings if f.severity == validate.ERROR] == []


def test_declared_geometry_must_match_the_plane(tmp_path):
    rgba, d16, cameras, frames = _scene()
    with pytest.raises(ValueError, match="declares"):
        wl.write(tmp_path / "bad.wurld.webm", cameras=cameras, frames=frames,
                 rgb=rgba, signals={"depth": d16},
                 specs={"depth": cz.inverse_depth_spec(NEAR, FAR)},
                 signal_meta=_meta(width=SW * 2, height=SH * 2))


def test_signal_meta_geometry_comes_together_or_not_at_all():
    with pytest.raises(ValueError, match="together"):
        SignalMeta("depth", "depth", width=32)


def test_streaming_writer_declares_signal_geometry(tmp_path):
    rgba, d16, cameras, frames = _scene()
    path = tmp_path / "stream.wurld.webm"
    with open(path, "wb") as fh:
        w = StreamWriter(fh.write, cameras=cameras,
                         signal_meta=_meta(width=SW, height=SH), has_rgb=True)
        for i, f in enumerate(frames):
            w.add_frame(f, rgb=rgba[i], signals={"depth": {"u16": d16[i]}})
        w.finish()
    seq = wl.read(path)
    assert seq.rgb.shape == (N, H, W, 4)
    np.testing.assert_array_equal(seq.signal("depth"), d16)
    assert seq.signal_resolution("depth") == (SW, SH)
    assert len(seq.frames) == N
    findings = validate.validate(path)
    assert [f for f in findings if f.severity == validate.ERROR] == []


def test_streaming_writer_rejects_a_wrong_size_plane(tmp_path):
    rgba, d16, cameras, frames = _scene()
    w = StreamWriter(lambda b: None, cameras=cameras,
                     signal_meta=_meta(width=SW, height=SH), has_rgb=True)
    with pytest.raises(Exception):
        w.add_frame(frames[0], rgb=rgba[0],
                    signals={"depth": {"u16": np.zeros((H, W), np.uint16)}})


def test_multi_stream_cameras_calibrate_at_their_own_resolution(tmp_path):
    """A stereo-style rig whose guide stream is smaller than the primary."""
    rng = np.random.default_rng(3)
    big = rng.integers(0, 255, (N, H, W, 4), dtype=np.uint8)
    small = rng.integers(0, 255, (N, H // 2, W // 2, 4), dtype=np.uint8)
    big[..., 3] = 255
    small[..., 3] = 255
    cameras = {
        "cam0": Camera("PINHOLE", W, H, [100.0, 100.0, 63.5, 47.5]),
        "cam1": Camera("PINHOLE", W // 2, H // 2, [50.0, 50.0, 31.5, 23.5]),
    }
    frames = [Frame(i, i / 30.0, "cam0", (1.0, 0, 0, 0), (0, 0, float(i)))
              for i in range(N)]
    path = tmp_path / "rig.wurld.webm"
    wl.write(path, cameras=cameras, frames=frames,
             rgb={"cam0": big, "cam1": small})
    seq = wl.read(path)
    assert seq.rgb_for("cam0").shape == (N, H, W, 4)
    assert seq.rgb_for("cam1").shape == (N, H // 2, W // 2, 4)
    findings = validate.validate(path)
    assert [f for f in findings if f.severity == validate.ERROR] == []


def test_multi_stream_camera_size_mismatch_is_rejected(tmp_path):
    rng = np.random.default_rng(3)
    big = rng.integers(0, 255, (N, H, W, 4), dtype=np.uint8)
    small = rng.integers(0, 255, (N, H // 2, W // 2, 4), dtype=np.uint8)
    cameras = {
        "cam0": Camera("PINHOLE", W, H, [100.0, 100.0, 63.5, 47.5]),
        # Calibrated at the primary size, but its stream is the small one.
        "cam1": Camera("PINHOLE", W, H, [100.0, 100.0, 63.5, 47.5]),
    }
    frames = [Frame(i, i / 30.0, "cam0", (1.0, 0, 0, 0), (0, 0, float(i)))
              for i in range(N)]
    with pytest.raises(ValueError, match="cam1"):
        wl.write(tmp_path / "bad.wurld.webm", cameras=cameras, frames=frames,
                 rgb={"cam0": big, "cam1": small})


def test_uniform_files_carry_no_geometry_keys(wl_file):
    """A file whose streams share the resolution stays exactly as before."""
    doc = json.loads(ebml.read_all_tags(wl_file.read_bytes())["WURLD"])
    (entry,) = doc["signals"]
    assert "width" not in entry and "height" not in entry
    seq = wl.read(wl_file)
    assert seq.signal_resolution("depth") == (seq.probe["width"], seq.probe["height"])
