import json

import chromapakz as cz
import numpy as np
import pytest

import wurld as wl
from wurld import conventions, ebml
from wurld.converters import colmap, detect, nerfstudio, tum
from tests.conftest import NEAR


# ---------- EBML layer ----------

def test_tag_roundtrip_preserves_chromapakz(scene):
    data = cz.encode({"depth": scene["d16"][:2]}, rgb=scene["rgba"][:2])
    doc = json.dumps({"format": "wurld", "n": 1})
    tagged = ebml.append_tag(data, "WURLD", doc)
    assert ebml.read_tag(tagged, "WURLD") == doc
    assert ebml.read_tag(tagged, "CHROMAPAKZ") is not None
    assert ebml.read_tag(tagged, "NOPE") is None
    out = cz.decode(tagged)
    assert np.array_equal(out["signals"]["depth"], scene["d16"][:2])


def test_tag_unicode_and_large():
    data = cz.encode({"x": np.zeros((1, 16, 16), np.uint16)})
    doc = json.dumps({"s": "καμερα 📷", "big": "x" * 300_000})
    tagged = ebml.append_tag(data, "WURLD", doc)
    assert ebml.read_tag(tagged, "WURLD") == doc


# ---------- conventions ----------

def test_quat_matrix_roundtrip():
    rng = np.random.default_rng(7)
    for _ in range(50):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        if q[0] < 0:
            q = -q
        R = conventions.quat_wxyz_to_matrix(q)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
        q2 = conventions.matrix_to_quat_wxyz(R)
        assert np.allclose(q, q2, atol=1e-9)


def test_gl_cv_is_involution():
    rng = np.random.default_rng(3)
    q = rng.normal(size=4)
    m = conventions.pose_to_matrix(q / np.linalg.norm(q), [1, 2, 3])
    assert np.allclose(conventions.c2w_gl_to_cv(conventions.c2w_cv_to_gl(m)), m)


def test_invert_pose():
    q = conventions.matrix_to_quat_wxyz(conventions.quat_wxyz_to_matrix([0.9, 0.1, 0.3, 0.2]))
    m = conventions.pose_to_matrix(q, [4, -2, 0.5])
    assert np.allclose(m @ conventions.invert_pose(m), np.eye(4), atol=1e-12)


def test_camera_axes_rdf():
    # A camera at origin looking down world +X, RDF: +Z forward maps to +X world.
    from wurld.synthetic import look_at_c2w

    c2w = look_at_c2w(np.zeros(3), np.array([1.0, 0.0, 0.0]))
    assert np.allclose(c2w[:3, 2], [1, 0, 0])  # forward
    assert np.allclose(c2w[:3, 1], [0, 0, -1])  # +Y down (world -Z)


# ---------- container ----------

def test_container_roundtrip(scene, wl_file):
    seq = wl.read(wl_file)
    assert seq.n_frames == 10
    assert np.array_equal(seq.signal("depth"), scene["d16"])
    for i in (0, 5, 9):
        assert np.allclose(seq.c2w(i), scene["frames"][i].c2w, atol=1e-12)
        assert seq.frames[i].t == scene["frames"][i].t
    dm = seq.depth_meters(0)
    valid = ~np.isnan(dm)
    ref = scene["depth_m"][0]
    assert np.nanmax(np.abs(dm[valid] - ref[valid])) < 0.05  # inverse-depth quantization error
    assert seq.world["metric_scale"] is True
    assert seq.cameras["0"].K[0, 0] == pytest.approx(0.75 * 128)


def test_validation_rejects_bad_pose(scene, tmp_path):
    bad = [wl.Frame(i=0, t=0.0, q_wxyz=(2.0, 0, 0, 0), tr=(0, 0, 0))]
    with pytest.raises(ValueError, match="not unit"):
        wl.write(tmp_path / "x.webm", cameras=scene["cameras"], frames=bad, rgb=scene["rgba"][:1])


def test_validation_rejects_nonmonotonic_t(scene, tmp_path):
    f = scene["frames"]
    bad = [f[0], wl.Frame(i=1, t=f[0].t - 1.0, q_wxyz=f[1].q_wxyz, tr=f[1].tr)]
    with pytest.raises(ValueError, match="timestamp decreases"):
        wl.write(tmp_path / "x.webm", cameras=scene["cameras"], frames=bad, rgb=scene["rgba"][:2])


def test_plain_chromapakz_reads_as_unposed(tmp_path, scene):
    p = tmp_path / "plain.webm"
    p.write_bytes(cz.encode({"depth": scene["d16"][:2]}, rgb=scene["rgba"][:2]))
    seq = wl.read(p)
    assert seq.frames == [] and seq.cameras == {}
    assert np.array_equal(seq.signal("depth"), scene["d16"][:2])


def test_info(wl_file):
    d = wl.info(wl_file)
    assert d["video"]["frames"] == 10
    assert d["wurld"]["posed_frames"] == 10
    assert d["chromapakz_signals"] == ["depth"]


# ---------- converters ----------

def test_tum_roundtrip(wl_file, scene, tmp_path):
    out = tmp_path / "tum"
    tum.to_tum(wl_file, out)
    assert detect(out) == "tum"
    back = tmp_path / "back.wl.webm"
    tum.from_tum(out, back, camera=scene["cameras"]["0"])
    seq = wl.read(back)
    assert len(seq.frames) == 10
    for i in (0, 4, 9):
        # poses go through text serialization (6 decimals) -> micro-level tolerance
        assert np.allclose(seq.c2w(i), scene["frames"][i].c2w, atol=1e-4)
        assert seq.frames[i].t == pytest.approx(scene["frames"][i].t, abs=1e-6)
    # depth: wl (inverse_depth map) -> TUM PNGs (1/5000 m) -> wl linear map
    dm_back = seq.depth_meters(0)
    dm_orig = scene["depth_m"][0]
    both = ~np.isnan(dm_back) & ~np.isnan(dm_orig)
    assert np.abs(dm_back[both] - dm_orig[both]).max() < 0.05


def test_tum_native_depth_bit_exact(tmp_path, scene, wl_file):
    # TUM -> wl -> TUM: raw u16 must be preserved exactly (linear native units).
    t1 = tmp_path / "t1"
    tum.to_tum(wl_file, t1)
    w2 = tmp_path / "w2.wl.webm"
    tum.from_tum(t1, w2, camera=scene["cameras"]["0"])
    t2 = tmp_path / "t2"
    tum.to_tum(w2, t2)
    from PIL import Image

    d1 = sorted((t1 / "depth").iterdir())
    d2 = sorted((t2 / "depth").iterdir())
    assert len(d1) == len(d2) == 10
    for a, b in zip(d1, d2):
        assert np.array_equal(np.asarray(Image.open(a)), np.asarray(Image.open(b)))


def test_transforms_roundtrip(wl_file, scene, tmp_path):
    out = tmp_path / "ns"
    nerfstudio.to_transforms(wl_file, out)
    assert detect(out) == "nerfstudio"
    doc = json.loads((out / "transforms.json").read_text())
    assert len(doc["frames"]) == 10 and "depth_file_path" in doc["frames"][0]
    back = tmp_path / "back.wl.webm"
    nerfstudio.from_transforms(out / "transforms.json", back)
    seq = wl.read(back)
    for i in (0, 3, 9):
        assert np.allclose(seq.c2w(i), scene["frames"][i].c2w, atol=1e-9)
    dm = seq.depth_meters(0)
    ref = scene["depth_m"][0]
    both = ~np.isnan(dm) & ~np.isnan(ref)
    assert np.abs(dm[both] - ref[both]).max() < 0.05  # mm rounding + inverse-depth quant


def test_colmap_roundtrip(wl_file, scene, tmp_path):
    out = tmp_path / "cm"
    colmap.to_colmap(wl_file, out)
    assert detect(out) == "colmap"
    back = tmp_path / "back.wl.webm"
    colmap.from_colmap(out, out / "images", back)
    seq = wl.read(back)
    assert seq.world["metric_scale"] is False  # COLMAP scale is arbitrary
    for i in (0, 5, 9):
        assert np.allclose(seq.c2w(i), scene["frames"][i].c2w, atol=1e-9)
    assert seq.cameras[next(iter(seq.cameras))].params == scene["cameras"]["0"].params


def test_colmap_bin_parsing(tmp_path, scene, wl_file):
    # to_colmap writes text; verify our bin reader against a bin model we synthesize
    # from the text model via struct packing of the same data.
    import struct

    out = tmp_path / "cm"
    colmap.to_colmap(wl_file, out, write_images=False)
    cams = colmap.read_cameras(out / "sparse" / "0")
    images = colmap.read_images(out / "sparse" / "0")
    base = tmp_path / "bin" / "sparse" / "0"
    base.mkdir(parents=True)
    with open(base / "cameras.bin", "wb") as f:
        f.write(struct.pack("<Q", len(cams)))
        for k, cam in cams.items():
            f.write(struct.pack("<iiQQ", int(k), colmap._MODEL_IDS[cam.model], cam.width, cam.height))
            f.write(struct.pack(f"<{len(cam.params)}d", *cam.params))
    with open(base / "images.bin", "wb") as f:
        f.write(struct.pack("<Q", len(images)))
        for im in images:
            f.write(struct.pack("<i", im["id"]))
            f.write(struct.pack("<4d", *im["qvec"]))
            f.write(struct.pack("<3d", *im["tvec"]))
            f.write(struct.pack("<i", int(im["camera"])))
            f.write(im["name"].encode() + b"\x00")
            f.write(struct.pack("<Q", 0))
    cams2 = colmap.read_cameras(base)
    images2 = colmap.read_images(base)
    assert {k: c.to_json() for k, c in cams2.items()} == {k: c.to_json() for k, c in cams.items()}
    assert [(i["qvec"], i["tvec"], i["name"]) for i in images2] == [
        (i["qvec"], i["tvec"], i["name"]) for i in images
    ]


# ---------- CLI ----------

def test_cli_demo_info_extract(tmp_path, capsys):
    from wurld.cli import main

    demo = tmp_path / "demo.wl.webm"
    assert main(["demo", str(demo), "--frames", "6", "--width", "96", "--height", "72"]) == 0
    capsys.readouterr()  # clear demo output
    assert main(["info", str(demo)]) == 0
    info_doc = json.loads(capsys.readouterr().out)
    assert info_doc["video"]["frames"] == 6
    assert main(["extract", str(demo), str(tmp_path / "tum_out"), "--format", "tum"]) == 0
    assert (tmp_path / "tum_out" / "groundtruth.txt").exists()
    back = tmp_path / "back.wl.webm"
    assert main(["convert", str(tmp_path / "tum_out"), str(back)]) == 0
    seq = wl.read(back)
    assert len(seq.frames) == 6
