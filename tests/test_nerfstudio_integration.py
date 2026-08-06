import numpy as np
import pytest

import wurld as wl
from wurld import conventions

pytest.importorskip("nerfstudio")

from wurld.integrations.nerfstudio_parser import WurldDataParserConfig  # noqa: E402


def test_dataparser_outputs(wl_file, scene):
    config = WurldDataParserConfig(data=wl_file)
    outputs = config.setup()._generate_dataparser_outputs()
    seq = wl.read(wl_file)

    assert len(outputs.image_filenames) == 10
    assert all(p.exists() for p in outputs.image_filenames)
    assert len(outputs.metadata["depth_filenames"]) == 10
    assert outputs.metadata["depth_unit_scale_factor"] == pytest.approx(1e-3)
    assert float(outputs.cameras.fx[0]) == pytest.approx(seq.K("0")[0, 0])

    # nerfstudio GL c2w converts back to the file's canonical RDF pose (f32 tensor)
    c2w_gl = np.eye(4)
    c2w_gl[:3, :4] = outputs.cameras.camera_to_worlds[3].numpy()
    assert np.abs(conventions.c2w_gl_to_cv(c2w_gl) - seq.c2w(3)).max() < 1e-5

    # cached depth PNGs carry millimeters
    from PIL import Image

    d = np.asarray(Image.open(outputs.metadata["depth_filenames"][0])).astype(np.float64) * 1e-3
    ref = seq.depth_meters(0)
    both = (d > 0) & ~np.isnan(ref)
    assert np.abs(d[both] - ref[both]).max() < 1.5e-3

    # second parse reuses the cache
    assert len(config.setup()._generate_dataparser_outputs().image_filenames) == 10
