"""wurld CLI: convert, info, extract."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _cmd_info(args) -> int:
    from . import container

    print(json.dumps(container.info(args.file), indent=2))
    return 0


def _cmd_convert(args) -> int:
    from . import converters

    kind = args.__dict__["from"] or converters.detect(args.source)
    if kind is None:
        print(f"error: could not detect source format of {args.source}; pass --from", file=sys.stderr)
        return 2
    src = Path(args.source)
    if kind == "tum":
        from .converters import tum

        tum.from_tum(src, args.out, rgb_kbps=args.rgb_kbps)
    elif kind == "nerfstudio":
        from .converters import nerfstudio

        nerfstudio.from_transforms(src, args.out, fps=args.fps, rgb_kbps=args.rgb_kbps)
    elif kind == "colmap":
        from .converters import colmap

        images = args.images or (src / "images")
        colmap.from_colmap(src, images, args.out, fps=args.fps, rgb_kbps=args.rgb_kbps)
    else:
        print(f"error: unknown source format {kind!r}", file=sys.stderr)
        return 2
    print(f"wrote {args.out} (from {kind})")
    return 0


def _cmd_extract(args) -> int:
    if args.format == "tum":
        from .converters import tum

        tum.to_tum(args.file, args.out)
    elif args.format == "transforms":
        from .converters import nerfstudio

        nerfstudio.to_transforms(args.file, args.out)
    elif args.format == "colmap":
        from .converters import colmap

        colmap.to_colmap(args.file, args.out)
    print(f"extracted {args.file} -> {args.out} ({args.format})")
    return 0


def _cmd_demo(args) -> int:
    import chromapakz as cz
    import numpy as np

    from . import container
    from .synthetic import make_sequence

    rgb, depth_m, cameras, frames = make_sequence(args.frames, args.width, args.height)
    near, far = 0.5, 40.0
    z = np.where(depth_m > 0, np.clip(depth_m, near, far), np.nan)
    d16 = cz.quantize_inverse(z, near=near, far=far)
    rgba = np.concatenate([rgb, np.full(rgb.shape[:3] + (1,), 255, np.uint8)], -1)
    container.write(
        args.out,
        cameras=cameras,
        frames=frames,
        rgb=rgba,
        signals={"depth": d16},
        specs={"depth": cz.inverse_depth_spec(near, far)},
        signal_meta=[
            container.SignalMeta(
                "depth", "depth", {"type": "inverse_depth", "near": near, "far": far, "levels": 65536, "invalid": 0}
            )
        ],
        world={"metric_scale": True, "gravity_in_world": [0, 0, -1], "description": "wurld synthetic demo scene"},
        rgb_kbps=args.rgb_kbps,
    )
    print(f"wrote {args.out}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="wurld", description="Posed sensor video in one playable WebM")
    sub = p.add_subparsers(dest="command", required=True)

    p_info = sub.add_parser("info", help="summarize a wurld file")
    p_info.add_argument("file")
    p_info.set_defaults(func=_cmd_info)

    p_conv = sub.add_parser("convert", help="convert a dataset (TUM / transforms.json / COLMAP) to wurld")
    p_conv.add_argument("source")
    p_conv.add_argument("out")
    p_conv.add_argument("--from", choices=["tum", "nerfstudio", "colmap"], default=None, help="override auto-detection")
    p_conv.add_argument("--images", default=None, help="COLMAP images directory (default: <source>/images)")
    p_conv.add_argument("--fps", type=float, default=30.0, help="synthesized fps for timestamp-less sources")
    p_conv.add_argument("--rgb-kbps", type=int, default=4000)
    p_conv.set_defaults(func=_cmd_convert)

    p_ext = sub.add_parser("extract", help="extract a wurld file back to a dataset layout")
    p_ext.add_argument("file")
    p_ext.add_argument("out")
    p_ext.add_argument("--format", choices=["tum", "transforms", "colmap"], required=True)
    p_ext.set_defaults(func=_cmd_extract)

    p_demo = sub.add_parser("demo", help="write a synthetic demo sequence")
    p_demo.add_argument("out", nargs="?", default="demo.wl.webm")
    p_demo.add_argument("--frames", type=int, default=90)
    p_demo.add_argument("--width", type=int, default=480)
    p_demo.add_argument("--height", type=int, default=360)
    p_demo.add_argument("--rgb-kbps", type=int, default=4000)
    p_demo.set_defaults(func=_cmd_demo)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
