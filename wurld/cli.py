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
    elif kind == "stray":
        from .converters import stray

        stray.from_stray(src, args.out, at=args.at, rgb_kbps=args.rgb_kbps)
    elif kind == "polycam":
        from .converters import polycam

        polycam.from_polycam(src, args.out, at=args.at, rgb_kbps=args.rgb_kbps)
    elif kind == "record3d":
        from .converters import record3d

        record3d.from_record3d(src, args.out, at=args.at, rgb_kbps=args.rgb_kbps)
    elif kind == "euroc":
        from .converters import euroc

        euroc.from_euroc(src, args.out, rgb_kbps=args.rgb_kbps)
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
    elif args.format == "mcap":
        from .converters import mcap_export

        mcap_export.to_mcap(args.file, args.out)
    print(f"extracted {args.file} -> {args.out} ({args.format})")
    return 0


def _cmd_trim(args) -> int:
    import numpy as np

    from . import container

    try:
        a, b = (int(v) if v else None for v in args.frames.split(":"))
    except ValueError:
        print("error: --frames must be START:STOP (e.g. 30:120)", file=sys.stderr)
        return 2
    seq = container.read(args.file)
    a = a or 0
    b = seq.probe["frames"] if b is None else b

    rgb_frames, signal_frames = [], {}
    for _, frame in seq.iter_frames(a, b):
        if frame["rgb"] is not None:
            rgb_frames.append(frame["rgb"])
        for sid, arr in frame["signals"].items():
            signal_frames.setdefault(sid, []).append(arr)
    if not rgb_frames and not signal_frames:
        print(f"error: no frames in range {a}:{b}", file=sys.stderr)
        return 2

    frames = [
        container.Frame(i=f.i - a, t=f.t, camera=f.camera, q_wxyz=f.q_wxyz,
                        tr=f.tr, pose_valid=f.pose_valid, params=f.params)
        for f in seq.frames if a <= f.i < b
    ]
    # rebuild chromapakz encode specs from the source quantization
    specs = {}
    for sig in seq.probe.get("signals", []):
        q = sig.get("quant") or {}
        if q.get("type") == "inverse-depth":
            specs[sig["id"]] = {"inverse_depth": True, "near": q["near"],
                                "far": q["far"], "levels": q.get("levels", 65536)}
    imu = []
    if frames:
        t0, t1 = frames[0].t, frames[-1].t
        for stream_id, stream in seq.imu.items():
            keep = stream.samples[(stream.samples[:, 0] >= t0) & (stream.samples[:, 0] <= t1)]
            if keep.size:
                imu.append(container.ImuStream(stream_id, keep, rate_hz=stream.rate_hz,
                                               extrinsics=stream.extrinsics,
                                               description=stream.description))

    container.write(
        args.out,
        cameras=seq.cameras,
        frames=frames,
        rgb=np.stack(rgb_frames) if rgb_frames else None,
        signals={sid: np.stack(v) for sid, v in signal_frames.items()} or None,
        specs=specs or None,
        signal_meta=seq.signals,
        rigs=seq.rigs,
        imu=imu or None,
        fps=seq.probe["fps"],
        world={**seq.world,
               "description": f"{seq.world.get('description', '')} [trimmed {a}:{b} of "
                              f"{seq.probe['frames']} frames]".strip()},
    )
    print(f"wrote {args.out} (frames {a}:{b}, {len(frames)} posed)")
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
    p_conv.add_argument("--from", choices=["tum", "nerfstudio", "colmap", "stray", "polycam", "record3d", "euroc"], default=None, help="override auto-detection")
    p_conv.add_argument("--images", default=None, help="COLMAP images directory (default: <source>/images)")
    p_conv.add_argument("--at", choices=["depth", "rgb"], default="depth", help="Stray/Polycam: resample RGB to the depth grid (default) or depth to the RGB grid")
    p_conv.add_argument("--fps", type=float, default=30.0, help="synthesized fps for timestamp-less sources")
    p_conv.add_argument("--rgb-kbps", type=int, default=4000)
    p_conv.set_defaults(func=_cmd_convert)

    p_ext = sub.add_parser("extract", help="extract a wurld file back to a dataset layout")
    p_ext.add_argument("file")
    p_ext.add_argument("out")
    p_ext.add_argument("--format", choices=["tum", "transforms", "colmap", "mcap"], required=True)
    p_ext.set_defaults(func=_cmd_extract)

    p_trim = sub.add_parser("trim", help="cut a frame range into a new wurld file")
    p_trim.add_argument("file")
    p_trim.add_argument("out")
    p_trim.add_argument("--frames", required=True, help="START:STOP frame range (STOP exclusive)")
    p_trim.set_defaults(func=_cmd_trim)

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
