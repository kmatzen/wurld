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

        euroc.from_euroc(src, args.out, rgb_kbps=args.rgb_kbps,
                         stereo=not args.mono)
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


def _vtt_timestamp(seconds: float) -> str:
    seconds = max(0.0, seconds)
    h, rem = divmod(seconds, 3600.0)
    m, s = divmod(rem, 60.0)
    return f"{int(h):02d}:{int(m):02d}:{s:06.3f}"


def _cmd_validate(args) -> int:
    """Check files against SPEC. Exit 1 if any MUST is violated."""
    from . import validate as v

    worst = 0
    for path in args.files:
        findings = v.validate(path)
        print(v.format_report(path, findings))
        if any(f.severity == v.ERROR for f in findings):
            worst = 1
        elif args.strict and findings:
            worst = 1
    return worst


def _cmd_poses(args) -> int:
    """Export poses as text, for readers that will not install anything.

    ffmpeg cannot see the binary pose tables (its Matroska demuxer drops
    TagBinary), so files written live or over ~10k frames are opaque to it.
    This writes the same poses as WebVTT or CSV; the WebVTT form can be muxed
    back in as a real subtitle track, which ffmpeg *can* read.
    """
    from . import container

    seq = container.read(args.file)
    frames = [f for f in seq.frames if f.pose_valid]
    if not frames:
        print(f"error: {args.file} carries no valid poses", file=sys.stderr)
        return 1

    fmt = args.format
    if fmt is None:
        fmt = "csv" if args.out and str(args.out).endswith(".csv") else "vtt"

    out = open(args.out, "w") if args.out else sys.stdout
    try:
        if fmt == "csv":
            print("i,t,qw,qx,qy,qz,tx,ty,tz,camera", file=out)
            for f in frames:
                q, tr = f.q_wxyz, f.tr
                print(f"{f.i},{f.t!r},{q[0]!r},{q[1]!r},{q[2]!r},{q[3]!r},"
                      f"{tr[0]!r},{tr[1]!r},{tr[2]!r},{f.camera}", file=out)
        else:
            # Cue times are rebased to the video timeline: sensor clocks are
            # absolute (ARKit hands out device uptime), and a cue at 71877s
            # would sit far past the end of the media. The true timestamp is
            # kept in the cue text, which stays authoritative.
            t0 = frames[0].t
            print("WEBVTT", file=out)
            print(f"NOTE wurld poses: camera_to_world, RDF axes, wxyz, metres, "
                  f"t in seconds (absolute; cues rebased by {t0!r})", file=out)
            for n, f in enumerate(frames):
                nxt = frames[n + 1].t if n + 1 < len(frames) else f.t + 1.0 / 30.0
                q, tr = f.q_wxyz, f.tr
                print(file=out)
                print(f"{_vtt_timestamp(f.t - t0)} --> {_vtt_timestamp(max(nxt - t0, f.t - t0 + 1e-3))}",
                      file=out)
                print(f"i={f.i} t={f.t!r} camera={f.camera} "
                      f"q_wxyz={q[0]!r},{q[1]!r},{q[2]!r},{q[3]!r} "
                      f"tr={tr[0]!r},{tr[1]!r},{tr[2]!r}", file=out)
    finally:
        if args.out:
            out.close()
    if args.out:
        print(f"wrote {len(frames)} poses to {args.out}", file=sys.stderr)
    return 0


def _cmd_pose_track(args) -> int:
    """Add a WebVTT pose track, preserving the tags ffmpeg would otherwise drop.

    Remuxing through ffmpeg by hand is destructive here: its Matroska demuxer
    never sees TagBinary, so a live-recorded file loses WURLD_POSES and
    WURLD_FRAMES and reads back with zero poses. This exports the track, remuxes,
    then re-injects every tag from the original and verifies the result before
    writing it.
    """
    import shutil
    import subprocess
    import tempfile

    from . import container, ebml

    if shutil.which("ffmpeg") is None:
        print("error: ffmpeg not found on PATH (needed to mux the track)", file=sys.stderr)
        return 2

    src = Path(args.file)
    original = src.read_bytes()
    tags = ebml.read_all_tags(original)
    before = len(container.read(src).frames)

    with tempfile.TemporaryDirectory() as tmp:
        vtt = Path(tmp) / "poses.vtt"
        rc = _cmd_poses(argparse.Namespace(file=str(src), out=str(vtt), format="vtt"))
        if rc != 0:
            return rc
        remuxed = Path(tmp) / "remuxed.webm"
        proc = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(src), "-i", str(vtt),
             "-map", "0", "-map", "1", "-c", "copy", "-c:s", "webvtt",
             "-metadata:s:s:0", "title=wurld-poses", "-y", str(remuxed)],
            capture_output=True, text=True)
        if proc.returncode != 0:
            print(f"error: ffmpeg failed: {proc.stderr.strip()[:400]}", file=sys.stderr)
            return 1
        # ffmpeg keeps TagString and silently drops TagBinary; put everything back.
        out_bytes = ebml.insert_header_tags(remuxed.read_bytes(), tags)

    Path(args.out).write_bytes(out_bytes)
    after = len(container.read(args.out).frames)
    if after != before:
        print(f"error: pose count changed ({before} -> {after}); refusing to keep {args.out}",
              file=sys.stderr)
        Path(args.out).unlink(missing_ok=True)
        return 1
    print(f"wrote {args.out}: {after} poses intact, plus a readable pose track",
          file=sys.stderr)
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


def _cmd_ros2(args) -> int:
    from .converters import ros2

    if args.direction == "export":
        out = ros2.to_rosbag2(args.src, args.out, storage=args.storage,
                              depth=not args.no_depth, images=not args.no_images)
        print(f"{out}: rosbag2 ({args.storage}) — /tf, camera_info, image_raw"
              f"{'' if args.no_depth else ', depth'}")
    else:
        from .container import read as _read

        out = ros2.from_rosbag2(args.src, args.out)
        seq = _read(out)
        posed = sum(1 for f in seq.frames if f.pose_valid)
        print(f"{out}: {len(seq.frames)} frames ({posed} posed), "
              f"{len(seq.cameras)} camera(s)")
    return 0


def _cmd_index(args) -> int:
    from .collection import build_manifest

    out = Path(args.out)
    manifest, failures = build_manifest(
        args.sources,
        pattern=args.pattern,
        checksum=args.checksum,
        relative_to=None if args.absolute else out.resolve().parent,
        on_error="skip" if args.skip_errors else "raise",
        description=args.description,
    )
    manifest.write(out)
    print(f"{out}: {len(manifest.members)} members, {manifest.total_frames} frames "
          f"({manifest.total_posed_frames} posed)")
    for uri, why in failures:
        print(f"  skipped {uri}: {why}", file=sys.stderr)
    # Unreadable members are a partial result, not a success.
    return 1 if failures else 0


def _cmd_collection(args) -> int:
    from .collection import Collection

    c = Collection.read(args.manifest)
    m = c.manifest
    print(f"{args.manifest}: {len(m.members)} members, {len(c)} frames "
          f"({m.total_posed_frames} posed)")
    if m.description:
        print(f"  {m.description}")
    cams = sorted({c for mem in m.members for c in mem.cameras})
    sigs = sorted({s for mem in m.members for s in mem.signals})
    sizes = sorted({(mem.width, mem.height) for mem in m.members})
    scales = sorted({str(mem.metric_scale) for mem in m.members})
    print(f"  cameras: {', '.join(cams) or '-'}")
    print(f"  signals: {', '.join(sigs) or '-'}")
    print(f"  resolutions: {', '.join(f'{w}x{h}' for w, h in sizes)}")
    print(f"  metric_scale: {', '.join(scales)}")
    if len(scales) > 1:
        print("  warning: mixed metric_scale — consumers requiring metres must filter",
              file=sys.stderr)
    if args.members:
        for i, mem in enumerate(m.members):
            print(f"  [{i}] {mem.uri}  {mem.frames} frames ({mem.posed_frames} posed)"
                  f"  {mem.width}x{mem.height}")

    if args.verify:
        drift = c.verify(checksum=args.checksum)
        if drift:
            print(f"\n{len(drift)} problem(s) — this manifest no longer describes "
                  "its files:", file=sys.stderr)
            for d in drift:
                print(f"  {d}", file=sys.stderr)
            print("  rebuild with: wurld index ...", file=sys.stderr)
            return 1
        print(f"  verified: {len(m.members)} members match their headers")
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
    p_conv.add_argument("--mono", action="store_true", help="EuRoC: carry cam0's pixels only, halving the file (calibration for both cameras is kept either way)")
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
    p_demo.add_argument("out", nargs="?", default="demo.wurld.webm")
    p_demo.add_argument("--frames", type=int, default=90)
    p_demo.add_argument("--width", type=int, default=480)
    p_demo.add_argument("--height", type=int, default=360)
    p_demo.add_argument("--rgb-kbps", type=int, default=4000)
    p_demo.set_defaults(func=_cmd_demo)

    p_val = sub.add_parser(
        "validate", help="check files against SPEC (exit 1 on a MUST violation)")
    p_val.add_argument("files", nargs="+")
    p_val.add_argument("--strict", action="store_true",
                       help="also fail on warnings and notes")
    p_val.set_defaults(func=_cmd_validate)

    p_poses = sub.add_parser(
        "poses", help="export poses as WebVTT or CSV (readable without wurld)")
    p_poses.add_argument("file")
    p_poses.add_argument("-o", "--out", help="output path (default: stdout)")
    p_poses.add_argument("--format", choices=["vtt", "csv"],
                         help="default: csv if -o ends in .csv, else vtt")
    p_poses.set_defaults(func=_cmd_poses)

    p_pt = sub.add_parser(
        "pose-track", help="copy a file, adding a WebVTT pose track ffmpeg can read")
    p_pt.add_argument("file")
    p_pt.add_argument("out")
    p_pt.set_defaults(func=_cmd_pose_track)

    p_ros = sub.add_parser(
        "ros2", help="convert to or from a rosbag2 (real ROS 2 messages)")
    p_ros.add_argument("direction", choices=["export", "import"])
    p_ros.add_argument("src", help="wurld file (export) or rosbag2 directory (import)")
    p_ros.add_argument("out", help="rosbag2 directory (export) or wurld file (import)")
    p_ros.add_argument("--storage", choices=["mcap", "sqlite3"], default="mcap")
    p_ros.add_argument("--no-depth", action="store_true",
                       help="skip the 32FC1 depth topic")
    p_ros.add_argument("--no-images", action="store_true",
                       help="poses and calibration only")
    p_ros.set_defaults(func=_cmd_ros2)

    p_idx = sub.add_parser(
        "index", help="build a collection manifest over many wurld files")
    p_idx.add_argument("sources", nargs="+",
                       help="files, directories (globbed recursively), or http(s) urls")
    p_idx.add_argument("-o", "--out", default="collection.json", help="manifest path")
    p_idx.add_argument("--pattern", default="*.webm",
                       help="glob used inside directories (covers .wurld.webm and plain .webm)")
    p_idx.add_argument("--checksum", action="store_true",
                       help="record sha256 per member (reads every byte; slow)")
    p_idx.add_argument("--absolute", action="store_true",
                       help="store absolute paths instead of paths relative to the manifest")
    p_idx.add_argument("--skip-errors", action="store_true",
                       help="report unreadable members instead of failing")
    p_idx.add_argument("--description", default="")
    p_idx.set_defaults(func=_cmd_index)

    p_cinfo = sub.add_parser("collection", help="summarize a collection manifest")
    p_cinfo.add_argument("manifest")
    p_cinfo.add_argument("--members", action="store_true", help="list every member")
    p_cinfo.add_argument("--verify", action="store_true",
                         help="re-read every member's header and report drift")
    p_cinfo.add_argument("--checksum", action="store_true",
                         help="with --verify, also compare recorded sha256 (reads every byte)")
    p_cinfo.set_defaults(func=_cmd_collection)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
