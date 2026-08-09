# Extracting wurld data without installing wurld

Everything below uses only `ffmpeg`/`ffprobe` plus `python3` for arithmetic. No
`pip install`, no `npm install`. The point of putting the metadata in standard
Matroska tags was that standard tools can read it — this file is that claim,
checked.

`$F` is a `.wl.webm` file throughout.

## What is in the file

```sh
ffprobe -hide_banner "$F"
```

Streams are self-describing — the titles tell you which is which:

```
Stream #0:0: Video: vp9, 640x480   title: rgb
Stream #0:1: Video: vp9, 640x480   title: signal-depth-hi
Stream #0:2: Video: vp9, 640x480   title: signal-depth-lo
Stream #0:3: Video: vp9, 640x480   title: signal-confidence-hi
Stream #0:4: Video: vp9, 640x480   title: signal-confidence-lo
```

Each `uint16` signal is stored as two lossless 8-bit planes (see
[depth](#depth-in-metres)). The RGB stream is ordinary VP9 — any player shows it.

## The metadata document

All calibration and semantics live in one Matroska tag:

```sh
ffprobe -v error -show_entries format_tags=WURLD -of default=nw=1:nk=1 "$F"
```

That prints the JSON document: `cameras`, `signals`, `conventions`, `world`, and
(for most files) the per-frame `frames` array. Pipe it to `python3 -m json.tool`
to read it, or to `jq`.

## Intrinsics

```sh
ffprobe -v error -show_entries format_tags=WURLD -of default=nw=1:nk=1 "$F" \
  | python3 -c '
import json,sys
d = json.load(sys.stdin)
for cid, c in d["cameras"].items():
    print(cid, c["model"], f'"'"'{c["width"]}x{c["height"]}'"'"', c["params"])
print(d["conventions"])'
```

```
0 PINHOLE 640x480 [480.0, 480.0, 319.5, 239.5]
{'camera_axes': 'RDF', 'pose_direction': 'camera_to_world',
 'quaternion_order': 'wxyz', 'units': 'meters', 'timestamp_units': 'seconds'}
```

`params` follows COLMAP ordering for the named model — `PINHOLE` is
`[fx, fy, cx, cy]`. The conventions are fixed by the spec, not negotiable per
file; they are written out so you can confirm rather than assume.

## Poses, as CSV

```sh
ffprobe -v error -show_entries format_tags=WURLD -of default=nw=1:nk=1 "$F" \
  | python3 -c '
import json,sys
d = json.load(sys.stdin)
print("i,t,qw,qx,qy,qz,tx,ty,tz")
for f in d["frames"]:
    if f.get("pose_valid") is False: continue
    print(f["i"], f["t"], *f["q_wxyz"], *f["tr"], sep=",")' > poses.csv
```

Quaternion is `wxyz`, translation is metres, and the pose is **camera-to-world**
with **RDF** camera axes (x right, y down, z forward — OpenCV/COLMAP). `t` is the
sensor timestamp in seconds and may be non-uniform; do not assume `i / fps`.

> **When `d["frames"]` is empty.** Files written by a streaming writer — anything
> recorded live, including the iPhone app — and any file over ~10k frames store
> poses in a compact binary tag (`WURLD_FRAMES`, 45 bytes per frame) instead of
> the JSON array. ffmpeg's Matroska demuxer maps `TagString` into metadata and
> **skips `TagBinary` entirely**, so those poses are invisible to it. Use the
> pose track below, or open the file in the
> [browser viewer](https://kmatzen.com/wurld/viewer/), which parses the binary
> table client-side.

## Poses as a subtitle track

Files may also carry poses as a WebVTT track — the same approach GoPro (GPMF),
MISB KLV and Apple's `mebx` take for per-frame metadata, because tags are for
file-level data and tracks are for per-frame data. ffmpeg reads it directly:

```sh
ffmpeg -i "$F" -map 0:s:0 -c copy poses.vtt
```

```
00:00.000 --> 00:00.033
i=0 t=71877.482882 camera=0 q_wxyz=0.646373,0.636629,-0.321544,-0.271136 tr=-0.296528,0.285562,0.575724
```

Readable with no parser at all. Cue times are rebased to the video timeline
because sensor clocks are absolute (ARKit hands out device uptime); the `t=`
field carries the true timestamp and is authoritative.

Add the track to a file that lacks one:

```sh
wurld pose-track in.wl.webm out.wl.webm      # needs ffmpeg on PATH
```

> **Do not do this by hand with `ffmpeg -i in.webm -i poses.vtt -c copy`.**
> Because ffmpeg cannot see `TagBinary`, remuxing drops `WURLD_POSES` and
> `WURLD_FRAMES` — a live-recorded file comes out the other side reading as
> **zero poses**, with the data surviving only as subtitle text. `wurld
> pose-track` remuxes, re-injects every original tag, and verifies the pose
> count round-trips before it will write the output.

## RGB frames

```sh
ffmpeg -i "$F" -map 0:v:0 frames/rgb_%05d.png      # stills
ffmpeg -i "$F" -map 0:v:0 -c copy rgb.webm          # just the colour track
```

Stream 0 is plain VP9. Nothing wurld-specific is involved.

## Depth, in metres

Depth is a `uint16` code per pixel, split across two lossless 8-bit planes with a
triangle fold (the low plane is reversed on odd high values, which keeps it
smooth and so compressible). Reconstruct the code, then dequantize with the
`value_map` from the metadata.

**Select the planes by title, not by index.** Track numbering is not fixed:
`0:v:1` is `signal-depth-hi` on a plain RGB+depth file, but on a stereo capture
it is the *second camera's colour*, and on a file that also carries confidence
the depth planes sit elsewhere again. Nothing errors — you get a colour plane
reinterpreted as depth codes, which dequantizes into plausible nonsense.

```sh
# What this file actually carries, in order:
ffprobe -v error -show_entries stream=index:stream_tags=title -of csv=p=0 "$F"
# 0,rgb
# 1,rgb-cam1            <- a stereo capture; not present on a mono one
# 2,signal-depth-hi
# 3,signal-depth-lo

hi=$(ffprobe -v error -show_entries stream=index:stream_tags=title -of csv=p=0 "$F" \
     | awk -F, '$2=="signal-depth-hi"{print $1}')
lo=$(ffprobe -v error -show_entries stream=index:stream_tags=title -of csv=p=0 "$F" \
     | awk -F, '$2=="signal-depth-lo"{print $1}')

ffmpeg -v error -i "$F" -map 0:"$hi" -frames:v 1 -pix_fmt gray -f rawvideo hi.raw -y
ffmpeg -v error -i "$F" -map 0:"$lo" -frames:v 1 -pix_fmt gray -f rawvideo lo.raw -y
```

Use `-pix_fmt gray -f rawvideo`, not PNG: it avoids any colourspace conversion
touching the values.

```python
import json, subprocess, numpy as np

F = "scene.wl.webm"
doc = json.loads(subprocess.run(
    ["ffprobe", "-v", "error", "-show_entries", "format_tags=WURLD",
     "-of", "default=nw=1:nk=1", F], capture_output=True, text=True).stdout)
vm = next(s for s in doc["signals"] if s["role"] == "depth")["value_map"]
W, H = doc["cameras"]["0"]["width"], doc["cameras"]["0"]["height"]

hi = np.fromfile("hi.raw", np.uint8)[:W * H].reshape(H, W).astype(np.uint16)
lo = np.fromfile("lo.raw", np.uint8)[:W * H].reshape(H, W).astype(np.uint16)

low  = np.where(hi & 1, 255 - lo, lo)        # undo the triangle fold
code = (hi << 8) | low                        # uint16 code; 0 means invalid

near, far, levels = vm["near"], vm["far"], vm.get("levels", 65536)
M, a, b = levels - 2, 1.0 / vm["near"], 1.0 / vm["far"]
depth = np.where(code == 0, np.nan,
                 1.0 / (((code.astype(np.float64) - 1) / M) * (a - b) + b))
```

`depth` is metres, `NaN` where the sensor had no return. Verified against the
reference reader on a 640×480 frame: identical invalid masks, maximum absolute
error **1.5 × 10⁻⁶ m** — float32 rounding, nothing more.

Note the `value_map` is per file. `type: "inverse_depth"` uses the formula above;
a linear map instead carries `scale`/`offset`, in which case
`depth = code * scale + offset` with `code == invalid` meaning no return.

## Confidence and other signals

Same two-plane layout, same fold. Confidence carries
`value_map: {"type": "labels", "labels": {"0": "low", "1": "medium", "2": "high"}}`,
so reconstruct the code exactly as above and look the integer up — no
dequantization.

Map the right streams by title rather than hardcoding indices:

```sh
ffprobe -v error -show_entries stream=index:stream_tags=title \
        -of csv=p=0 "$F"
```

## Timestamps

Take them from the `frames` array (`t`, seconds), not from container timing.
Sensor timestamps are authoritative and frequently non-uniform — a real handheld
capture drops frames, and the gaps are information, not error.

## What you cannot get this way

- **Binary pose tables** — see the limitation above.
- **IMU streams** — stored in binary `WURLD_IMU_<id>` tags, same situation.
- **Rig extrinsics** — these *are* in the JSON document, under `rigs`.

For those, either `pip install wurld` or use the browser viewer, which needs no
install at all. The viewer exports exactly the pieces ffmpeg cannot reach:

| Button | File | Contents |
|---|---|---|
| `poses.csv` | `poses.csv` | `i,t,qw,qx,qy,qz,tx,ty,tz,camera` — one row per **posed** frame, read through the full SPEC §9 precedence chain, so binary tables come out the same as JSON ones |
| `imu.csv` | `imu_<id>.csv` | `t,gx,gy,gz,ax,ay,az` at the IMU's own rate, one file per stream; the button is hidden when the file has no IMU |
| `json` | `wurld.json` | the whole `WURLD` document, including `rigs` and `value_map` |
| `depth.npy` | `depth_NNNNN.npy` | the current frame's **metric** depth, `float32` metres, `NaN` where there was no return — `np.load` reads it directly |

Unposed frames are omitted from `poses.csv` rather than written as identity, and
the `i` column keeps the original frame numbering, so a gap in it is visible
rather than silently renumbered.

The depth export is per-frame by design: it is for checking a frame against
another tool, not for bulk extraction. For a whole sequence use the Python
reader, which streams instead of holding every frame in a tab.
